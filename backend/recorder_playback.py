"""On-demand, bounded NVR playback. No video files are written to disk."""
import asyncio
import re
import string
import threading
import anyio
from datetime import timedelta, timezone
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from database import RecorderConfig
from password_crypto import decrypt_password
from recorder_connection import validated_host

SLOTS = threading.BoundedSemaphore(2)
HIKVISION_PATH = '/Streaming/tracks/{channel}01?starttime={start_device}&endtime={end_device}'
FIELDS = {'channel', 'start_utc', 'end_utc', 'start_local', 'end_local', 'start_epoch', 'end_epoch', 'start_device', 'end_device'}


def playback_mode(config):
    if 'playback_mode' in config: return config['playback_mode']
    vendor = str(config.get('vendor', '')).casefold()
    return 'hikvision' if vendor in ('hikvision', '海康', '海康威视') or config.get('connection_protocol') == 'hikvision_isapi' else 'disabled'


def validate_template(value):
    if not value: return value
    if not value.startswith('/') or value.startswith('//') or re.search(r'[\s#\\]', value):
        raise ValueError('回放模板只填写以 / 开头的相对路径，不含地址、账号或空格')
    fields = []
    for _, field, spec, conversion in string.Formatter().parse(value):
        if field is not None:
            if field not in FIELDS or spec or conversion: raise ValueError('回放模板包含不支持的占位符')
            fields.append(field)
    if not any(f.startswith('start_') for f in fields) or not any(f.startswith('end_') for f in fields):
        raise ValueError('回放模板必须包含开始和结束时间')
    return value


def window(row):
    at = row.alerted_at or row.confirmed_at
    if at is None: raise HTTPException(422, '告警缺少时间，无法查询录像')
    # Historical event timestamps were created with datetime.fromtimestamp (server local time).
    at = at.astimezone(timezone.utc).replace(microsecond=0)
    return at - timedelta(seconds=30), at + timedelta(seconds=30)


def recording_source(db, row):
    match = re.fullmatch(r'recorder:([^:]+):channel:([^:]+)', str(row.source_id or ''))
    if not match: raise HTTPException(422, '此告警未关联录像机通道，无法查询录像')
    recorder = db.query(RecorderConfig).filter(RecorderConfig.id == match[1], RecorderConfig.owner_id == str(row.owner_id)).first()
    if recorder is None: raise HTTPException(404, '告警关联的录像机配置已不存在')
    cfg = recorder.config_json or {}
    if not any(str(c['number']) == match[2] for c in cfg.get('channels', [])):
        raise HTTPException(404, '告警关联的录像机通道已不存在')
    mode = playback_mode(cfg)
    template = HIKVISION_PATH if mode == 'hikvision' else cfg.get('playback_path', '') if mode == 'custom' else ''
    if not template: raise HTTPException(422, '请先在监控配置中设置录像机回放方式')
    try: validate_template(template)
    except ValueError: raise HTTPException(422, '录像机回放模板无效') from None
    return recorder, match[2], template


def playback_url(db, row):
    recorder, channel, template = recording_source(db, row)
    start, end = window(row)
    cfg = recorder.config_json
    local = ZoneInfo(cfg.get('timezone', 'Asia/Shanghai'))
    values = {'channel':quote(channel, safe=''),
              # This recorder's RTSP dialect requires Z syntax but interprets the digits in device local time.
              # Keep actual UTC placeholders unchanged for custom adapters.
              'start_device':start.astimezone(local).strftime('%Y%m%dT%H%M%SZ'),
              'end_device':end.astimezone(local).strftime('%Y%m%dT%H%M%SZ'),
              'start_utc':start.strftime('%Y%m%dT%H%M%SZ'), 'end_utc':end.strftime('%Y%m%dT%H%M%SZ'),
              'start_local':start.astimezone(local).strftime('%Y_%m_%d_%H_%M_%S'),
              'end_local':end.astimezone(local).strftime('%Y_%m_%d_%H_%M_%S'),
              'start_epoch':str(int(start.timestamp())), 'end_epoch':str(int(end.timestamp()))}
    try:
        host = validated_host(cfg['host'])
        password = decrypt_password(recorder.password_ciphertext)
        return f"rtsp://{quote(cfg['username'], safe='')}:{quote(password, safe='')}@{host}:{cfg.get('rtsp_port',554)}{template.format(**values)}"
    except Exception:
        raise HTTPException(503, '录像机地址或凭据不可用，请检查配置') from None


async def stream_recording(url):
    if not SLOTS.acquire(blocking=False): raise HTTPException(429, '录像回放繁忙，请关闭其他回放后重试')
    process = None
    released = False
    async def cleanup():
        nonlocal released
        try:
            with anyio.CancelScope(shield=True):
                if process is not None and process.returncode is None:
                    try: process.kill()
                    except ProcessLookupError: pass
                    await process.wait()
        finally:
            if not released:
                released = True
                SLOTS.release()
    try:
        process = await asyncio.create_subprocess_exec(
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-rtsp_transport', 'tcp',
            '-timeout', '10000000', '-i', url, '-t', '60', '-map', '0:v:0', '-an',
            '-vf', r'scale=trunc(min(1280\,iw)/2)*2:-2', '-c:v', 'libx264', '-preset', 'ultrafast',
            '-threads', '2', '-pix_fmt', 'yuv420p', '-g', '25', '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
            '-f', 'mp4', 'pipe:1', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        first = await asyncio.wait_for(process.stdout.read(65536), timeout=15)
        if not first: raise HTTPException(502, '无法读取该时段录像，请检查录像计划、保留时间及回放权限')
    except BaseException as exc:
        await cleanup()
        if isinstance(exc, (HTTPException, asyncio.CancelledError)): raise
        raise HTTPException(502, '录像机回放连接失败或超时，请检查配置及录像是否存在') from None

    async def chunks():
        try:
            yield first
            async with asyncio.timeout(90):
                while chunk := await asyncio.wait_for(process.stdout.read(65536), timeout=15):
                    yield chunk
        except (TimeoutError, ConnectionError):
            return
        finally: await cleanup()
    return StreamingResponse(chunks(), media_type='video/mp4', background=BackgroundTask(cleanup),
        headers={'Cache-Control':'no-store', 'X-Accel-Buffering':'no', 'Accept-Ranges':'none'})

from urllib.parse import quote
from .streams import latest_frame


def resolve_camera(db, owner_id, recorder_id, channel_number):
    from database import RecorderConfig
    from password_crypto import decrypt_password
    from recorder_connection import validated_host
    recorder = db.query(RecorderConfig).filter(RecorderConfig.id == recorder_id, RecorderConfig.owner_id == str(owner_id)).first()
    if recorder is None:
        raise ValueError('摄像头不存在或无权访问')
    cfg = recorder.config_json
    channel = next((c for c in cfg.get('channels', []) if c['number'] == channel_number), None)
    if not cfg.get('enabled') or not channel or not channel.get('enabled') or not channel.get('rtsp_path'):
        raise ValueError('摄像头未启用或未配置视频路径')
    host = validated_host(cfg['host'])
    url = f"rtsp://{quote(cfg['username'], safe='')}:{quote(decrypt_password(recorder.password_ciphertext), safe='')}@{host}:{cfg['rtsp_port']}{channel['rtsp_path']}"
    name = channel['name']
    db.rollback()
    return url, name


def read_camera(db, owner_id, recorder_id, channel_number):
    url, name = resolve_camera(db, owner_id, recorder_id, channel_number)
    return latest_frame(str(owner_id), recorder_id, channel_number, url), name

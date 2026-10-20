"""Bounded ONVIF and optional vendor management connection checks."""
import ipaddress
import socket
import time
import base64
import hashlib
import os
import subprocess
import sys
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit, quote
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests
from requests.auth import HTTPDigestAuth


class ProbeFailure(Exception):
    def __init__(self, code, message):
        self.code, self.message = code, message


def validated_host(host):
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        raise ProbeFailure('dns_failed', '无法解析录像机地址') from None
    for entry in addresses:
        address = ipaddress.ip_address(entry[4][0])
        if address.is_loopback or address.is_link_local or address.is_multicast or address.is_unspecified:
            raise ProbeFailure('invalid_target', '该地址不能作为录像机地址')
    return f'[{host}]' if ':' in host else host


def read_xml(session, url, expected, soap=None, action=None):
    kwargs = dict(timeout=(3, 4), allow_redirects=False, stream=True)
    response = (session.post(url, data=soap, headers={'Content-Type': f'application/soap+xml; charset=utf-8; action="{action}"'}, **kwargs)
                if soap else session.get(url, **kwargs))
    with response:
        if response.status_code in (401, 403):
            raise ProbeFailure('authentication_failed', '账号密码错误、账号锁定或没有设备接口权限')
        if response.status_code != 200:
            raise ProbeFailure('protocol_error', f'设备接口返回 HTTP {response.status_code}，请确认所选接入协议已启用')
        data = bytearray()
        deadline = time.monotonic() + 5
        for chunk in response.iter_content(8192):
            data.extend(chunk)
            if len(data) > 262144 or time.monotonic() > deadline:
                raise ProbeFailure('response_limit', '设备响应超出限制')
    if b'<!DOCTYPE' in data or b'<!ENTITY' in data:
        raise ProbeFailure('invalid_response', '设备响应格式不符合要求')
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        raise ProbeFailure('invalid_response', '设备未返回有效的协议数据') from None
    if expected == 'Envelope':
        fault = next((n for n in root.iter() if n.tag.split('}')[-1] == 'Fault'), None)
        if fault is not None:
            raise ProbeFailure('onvif_fault', 'ONVIF 拒绝请求，请检查专用账号、权限、设备时间及协议是否启用')
    if root.tag.split('}')[-1] != expected:
        raise ProbeFailure('invalid_response', '设备未返回预期的协议数据')
    return root


def onvif_call(session, url, username, password, operation, service='device', content=None):
    soap_ns = 'http://www.w3.org/2003/05/soap-envelope'
    wsse = 'http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd'
    wsu = 'http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd'
    namespace = f'http://www.onvif.org/ver10/{service}/wsdl'
    envelope = ET.Element(f'{{{soap_ns}}}Envelope')
    header = ET.SubElement(envelope, f'{{{soap_ns}}}Header')
    security = ET.SubElement(header, f'{{{wsse}}}Security', {f'{{{soap_ns}}}mustUnderstand': 'true'})
    token = ET.SubElement(security, f'{{{wsse}}}UsernameToken')
    nonce = os.urandom(20)
    created = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + password.encode()).digest()).decode()
    ET.SubElement(token, f'{{{wsse}}}Username').text = username
    ET.SubElement(token, f'{{{wsse}}}Password', {'Type': 'http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest'}).text = digest
    ET.SubElement(token, f'{{{wsse}}}Nonce', {'EncodingType': 'http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary'}).text = base64.b64encode(nonce).decode()
    ET.SubElement(token, f'{{{wsu}}}Created').text = created
    body = ET.SubElement(envelope, f'{{{soap_ns}}}Body')
    request = ET.SubElement(body, f'{{{namespace}}}{operation}')
    if content:
        ET.SubElement(request, f'{{{namespace}}}{content[0]}').text = content[1]
    response = read_xml(session, url, 'Envelope', ET.tostring(envelope, encoding='utf-8'), namespace + '/' + operation)
    if not any(n.tag == f'{{{namespace}}}{operation}Response' for n in response.iter()):
        raise ProbeFailure('invalid_response', '设备未返回预期的 ONVIF 响应')
    return response


def onvif_connect(session, base, config, password, result):
    endpoint = base + config.get('onvif_path', '/onvif/device_service')
    info = onvif_call(session, endpoint, config['username'], password, 'GetDeviceInformation')
    result['authenticated'] = True
    result['device'] = {n.tag.split('}')[-1]: (n.text or '')[:128] for n in info.iter()
                        if n.tag.split('}')[-1] in ('Manufacturer', 'Model', 'FirmwareVersion')}
    capabilities = onvif_call(session, endpoint, config['username'], password, 'GetCapabilities', content=('Category','Media'))
    media = next((n for n in capabilities.iter() if n.tag.split('}')[-1] == 'Media'), None)
    address = next((n.text for n in media.iter() if n.tag.split('}')[-1] == 'XAddr'), None) if media is not None else None
    if not address:
        raise ProbeFailure('media_unavailable', '设备连接成功，但未提供 ONVIF Media 服务')
    parsed, origin = urlsplit(address), urlsplit(base)
    if parsed.scheme != origin.scheme or parsed.hostname != origin.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ProbeFailure('unsafe_media_endpoint', '设备返回的媒体服务地址不属于配置的录像机或协议不一致')
    profiles = onvif_call(session, address, config['username'], password, 'GetProfiles', service='media')
    result['profiles'] = []
    for profile in profiles.iter():
        if profile.tag.split('}')[-1] != 'Profiles':
            continue
        name = next((n.text for n in profile if n.tag.split('}')[-1] == 'Name'), '')
        result['profiles'].append({'token': profile.attrib.get('token','')[:128], 'name': (name or '')[:100]})
        if len(result['profiles']) >= 256:
            break


RTSP_PROBE = '''
import sys, json, os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
import cv2
cv2.setNumThreads(1)
url = json.load(sys.stdin)['url']
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
try:
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    print(json.dumps({'decoded': bool(ok and frame is not None and frame.size),
                      'width': int(frame.shape[1]) if ok and frame is not None else 0,
                      'height': int(frame.shape[0]) if ok and frame is not None else 0}))
finally:
    cap.release()
'''


def rtsp_connect(config, password, host, result, channel):
    if not channel or not channel.get('rtsp_path'):
        raise ProbeFailure('rtsp_path_required', '该通道未填写 RTSP 路径（含设备要求的查询参数）')
    path = channel['rtsp_path']
    if not path.startswith('/') or any(c.isspace() for c in path) or '#' in path:
        raise ProbeFailure('invalid_rtsp_path', 'RTSP 路径格式不正确')
    url = f"rtsp://{quote(config['username'], safe='')}:{quote(password, safe='')}@{host}:{config['rtsp_port']}{path}"
    result['tested_channel'] = channel['number']
    try:
        process = subprocess.run([sys.executable, '-c', RTSP_PROBE], input=json.dumps({'url': url}),
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15)
        probe = json.loads(process.stdout) if process.returncode == 0 else {}
    except subprocess.TimeoutExpired:
        raise ProbeFailure('rtsp_timeout', 'RTSP 取流超时，请检查网络、端口、路径和设备连接数') from None
    except (ValueError, OSError):
        raise ProbeFailure('rtsp_probe_failed', '服务器无法完成视频解码测试') from None
    if not probe.get('decoded'):
        raise ProbeFailure('rtsp_decode_failed', '未取得可解码画面，请核对账号权限、RTSP 路径、通道在线状态及编码支持')
    result.update(status='connected', code='ok', authenticated=True, rtsp_reachable=True,
                  video_decoded=True, width=probe['width'], height=probe['height'],
                  message='RTSP 取流及画面解码成功')


def rtsp_connect_all(config, password, result):
    channels = [c for c in config.get('channels', []) if c.get('enabled', True)]
    if not channels:
        raise ProbeFailure('no_enabled_channels', '没有启用的通道，请先启用至少一个通道')
    # Resolve once; common target errors still produce a row for every enabled channel.
    host_error = None
    try:
        host = validated_host(config['host'])
    except ProbeFailure as exc:
        host, host_error = None, exc
    def probe(channel):
        item = {'number': channel['number'], 'name': channel.get('name', ''),
                'status': 'failed', 'video_decoded': False}
        try:
            if host_error:
                raise host_error
            rtsp_connect(config, password, host, item, channel)
        except ProbeFailure as exc:
            item.update(code=exc.code, message=exc.message)
        except Exception:
            # Isolate an unexpected decoder error without exposing credential-bearing output.
            item.update(code='rtsp_probe_failed', message='该通道视频测试失败，请重试或检查服务器解码支持')
        return item
    with ThreadPoolExecutor(max_workers=min(8, len(channels))) as pool:
        rows = list(pool.map(probe, channels))
    success = sum(row['status'] == 'connected' for row in rows)
    result.update(channel_results=rows, total=len(rows), succeeded=success, failed=len(rows)-success,
                  status='connected' if success == len(rows) else 'partial' if success else 'failed',
                  code='ok' if success == len(rows) else 'partial_failure' if success else 'all_failed',
                  authenticated=success > 0, rtsp_reachable=success > 0, video_decoded=success == len(rows),
                  message=f'已测试全部 {len(rows)} 个启用通道：成功 {success} 个，失败 {len(rows)-success} 个；录像回放尚未验证')


def connect_recorder(config, password):
    result = {'status': 'failed', 'checked_at': datetime.now(timezone.utc).isoformat(),
              'authenticated': False, 'rtsp_reachable': False, 'channels': [], 'device': {}}
    if not config.get('enabled'):
        return {**result, 'code': 'disabled', 'message': '请先启用录像机配置'}
    try:
        protocol = config.get('connection_protocol', 'rtsp')
        result['protocol'] = protocol
        if protocol == 'rtsp':
            rtsp_connect_all(config, password, result)
            return result
        host = validated_host(config['host'])
        base = f"{config['management_scheme']}://{host}:{config['management_port']}"
        with requests.Session() as session:
            session.trust_env = False
            session.auth = HTTPDigestAuth(config['username'], password)
            if protocol == 'onvif':
                onvif_connect(session, base, config, password, result)
            else:
                hikvision_connect(session, base, result)
        try:
            with socket.create_connection((config['host'], config['rtsp_port']), timeout=3):
                result['rtsp_reachable'] = True
        except OSError:
            pass
        result.update(status='connected', code='ok', message='设备认证及通道查询成功；视频解码与录像回放尚未验证')
    except ProbeFailure as exc:
        result.update(code=exc.code, message=exc.message)
    except requests.exceptions.SSLError:
        result.update(code='tls_error', message='HTTPS 证书验证失败，请配置受信任证书或确认设备实际支持的管理协议')
    except requests.exceptions.Timeout:
        result.update(code='timeout', message='连接录像机超时，请检查网络和管理端口')
    except (requests.exceptions.RequestException, OSError, ValueError):
        result.update(code='connection_failed', message='无法连接管理接口，请核对 HTTP/HTTPS 和管理端口')
    return result


def hikvision_connect(session, base, result):
    info = read_xml(session, base + '/ISAPI/System/deviceInfo', 'DeviceInfo')
    result['authenticated'] = True
    result['device'] = {n.tag.split('}')[-1]: (n.text or '')[:128] for n in info
                        if n.tag.split('}')[-1] in ('model', 'deviceType', 'firmwareVersion')}
    streams = read_xml(session, base + '/ISAPI/Streaming/channels', 'StreamingChannelList')
    grouped = {}
    for stream in streams:
        fields = {n.tag.split('}')[-1]: n.text for n in stream}
        stream_id = fields.get('id', '')
        if not str(stream_id).isdigit() or int(stream_id) < 100:
            continue
        number, kind = divmod(int(stream_id), 100)
        if kind not in (1, 2):
            continue
        item = grouped.setdefault(str(number), {'number': str(number), 'name': (fields.get('channelName') or f'通道 {number}')[:100], 'streams': []})
        item['streams'].append('main' if kind == 1 else 'sub')
    result['channels'] = list(grouped.values())[:256]


def hikvision_channel_inventory(config, password):
    """Read the current Hikvision NVR digital-channel metadata without changing it."""
    host = validated_host(config['host'])
    base = f"http://{host}:80"

    def text(node, name):
        return next((child.text for child in node.iter() if child.tag.split('}')[-1] == name and child.text), '')

    try:
        with requests.Session() as session:
            session.trust_env = False
            session.auth = HTTPDigestAuth(config['username'], password)
            document = read_xml(session, base + '/ISAPI/ContentMgmt/InputProxy/channels', 'InputProxyChannelList')
            states = read_xml(session, base + '/ISAPI/ContentMgmt/InputProxy/channels/status', 'InputProxyChannelStatusList')
    except ProbeFailure:
        raise
    except requests.exceptions.Timeout:
        raise ProbeFailure('timeout', '读取海康通道配置超时') from None
    except (requests.exceptions.RequestException, OSError, ValueError):
        raise ProbeFailure('channel_discovery_failed', '无法读取海康通道配置，请检查录像机网络和账号权限') from None

    status_by_number = {
        text(channel, 'id').strip(): text(channel, 'online').strip().lower()
        for channel in states if channel.tag.split('}')[-1] == 'InputProxyChannelStatus'
    }
    items = []
    for channel in document:
        if channel.tag.split('}')[-1] != 'InputProxyChannel':
            continue
        number = text(channel, 'id').strip()
        if not number:
            continue
        online = status_by_number.get(number, '')
        items.append({
            'number': number[:64],
            'name': (text(channel, 'name') or f'通道 {number}')[:100],
            'ip_address': text(channel, 'ipAddress')[:64],
            'status': 'online' if online == 'true' else 'offline' if online == 'false' else 'unknown',
        })
        if len(items) >= 256:
            break
    return items

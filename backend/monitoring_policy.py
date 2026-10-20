"""Recorder monitoring output policy; never restrict evidence used by rule evaluation."""
import copy
import math
from urllib.parse import quote

from fastapi import HTTPException
from database import RecorderConfig, ModelRegistry
from algorithm_catalog import bindings, enabled_selection, configuration
from password_crypto import decrypt_password, PasswordEncryptionError
from recorder_connection import validated_host, ProbeFailure

EVENTS = ('phone_use', 'missing_badge', 'fall_down')
CLASSES = {
    'person': {'person', 'people', 'human', '人', '人员'},
    'phone': {'phone', 'cell phone', 'cellphone', 'mobilephone', 'mobile phone', 'smartphone', '手机'},
    'badge': {'badge', 'work_badge', 'id_badge', 'id card', '工牌'},
}


def selected_events(value, db=None):
    allowed = [item['code'] for item in configuration(db)['algorithms']] if db is not None else list(EVENTS)
    if not isinstance(value,list) or len(value)>100 or any(not isinstance(v,str) or v not in allowed for v in value):
        raise HTTPException(422, '请选择告警算法目录中的有效告警')
    return list(dict.fromkeys(value))


def filter_detections(detections, events, behavior=None):
    if not events:
        return []
    behavior = behavior or {}
    allowed = set(CLASSES['person'])
    if 'phone_use' in events:
        allowed.update(CLASSES['phone'])
        allowed.update(str(v).strip().casefold() for v in behavior.get('phone_classes', []))
    if 'missing_badge' in events:
        allowed.update(CLASSES['badge'])
        allowed.update(str(v).strip().casefold() for v in behavior.get('badge_classes', []))
    return [copy.deepcopy(d) for d in detections
            if str(d.get('class') or d.get('class_name') or '').strip().casefold() in allowed][:256]


def filter_events(result, events):
    out = dict(result)
    for key in ('action_events', 'active_events', 'alarm_events', 'temporal_shadow_events', 'temporal_score_snapshots'):
        if key in out:
            out[key] = [e for e in out[key] or [] if e.get('event_type') in events]
    return out


def resolve_source(db, source, owner_id):
    row = db.query(RecorderConfig).filter(RecorderConfig.id == source.get('recorder_id'),
                                        RecorderConfig.owner_id == str(owner_id)).first()
    if row is None:
        raise HTTPException(404, '录像机配置不存在或不属于当前账号')
    cfg = row.config_json or {}
    channel = next((c for c in cfg.get('channels', []) if c['number'] == source.get('channel_number')), None)
    if not cfg.get('enabled') or not channel or not channel.get('enabled'):
        raise HTTPException(422, '录像机或摄像头通道已停用')
    events = selected_events(channel.get('algorithms', []), db)
    if not events:
        raise HTTPException(422, '摄像头尚未选择算法')
    from algorithm_catalog import enabled_selection, configuration, bindings
    selected_entries = enabled_selection(db, events)
    algorithm_bindings = bindings(db, events)
    from alarm_flow import flow_model_ids
    algorithm_bindings['flows'] = [copy.deepcopy(item) for item in selected_entries if item.get('flow')]
    algorithm_bindings['model_ids'] = list(dict.fromkeys(model_id for item in selected_entries for model_id in flow_model_ids(item)))
    import continuous_monitoring as cm
    algorithm_bindings['detectors'] = []
    for model_id in algorithm_bindings['model_ids']:
        record = db.get(ModelRegistry, model_id)
        if record:
            algorithm_bindings['detectors'].append(str(cm.runtime.api.resolve_managed_model_path(record.path, record)))
    if any(not flow_model_ids(item) for item in selected_entries):
        raise HTTPException(422, '所选告警尚未配置检测模型，请管理员先完成模型选择')
    pose_model_id = configuration(db).get('pose_model_id')
    if not pose_model_id and any(not item.get('flow') for item in selected_entries):
        raise HTTPException(422, '请管理员在告警算法配置页选择行为姿态模型')
    from inference_service.catalog import ensure_current_models
    algorithm_bindings['revision'] = ensure_current_models(
        db, [*algorithm_bindings['model_ids'], pose_model_id])
    if algorithm_bindings.get('pose'):
        algorithm_bindings['pose_model_id'] = configuration(db).get('pose_model_id')
    path = channel.get('rtsp_path', '')
    if not path.startswith('/') or '#' in path or any(c.isspace() for c in path):
        raise HTTPException(422, '请先配置通道的 RTSP 路径')
    try:
        host = validated_host(cfg['host'])
        password = decrypt_password(row.password_ciphertext)
    except (ProbeFailure, PasswordEncryptionError):
        raise HTTPException(422, '无法读取录像机凭据或地址，请检查监控配置') from None
    return {'type': 'rtsp', 'url': f"rtsp://{quote(cfg['username'], safe='')}:{quote(password, safe='')}@{host}:{cfg['rtsp_port']}{path}",
            'name': f"{cfg['name']} / {channel['name']}", 'recorder_id': row.id,
            'channel_number': channel['number'], 'events': events, 'algorithm_bindings': algorithm_bindings}


def public_event(event):
    return {key: copy.deepcopy(event[key]) for key in ('event_id', 'event_type', 'state', 'should_alert',
            'started_at', 'confirmed_at', 'resolved_at', 'confidence') if key in event}


def public_detections(detections):
    output = []
    for detection in detections[:256]:
        box = detection.get('bbox') or {}
        if isinstance(box, (list, tuple)) and len(box) == 4:
            box = dict(zip(('x1','y1','x2','y2'), box))
        if not isinstance(box, dict) or not all(isinstance(box.get(k), (int,float)) and math.isfinite(box[k]) for k in ('x1','y1','x2','y2')):
            continue
        item = {key: copy.deepcopy(detection[key]) for key in ('class','class_name','confidence','track_id') if key in detection}
        item['bbox'] = {key: box[key] for key in ('x1','y1','x2','y2')}
        output.append(item)
    return output


def behavior_detections(persons, primary, scene, events, behavior):
    detections = list(primary or [])
    if not any(str(d.get('class') or d.get('class_name')).casefold() in CLASSES['person'] for d in detections):
        detections += [{'class':'person', 'bbox':p['bbox'], 'track_id':track, **({'confidence':p['confidence']} if p.get('confidence') is not None else {})}
                       for track,p in persons.items()]
    classes = {str(d.get('class') or d.get('class_name')).casefold() for d in detections}
    detections += [d for d in scene or [] if str(d.get('class') or d.get('class_name')).casefold() not in classes]
    return public_detections(filter_detections(detections, events, behavior))

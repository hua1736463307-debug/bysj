"""User-facing monitoring controls using recorder references, not model or stream URLs."""
import copy
import json
import os
import uuid
from urllib.parse import quote
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from database import ContinuousMonitor, RecorderConfig, get_db
from auth_utils import require_any_menu
from monitoring_policy import selected_events
from monitoring_live_frames import live_frame_pool
import continuous_monitoring as cm

router = APIRouter(prefix='/monitoring', tags=['monitoring-workspace'])
read_access = require_any_menu('/monitoring/config', '/monitoring/live')
algorithm_access = require_any_menu('/monitoring/config')
live_access = require_any_menu('/monitoring/live')


def current(db, user):
    return db.query(ContinuousMonitor).filter(ContinuousMonitor.owner_id == str(user['id'])).first()


def safe_task(row):
    if row is None:
        return None
    out = cm.snapshot(row)
    cfg = row.config_json or {}
    out['recorder_monitor'] = cfg.get('recorder_monitor', False)
    out['sources'] = camera_states(row)
    out['source_count'] = sum(s['desired_state'] == 'running' for s in out['sources'])
    return out


def camera_states(row):
    if row is None:
        return []
    result = (row.result_json or {}).get('cells', {})
    output = []
    for position, source in enumerate((row.config_json or {}).get('video_sources', [])):
        run_id = source.get('run_id', f'legacy-{position}')
        stats = result.get(str(position), result.get(position, {}))
        acknowledged = stats.get('run_id') == run_id
        desired = source.get('desired_state', 'running') if row.desired_state == 'running' else 'stopped'
        if desired == 'stopped':
            status = 'stopped' if row.status == 'stopped' or (acknowledged and stats.get('status') == 'stopped') else 'stopping'
        elif cm.snapshot(row)['status'] == 'recovering':
            status = 'recovering'
        else:
            status = stats.get('status', 'starting') if acknowledged else 'starting'
        output.append({**{k: source[k] for k in ('name','recorder_id','channel_number','events') if k in source},
                       'position': position, 'run_id': run_id, 'desired_state': desired, 'status': status})
    return output


def matches(source, recorder_id, channel_number):
    return source.get('recorder_id') == recorder_id and source.get('channel_number') == channel_number


@router.get('/algorithms')
def list_algorithms(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                    db=Depends(get_db), user=Depends(read_access)):
    query = db.query(RecorderConfig).filter(RecorderConfig.owner_id == str(user['id']))
    rows = query.order_by(RecorderConfig.created_at.desc(), RecorderConfig.id).offset((page-1)*page_size).limit(page_size).all()
    return {'total': query.count(), 'items': [
        {'id': r.id, 'name': r.config_json['name'], 'host': r.config_json['host'], 'enabled': r.config_json['enabled'],
         'channels': [{k: c.get(k, [] if k == 'algorithms' else None) for k in ('number','name','stream','enabled','algorithms')}
                      for c in r.config_json.get('channels', [])]} for r in rows]}


class AlgorithmUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    algorithms: list[str] = Field(max_length=100)


@router.put('/algorithms/{recorder_id}/{channel_number}')
def save_algorithms(recorder_id: str, channel_number: str, body: AlgorithmUpdate, db=Depends(get_db), user=Depends(algorithm_access)):
    events = selected_events(body.algorithms, db)
    cm.admission_lock(db)
    from algorithm_catalog import enabled_selection
    enabled_selection(db, events)
    task = current(db, user)
    if task and (task.desired_state == 'running' or task.status == 'stopping'):
        if not (task.config_json or {}).get('recorder_monitor') or any(
                matches(s, recorder_id, channel_number) and s['status'] != 'stopped' for s in camera_states(task)):
            raise HTTPException(409, '请先停止本路摄像头，待资源释放后修改算法')
    row = db.query(RecorderConfig).filter(RecorderConfig.id == recorder_id, RecorderConfig.owner_id == str(user['id'])).with_for_update().first()
    if row is None:
        raise HTTPException(404, '录像机配置不存在')
    cfg = copy.deepcopy(row.config_json)
    channel = next((c for c in cfg.get('channels', []) if c['number'] == channel_number), None)
    if channel is None:
        raise HTTPException(404, '摄像头通道不存在')
    channel['algorithms'] = events
    row.config_json = cfg
    db.commit()
    return {'algorithms': events}


@router.get('/tasks')
def list_tasks(db=Depends(get_db), user=Depends(read_access)):
    row = current(db, user)
    cfg = (row.config_json if row else {}) or {}
    from algorithm_catalog import configuration
    catalog = configuration(db)
    ready = bool((cfg.get('shared_model') or os.getenv('MONITOR_OBJECT_MODEL') or any(a['enabled'] and a['model_id'] for a in catalog['algorithms'])) and
                 ((cfg.get('shared_behavior_config') or {}).get('pose_model_path') or os.getenv('MONITOR_POSE_MODEL') or catalog['pose_model_id']))
    return {'items': [safe_task(row)] if row else [], 'ready': ready}


class SourceRef(BaseModel):
    model_config = ConfigDict(extra='forbid')
    recorder_id: str
    channel_number: str


class LiveFrameRef(SourceRef):
    run_id: str = Field(default='', max_length=64)
    frame_sequence: int = Field(default=0, ge=0)
    probe_only: bool = False


class LiveFrameRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    cameras: list[LiveFrameRef] = Field(max_length=9)


@router.post('/live-frames')
def live_frames(body: LiveFrameRequest, db=Depends(get_db), user=Depends(live_access)):
    """Preview configured channels without starting an alarm task or running inference."""
    from password_crypto import decrypt_password
    from recorder_connection import validated_host
    recorders = {}
    output = []
    seen = set()
    for ref in body.cameras:
        key = (ref.recorder_id, ref.channel_number)
        if key in seen:
            continue
        seen.add(key)
        recorder = recorders.get(ref.recorder_id)
        if recorder is None:
            recorder = db.query(RecorderConfig).filter(
                RecorderConfig.id == ref.recorder_id,
                RecorderConfig.owner_id == str(user['id']),
            ).first()
            recorders[ref.recorder_id] = recorder
        if recorder is None:
            output.append({**ref.model_dump(include={'recorder_id','channel_number'}), 'status': 'unavailable'})
            continue
        cfg = recorder.config_json or {}
        channel = next((c for c in cfg.get('channels', []) if c.get('number') == ref.channel_number), None)
        path = (channel or {}).get('rtsp_path') or ''
        if not cfg.get('enabled') or not channel or not channel.get('enabled') or not path.startswith('/') or '#' in path or any(c.isspace() for c in path):
            output.append({**ref.model_dump(include={'recorder_id','channel_number'}), 'status': 'unavailable'})
            continue
        try:
            host = validated_host(cfg['host'])
            password = decrypt_password(recorder.password_ciphertext)
            url = f"rtsp://{quote(cfg['username'], safe='')}:{quote(password, safe='')}@{host}:{cfg['rtsp_port']}{path}"
            frame = live_frame_pool.frame(user['id'], ref.recorder_id, ref.channel_number,
                                          url, ref.run_id, ref.frame_sequence, ref.probe_only)
        except Exception:
            frame = {'status': 'unavailable'}
        output.append({**ref.model_dump(include={'recorder_id','channel_number'}), **frame})
    return {'items': output}


class StartTask(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(default='实时监控', min_length=1, max_length=100)
    sources: list[SourceRef] = Field(min_length=1, max_length=9)


@router.post('/tasks/start')
def start_task(body: StartTask, db=Depends(get_db), user=Depends(algorithm_access)):
    cm.admission_lock(db)
    row = current(db, user)
    if row and row.desired_state == 'running':
        return safe_task(row)
    previous = (row.config_json if row else {}) or {}
    from algorithm_catalog import service_defaults
    selected = []
    for source in body.sources:
        recorder = db.query(RecorderConfig).filter(RecorderConfig.id == source.recorder_id, RecorderConfig.owner_id == str(user['id'])).first()
        channel = next((c for c in (recorder.config_json if recorder else {}).get('channels', []) if c['number'] == source.channel_number), None)
        if channel is None:
            raise HTTPException(404, '摄像头配置不存在')
        selected.extend(channel.get('algorithms', []))
    default_model, default_pose = service_defaults(db, list(dict.fromkeys(selected)))
    behavior = copy.deepcopy(previous.get('shared_behavior_config') or {})
    model = default_model or previous.get('shared_model') or os.getenv('MONITOR_OBJECT_MODEL')
    behavior['pose_model_path'] = default_pose or behavior.get('pose_model_path') or os.getenv('MONITOR_POSE_MODEL')
    if not model or not behavior['pose_model_path']:
        raise HTTPException(503, '监控服务尚未配置，请联系管理员完成服务配置')
    if not published.get('pose_model_id'):
        raise HTTPException(503, '请先在告警算法配置页选择姿态模型')
    cfg = {'recorder_monitor': True, 'video_sources': [s.model_dump() for s in body.sources],
           'shared_model': model, 'max_fps': min(8, previous.get('max_fps', 8)),
           'shared_confidence': previous.get('shared_confidence', .25),
           'shared_iou_threshold': previous.get('shared_iou_threshold', .7), 'shared_behavior_config': behavior}
    cm.start_monitor(cm.StartRequest(name=body.name, config=cfg), db, user)
    return safe_task(current(db, user))


@router.post('/tasks/{task_id}/stop')
def stop_task(task_id: str, db=Depends(get_db), user=Depends(algorithm_access)):
    row = current(db, user)
    if row is None or row.id != task_id:
        raise HTTPException(404, '监控任务不存在')
    cm.stop_monitor(task_id, db, user)
    return safe_task(row)


@router.post('/tasks/{task_id}/resume')
def resume_task(task_id: str, db=Depends(get_db), user=Depends(algorithm_access)):
    row = current(db, user)
    if row is None or row.id != task_id:
        raise HTTPException(404, '监控任务不存在')
    if not (row.config_json or {}).get('recorder_monitor'):
        raise HTTPException(422, '旧版任务请在本页选择已配置算法的摄像头后重新启动')
    if (row.config_json or {}).get('per_camera'):
        raise HTTPException(422, '请在监控配置页逐路启动摄像头')
    cm.resume_monitor(task_id, db, user)
    return safe_task(row)


@router.get('/tasks/{task_id}/preview')
def preview(task_id: str, since: str = Query('{}', max_length=2000), db=Depends(get_db), user=Depends(live_access)):
    try:
        known = json.loads(since)
        if not isinstance(known, dict): raise ValueError()
    except (ValueError, TypeError):
        raise HTTPException(422, '预览帧游标无效') from None
    row = current(db, user)
    if row is None or row.id != task_id:
        raise HTTPException(404, '监控任务不存在')
    if row.desired_state != 'running':
        return {'cells': {}}
    data = cm.runtime.preview(task_id)
    active = {s['position']: s for s in camera_states(row) if s['desired_state'] == 'running'}
    def allowed(position, value):
        source = active.get(int(position))
        return source and value.get('run_id', f'legacy-{position}') == source['run_id']
    def delta(position, frame):
        key = f"{frame.get('run_id')}:{frame.get('frame_sequence')}"
        return {'unchanged': True, 'run_id': frame.get('run_id'), 'frame_sequence': frame.get('frame_sequence')} if known.get(str(position)) == key else frame
    return {'cells': {p: delta(p, v) for p, v in data.get('cells', {}).items() if allowed(p, v)},
            'alarms': [v for v in data.get('alarms', []) if allowed(v['position'], v)]}


@router.post('/cameras/{recorder_id}/{channel_number}/start')
def start_camera(recorder_id: str, channel_number: str, db=Depends(get_db), user=Depends(algorithm_access)):
    cm.admission_lock(db)
    recorder = db.query(RecorderConfig).filter(RecorderConfig.id == recorder_id, RecorderConfig.owner_id == str(user['id'])).first()
    if recorder is None or not any(c['number'] == channel_number for c in recorder.config_json.get('channels', [])):
        raise HTTPException(404, '摄像头配置不存在')
    row = db.query(ContinuousMonitor).filter(ContinuousMonitor.owner_id == str(user['id'])).with_for_update().first()
    if row and row.status == 'stopping':
        raise HTTPException(409, '任务正在释放资源，请稍后启动')
    previous = copy.deepcopy((row.config_json if row else {}) or {})
    running = bool(row and row.desired_state == 'running')
    if running and not previous.get('recorder_monitor'):
        raise HTTPException(409, '请先停止旧版监控任务')
    states = camera_states(row)
    same = next((s for s in states if matches(s, recorder_id, channel_number)), None)
    if running and same and same['desired_state'] == 'running':
        return safe_task(row)
    if same and same['status'] == 'stopping':
        raise HTTPException(409, '本路正在释放资源，请稍后启动')
    from algorithm_catalog import bindings
    channel = next(c for c in recorder.config_json['channels'] if c['number'] == channel_number)
    published = bindings(db, channel.get('algorithms', []))
    from inference_service.catalog import current_switch
    published['revision'] = current_switch(db)['revision']
    behavior = copy.deepcopy(previous.get('shared_behavior_config') or {})
    model = next((p for p in published['detectors'] if p), None) or previous.get('shared_model') or os.getenv('MONITOR_OBJECT_MODEL')
    behavior['pose_model_path'] = published['pose'] or behavior.get('pose_model_path') or os.getenv('MONITOR_POSE_MODEL')
    if not model or not behavior['pose_model_path']:
        raise HTTPException(503, '监控服务尚未配置，请联系管理员完成服务配置')
    validated = cm.runtime.validate({**previous, 'recorder_monitor': True, 'shared_model': model,
        'shared_behavior_config': behavior, 'video_sources': [{'recorder_id': recorder_id, 'channel_number': channel_number}]}, db, user)
    source = {**validated['video_sources'][0], 'run_id': uuid.uuid4().hex, 'desired_state': 'running',
              'inference_revision': published['revision']}
    if running:
        sources = previous['video_sources']
        reusable = same['position'] if same else next((s['position'] for s in states if s['status'] == 'stopped'), None)
        if reusable is not None:
            sources[reusable] = source
        elif len(sources) < 9:
            sources.append(source)
        else:
            raise HTTPException(409, '最多同时运行 9 路摄像头，请先停止一路并等待资源释放')
        cfg = previous
    else:
        cfg = validated
        cfg['video_sources'] = [source]
        if row is None:
            row = ContinuousMonitor(owner_id=str(user['id']), owner_username=user['username'], name='摄像头检测')
            db.add(row)
            db.flush()
        row.result_json = {}
    cfg['per_camera'] = True
    if running:
        cm.check_admission(db, row, cfg)
    else:
        cm.reserve(db, row, cfg)
    row.config_json = cfg
    db.commit()
    return safe_task(row)


@router.post('/cameras/{recorder_id}/{channel_number}/stop')
def stop_camera(recorder_id: str, channel_number: str, db=Depends(get_db), user=Depends(algorithm_access)):
    cm.admission_lock(db)
    row = db.query(ContinuousMonitor).filter(ContinuousMonitor.owner_id == str(user['id'])).with_for_update().first()
    if row is None or not any(matches(s, recorder_id, channel_number) for s in camera_states(row)):
        raise HTTPException(404, '本路摄像头没有检测任务')
    cfg = copy.deepcopy(row.config_json)
    for source in cfg['video_sources']:
        if matches(source, recorder_id, channel_number):
            source['desired_state'] = 'stopped'
    cfg['per_camera'] = True
    row.config_json = cfg
    if row.desired_state == 'running' and not any(s.get('desired_state', 'running') == 'running' for s in cfg['video_sources']):
        row.desired_state, row.status = 'stopped', 'stopping'
    db.commit()
    return safe_task(row)

"""Administrator-owned algorithm catalog. Public projection never includes model bindings."""
import copy
import uuid
import datetime
import threading
import subprocess
import sys
import os
import json
from pathlib import PurePosixPath
from sqlalchemy import String, or_
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from database import AlgorithmCatalog, ContinuousMonitor, DatasetVersion, ModelRegistry, RecorderConfig, get_db
from auth_utils import require_admin, require_any_menu
from model_tasks import registry_task

CODES = ('phone_use', 'missing_badge', 'fall_down')
NAMES = ('使用手机', '未佩戴工牌', '跌倒')
LEGACY_CATEGORIES = {'fall_down': 'behavior', 'phone_use': 'object', 'missing_badge': 'violation'}
router = APIRouter(tags=['algorithm-catalog'])
reader = require_any_menu('/monitoring/config', '/monitoring/live')


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    confidence: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(default=0.7, ge=0, le=1)
    detection_fps: int = Field(default=8, ge=1, le=25)
    behavior_fps: int = Field(default=6, ge=1, le=10)
    preview_fps: int = Field(default=8, ge=1, le=25)


def configuration(db):
    row = db.get(AlgorithmCatalog, 'global')
    cfg = copy.deepcopy(row.config_json) if row else {
        'pose_model_id': None,
        'algorithms': [{'code':code, 'name':name, 'category':LEGACY_CATEGORIES[code], 'enabled':True, 'model_id':None} for code, name in zip(CODES, NAMES)]}
    cfg['algorithms'] = [{
        'code': item.get('code', ''),
        'name': item.get('name', ''),
        'category': item.get('category') or LEGACY_CATEGORIES.get(item.get('code')),
        'labels': item.get('labels') or [],
        'enabled': bool(item.get('enabled', True)),
        'model_id': item.get('model_id'),
        **({'flow':copy.deepcopy(item['flow'])} if item.get('flow') else {}),
    } for item in cfg.get('algorithms', [])]
    cfg['runtime'] = RuntimeSettings(**cfg.get('runtime', {})).model_dump()
    cfg.pop('_deleted_algorithms', None)
    cfg.pop('_inference_revision', None)
    return cfg


def enabled_selection(db, events):
    entries = {item['code']:item for item in configuration(db)['algorithms']}
    if any(code not in entries or not entries[code]['enabled'] for code in events):
        raise HTTPException(422, '所选算法已停用，请重新选择并保存本路算法')
    return [entries[code] for code in events]


def model_path(db, model_id, kind='detect'):
    if not model_id:
        return None
    record = db.get(ModelRegistry, model_id)
    if not record or record.status != 'active' or record.source not in ('pretrained','trained') or record.format != 'pt':
        raise HTTPException(422, '请选择有效的已注册 PT 模型')
    task = registry_task(record)
    if task and task not in (('detect', 'segment') if kind=='detect' else (kind,)):
        raise HTTPException(422, '模型任务类型与配置用途不匹配')
    import continuous_monitoring as cm
    if cm.runtime is None:
        raise HTTPException(503, '监控服务尚未初始化')
    return str(cm.runtime.api.resolve_managed_model_path(record.path, record))


def bindings(db, events):
    """Resolve only administrator-published model IDs; callers never supply paths."""
    entries = enabled_selection(db, events)
    try:
        return {'detectors':[model_path(db, model_id, kind) for item in entries for model_id,kind in __import__('alarm_flow').flow_model_specs(item)],
                'pose':model_path(db, configuration(db)['pose_model_id'], 'pose'),
                'face_required_algorithms':[item['code'] for item in entries if item['category'] in ('object','violation')],
                'runtime':configuration(db)['runtime']}
    except Exception:
        # Model resolution may include filesystem paths in errors. Never forward them.
        raise HTTPException(503, '算法服务暂不可用，请联系管理员检查配置') from None


def service_defaults(db, events):
    values = bindings(db, events)
    return next((path for path in values['detectors'] if path), None), values['pose']


@router.get('/monitoring/algorithm-catalog')
def public_catalog(db=Depends(get_db), user=Depends(reader)):
    return {'items':[{k:item[k] for k in ('code','name','category')} for item in configuration(db)['algorithms'] if item['enabled']]}


@router.get('/system/algorithms')
def admin_catalog(db=Depends(get_db), admin=Depends(require_admin)):
    row = db.get(AlgorithmCatalog, 'global')
    return {**configuration(db), 'revision':row.revision if row else 0}


@router.get('/system/algorithms/models')
def admin_models(search: str = Query('', max_length=100), page: int = Query(1, ge=1),
                 db=Depends(get_db), admin=Depends(require_admin)):
    query = db.query(ModelRegistry).filter(ModelRegistry.status == 'active', ModelRegistry.source.in_(['trained','pretrained']), ModelRegistry.format == 'pt')
    if search.strip():
        term = '%'+search.strip()+'%'
        query = query.filter(or_(ModelRegistry.name.ilike(term), ModelRegistry.metadata_json.cast(String).ilike(term), ModelRegistry.path.ilike(term)))
    rows = query.order_by(ModelRegistry.updated_at.desc(), ModelRegistry.id).offset((page-1)*100).limit(100).all()
    return {'total':query.count(), 'items':[model_option(db, r) for r in rows]}


def _ordered_labels(value):
    if isinstance(value, dict):
        def sort_key(item):
            key = str(item[0])
            return (0, int(key)) if key.isdigit() else (1, key.casefold())
        value = [item[1] for item in sorted(value.items(), key=sort_key)]
    if not isinstance(value, (list, tuple, set)):
        return []
    return sorted({str(label).strip() for label in value if str(label).strip()}, key=str.casefold)


def model_labels(db, record):
    metadata = record.metadata_json or {}
    for value in (
        metadata.get('trained_class_names'), metadata.get('class_names'),
        metadata.get('classes'), metadata.get('names'), metadata.get('class_map'),
    ):
        labels = _ordered_labels(value)
        if labels:
            return labels
    if record.source == 'trained' and record.dataset_version_id:
        version = db.get(DatasetVersion, record.dataset_version_id)
        labels = _ordered_labels(version.class_map_json if version else None)
        if labels:
            return labels
    return []


def model_option(db, record):
    metadata = record.metadata_json or {}
    path = PurePosixPath(str(record.path).replace('\\', '/'))
    experiment = metadata.get('experiment_name') or (path.parent.parent.name if path.parent.name == 'weights' else path.parent.name)
    display = metadata.get('display_name') or record.name
    if record.source == 'trained' and display.lower() in ('best', 'last', 'best.pt', 'last.pt', record.name.lower()):
        display = f'{experiment} - {path.stem}' if experiment else record.name
    labels = model_labels(db, record)
    return {'id':record.id, 'name':record.name, 'display_name':display,
            'experiment_name':experiment if record.source == 'trained' else '', 'source':record.source,
            'base_model':PurePosixPath(str(metadata.get('base_model') or '').replace('\\', '/')).name,
            'dataset':PurePosixPath(str(metadata.get('dataset') or '').replace('\\', '/')).name,
            'epochs':metadata.get('epochs') or '',
            'updated_at':record.updated_at.isoformat() if record.updated_at else None,
            'task':registry_task(record), 'labels': labels}


_class_slots = threading.BoundedSemaphore(1)
_class_cache = {}
_class_cache_lock = threading.Lock()


def read_registered_classes(path):
    """Read trusted registered weights in a bounded CPU child, without inference."""
    from pathlib import Path
    path = Path(path)
    stat = path.stat()
    if stat.st_size > 512 * 1024 * 1024:
        raise HTTPException(422, '模型过大，请先完善模型目录中的类别元数据')
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _class_cache_lock:
        if key in _class_cache: return copy.deepcopy(_class_cache[key])
    if not _class_slots.acquire(blocking=False):
        raise HTTPException(503, '模型类别正在读取，请稍后重试')
    try:
        code = "import sys,json;from ultralytics import YOLO;m=YOLO(sys.argv[1]);names=m.names;print('MODEL_CLASSES_RESULT='+json.dumps({'classes':list(names.values()) if isinstance(names,dict) else list(names),'task':m.task}))"
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'}
        result = subprocess.run([sys.executable, '-c', code, str(path)], env=env, capture_output=True, text=True, timeout=45)
        if result.returncode or 'MODEL_CLASSES_RESULT=' not in result.stdout:
            raise HTTPException(422, '无法读取所选模型类别，请检查注册模型文件')
        data = json.loads(result.stdout.rsplit('MODEL_CLASSES_RESULT=',1)[1].splitlines()[0])
        data['classes'] = _ordered_labels(data['classes'])
        if not data['classes']: raise HTTPException(422, '所选模型没有可用识别类别')
        with _class_cache_lock:
            if len(_class_cache) >= 128: _class_cache.pop(next(iter(_class_cache)))
            _class_cache[key] = copy.deepcopy(data)
        return data
    except subprocess.TimeoutExpired:
        raise HTTPException(503, '读取模型类别超时，请稍后重试') from None
    finally:
        _class_slots.release()


@router.get('/system/algorithms/models/{model_id}/classes')
def registered_classes(model_id: str, db=Depends(get_db), admin=Depends(require_admin)):
    record = db.get(ModelRegistry, model_id)
    if not record or record.status != 'active' or record.format != 'pt' or record.source not in ('trained','pretrained'):
        raise HTTPException(404, '模型不存在或不可用')
    classes = model_labels(db, record)
    metadata = record.metadata_json or {}
    if classes: return {'model_id':model_id, 'classes':classes, 'task':registry_task(record), 'source':'registry'}
    try: data = read_registered_classes(model_path(db,model_id,'pose' if registry_task(record)=='pose' else 'detect'))
    except (OSError, ValueError, KeyError):
        raise HTTPException(422, '无法读取注册模型的类别，请检查模型文件和元数据') from None
    return {**data,'model_id':model_id,'source':'registered_weights'}


class AlgorithmEntry(BaseModel):
    model_config = ConfigDict(extra='forbid')
    code: str = Field(min_length=1, max_length=64, pattern=r'^[a-z0-9_]+$')
    name: str = Field(min_length=1, max_length=64)
    category: Literal['behavior','object','violation']
    labels: list[str] = Field(default_factory=list, max_length=64)
    enabled: bool
    model_id: str | None = Field(default=None, max_length=36)
    flow: dict | None = None

    @field_validator('name')
    @classmethod
    def trimmed_name(cls, value):
        if not value.strip(): raise ValueError('告警名称不能为空')
        return value.strip()

    @field_validator('labels')
    @classmethod
    def normalized_labels(cls, value):
        labels = []
        for item in value:
            label = str(item).strip()
            if label and label.casefold() not in {existing.casefold() for existing in labels}:
                labels.append(label)
        return labels


class CatalogUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: int = Field(ge=0)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    pose_model_id: str | None = Field(default=None, max_length=36)
    algorithms: list[AlgorithmEntry] = Field(max_length=100)


class CatalogCreate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    category: Literal['behavior','object','violation']
    name: str = Field(min_length=1, max_length=64)
    model_id: str | None = Field(default=None, max_length=36)
    flow: dict | None = None
    labels: list[str] = Field(default_factory=list, max_length=64)

    @field_validator('name')
    @classmethod
    def trimmed_name(cls, value):
        if not value.strip(): raise ValueError('告警名称不能为空')
        return value.strip()

    @field_validator('labels')
    @classmethod
    def create_labels(cls, value):
        labels = AlgorithmEntry.normalized_labels(value)
        return labels


class CatalogRevision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: int = Field(ge=0)


def editable_catalog(db, revision, *, allow_monitoring=False):
    from inference_service.catalog import inference_switch_lock
    import continuous_monitoring as cm
    if inference_switch_lock.locked(): raise HTTPException(409, '模型正在切换，请稍后重试')
    cm.admission_lock(db)
    if not allow_monitoring and db.query(ContinuousMonitor).filter((ContinuousMonitor.desired_state=='running') | (ContinuousMonitor.status=='stopping')).first():
        raise HTTPException(409, '请先停止监控并等待资源释放，再修改告警配置')
    row=db.query(AlgorithmCatalog).filter(AlgorithmCatalog.id=='global').with_for_update().first()
    if row is None or row.revision!=revision: raise HTTPException(409, '配置已更新，请刷新后重试')
    return row,configuration(db)


def commit_catalog_change(db,row,cfg,admin,*,metadata_only=False):
    if metadata_only:
        from inference_service.catalog import inference_revision
        cfg['_inference_revision']=inference_revision(row)
    else:
        cfg.pop('_inference_revision',None)
    row.config_json=cfg;row.revision+=1;row.updated_by=str(admin['id']);db.commit()
    if not metadata_only:
        from inference_service.catalog import enqueue_configuration,published_models
        try: enqueue_configuration(row.revision,published_models(db))
        except Exception: pass
    return {**configuration(db),'revision':row.revision}


def references_algorithm(value,code):
    if isinstance(value,dict):
        return any(references_algorithm(item,code) for item in value.values())
    if isinstance(value,(list,tuple)):
        return any(references_algorithm(item,code) for item in value)
    return value==code


def ensure_deletable_during_monitoring(db,item):
    import continuous_monitoring as cm
    active=db.query(ContinuousMonitor).filter((ContinuousMonitor.desired_state=='running') | (ContinuousMonitor.status=='stopping')).all()
    if active and item['enabled']:
        raise HTTPException(409,'请先停止监控并等待资源释放，再删除已启用的告警')
    contexts=getattr(getattr(cm.runtime,'api',None),'behavior_session_contexts',{})
    if any(references_algorithm(row.config_json,item['code']) for row in active) or any(
        references_algorithm(context,item['code']) for context in list(contexts.values())
    ):
        raise HTTPException(409,'该告警仍被监控引用，请先停止相关监控并等待资源释放')


@router.get('/system/algorithms/deleted')
def deleted_catalog_entries(db=Depends(get_db), admin=Depends(require_admin)):
    row=db.get(AlgorithmCatalog,'global')
    deleted=(row.config_json or {}).get('_deleted_algorithms',[]) if row else []
    return {'items':[{'code':d['item']['code'],'name':d['item']['name'],'category':d['item']['category'],'deleted_at':d['deleted_at']} for d in deleted]}


@router.delete('/system/algorithms/{code}')
def delete_catalog_entry(code: str, body: CatalogRevision, db=Depends(get_db), admin=Depends(require_admin)):
    row,cfg=editable_catalog(db,body.revision,allow_monitoring=True)
    item=next((a for a in cfg['algorithms'] if a['code']==code),None)
    if item is None: raise HTTPException(404,'告警不存在或已删除')
    for recorder in db.query(RecorderConfig).all():
        if any(code in (channel.get('algorithms') or []) for channel in (recorder.config_json or {}).get('channels',[])):
            raise HTTPException(409,'该告警仍被摄像头绑定，请先在监控配置中取消绑定')
    ensure_deletable_during_monitoring(db,item)
    cfg['algorithms']=[a for a in cfg['algorithms'] if a['code']!=code]
    deleted=copy.deepcopy(row.config_json.get('_deleted_algorithms',[]))
    deleted.append({'item':copy.deepcopy(item),'deleted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deleted_by':str(admin['id'])})
    cfg['_deleted_algorithms']=deleted
    return commit_catalog_change(db,row,cfg,admin,metadata_only=not item['enabled'])


@router.post('/system/algorithms/{code}/restore')
def restore_catalog_entry(code: str, body: CatalogRevision, db=Depends(get_db), admin=Depends(require_admin)):
    row,cfg=editable_catalog(db,body.revision)
    deleted=copy.deepcopy(row.config_json.get('_deleted_algorithms',[]));record=next((d for d in deleted if d['item']['code']==code),None)
    if record is None: raise HTTPException(404,'没有可恢复的告警')
    if len(cfg['algorithms'])>=100: raise HTTPException(422,'告警数量已达上限')
    if any(a['code']==code or a['name'].casefold()==record['item']['name'].casefold() for a in cfg['algorithms']):
        raise HTTPException(409,'已有同名告警，请先修改名称再恢复')
    item=copy.deepcopy(record['item']);item['enabled']=False;cfg['algorithms'].append(item)
    cfg['_deleted_algorithms']=[d for d in deleted if d['item']['code']!=code]
    return commit_catalog_change(db,row,cfg,admin)


@router.post('/system/algorithms')
def create_catalog_entry(body: CatalogCreate, db=Depends(get_db), admin=Depends(require_admin)):
    from inference_service.catalog import inference_switch_lock
    if inference_switch_lock.locked():
        raise HTTPException(409, '模型仍在切换中，请等待完成后再新增告警')
    active = db.query(ContinuousMonitor).filter((ContinuousMonitor.desired_state == 'running') | (ContinuousMonitor.status == 'stopping')).first()
    if active:
        raise HTTPException(409, '请先停止监控并等待资源释放，再修改告警配置')
    row = db.query(AlgorithmCatalog).filter(AlgorithmCatalog.id == 'global').with_for_update().first()
    cfg = configuration(db)
    if len(cfg['algorithms']) >= 100:
        raise HTTPException(422, '告警数量已达上限')
    if any(item['name'].casefold() == body.name.casefold() for item in cfg['algorithms']):
        raise HTTPException(422, '告警名称不能重复')
    if body.flow:
        if any(n.get('config',{}).get('classifier') for n in body.flow.get('nodes',[])):
            raise HTTPException(422,'请在单帧姿态分类中测试后启用倒地分类器')
        validate_registered_flow(db, body.flow, body.category)
    if body.category in {'object', 'violation'} and body.labels and not body.model_id and not body.flow:
        raise HTTPException(422, '请选择可提供标注标签的检测模型')
    if body.category in {'object', 'violation'} and body.model_id and not body.labels and not body.flow:
        raise HTTPException(422, '至少选择一个标注标签')
    if not body.flow: model_path(db, body.model_id)
    item = {'code': f'alarm_{uuid.uuid4().hex}', 'name': body.name, 'category': body.category,
            'labels': [] if body.flow else body.labels, 'enabled': False, 'model_id': None if body.flow else body.model_id, **({'flow':body.flow} if body.flow else {})}
    cfg['algorithms'].append(item)
    if row is None:
        row = AlgorithmCatalog(id='global', config_json=cfg, revision=1, updated_by=str(admin['id']))
        db.add(row)
    else:
        if row.config_json.get('_deleted_algorithms'): cfg['_deleted_algorithms'] = copy.deepcopy(row.config_json['_deleted_algorithms'])
        row.config_json = cfg; row.revision += 1; row.updated_by = str(admin['id'])
    db.commit()
    from inference_service.catalog import enqueue_configuration, published_models
    try:
        operation_id = enqueue_configuration(row.revision, published_models(db)) if any(algorithm.get('enabled') for algorithm in cfg['algorithms']) else None
    except Exception:
        operation_id = None
    return {'catalog': {**configuration(db), 'revision': row.revision}, 'created': item,
            'inference_switch': {'operation_id': operation_id, 'state': 'loading' if operation_id else 'unchanged'}}


@router.put('/system/algorithms')
def save_catalog(body: CatalogUpdate, db=Depends(get_db), admin=Depends(require_admin)):
    from inference_service.catalog import inference_switch_lock
    if inference_switch_lock.locked():
        raise HTTPException(409, '模型仍在切换中，请等待完成后再保存新配置')
    import continuous_monitoring as cm
    cm.admission_lock(db)
    row = db.query(AlgorithmCatalog).filter(AlgorithmCatalog.id == 'global').with_for_update().first()
    if body.revision != (row.revision if row else 0):
        raise HTTPException(409, '配置已被其他管理员修改，请刷新后重试')
    old = configuration(db)
    incoming_codes = [item.code for item in body.algorithms]
    if len(set(incoming_codes)) != len(incoming_codes):
        raise HTTPException(422, '告警标识不能重复')
    if set(incoming_codes) != {item['code'] for item in old['algorithms']}:
        raise HTTPException(422, '请通过新增告警创建配置项，现有告警不可在保存时删除')
    if len({item.name.casefold() for item in body.algorithms}) != len(body.algorithms):
        raise HTTPException(422, '告警名称不能重复')
    incoming = {item.code:{k:v for k,v in item.model_dump().items() if k!='flow' or v is not None} for item in body.algorithms}
    from pose_training.inference import guard_catalog_bindings
    try:guard_catalog_bindings(old,incoming)
    except ValueError as exc:raise HTTPException(422,str(exc)) from None
    for item in incoming.values():
        if item.get('flow'): item['model_id']=None; item['labels']=[]
    cfg = {'algorithms':[incoming[item['code']] for item in old['algorithms']], 'pose_model_id':body.pose_model_id, 'runtime':body.runtime.model_dump()}
    if row and row.config_json.get('_deleted_algorithms'): cfg['_deleted_algorithms'] = copy.deepcopy(row.config_json['_deleted_algorithms'])
    active = db.query(ContinuousMonitor).filter((ContinuousMonitor.desired_state == 'running') | (ContinuousMonitor.status == 'stopping')).first()
    if active:
        raise HTTPException(409, '请先停止监控并等待资源释放，再切换推理模型')
    for item in body.algorithms:
        if item.flow:
            validate_registered_flow(db, item.flow, item.category)
        if item.enabled and not item.model_id and not item.flow:
            raise HTTPException(422, f'启用的告警“{item.name}”必须选择检测模型')
        if item.model_id and not item.flow:
            model_path(db, item.model_id)
    if any(item.enabled and not item.flow for item in body.algorithms) and not body.pose_model_id:
        raise HTTPException(422, '启用监控告警时必须选择行为姿态模型')
    if body.pose_model_id and body.pose_model_id in {model_id for item in body.algorithms if item.enabled for model_id,kind in __import__('alarm_flow').flow_model_specs(item.model_dump()) if kind=='detect'}:
        raise HTTPException(422, '行为姿态模型不能同时作为告警检测模型')
    model_path(db, body.pose_model_id, 'pose')
    from inference_service.catalog import published_models
    models = published_models(db, cfg)
    if row is None:
        row = AlgorithmCatalog(id='global', config_json=cfg, revision=1, updated_by=str(admin['id']))
        db.add(row)
    else:
        row.config_json = cfg; row.revision += 1; row.updated_by = str(admin['id'])
    db.commit()
    from inference_service.catalog import enqueue_configuration
    try:
        operation_id = enqueue_configuration(row.revision, models)
        switch_state = 'loading'
    except Exception:
        operation_id = None; switch_state = 'failed'
    return {**configuration(db), 'revision':row.revision,
            'inference_switch':{'operation_id':operation_id, 'state':switch_state,
                                'message':None if operation_id else '启用的告警必须先选择检测模型'}}


@router.get('/system/algorithms/inference-switch/{operation_id}')
def get_inference_switch(operation_id: str, admin=Depends(require_admin)):
    from inference_service.catalog import switch_status
    return switch_status(operation_id)


def migrate_algorithm_menu(db, Menu, SystemSeedState):
    """Ensure the administrator algorithm catalog lives under monitoring management."""
    key = 'monitoring_alarm_algorithm_catalog_v2'
    if db.get(SystemSeedState, key):
        return False

    from monitoring_menu_migration import _create_menu

    parent = (
        db.query(Menu)
        .filter(Menu.path == '/monitoring', Menu.menu_type == 'directory')
        .one_or_none()
    )
    if parent is None:
        return False

    existing = (
        db.query(Menu)
        .filter(Menu.path == '/system/algorithms')
        .one_or_none()
    )
    if existing and existing.menu_type != 'menu':
        raise RuntimeError('算法配置菜单路径冲突')

    if not existing:
        existing = _create_menu(
            db,
            Menu,
            parent_id=parent.id,
            name='告警算法配置',
            path='/system/algorithms',
            icon='DataAnalysis',
            menu_type='menu',
            sort_order=6,
            status='active',
            visible='visible',
        )
        db.add(existing)
    else:
        existing.parent_id = parent.id
        existing.name = '告警算法配置'
        existing.icon = 'DataAnalysis'
        existing.sort_order = 6

    db.add(SystemSeedState(key=key))
    return True


def validate_registered_flow(db, flow, category=None):
    from alarm_flow import validate_flow, FlowError
    try:
        nodes = validate_flow(flow)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)[:300]) from None
    if category is not None and category!='behavior' and any(n['type']=='behavior_condition' for n in nodes):raise HTTPException(422,'行为判断只用于行为类告警')
    for node in nodes:
        if node['type'] not in ('recognize','pose'): continue
        c = node['config']; record = db.get(ModelRegistry, c['modelRef'])
        model_path(db, c['modelRef'], 'pose' if node['type']=='pose' else 'detect')
        labels = model_labels(db, record)
        if labels and not set(c['targets']).issubset(labels):
            raise HTTPException(422, '识别类别不在所选模型的实际训练类别中')

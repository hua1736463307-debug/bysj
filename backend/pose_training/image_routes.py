"""Bounded, owner-scoped local image tests and explicit fall-only activation."""
import copy
import hashlib
import io
import json
import math
import time
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from auth_utils import require_admin, require_any_menu
from database import AlgorithmCatalog, ContinuousMonitor, JobTask, ModelRegistry, get_db
from job_store import _append_event_locked
from .data import digest_file, run_root
from .features import FEATURE_SCHEMA
from .inference import classifier_nodes, flow_fingerprint, validate_spec

router=APIRouter()
permission=require_any_menu('/model-training','/training')
MAX_BYTES=5*1024*1024

def model_spec(db,task_id,user):
    from .routes import visible
    task=visible(db,task_id,user)
    model=db.get(ModelRegistry,task_id)
    if task.type!='pose_training' or task.status!='completed' or not model or model.source!='pose_classifier' or model.status not in ('experimental','active'):
        raise HTTPException(409,'请先完成姿态训练')
    locked=db.query(JobTask).filter(JobTask.type=='pose_evaluation',JobTask.status=='completed',JobTask.input_json['training_task_id'].as_string()==task_id).first()
    if not locked:raise HTTPException(409,'请先完成锁定测试')
    path=(run_root()/task_id/'training-report.json').resolve()
    weights=(run_root()/task_id/'static-pose.pt').resolve()
    if not path.is_relative_to(run_root()) or not weights.is_relative_to(run_root()) or str(weights)!=str(Path(model.path).resolve()):raise HTTPException(409,'模型文件不在受管目录')
    try:
        if path.stat().st_size>4*1024*1024 or weights.stat().st_size>4*1024*1024:raise ValueError()
        report=json.loads(path.read_text(encoding='utf-8'))
        if model.sha256!=report['weights_sha256'] or digest_file(weights)!=model.sha256 or model.metadata_json.get('feature_schema')!=FEATURE_SCHEMA:raise ValueError()
        if not report.get('signature') or locked.result_json.get('training_signature')!=report['signature'] or locked.result_json.get('threshold')!=report['threshold']:raise ValueError()
        spec=dict(model_id=task_id,weights_sha256=model.sha256,threshold=report['threshold'],feature_schema=FEATURE_SCHEMA,pipeline_sha256='0'*64)
        validate_spec(spec)
    except (OSError,ValueError,KeyError,TypeError):raise HTTPException(409,'模型或阈值校验失败，请检查训练产物') from None
    return task,model,spec

def fall_flow(db):
    from algorithm_catalog import configuration
    cfg=configuration(db)
    item=next((a for a in cfg['algorithms'] if a['code']=='fall_down'),None)
    if not item or not item.get('flow') or len(classifier_nodes(item['flow']))!=1:raise HTTPException(409,'突然倒地告警需要一个有效的姿态行为判断流程')
    return copy.deepcopy(item['flow'])

def image_row(db,test_id,user):
    row=db.get(JobTask,test_id)
    if not row or row.type!='pose_image_test' or (row.owner_username!=user['username'] and user.get('role')!='admin'):raise HTTPException(404,'图片测试不存在')
    return row

def history_archived(row):
    return (row.result_json or {}).get('history_archived') is True

def history_query(db,task_id,user,archived=False):
    query=db.query(JobTask).filter(JobTask.type=='pose_image_test',JobTask.owner_username==user['username'],JobTask.input_json['training_task_id'].as_string()==task_id)
    flag=JobTask.result_json['history_archived'].as_boolean()
    return query.filter(flag.is_(True) if archived else flag.isnot(True))

def decode_image(raw):
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(raw)) as img:
            if img.format not in ('JPEG','PNG','WEBP') or img.width*img.height>12000000 or min(img.size)<16:raise ValueError()
            # Match alarm frame pixels; ignore EXIF transforms rather than shifting coordinates.
            return np.asarray(img.convert('RGB')).copy()
    except (OSError,ValueError,UnidentifiedImageError,Image.DecompressionBombError):raise HTTPException(422,'请上传不超过 1200 万像素的 JPG、PNG 或 WebP 图片') from None

def decision_threshold(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not .01<=value<=.99:
        raise HTTPException(422,'倒地阈值必须是 0.01–0.99 之间的有限数值')
    return float(value)

def image_test_threshold(spec,flow,override=None):
    if override is not None:return decision_threshold(override)
    active=classifier_nodes(flow)[0]['config'].get('classifier')
    if active and all(active.get(k)==spec.get(k) for k in ('model_id','weights_sha256','pipeline_sha256')):
        return decision_threshold(active['threshold'])
    return decision_threshold(spec['threshold'])

def run_image_test(db,task_id,user,raw,request_id,threshold=None):
    if not raw or len(raw)>MAX_BYTES:raise HTTPException(413,'单张图片最大 5 MiB')
    if not request_id or len(request_id)>100:raise HTTPException(422,'图片测试请求标识无效')
    # Hold the same admission transaction used by monitor/training through inference and persistence.
    from .gpu import GPU_ADMISSION_KEY
    if db.bind.dialect.name=='postgresql' and not db.execute(text('select pg_try_advisory_xact_lock(:key)'),{'key':GPU_ADMISSION_KEY}).scalar():raise HTTPException(409,'GPU 正在处理其他操作，请稍后重试')
    task,model,spec=model_spec(db,task_id,user);flow=fall_flow(db)
    spec['pipeline_sha256']=flow_fingerprint(flow)
    recommended_threshold=spec['threshold'];spec['threshold']=image_test_threshold(spec,flow,threshold)
    identity=str(uuid.uuid5(uuid.NAMESPACE_URL,'pose-local-image:'+user['username']+':'+request_id))
    image_sha=hashlib.sha256(raw).hexdigest();existing=db.get(JobTask,identity)
    if existing:
        if existing.type!='pose_image_test' or existing.owner_username!=user['username'] or existing.input_json!=dict(training_task_id=task_id,image_sha256=image_sha,spec=spec):raise HTTPException(409,'请求标识已用于其他图片或模型配置')
        if history_archived(existing):raise HTTPException(409,'该测试记录已清空，请重新选择图片后测试，或先恢复记录')
        return existing.to_dict()
    import continuous_monitoring as cm
    if db.query(ContinuousMonitor.id).filter((ContinuousMonitor.desired_state=='running') | (ContinuousMonitor.status=='stopping')).first():raise HTTPException(409,'请先停止持续监控，再进行本地图片测试')
    if db.query(JobTask.id).filter(JobTask.type.in_(cm.ACTIVE_JOBS),JobTask.status.in_(['running','cancelling'])).first():raise HTTPException(409,'GPU 训练或分析任务正在运行，请等待完成')
    image=decode_image(raw)
    classifier_nodes(flow)[0]['config']['classifier']=spec
    from alarm_flow import execute_flow
    from inference_service.client import inference_client
    from inference_service.catalog import inference_revision
    revision=inference_revision(db.get(AlgorithmCatalog,'global'));started=time.monotonic();calls=0;limited=False
    def infer(img,config):
        nonlocal calls,limited
        calls+=1
        if calls>8 or time.monotonic()-started>45:
            limited=True;raise ValueError('图片测试超出单次处理上限')
        return inference_client.infer_flow_sync(img,config,revision=revision)
    result=execute_flow(flow,image,infer)
    if limited:raise HTTPException(422,'图片人员过多或处理超时，请裁剪后重试；未保存部分结果')
    if any(not entry['success'] for entries in result['modules'].values() for entry in entries):raise HTTPException(503,'人员或姿态识别失败，请检查推理服务后重试')
    people=[]
    for decision in result['decisions']:
        people.append(dict(person=len(people)+1,bbox=decision['entity']['bbox'],**decision['posture']))
    from PIL import Image, ImageDraw
    preview=Image.fromarray(image);draw=ImageDraw.Draw(preview)
    for person in people:
        color='red' if person['fallen'] else 'green' if person['valid'] else 'orange'
        draw.rectangle(person['bbox'],outline=color,width=max(2,preview.width//500))
        score=person.get('probability');label=f"#{person['person']} "+('FALL' if person['fallen'] else 'NON-FALL' if person['valid'] else 'REJECT')+(f' {score:.3f}' if score is not None else '')
        draw.text((person['bbox'][0],max(0,person['bbox'][1]-12)),label,fill=color)
    preview.thumbnail((1600,1600));folder=run_root()/identity;folder.mkdir(exist_ok=True)
    preview.save(folder/'image-test.png')
    report=dict(training_task_id=task_id,spec=spec,recommended_threshold=recommended_threshold,threshold_source='manual' if spec['threshold']!=recommended_threshold else 'training_recommended',people=people,person_count=len(people),valid_people=sum(p['valid'] for p in people),fall_count=sum(p['fallen'] for p in people),message='NO_PERSON_OR_UNASSOCIATED' if not people else 'TEST_COMPLETE',elapsed_seconds=round(time.monotonic()-started,3),image_sha256=image_sha,alarm_sent=False)
    (folder/'image-test.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8')
    now=datetime.now();row=JobTask(id=identity,type='pose_image_test',status='completed',stage='completed',progress=100,message='本地图片测试完成（未发送告警）',owner_id=str(user['id']),owner_username=user['username'],input_json=dict(training_task_id=task_id,image_sha256=image_sha,spec=spec),result_json=report,created_at=now,started_at=now,finished_at=now,heartbeat_at=now)
    db.add(row);_append_event_locked(db,identity,event_type='completed',message=row.message,payload=dict(model_id=task_id,image_sha256=image_sha));db.commit()
    return row.to_dict()

@router.post('/jobs/{task_id}/image-tests')
def test_image(task_id:str,file:UploadFile=File(...),request_id:str=Form(...),threshold:float|None=Form(None,ge=.01,le=.99),db=Depends(get_db),user=Depends(permission)):
    try:return run_image_test(db,task_id,user,file.file.read(MAX_BYTES+1),request_id,threshold)
    finally:file.file.close()

@router.get('/jobs/{task_id}/image-tests')
def image_tests(task_id:str,db=Depends(get_db),user=Depends(permission),archived:bool=False):
    _,model,spec=model_spec(db,task_id,user);flow=fall_flow(db)
    row=db.get(AlgorithmCatalog,'global');active=classifier_nodes(flow)[0]['config'].get('classifier')
    recommended_threshold=spec['threshold'];spec['pipeline_sha256']=flow_fingerprint(flow)
    spec['threshold']=image_test_threshold(spec,flow)
    query=history_query(db,task_id,user,archived)
    rows=query.order_by(JobTask.created_at.desc(),JobTask.id).limit(20).all()
    return dict(revision=row.revision,model_id=model.id,threshold=spec['threshold'],recommended_threshold=recommended_threshold,active=active,label_mode=model.metadata_json.get('label_mode'),pipeline_sha256=spec['pipeline_sha256'],archived=archived,total=query.count(),active_count=history_query(db,task_id,user).count(),archived_count=history_query(db,task_id,user,True).count(),tests=[r.to_dict() for r in rows])

@router.delete('/jobs/{task_id}/image-tests')
def clear_image_tests(task_id:str,db=Depends(get_db),user=Depends(permission),before:datetime|None=None):
    from .routes import visible
    if visible(db,task_id,user).type!='pose_training':raise HTTPException(404,'姿态训练不存在')
    cutoff=before or datetime.now()
    if cutoff.tzinfo is not None:raise HTTPException(422,'清空时间标识无效，请刷新后重试')
    query=history_query(db,task_id,user).filter(JobTask.status=='completed',JobTask.created_at<=cutoff)
    rows=query.order_by(JobTask.created_at,JobTask.id).limit(500).with_for_update().all()
    for row in rows:
        row.result_json={**row.result_json,'history_archived':True,'history_archived_at':datetime.now().isoformat(),'history_archived_by':user['username']}
        _append_event_locked(db,row.id,event_type='history_archived',message='已清空本地图片测试历史（可恢复）',payload=dict(actor=user['username'],training_task_id=task_id))
    db.flush();remaining=query.count();db.commit()
    return dict(archived_count=len(rows),remaining_count=remaining,before=cutoff.isoformat())

@router.post('/image-tests/{test_id}/restore')
def restore_image_test(test_id:str,db=Depends(get_db),user=Depends(permission)):
    row=db.query(JobTask).filter(JobTask.id==test_id,JobTask.type=='pose_image_test',JobTask.owner_username==user['username']).with_for_update().first()
    if not row:raise HTTPException(404,'图片测试不存在')
    changed=history_archived(row)
    if changed:
        result=dict(row.result_json)
        for key in ('history_archived','history_archived_at','history_archived_by'):result.pop(key,None)
        row.result_json=result
        _append_event_locked(db,row.id,event_type='history_restored',message='已恢复本地图片测试历史',payload=dict(actor=user['username']))
        db.commit()
    return dict(restored=changed,test=row.to_dict())

@router.get('/image-tests/{test_id}/files/{artifact}')
def image_file(test_id:str,artifact:str,db=Depends(get_db),user=Depends(permission)):
    row=image_row(db,test_id,user)
    if artifact not in ('preview','report') or row.status!='completed':raise HTTPException(404,'结果不存在')
    path=(run_root()/row.id/('image-test.png' if artifact=='preview' else 'image-test.json')).resolve()
    if not path.is_relative_to(run_root()) or not path.is_file():raise HTTPException(404,'结果不存在')
    return FileResponse(path,filename=path.name)

class BindingRequest(BaseModel):
    revision:int=Field(ge=0)
    test_id:str|None=Field(default=None,max_length=36)
    enable:bool=True
    reviewed:bool=False

def crop_classifier_flow(db):
    """This configuration entry explicitly operates on detected-person crops."""
    from alarm_flow import validate_flow
    flow=fall_flow(db);nodes={n['id']:n for n in validate_flow(flow)}
    judge=classifier_nodes(flow)[0];pose=nodes[judge['config']['sourceNodeId']]
    if pose['type']!='pose' or pose['config'].get('range')!='target_roi' or pose['config'].get('sourceTarget')!='person':
        raise HTTPException(409,'请先保存“人员识别 → 按人员框裁剪的姿态识别 → 倒地判断”流程')
    return flow,pose

@router.get('/alarm-classifier')
def alarm_classifier_options(db=Depends(get_db),admin=Depends(require_admin)):
    flow,pose=crop_classifier_flow(db);models=[]
    candidates=db.query(ModelRegistry).filter(ModelRegistry.source=='pose_classifier',ModelRegistry.status.in_(['experimental','active'])).order_by(ModelRegistry.created_at.desc()).limit(100).all()
    for model in candidates:
        try:
            _,_,spec=model_spec(db,model.id,admin)
        except HTTPException:
            continue
        models.append(dict(id=model.id,name=model.name,threshold=spec['threshold'],label_mode=(model.metadata_json or {}).get('label_mode')))
    return dict(revision=db.get(AlgorithmCatalog,'global').revision,active=classifier_nodes(flow)[0]['config'].get('classifier'),models=models,pose_node_id=pose['id'],crop=pose['config']['crop'],pipeline_sha256=flow_fingerprint(flow))

class AlarmClassifierRequest(BaseModel):
    revision:int=Field(ge=0)
    model_id:str|None=Field(default=None,max_length=36)
    threshold:float|None=Field(default=None,ge=.01,le=.99,allow_inf_nan=False)

@router.put('/alarm-classifier')
def configure_alarm_classifier(body:AlarmClassifierRequest,db=Depends(get_db),admin=Depends(require_admin)):
    """Explicit admin configuration; completed locked evaluation is mandatory."""
    from algorithm_catalog import editable_catalog,commit_catalog_change
    row,_=editable_catalog(db,body.revision)
    flow,_=crop_classifier_flow(db);config=classifier_nodes(flow)[0]['config']
    previous=copy.deepcopy(config.get('classifier'));spec=None
    if body.model_id:
        _,model,spec=model_spec(db,body.model_id,admin)
        spec['pipeline_sha256']=flow_fingerprint(flow)
        selected_threshold=body.threshold if body.threshold is not None else previous['threshold'] if previous and previous['model_id']==spec['model_id'] and previous['weights_sha256']==spec['weights_sha256'] else spec['threshold']
        spec['threshold']=decision_threshold(selected_threshold)
        config['classifier']=spec
    else:
        if body.threshold is not None:raise HTTPException(422,'停用分类器时不能设置倒地阈值')
        config.pop('classifier',None)
    if previous==spec:return {**alarm_classifier_options(db,admin),'changed':False}
    cfg=copy.deepcopy(row.config_json)
    next(a for a in cfg['algorithms'] if a['code']=='fall_down')['flow']=flow
    audit_task=body.model_id or (previous or {}).get('model_id')
    if audit_task:
        db.query(JobTask).filter(JobTask.id==audit_task).with_for_update().one()
        _append_event_locked(db,audit_task,event_type='pose_alarm_binding',message='算法配置：启用裁剪姿态分类器' if spec else '算法配置：恢复原几何规则',payload=dict(actor=admin['username'],before=previous,after=spec,algorithm_code='fall_down',source='algorithm_configuration',input='person_roi_pose'))
    commit_catalog_change(db,row,cfg,admin,metadata_only=True)
    return {**alarm_classifier_options(db,admin),'changed':True}

@router.post('/jobs/{task_id}/alarm-binding')
def alarm_binding(task_id:str,body:BindingRequest,db=Depends(get_db),admin=Depends(require_admin)):
    from algorithm_catalog import editable_catalog,commit_catalog_change
    row,_=editable_catalog(db,body.revision)
    _,model,spec=model_spec(db,task_id,admin);cfg=copy.deepcopy(row.config_json)
    item=next((a for a in cfg['algorithms'] if a['code']=='fall_down'),None)
    if not item or len(classifier_nodes(item.get('flow') or {}))!=1:raise HTTPException(409,'突然倒地流程不可用于绑定')
    config=classifier_nodes(item['flow'])[0]['config'];previous=copy.deepcopy(config.get('classifier'))
    if body.enable:
        if not body.reviewed or not body.test_id:raise HTTPException(422,'请先查看本地图片测试结果并确认用于倒地告警')
        test=image_row(db,body.test_id,admin)
        if history_archived(test):raise HTTPException(409,'该图片测试已清空，请先恢复记录或重新测试')
        spec['pipeline_sha256']=flow_fingerprint(item['flow'])
        spec['threshold']=decision_threshold(test.result_json.get('spec',{}).get('threshold'))
        if test.status!='completed' or test.owner_username!=admin['username'] or test.input_json.get('training_task_id')!=task_id or test.result_json.get('spec')!=spec or not test.result_json.get('valid_people'):
            raise HTTPException(409,'测试结果与当前人员/姿态流程或模型不一致，请重新测试')
        config['classifier']=spec
    else:
        if previous and previous['model_id']!=task_id:raise HTTPException(409,'当前启用的是其他分类器，请刷新后重试')
        config.pop('classifier',None)
    db.query(JobTask).filter(JobTask.id==task_id).with_for_update().one()
    _append_event_locked(db,task_id,event_type='pose_alarm_binding',message='启用单帧姿态倒地分类器' if body.enable else '停用分类器并恢复原几何规则',payload=dict(actor=admin['username'],before=previous,after=config.get('classifier'),test_id=body.test_id,algorithm_code='fall_down',label_mode=model.metadata_json.get('label_mode')))
    # Only CPU classifier selection changed; detector/pose GPU configuration is identical.
    return commit_catalog_change(db,row,cfg,admin,metadata_only=True)

import os
import time
from datetime import datetime
from uuid import uuid4

from sqlalchemy import text

from database import ContinuousMonitor, JobTask, ModelRegistry, SessionLocal
from job_store import ACTIVE_STATUSES, _append_event_locked, create_job, update_job
from .data import load_dataset, run_root, digest_file
from .trainer import config_hash, train, evaluate
from .gpu import GPU_ADMISSION_KEY, GPUWorkerRestart, execution_device, snapshot as gpu_snapshot
from gpu_probe_service import report_gpu_fault

JOB_TYPES = ('pose_training', 'pose_evaluation')
LEASE_KEY = 4924218313798924700


class Cancelled(Exception):
    pass


class WorkerStopping(Exception):
    pass


class LeaseLost(Exception):
    pass


def locked_submission(db, owner, kind, inputs, request_id=None):
    identity = dict(owner=owner['username'], type=kind, request_id=request_id)
    if not request_id:
        identity['inputs'] = inputs
    key = config_hash(identity)
    # All submissions for one owner serialize, including train and its held-out evaluation.
    if db.bind.dialect.name == 'postgresql':
        owner_key = int(config_hash({'owner':owner['username']})[:15],16)
        db.execute(text('select pg_advisory_xact_lock(:key)'), {'key':owner_key})
    query = db.query(JobTask).filter(JobTask.type==kind, JobTask.owner_username==owner['username'])
    if kind == 'pose_evaluation':
        query = query.filter(JobTask.input_json['training_task_id'].as_string()==inputs['training_task_id'])
    else:
        query = query.filter(JobTask.input_json['request_key'].as_string()==key)
        if not request_id:
            query = query.filter(JobTask.status.in_(ACTIVE_STATUSES))
    existing = query.order_by(JobTask.created_at.desc()).first()
    if existing:
        db.rollback()
        if kind=='pose_training' and existing.input_json.get('submission_signature')!=config_hash(inputs):
            raise ValueError('同一请求 ID 不允许修改训练参数')
        return existing, True
    # Bound outstanding submissions per user, separate from other workflow queues.
    if db.query(JobTask).filter(JobTask.owner_username==owner['username'],JobTask.type.in_(JOB_TYPES),JobTask.status.in_(ACTIVE_STATUSES)).count() >= 5:
        db.rollback()
        raise ValueError('姿态任务队列已满，请等待已有任务完成')
    if db.query(JobTask).filter(JobTask.type.in_(JOB_TYPES),JobTask.status.in_(ACTIVE_STATUSES)).count() >= 100:
        db.rollback()
        raise ValueError('姿态任务总队列已满')
    return create_job(db,job_type=kind,owner_username=owner['username'],owner_id=owner.get('id'),
                      input_data={**inputs,'request_key':key,'submission_signature':config_hash(inputs),'attempts':0},message='单帧姿态任务已排队'),False


def claim(db, gpu=None):
    if db.bind.dialect.name=='postgresql' and not db.execute(text('select pg_try_advisory_xact_lock(:key)'),{'key':GPU_ADMISSION_KEY}).scalar():
        db.rollback();return None
    task = db.query(JobTask).filter(JobTask.type.in_(JOB_TYPES),JobTask.status=='pending',JobTask.cancel_requested.is_(False)).order_by(JobTask.created_at).with_for_update(skip_locked=True).first()
    if not task:
        db.rollback()
        return None
    if float(task.input_json.get('_gpu_retry_after',0))>time.time():
        db.rollback();return None
    busy=db.query(JobTask.id).filter(JobTask.type.in_(('training','behavior_analysis','behavior_temporal_training','annotation','model_export')),
                                    JobTask.status.in_(('running','cancelling'))).first()
    monitor=db.query(ContinuousMonitor.id).filter((ContinuousMonitor.desired_state=='running') | (ContinuousMonitor.status=='stopping')).first()
    state=gpu if gpu is not None else gpu_snapshot()
    reason='GPU 被持续监控占用，停止监控后自动开始' if monitor else 'GPU 被其他训练或分析任务占用，等待资源释放' if busy else 'GPU 不可用或显存不足，恢复后自动开始' if state.get('gpu_status')!='ready' else None
    if reason:
        if task.stage!='waiting_gpu' or task.message!=reason:
            update_job(db,task.id,stage='waiting_gpu',message=reason,error_code='POSE_GPU_WAITING',event_type='waiting_gpu')
        else:db.rollback()
        return None
    task.input_json={**dict(task.input_json or {}),'claim_token':uuid4().hex}
    task.error_code=None;task.error_detail=None
    return update_job(db,task.id,status='running',stage='loading',worker_pid=os.getpid(),message='GPU 姿态任务已领取，正在校验数据',error_code=None,error_detail=None,event_type='claimed')


def reconcile(db):
    """Called only while holding the independent worker's exclusive scheduler lease."""
    jobs = db.query(JobTask).filter(JobTask.type.in_(JOB_TYPES),JobTask.status.in_({'running','cancelling','interrupted','pending'})).with_for_update().all()
    for task in jobs:
        if task.cancel_requested:
            task.status='cancelled';task.finished_at=datetime.now();task.stage='cancelled';task.message='姿态任务已取消'
        elif task.status in ('running','interrupted'):
            inputs = dict(task.input_json or {});attempts=int(inputs.get('attempts',0))+1;inputs['attempts']=attempts;task.input_json=inputs
            task.status='pending' if attempts<=3 else 'failed';task.stage='queued' if attempts<=3 else 'failed'
            task.worker_pid=None;task.message='从已保存轮次恢复训练' if attempts<=3 else '任务恢复次数超限'
            if attempts>3:
                task.finished_at=datetime.now();task.error_code='pose_recovery_exhausted'
        else:
            continue
        task.updated_at=datetime.now();task.heartbeat_at=datetime.now()
        _append_event_locked(db,task.id,event_type='pose_reconciled',message=task.message,payload={'status':task.status})
    db.commit()


def perform(task_id, stop_requested=lambda: False, lease_check=lambda: None):
    last_heartbeat = 0.
    def check():
        nonlocal last_heartbeat
        lease_check()
        if stop_requested():
            raise WorkerStopping()
        with SessionLocal() as db:
            task=db.query(JobTask).filter(JobTask.id==task_id).first()
            if task and task.input_json.get('claim_token')!=claim_token:
                raise LeaseLost('Pose task ownership changed')
            if not task or task.cancel_requested or task.status=='cancelling':
                raise Cancelled()
            if time.monotonic()-last_heartbeat>5:
                task.heartbeat_at=datetime.now();task.updated_at=datetime.now();db.commit();last_heartbeat=time.monotonic()
    def progress(value,message,payload):
        check()
        with SessionLocal() as db:
            update_job(db,task_id,progress=value,stage='training',message=message,event_payload=payload)
    with SessionLocal() as db:
        task=db.get(JobTask,task_id);inputs=dict(task.input_json);kind=task.type;owner=task.owner_username;claim_token=inputs.get('claim_token')
    try:
        check();rows,info=load_dataset(inputs['dataset_id'],inputs['index_sha256'],check)
        output=run_root()/task_id;device=execution_device()
        if kind=='pose_training':
            report=train(rows,info,inputs['config'],output,check,progress,device=device)
            check()
            with SessionLocal() as db:
                task=db.query(JobTask).filter(JobTask.id==task_id).with_for_update().one()
                if task.cancel_requested:
                    db.rollback();raise Cancelled()
                # Separate namespace/status prevents YOLO/Pose/annotation activation or filesystem reconciliation.
                path=output/'static-pose.pt';model=db.query(ModelRegistry).filter(ModelRegistry.path==str(path)).first()
                if not model:
                    model=ModelRegistry(id=task_id,path=str(path));db.add(model)
                model.name=f"{inputs['name']} - 单帧姿态实验";model.sha256=digest_file(path);model.size=path.stat().st_size
                model.source='pose_classifier';model.status='experimental';model.format='pt';model.task_id=task_id;model.created_by=owner
                model.metadata_json=dict(model_family='static_pose_classifier',task='pose_classification',production_compatible=False,
                    supports_text_prompts=False,label_mode=info['label_mode'],feature_schema=info['feature_schema'],index_sha256=info['index_sha256'],
                    dataset_id=inputs['dataset_id'],display_name=model.name,execution=report.get('execution'))
                db.flush()
                update_job(db,task_id,status='completed',stage='completed',progress=100,message='姿态训练完成，模型已登记为实验版本',
                    result=dict(model_id=model.id,threshold=report['threshold'],threshold_status=report['threshold_status'],validation=report['validation'],
                                dataset=info,production_compatible=False,execution=report.get('execution')),event_type='completed')
        else:
            parent_id=inputs['training_task_id']
            with SessionLocal() as db:
                parent=db.get(JobTask,parent_id)
                if not parent or parent.type!='pose_training' or parent.status!='completed' or parent.owner_username!=owner:
                    raise ValueError('关联训练任务不可用于测试')
            report=evaluate(rows,info,run_root()/parent_id,output,check,device=device)
            check()
            with SessionLocal() as db:
                task=db.query(JobTask).filter(JobTask.id==task_id).with_for_update().one()
                if task.cancel_requested:
                    db.rollback();raise Cancelled()
                update_job(db,task_id,status='completed',stage='completed',progress=100,message='锁定测试完成',result=report,event_type='completed')
    except LeaseLost:
        # A successor owns recovery; the stale worker must not mutate its task.
        raise
    except (Cancelled,WorkerStopping) as e:
        with SessionLocal() as db:
            update_job(db,task_id,status='cancelled' if isinstance(e,Cancelled) else 'interrupted',
                       stage='cancelled' if isinstance(e,Cancelled) else 'interrupted',message='姿态任务已取消' if isinstance(e,Cancelled) else '等待 worker 恢复',event_type='stopped')
    except Exception as e:
        with SessionLocal() as db:
            task=db.query(JobTask).filter(JobTask.id==task_id).with_for_update().one()
            # Cancellation wins if it arrived while an operation failed.
            if task.cancel_requested or task.status=='cancelling':
                update_job(db,task_id,status='cancelled',stage='cancelled',message='姿态任务已取消',event_type='cancelled')
            elif report_gpu_fault(e):
                retries=int(task.input_json.get('_gpu_retry_count',0))+1
                task.input_json={**task.input_json,'_gpu_retry_count':retries,'_gpu_retry_after':time.time()+min(300,15*2**min(retries-1,5))}
                task.worker_pid=None
                update_job(db,task_id,status='interrupted' if retries<=3 else 'failed',stage='waiting_gpu' if retries<=3 else 'failed',
                           message='GPU 异常，等待恢复后从检查点继续' if retries<=3 else 'GPU 恢复次数超限',
                           error_code='POSE_GPU_WAITING' if retries<=3 else 'pose_gpu_recovery_exhausted',error_detail=str(e)[:1000],event_type='waiting_gpu')
                if retries<=3:
                    task=db.query(JobTask).filter(JobTask.id==task_id).with_for_update().populate_existing().one()
                    update_job(db,task_id,status='cancelled' if task.cancel_requested else 'pending',stage='cancelled' if task.cancel_requested else 'waiting_gpu',
                               message='姿态任务已取消' if task.cancel_requested else 'GPU 异常，等待恢复后从检查点继续',event_type='gpu_retry_queued')
                raise GPUWorkerRestart() from e
            else:
                update_job(db,task_id,status='failed',stage='failed',message='姿态任务失败',error_code='pose_task_failed',error_detail=str(e)[:1000],event_type='failed')

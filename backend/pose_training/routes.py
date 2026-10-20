import json
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from auth_utils import require_any_menu
from database import JobEvent, JobTask, get_db
from job_store import request_job_cancel
from .data import catalog, load_dataset, run_root, release_lock
from .jobs import JOB_TYPES, locked_submission
from .worker import health

router=APIRouter(prefix='/pose-training',tags=['pose-training'])
from .image_routes import router as image_router
router.include_router(image_router)
permission=require_any_menu('/model-training','/training')


class TrainingRequest(BaseModel):
    device: Literal['cuda:0'] = 'cuda:0'
    dataset_id: str = Field(min_length=1,max_length=100)
    index_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    name: str = Field(default='单帧姿态实验',min_length=1,max_length=100)
    epochs: int = Field(default=30,ge=1,le=100)
    batch_size: int = Field(default=32,ge=1,le=256)
    learning_rate: float = Field(default=.001,ge=.000001,le=.1)
    max_false_positive_rate: float = Field(default=.05,ge=0,le=1)
    seed: int = Field(default=20261018,ge=0,le=2147483647)
    request_id: Optional[str] = Field(default=None,max_length=100)


def visible(db,task_id,user):
    task=db.get(JobTask,task_id)
    if not task or task.type not in JOB_TYPES or (user.get('role')!='admin' and task.owner_username!=user['username']):
        raise HTTPException(404,'姿态任务不存在')
    return task


@router.get('/datasets')
def datasets(user=Depends(permission)):
    return dict(datasets=catalog(),worker=health())


@router.get('/datasets/{dataset_id}/preflight')
def preflight(dataset_id: str,user=Depends(permission)):
    try:
        _,info=load_dataset(dataset_id)
        return info
    except (ValueError,OSError,KeyError,TypeError) as e:
        raise HTTPException(422,str(e))


@router.post('/jobs')
def submit(body:TrainingRequest,user=Depends(permission),db=Depends(get_db)):
    if release_lock().exists():raise HTTPException(503,'姿态训练版本正在切换，请稍后重试')
    try:
        _,info=load_dataset(body.dataset_id,body.index_sha256)
        if not health()['available']:
            raise HTTPException(503,'姿态训练 worker 未就绪')
        config={k:getattr(body,k) for k in ('epochs','batch_size','learning_rate','max_false_positive_rate','seed','device')}
        task,reused=locked_submission(db,user,'pose_training',dict(dataset_id=body.dataset_id,index_sha256=info['index_sha256'],name=body.name,config=config),body.request_id)
        return dict(task=task.to_dict(),reused=reused)
    except (ValueError,OSError,KeyError,TypeError) as e:
        raise HTTPException(422,str(e))


@router.get('/jobs')
def tasks(page:int=Query(1,ge=1),page_size:int=Query(20,ge=1,le=100),user=Depends(permission),db=Depends(get_db)):
    q=db.query(JobTask).filter(JobTask.type.in_(JOB_TYPES))
    if user.get('role')!='admin':q=q.filter(JobTask.owner_username==user['username'])
    return dict(total=q.count(),tasks=[t.to_dict() for t in q.order_by(JobTask.created_at.desc()).offset((page-1)*page_size).limit(page_size).all()])


@router.get('/jobs/{task_id}')
def status(task_id:str,user=Depends(permission),db=Depends(get_db)):
    return visible(db,task_id,user).to_dict()


@router.get('/jobs/{task_id}/events')
def events(task_id:str,after_sequence:int=Query(0,ge=0),user=Depends(permission),db=Depends(get_db)):
    visible(db,task_id,user)
    items=db.query(JobEvent).filter(JobEvent.task_id==task_id,JobEvent.sequence>after_sequence).order_by(JobEvent.sequence).limit(100).all()
    return dict(events=[e.to_dict() for e in items],next_sequence=items[-1].sequence if items else after_sequence)


@router.post('/jobs/{task_id}/cancel')
def cancel(task_id:str,user=Depends(permission),db=Depends(get_db)):
    visible(db,task_id,user)
    return request_job_cancel(db,task_id).to_dict()


@router.post('/jobs/{task_id}/evaluate')
def test(task_id:str,user=Depends(permission),db=Depends(get_db)):
    if release_lock().exists():raise HTTPException(503,'姿态训练版本正在切换，请稍后重试')
    task=visible(db,task_id,user)
    if task.type!='pose_training' or task.status!='completed' or task.result_json.get('threshold') is None:
        raise HTTPException(409,'训练尚未完成或验证未产生可用阈值')
    # Admin visibility does not permit changing the owning user's experiment lineage.
    if task.owner_username!=user['username']:
        raise HTTPException(403,'仅任务所有者可启动锁定测试')
    inputs=task.input_json
    if not health()['available']:raise HTTPException(503,'姿态训练 worker 未就绪')
    try:
        evaluation,reused=locked_submission(db,user,'pose_evaluation',dict(dataset_id=inputs['dataset_id'],index_sha256=inputs['index_sha256'],training_task_id=task.id))
        return dict(task=evaluation.to_dict(),reused=reused)
    except ValueError as e:raise HTTPException(422,str(e))


@router.get('/jobs/{task_id}/files/{artifact}')
def download(task_id:str,artifact:str,user=Depends(permission),db=Depends(get_db)):
    task=visible(db,task_id,user)
    names={'report':'training-report.json','weights':'static-pose.pt'} if task.type=='pose_training' else {'report':'test-report.json','predictions':'predictions.jsonl'}
    if task.status!='completed' or artifact not in names:raise HTTPException(404,'结果尚不可下载')
    path=(run_root()/task.id/names[artifact]).resolve()
    if not path.is_relative_to(run_root()) or not path.is_file():raise HTTPException(404,'结果文件不存在')
    return FileResponse(path,filename=f'{task.id}-{path.name}')

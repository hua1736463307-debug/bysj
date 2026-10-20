"""Exclusive GPU scheduler; CUDA contexts live in disposable task children."""
import json
import os
import signal
import subprocess
import sys
import threading
import time

from .data import atomic_json, run_root


def health():
    try:
        path=run_root()/'worker-health.json'
        if path.stat().st_size>16384:raise ValueError('health size')
        value=json.loads(path.read_text(encoding='utf-8'))
        return dict(available=value.get('status') in ('ready','running') and 0<=time.time()-float(value['at'])<30,
                    status=value.get('status'),active_task_id=value.get('active_task_id'),device='cuda:0',
                    gpu_status=value.get('gpu_status','unavailable'),gpu_reason=value.get('gpu_reason'))
    except (OSError,ValueError,KeyError,TypeError):
        return dict(available=False,status='offline',active_task_id=None,device='cuda:0',gpu_status='unavailable')


def run():
    from sqlalchemy import text
    from database import engine, SessionLocal, JobTask
    from job_store import update_job
    from .jobs import LEASE_KEY, LeaseLost, claim, perform, reconcile
    from .gpu import snapshot as gpu_snapshot
    stop=threading.Event();active={'id':None};heartbeat_stop=threading.Event()
    for name in (signal.SIGTERM,signal.SIGINT):signal.signal(name,lambda *_:stop.set())
    lease=engine.connect()
    if engine.dialect.name!='postgresql':
        lease.close();raise RuntimeError('Pose worker requires PostgreSQL scheduler leases')
    if not lease.execute(text('select pg_try_advisory_lock(:key)'),{'key':LEASE_KEY}).scalar():
        lease.close();raise RuntimeError('Another pose worker already owns the scheduler lease')
    lease.commit()
    lease_pid=lease.execute(text('select pg_backend_pid()')).scalar();lease.commit()
    def check_lease():
        try:
            held=lease.execute(text("select exists(select 1 from pg_locks where locktype='advisory' and pid=pg_backend_pid() and classid::bigint=:hi and objid::bigint=:lo and granted)"),
                               {'hi':LEASE_KEY>>32,'lo':LEASE_KEY&0xffffffff}).scalar()
            lease.commit()
            if not held:raise RuntimeError('Lease no longer held')
        except Exception as e:
            raise LeaseLost('Pose worker scheduler lease lost') from e
    publish_lock=threading.Lock()
    def publish():
        with publish_lock:
            state=gpu_snapshot()
            atomic_json(run_root()/'worker-health.json',dict(status='running' if active['id'] else 'ready',at=time.time(),pid=os.getpid(),active_task_id=active['id'],
                        device='cuda:0',gpu_status=state.get('gpu_status','unavailable'),gpu_reason=state.get('gpu_reason')))
    def heartbeat_loop():
        while not heartbeat_stop.wait(5):publish()
    pulse=threading.Thread(target=heartbeat_loop,daemon=True)
    try:
        with SessionLocal() as db:reconcile(db)
        publish();pulse.start()
        while not stop.is_set():
            # A lost session invalidates exclusive ownership; fail before the next claim.
            check_lease()
            with SessionLocal() as db:
                for task in db.query(JobTask).filter(JobTask.type.in_(('pose_training','pose_evaluation')),JobTask.status=='cancelling',JobTask.worker_pid.is_(None)).all():
                    update_job(db,task.id,status='cancelled',stage='cancelled',message='排队姿态任务已取消',event_type='cancelled')
                task=claim(db,gpu_snapshot());active['id']=task.id if task else None
            publish()
            if active['id']:
                child=subprocess.Popen([sys.executable,'-m','pose_training.task',active['id'],str(lease_pid),str(os.getpid())])
                stopping_at=None
                try:
                    while child.poll() is None:
                        check_lease()
                        if stop.is_set() and stopping_at is None:
                            child.terminate();stopping_at=time.monotonic()
                        if stopping_at is not None and time.monotonic()-stopping_at>35:child.kill()
                        time.sleep(.2)
                finally:
                    if child.poll() is None:child.kill()
                    child.wait(timeout=10)
                with SessionLocal() as db:
                    # An unhandled child failure can only recover under this scheduler lease.
                    item=db.get(JobTask,active['id'])
                    if item and item.status in ('running','interrupted','cancelling'):reconcile(db)
                active['id']=None
            else:stop.wait(2)
    finally:
        heartbeat_stop.set()
        if pulse.is_alive():pulse.join(6)
        atomic_json(run_root()/'worker-health.json',dict(status='stopped',at=time.time(),active_task_id=None))
        try:lease.execute(text('select pg_advisory_unlock(:key)'),{'key':LEASE_KEY});lease.commit()
        finally:lease.close()


if __name__=='__main__':
    import sys
    if '--health' in sys.argv:
        raise SystemExit(0 if health()['available'] else 1)
    run()

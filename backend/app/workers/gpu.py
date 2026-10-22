"""Shared GPU queue ownership; no two-minute failure for normal queue waits."""
import time
from app.core.errors import ApiError
from app.domains import lease_service


def acquire(ctx):
    holder = f"{ctx.worker_id}:{ctx.job.id}:{ctx.attempt}"
    started = time.monotonic()
    while True:
        ctx.check_cancelled()
        token = lease_service.acquire_lease(ctx.db, lease_service.GPU_RESOURCE,
            holder=holder, job_id=ctx.job.id, ttl_seconds=ctx.settings.gpu_lease_seconds)
        if token is not None:
            ctx.guard.set_gpu(holder, token)
            return holder, token
        limit = ctx.settings.gpu_wait_seconds
        if limit and time.monotonic() - started > limit:
            raise ApiError(503, "gpu_wait_timeout", "GPU 等待超过配置上限；检查队列或待恢复租约")
        ctx.progress(None, "waiting_gpu", "等待 GPU；失联占用须确认旧计算退出后恢复")
        time.sleep(min(2, max(0.1, ctx.settings.worker_poll_seconds)))


def release(ctx, ownership):
    ctx.guard.clear_gpu()
    holder, token = ownership
    # Called only after the handler has released its model / reaped its child.
    lease_service.release_lease(ctx.db, lease_service.GPU_RESOURCE,
        holder=holder, generation=token, job_id=ctx.job.id)

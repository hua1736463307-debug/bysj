"""GPU-only execution; availability is distinct from scheduler liveness."""
import os

from gpu_runtime import probe_gpu
from gpu_probe_service import GPUUnavailable

GPU_ADMISSION_KEY = 4924218313798924623


class GPUWorkerRestart(Exception):
    """The isolated CUDA child must exit before retrying a damaged context."""


def snapshot():
    return probe_gpu(min_free_memory_mb=float(os.getenv('POSE_GPU_MIN_FREE_MIB', '6144')))


def execution_device():
    import torch
    state = snapshot()
    if state.get('gpu_status') != 'ready' or not torch.cuda.is_available():
        raise GPUUnavailable('GPU_RUNTIME_UNAVAILABLE: GPU 未就绪，等待恢复')
    device = torch.device('cuda:0')
    budget = float(os.getenv('POSE_GPU_MEMORY_LIMIT_MIB', '1024')) * 1024**2
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.25, budget / total), device)
    return device

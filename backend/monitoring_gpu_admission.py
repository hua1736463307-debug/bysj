"""Narrow, measured admission for isolated recorder inference during training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import requests
from fastapi import HTTPException

PROFILE = json.loads(Path(__file__).with_name('monitoring_gpu_profile.json').read_text())


def flow_hash(flow):
    return hashlib.sha256(json.dumps(flow, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def free_memory():
    # Read driver counters only; do not create a CUDA context in the API process.
    if os.getenv('REALTIME_INFERENCE_DEVICE', '0') != '0':
        raise ValueError('unprofiled device')
    output = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True, timeout=3).stdout
    rows = [line.split(',') for line in output.strip().splitlines()]
    if len(rows) != 1 or rows[0][0].strip() != PROFILE['gpu_uuid']:
        raise ValueError('unprofiled GPU mapping')
    return int(rows[0][1].strip())


def validate_profile(config, bindings, client, available):
    sources = [s for s in config.get('video_sources', []) if s.get('desired_state', 'running') == 'running']
    if not config.get('recorder_monitor') or not config.get('per_camera') or len(sources) != 1:
        raise HTTPException(409, '训练期间仅支持单路独立摄像头告警推理')
    flows = bindings.get('flows') or []
    if not flows or any(PROFILE['flows'].get(f['code']) != flow_hash(f.get('flow')) for f in flows):
        raise HTTPException(409, '当前算法流程尚未完成训练并行显存评估')
    with client._lock:
        specs = [dict(s) for s in client._models.values()]
        if not client.token or client._switching or client._revision != PROFILE['revision'] or bindings.get('revision') != PROFILE['revision']:
            raise HTTPException(409, '独立推理配置未就绪或需要重新评估显存')
    expected = [{k: m[k] for k in ('model_id', 'path', 'kind')} for m in PROFILE['models']]
    if sorted(specs, key=lambda m: m['model_id']) != sorted(expected, key=lambda m: m['model_id']):
        raise HTTPException(409, '当前模型组合尚未完成训练并行显存评估')
    if available < PROFILE['inference_budget_mib'] + PROFILE['reserve_mib']:
        raise HTTPException(409, '显存不足：训练并行告警推理需要至少 7168 MiB 空闲显存')


def admit_training(config, db, owner_id):
    from monitoring_policy import resolve_source
    from inference_service.client import inference_client
    sources = [s for s in config.get('video_sources', []) if s.get('desired_state', 'running') == 'running']
    if not config.get('recorder_monitor') or not config.get('per_camera') or len(sources) != 1:
        raise HTTPException(409, '训练期间仅支持单路独立摄像头告警推理')
    bindings = resolve_source(db, sources[0], owner_id)['algorithm_bindings']
    try:
        available = free_memory()
        validate_profile(config, bindings, inference_client, available)
        for model in PROFILE['models']:
            if hashlib.sha256(Path(model['path']).read_bytes()).hexdigest() != model['sha256']:
                raise HTTPException(409, '模型权重已改变，请重新评估训练并行显存')
        response = requests.get(inference_client.base_url + '/health', timeout=3)
        response.raise_for_status()
        status = response.json()
        if status.get('state') != 'ready' or status.get('revision') != PROFILE['revision'] or status.get('error'):
            raise ValueError('service not ready')
        # Recheck after disk and health reads; admission lock is still held.
        validate_profile(config, bindings, inference_client, free_memory())
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, '无法确认显存或独立推理状态，本路未启动') from None
    return {'profile_revision': PROFILE['revision'], 'inference_budget_mib': PROFILE['inference_budget_mib'],
            'reserve_mib': PROFILE['reserve_mib'], 'admitted_free_mib': available}


def guard_reason(client):
    try:
        if client.revision != PROFILE['revision']:
            return '推理配置已改变，请重新评估训练并行显存'
        if free_memory() < PROFILE['reserve_mib']:
            return '空闲显存低于 4096 MiB，已停止本路告警推理以保护训练'
    except Exception:
        return '无法确认安全显存余量，已停止本路告警推理'
    return None

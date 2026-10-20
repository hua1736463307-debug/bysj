"""Pinned static-pose inference shared by image tests and fall alarm decisions."""
import copy
import hashlib
import json
import math
import re
from functools import lru_cache
from .data import run_root, digest_file
from .features import FEATURE_SCHEMA, encode_coco17

def validate_spec(spec):
    if not isinstance(spec,dict) or set(spec)!={'model_id','weights_sha256','threshold','feature_schema','pipeline_sha256'}:
        raise ValueError('姿态分类器绑定格式无效')
    if not isinstance(spec['model_id'],str) or not re.fullmatch(r'[a-f0-9-]{36}',spec['model_id']):raise ValueError('姿态分类器模型标识无效')
    for key in ('weights_sha256','pipeline_sha256'):
        if not isinstance(spec[key],str) or not re.fullmatch(r'[a-f0-9]{64}',spec[key]):raise ValueError('姿态分类器指纹无效')
    if type(spec['threshold']) not in (int,float) or not math.isfinite(spec['threshold']) or not 0<spec['threshold']<1:raise ValueError('姿态分类器阈值无效')
    if spec['feature_schema']!=FEATURE_SCHEMA:raise ValueError('姿态特征版本不匹配')

def flow_fingerprint(flow):
    from alarm_flow import validate_flow
    nodes=[]
    for node in validate_flow(flow):
        # Face routing changes neither the detector/pose pixels nor classifier features.
        # Preserve existing locked-test bindings when only this module changes.
        if node['type']=='face_capture':continue
        config=copy.deepcopy(node['config']);config.pop('classifier',None)
        nodes.append(dict(id=node['id'],type=node['type'],config=config))
    return hashlib.sha256(json.dumps(nodes,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def classifier_nodes(flow):
    return [n for n in flow.get('nodes',[]) if n.get('type')=='behavior_condition']

def guard_catalog_bindings(old,incoming):
    """Generic flow editing cannot silently activate, remove or retarget a model."""
    previous={a['code']:a for a in old['algorithms']}
    for code,item in incoming.items():
        nodes=classifier_nodes(item.get('flow') or {})
        before={n['id']:n['config'].get('classifier') for n in classifier_nodes(previous.get(code,{}).get('flow') or {}) if n['config'].get('classifier')}
        after={n['id']:n['config'].get('classifier') for n in nodes if n['config'].get('classifier')}
        if before!=after:raise ValueError('请在单帧姿态分类的本地图片测试中启用或停用分类器')
        if after:
            if code!='fall_down':raise ValueError('姿态分类器仅用于突然倒地告警')
            fingerprint=flow_fingerprint(item['flow'])
            if any(spec['pipeline_sha256']!=fingerprint for spec in after.values()):raise ValueError('倒地推理流程已改变，请先停用分类器并重新进行本地图片测试')

@lru_cache(maxsize=4)
def _load(model_id,sha,size,mtime):
    import torch
    from .network import StaticPoseNet
    path=(run_root()/model_id/'static-pose.pt').resolve()
    if not path.is_relative_to(run_root()) or size>4*1024*1024 or digest_file(path)!=sha:raise ValueError('分类器文件校验失败')
    model=StaticPoseNet();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
    return model

def predict_person(detection,spec):
    result=dict(valid=False,fallen=False,probability=None,reason='CLASSIFIER_UNAVAILABLE',model_id=spec.get('model_id'),weights_sha256=spec.get('weights_sha256'),threshold=spec.get('threshold'),feature_schema=FEATURE_SCHEMA)
    try:
        validate_spec(spec)
        features=encode_coco17(detection.get('keypoints'),detection.get('bbox'))
        result.update(keypoints_available=features['keypoints_available'],reason=features['reason'])
        if not features['sufficient']:return result
        import torch
        path=(run_root()/spec['model_id']/'static-pose.pt').resolve()
        if not path.is_relative_to(run_root()):raise ValueError('模型路径无效')
        stat=path.stat();model=_load(spec['model_id'],spec['weights_sha256'],stat.st_size,stat.st_mtime_ns)
        with torch.inference_mode():score=float(torch.softmax(model(torch.from_numpy(features['vector']).unsqueeze(0)),dim=1)[0,1])
        if not math.isfinite(score):raise ValueError('分类分数无效')
        result.update(valid=True,fallen=score>=spec['threshold'],probability=score,reason='CLASSIFIER_FALL' if score>=spec['threshold'] else 'CLASSIFIER_NON_FALL')
    except Exception:result.update(valid=False,fallen=False,probability=None,reason='CLASSIFIER_UNAVAILABLE')
    return result

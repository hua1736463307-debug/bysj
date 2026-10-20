"""CUDA training with portable epoch checkpoints and validation-only thresholds."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .data import atomic_json, digest_file


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save_torch(path, value):
    import torch
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def metrics(rows, probabilities, threshold):
    y = np.asarray([r['label'] for r in rows]);valid = np.isfinite(probabilities)
    pred = valid & (probabilities >= threshold)
    tp = int((pred & (y == 1)).sum());fp = int((pred & (y == 0)).sum())
    positive = int((y == 1).sum());negative = int((y == 0).sum())
    return dict(total=len(rows), usable=int(valid.sum()), coverage=float(valid.mean()), tp=tp, fp=fp,
                fn_including_abstention=positive-tp, tn=int((valid & ~pred & (y==0)).sum()),
                abstained_positive=int((~valid & (y==1)).sum()), abstained_negative=int((~valid & (y==0)).sum()),
                recall=tp/positive if positive else None, false_positive_rate=fp/negative if negative else None,
                precision=tp/(tp+fp) if tp+fp else None)


def probabilities(model, rows, check_cancel=lambda: None):
    import torch
    result = np.full(len(rows), np.nan)
    indices = [i for i,r in enumerate(rows) if r['usable']]
    device = next(model.parameters()).device
    # Bound inference tensors even for the largest permitted index.
    for offset in range(0, len(indices), 256):
        check_cancel()
        chunk = indices[offset:offset+256]
        with torch.inference_mode():
            x = torch.from_numpy(np.stack([rows[i]['vector'] for i in chunk])).to(device)
            result[chunk] = torch.softmax(model(x),dim=1)[:,1].detach().cpu().numpy()
    return result


def cpu_state(value):
    import torch
    if isinstance(value, torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value, dict):return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value, list):return [cpu_state(v) for v in value]
    if isinstance(value, tuple):return tuple(cpu_state(v) for v in value)
    return value


def execution_info(device, started, resumed_from=None):
    import torch
    cuda = device.type == 'cuda'
    if cuda:torch.cuda.synchronize(device)
    return dict(device=str(device),device_name=torch.cuda.get_device_name(device) if cuda else 'CPU test execution',
                cuda_version=torch.version.cuda if cuda else None,elapsed_seconds=round(time.monotonic()-started,3),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0,
                resumed_from_device=resumed_from)


def train(rows, info, config, output, check_cancel=lambda: None, progress=lambda *a: None, *, device='cuda:0'):
    import torch
    from .network import StaticPoseNet
    device=torch.device(device);started=time.monotonic()
    torch.set_num_threads(2);torch.manual_seed(config['seed'])
    if device.type=='cuda':
        torch.cuda.init();torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    signature = config_hash(dict(config=config,index_sha256=info['index_sha256'],schema=info['feature_schema']))
    report_path = output/'training-report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report['signature'] != signature or digest_file(output/'static-pose.pt') != report['weights_sha256']:
            raise ValueError('输出版本与当前配置不一致')
        return report
    selected = [r for r in rows if r['split']=='train' and r['usable']]
    validation = [r for r in rows if r['split']=='validation']
    v = [r for r in validation if r['usable']]
    x = torch.from_numpy(np.stack([r['vector'] for r in selected]));y = torch.tensor([r['label'] for r in selected])
    vx = torch.from_numpy(np.stack([r['vector'] for r in v]));vy = torch.tensor([r['label'] for r in v])
    counts = torch.bincount(y,minlength=2);weights = (len(y)/(2*counts)).float().to(device)
    model = StaticPoseNet().to(device);optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'])
    checkpoint = output/'checkpoint.pt';history=[];best_loss=float('inf');best_state=None;start=0;resumed_from=None
    if checkpoint.exists():
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['signature'] != signature:
            raise ValueError('恢复检查点的数据/配置不一致')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng']);history=saved['history'];start=saved['epoch']
        best_loss=saved['best_loss'];best_state=saved['best_state'];resumed_from=saved.get('device','cpu')
        if device.type=='cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device)
    for epoch in range(start,config['epochs']):
        check_cancel();model.train();order=torch.randperm(len(selected));losses=[]
        for offset in range(0,len(order),config['batch_size']):
            check_cancel();chosen=order[offset:offset+config['batch_size']];optimizer.zero_grad()
            loss=torch.nn.functional.cross_entropy(model(x[chosen].to(device)),y[chosen].to(device),weight=weights)
            if not torch.isfinite(loss):
                raise ValueError('训练损失非有限值')
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        vl=0.
        with torch.inference_mode():
            for offset in range(0,len(v),256):
                check_cancel()
                vl+=float(torch.nn.functional.cross_entropy(model(vx[offset:offset+256].to(device)),vy[offset:offset+256].to(device),reduction='sum'))
        vl/=len(v)
        if not np.isfinite(vl):
            raise ValueError('验证损失非有限值')
        if vl<best_loss:
            best_loss=vl;best_state=cpu_state(model.state_dict())
        history.append(dict(epoch=epoch+1,training_loss=float(np.mean(losses)),validation_loss=vl))
        check_cancel()
        save_torch(checkpoint,dict(signature=signature,model=cpu_state(model.state_dict()),optimizer=cpu_state(optimizer.state_dict()),
                                  rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                                  device=str(device),history=history,epoch=epoch+1,best_loss=best_loss,best_state=best_state))
        progress(10+80*(epoch+1)/config['epochs'],f"训练 {epoch+1}/{config['epochs']} 轮",history[-1])
    check_cancel();model.load_state_dict(best_state);model.eval();scores=probabilities(model,validation,check_cancel);options=[]
    for threshold in np.linspace(.05,.99,95):
        check_cancel();m=metrics(validation,scores,float(threshold))
        if m['false_positive_rate'] is not None and m['false_positive_rate']<=config['max_false_positive_rate']:
            options.append((m['recall'] or 0,-m['false_positive_rate'],float(threshold),m))
    choice=max(options,key=lambda a:a[:2]) if options else None
    save_torch(output/'static-pose.pt',best_state)
    report=dict(signature=signature,config=config,dataset=info,history=history,execution=execution_info(device,started,resumed_from),weights_sha256=digest_file(output/'static-pose.pt'),
                threshold=choice[2] if choice else None,validation=choice[3] if choice else None,
                threshold_status='selected_on_validation' if choice else 'no_feasible_threshold',
                production_compatible=False,model_family='static_pose_classifier',architecture='static_mlp_v1',
                calibration='validation operating threshold; probabilities are uncalibrated',
                metric_target='source fall/nofall agreement' if info['label_mode']=='source_proxy' else 'verified static posture labels')
    check_cancel();atomic_json(report_path,report);return report


def evaluate(rows, info, training_output, output, check_cancel=lambda: None, *, device='cuda:0'):
    import torch
    from .network import StaticPoseNet
    device=torch.device(device);started=time.monotonic()
    torch.set_num_threads(2);training_output=Path(training_output);output=Path(output)
    if device.type=='cuda':
        torch.cuda.init();torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    report=json.loads((training_output/'training-report.json').read_text(encoding='utf-8'))
    if report['dataset']['index_sha256']!=info['index_sha256'] or digest_file(training_output/'static-pose.pt')!=report['weights_sha256']:
        raise ValueError('训练数据或权重版本已变化')
    if report['threshold'] is None:
        raise ValueError('验证集未产生可用阈值；禁止使用测试集调阈值')
    path=output/'test-report.json'
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        if saved['training_signature']!=report['signature']:
            raise ValueError('测试输出版本不一致')
        return saved
    check_cancel();model=StaticPoseNet().to(device);model.load_state_dict(torch.load(training_output/'static-pose.pt',map_location='cpu',weights_only=True));model.eval()
    selected=[r for r in rows if r['split']=='test'];scores=probabilities(model,selected,check_cancel)
    result=metrics(selected,scores,report['threshold']);result.update(training_signature=report['signature'],threshold=report['threshold'],
                index_sha256=info['index_sha256'],production_compatible=False,metric_target=report['metric_target'],execution=execution_info(device,started))
    output.mkdir(parents=True,exist_ok=True)
    # Bounded per-image results stay in a file, never in the task poll response.
    tmp=output/'predictions.jsonl.tmp'
    with tmp.open('w',encoding='utf-8') as f:
        for r,p in zip(selected,scores):
            check_cancel();f.write(json.dumps(dict(image_id=r['image_id'],person_id=r['person_id'],label=r['label'],
                    probability=float(p) if np.isfinite(p) else None,prediction=int(p>=report['threshold']) if np.isfinite(p) else 'insufficient_evidence'))+'\n')
    os.replace(tmp,output/'predictions.jsonl');check_cancel();atomic_json(path,result);return result

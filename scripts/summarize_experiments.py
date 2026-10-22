"""Build evidence-bounded E1/E2 tables from immutable protocols and seed records.

Missing/failed seeds remain visible and are never scored as zero. Paired deltas
are descriptive; a shared small test set and three training seeds do not provide
patient-level uncertainty or clinical validation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from run_experiments import config_hash, require_metrics, write_json, METRICS


def stats(values):
    return {'n':len(values),'mean':statistics.mean(values) if values else None,
            'std':statistics.stdev(values) if len(values)>1 else None,'values':values}


def read_group(root):
    frozen=root/'config.frozen.json';protocol=json.loads(frozen.read_text(encoding='utf-8'))
    digest=config_hash(protocol)
    if root.name!=digest:raise ValueError('protocol path/hash mismatch')
    config=protocol['config'];records=[];completed=[]
    for seed in config['seeds']:
        path=root/f'seed-{seed}.json'
        if not path.exists():records.append({'seed':seed,'status':'not_run'});continue
        record=json.loads(path.read_text(encoding='utf-8'))
        if record['seed']!=seed:raise ValueError('seed identity mismatch')
        if record['status']=='completed':
            train=record['training'];evaluation=record['evaluation'];result=train['result']
            if train['status']!='completed' or record['evaluation_job']['status']!='completed':raise ValueError('terminal success missing')
            if evaluation['status']!='completed' or evaluation['split']!='test' or evaluation['model_id']!=result['model_id'] or evaluation['dataset_version_id']!=config['dataset_version_id']:
                raise ValueError('independent test evaluation identity mismatch')
            if 'protocol='+digest not in evaluation['purpose']:raise ValueError('evaluation protocol identity missing')
            require_metrics(evaluation['metrics'])
            if result['environment']['code_sha']!=protocol['build']['code_sha']:raise ValueError('code provenance mismatch')
            for key in ('requirements_lock_sha256','ml_lock_sha256'):
                if result['environment']['locks'].get(key)!=protocol['build'][key]:raise ValueError('dependency provenance mismatch')
            if result['dataset_manifest_sha256']!=protocol['version']['manifest_sha256']:raise ValueError('dataset provenance mismatch')
            requested=next(r['params'] for r in protocol['effective_requests'] if r['seed']==seed)
            effective=result['effective_args']
            for key in ('seed','epochs','batch','imgsz','optimizer','lr0','lrf','momentum','weight_decay','warmup_epochs'):
                if effective.get(key)!=requested.get(key):raise ValueError('effective training argument mismatch: '+key)
            compact={'seed':seed,'status':'completed','training_job_id':train['job_id'],'evaluation_id':record['evaluation_id'],
                'evaluation_job_id':record['evaluation_job_id'],'model_id':result['model_id'],'model_sha256':result['model_sha256'],
                'epochs_completed':result['epochs_completed'],'training_seconds':result['training_seconds'],
                'peak_gpu_allocated_bytes':result['peak_gpu_allocated_bytes'],'peak_gpu_reserved_bytes':result['peak_gpu_reserved_bytes'],
                'metrics':evaluation['metrics'],'seed_record_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            completed.append(compact)
        records.append({'seed':seed,'status':record['status'],'job_id':record.get('training_job_id'),'error':record.get('error')})
    per_class={}
    for class_id in config['class_map']:
        per_class[class_id]={}
        for metric in METRICS:
            values=[r['metrics']['per_class'].get(class_id,{}).get(metric) for r in completed]
            if any(type(v) not in (float,int) or not math.isfinite(v) for v in values):raise ValueError('per-class metric missing/nonfinite')
            per_class[class_id][metric]=stats(values)
    return {'status':'completed' if len(completed)==len(config['seeds']) else 'partial',
        'protocol_sha256':digest,'protocol':protocol,'seeds':records,'completed':completed,
        'aggregate':{m:stats([r['metrics'][m] for r in completed]) for m in METRICS},'per_class':per_class,
        'training_seconds':stats([r['training_seconds'] for r in completed])}


def compare(groups):
    if 'E1' not in groups or 'E2' not in groups:return {}
    left,right=groups['E1'],groups['E2'];a,b=left['protocol'],right['protocol']
    for key in ('build','base'):
        if a[key]!=b[key]:raise ValueError('comparison provenance mismatch: '+key)
    for split in ('val','test'):
        if a['split_asset_sha256'][split]!=b['split_asset_sha256'][split]:raise ValueError('comparison '+split+' differs')
    if a['effective_requests']!=b['effective_requests']:raise ValueError('comparison training parameters differ')
    if a['config']['class_map']!=b['config']['class_map']:raise ValueError('comparison classes differ')
    pairs=[]
    for row in left['completed']:
        match=next((v for v in right['completed'] if v['seed']==row['seed']),None)
        if match:pairs.append({'seed':row['seed'],**{m:match['metrics'][m]-row['metrics'][m] for m in METRICS}})
    return {'direction':'E2 minus E1','paired_seeds':[v['seed'] for v in pairs],
        'metrics':{m:stats([v[m] for v in pairs]) for m in METRICS},'pairs':pairs,
        'note':'Equal epoch ceiling, not equal optimizer steps/compute. No significance or clinical claim.'}


def number(value):return '—' if value is None else f'{value:.4f}'
def cell(value):return f"{number(value['mean'])} ± {number(value['std'])} (n={value['n']})"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e1',type=Path,required=True);parser.add_argument('--e2',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();groups={'E1':read_group(args.e1),'E2':read_group(args.e2)};comparison=compare(groups)
    args.output.mkdir(parents=True,exist_ok=True)
    report={'status':'completed' if all(g['status']=='completed' for g in groups.values()) else 'partial',
        'groups':groups,'comparison':comparison}
    # Full curves stay in raw artifacts; publish only numerical evidence and
    # confusion matrices needed to reproduce tables and plots.
    for group in groups.values():
        for row in group['completed']:row['metrics'].pop('curves',None)
    write_json(args.output/'results.json',report)
    rows=[{'group':name,**{k:r[k] for k in ('seed','epochs_completed','training_seconds','model_id','model_sha256')},
           **{m:r['metrics'][m] for m in METRICS}} for name,g in groups.items() for r in g['completed']]
    if rows:
        with (args.output/'seeds.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# E1/E2 实验结果（真实记录汇总）','',f"状态：**{report['status']}**。缺失种子不计零分；n 不足 3 时为中间结果。",'',
        '## 协议与比较边界','',
        'E1 为完整原图训练；E2 只对训练集做 320 像素、0.2 重叠切片，最小可见比例 0.5。共同 val/test 使用原图与相同标注、整图推理。两组模型、优化器、batch、epoch 上限和 seeds 一致；E2 训练样本为 6 倍，不是等计算量对比。','',
        '80 epoch 上限，batch=8，imgsz=640，seeds=42/43/44；AdamW，lr0=0.001，patience=20，仅按 val/mAP50–95 早停与选权重。test 不用于参数选择。','',
        '| 组 | 状态 | P | R | mAP50 | mAP50–95 |','|---|---|---|---|---|---|']
    for name,g in groups.items():lines.append('| '+name+' | '+g['status']+' | '+' | '.join(cell(g['aggregate'][m]) for m in METRICS)+' |')
    lines+=['','表内为训练种子均值 ± 样本标准差（n−1）。不是患者级置信区间。','',
        '## 分类别结果','', '| 组 | 类别 | P | R | mAP50 | mAP50–95 |','|---|---|---|---|---|---|']
    for name,g in groups.items():
        for cid,label in g['protocol']['config']['class_map'].items():lines.append('| '+name+' | '+label+' | '+' | '.join(cell(g['per_class'][cid][m]) for m in METRICS)+' |')
    lines+=['','## 种子与计算成本','', '| 组 | seed | 状态 | 实际 epoch | 训练秒数 |','|---|---|---|---|---|']
    for name,g in groups.items():
        for seed in g['seeds']:
            row=next((r for r in g['completed'] if r['seed']==seed['seed']),{})
            lines.append(f"| {name} | {seed['seed']} | {seed['status']} | {row.get('epochs_completed','—')} | {number(row.get('training_seconds'))} |")
    lines+=['','## 配对差值','',f"E2−E1 的配对种子：{comparison['paired_seeds']}。仅对两组均成功的相同 seed 计算。",'']
    for m in METRICS:lines.append(f"- {m}：{cell(comparison['metrics'][m])}")
    lines+=['','## 证据与限制','']
    for name,g in groups.items():lines.append(f"- {name} 协议 SHA256：`{g['protocol_sha256']}`。")
    lines += [f"- 共同 test 摘要：`{groups['E1']['protocol']['split_asset_sha256']['test']}`。",
        f"- 源码：`{groups['E1']['protocol']['build']['code_sha']}`；具体锁、任务/模型/评估 ID 和 seed 文件 hash 见 results.json。",
        '- 原始训练 CSV、混淆矩阵、曲线和评估产物保存在受控卷与 artifacts，未伪造缺失结果。',
        '- BCCD 仅 364 图、test 36 图，无患者/载玻片分组，未验证近重复或跨设备泛化；本结果不构成临床能力结论。',
        '- E3 缺少真实人工审核记录，仍待补；E4 单独按预登记性能方案测量。','']
    (args.output/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':report['status'],'completed':{k:len(v['completed']) for k,v in groups.items()},'output':str(args.output)}))


if __name__=='__main__':main()

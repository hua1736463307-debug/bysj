"""E4: isolated sequential import, annotation, training API and GPU inference.

Use 100 fixed original TRAIN images, three import repetitions, 30 annotation
saves, three one-epoch engineering training jobs, and cold/warm model inference.
Never modifies the formal training dataset or reads its test labels. Credentials
are read from BYSJ_USER/BYSJ_PASSWORD. No workload may be active at the start.
Resource sampling is bounded to the explicit local bysj Compose project.
"""
import argparse
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import zipfile
import httpx
from benchmark_medical import benchmark, describe
from run_experiments import freeze_protocol, request_data, training_body, write_json


RSS_PROBE="""import json,os,pathlib
values=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:values.append(int((p/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'))
 except (OSError,IndexError,ValueError):pass
print(json.dumps({'rss_sum_bytes':sum(values),'processes':len(values)}))
"""


class Sampler:
    def __init__(self, project, output, interval=2):
        if not re.fullmatch('bysj-[a-z0-9-]+',project):raise ValueError('explicit local bysj-* project required')
        self.project=project;self.output=output;self.interval=interval;self.samples=[];self.errors=[]
        self.stop=threading.Event();self.phase='idle';self.thread=threading.Thread(target=self.run,daemon=True)
        self.containers={}
        for role in ('api','dataset-worker','worker','inference-worker'):
            name=project+'-'+role+'-1'
            item=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
            if item['Config']['Labels'].get('com.docker.compose.project')!=project or not item['State']['Running']:
                raise ValueError('container ownership/running check failed')
            self.containers[role]=item['Id']
    def run(self):
        while not self.stop.is_set():
            row={'time':time.time(),'phase':self.phase,'rss':{}}
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                memory,util=map(float,raw.split(','));row.update(device_memory_mib=memory,gpu_util_percent=util)
            except Exception as exc:self.errors.append({'time':time.time(),'sampler':'gpu','error':type(exc).__name__})
            for role,identity in self.containers.items():
                try:row['rss'][role]=json.loads(subprocess.check_output(['docker','exec',identity,'python','-c',RSS_PROBE],timeout=5))
                except Exception as exc:self.errors.append({'time':time.time(),'sampler':role,'error':type(exc).__name__})
            self.samples.append(row)
            write_json(self.output/'resources.json',{'samples':self.samples,'errors':self.errors})
            self.stop.wait(self.interval)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=30)
    def summary(self):
        return {'samples':len(self.samples),'errors':self.errors,
            'device_memory_peak_mib':max((v['device_memory_mib'] for v in self.samples if 'device_memory_mib' in v),default=None),
            'gpu_util_percent':describe([v.get('gpu_util_percent') for v in self.samples]),
            'rss_sum_peak_bytes_by_role':{role:max((v['rss'][role]['rss_sum_bytes'] for v in self.samples if role in v['rss']),default=None) for role in self.containers},
            'notes':['Host device memory includes display/other processes.',
                     'Container process RSS sum can double-count shared pages; it is not private working set.',
                     'Peaks are sampled lower bounds. Sample timestamps give actual interval; configured delay follows each sweep.']}


def selected_training_inputs(prepared):
    manifest=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    selected=sorted([a for a in manifest['assets'] if a['split']=='train'],key=lambda a:a['relative_path'])[:100]
    if len(selected)!=100:raise ValueError('need 100 actual train images')
    paths=[];archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for asset in selected:
            relative=Path(asset['relative_path']);image=(prepared/'images'/relative).resolve();image.relative_to((prepared/'images').resolve())
            if hashlib.sha256(image.read_bytes()).hexdigest()!=asset['sha256']:raise ValueError('image hash mismatch')
            label=(prepared/'labels'/relative.with_suffix('.txt')).resolve();label.relative_to((prepared/'labels').resolve())
            expected=sorted((float(a['class_id']),*(float(a[k]) for k in ('cx','cy','w','h'))) for a in asset['annotations'])
            actual=sorted(tuple(map(float,line.split())) for line in label.read_text().splitlines() if line.strip())
            if actual!=expected:raise ValueError('label/manifest mismatch')
            paths.append(image);bundle.write(image,relative.as_posix());bundle.write(label,relative.with_suffix('.txt').as_posix())
    return manifest,selected,paths,archive.getvalue()


def wait_measured(client,api,headers,identity,timeout=600):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        job=request_data(client,'GET',api+'/jobs/'+identity,headers=headers)
        if job['status'] in {'completed','failed','cancelled','interrupted'}:return job
        time.sleep(.25)
    raise TimeoutError('job still active; inspect recorded job id before rerun')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:18004')
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--project',required=True)
    parser.add_argument('--model-id',required=True,help='Preregistered E1 seed-42 model, never chosen by test score')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if httpx.URL(args.base_url).host not in {'127.0.0.1','localhost'}:raise ValueError('local isolated deployment required')
    manifest,selected,paths,archive=selected_training_inputs(args.prepared)
    api=args.base_url.rstrip('/')+'/api/v1';raw=[]
    with httpx.Client(timeout=180,trust_env=False) as client:
        token=request_data(client,'POST',api+'/auth/login',json={'username':os.environ['BYSJ_USER'],'password':os.environ['BYSJ_PASSWORD']})['token'];headers={'Authorization':'Bearer '+token}
        for status in ('pending','running','cancelling'):
            if request_data(client,'GET',api+'/jobs',headers=headers,params={'status':status,'page_size':1})['total']:raise ValueError('other jobs active; performance run must be isolated')
        model=request_data(client,'GET',api+'/models/'+args.model_id,headers=headers)
        build=request_data(client,'GET',api+'/health/live')
        protocol={'kind':'E4-workflow','input_count':100,'input_split':'train','import_repeats':3,'annotation_saves':30,
            'training_api_repeats':3,'engineering_train_epochs':1,'concurrency':1,'poll_seconds':.25,
            'model_id':model['model_id'],'model_sha256':model['sha256'],'build':build,
            'inputs':[{k:a[k] for k in ('relative_path','sha256','source_group')} for a in selected],
            'inference':{'count':100,'repeat_count':3,'warmup_count':5,'imgsz':640,'device':'0','confidence':.25,'iou':.45},
            'model_selection':'E1 seed=42 fixed before reading test results',
            'hardware':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()}
        frozen,digest=freeze_protocol(args.output,protocol);output=frozen/str(time.time_ns());output.mkdir()
        status='failed';failure=None
        with Sampler(args.project,output) as sampler:
            try:
                def measured(operation,fn):
                    started=time.perf_counter()
                    try:value=fn()
                    except Exception as exc:
                        raw.append({'operation':operation,'status':'failed','milliseconds':(time.perf_counter()-started)*1000,'error':type(exc).__name__});write_json(output/'requests.json',raw);raise
                    row={'operation':operation,'status':'passed','milliseconds':(time.perf_counter()-started)*1000}
                    if isinstance(value,dict):
                        for key in ('job_id','revision','run_id'): 
                            if key in value:row[key]=value[key]
                    raw.append(row);write_json(output/'requests.json',raw);return value
                imports=[]
                sampler.phase='import_and_annotation'
                for repeat in range(3):
                    start=time.perf_counter()
                    job=measured('import_accept',lambda:request_data(client,'POST',api+'/datasets/import',headers=headers,
                        data={'name':f'E4 engineering 100 train originals {time.time_ns()}','class_map':json.dumps(manifest['class_map'])},files={'files':('inputs.zip',archive,'application/zip')}))
                    done=wait_measured(client,api,headers,job['job_id']);write_json(output/f'import-{repeat}.json',done)
                    raw.append({'operation':'import_completion','status':done['status'],'milliseconds':(time.perf_counter()-start)*1000,'job_id':job['job_id']});write_json(output/'requests.json',raw)
                    if done['status']!='completed':raise ValueError('import failed')
                    result=done['result'];imports.append(result)
                    assets=request_data(client,'GET',api+f"/datasets/{result['dataset_id']}/assets",headers=headers,params={'version_id':result['version_id'],'page_size':100})
                    if assets['total']!=100:raise ValueError('import count mismatch')
                    for asset in assets['items'][:10]:
                        state=request_data(client,'GET',api+f"/annotations/{asset['asset_id']}",headers=headers)
                        boxes=[{k:a[k] for k in ('class_id','cx','cy','w','h')} for a in state['annotations']]
                        measured('annotation_save',lambda:request_data(client,'PUT',api+f"/annotations/{asset['asset_id']}",headers=headers,json={'expected_revision':state['revision'],'boxes':boxes}))
                first=imports[0]
                freeze=request_data(client,'POST',api+f"/datasets/{first['dataset_id']}/versions",headers=headers,json={'base_version_id':first['version_id'],'split_seed':42})
                if wait_measured(client,api,headers,freeze['job_id'])['status']!='completed':raise ValueError('engineering version freeze failed')
                sampler.phase='training_api_and_queue'
                for repeat in range(3):
                    body=training_body({'dataset_version_id':first['version_id'],'approved_model_id':model['model_id'],'epochs':1,'batch':8,'imgsz':640,'device':'0','optimizer':'AdamW'},repeat)
                    job=measured('training_accept',lambda:request_data(client,'POST',api+'/training/jobs',headers={**headers,'Idempotency-Key':f'e4-{output.name}-{repeat}'},json=body))
                    done=wait_measured(client,api,headers,job['job_id']);write_json(output/f'training-{repeat}.json',done)
                    if done['status']!='completed':raise ValueError('engineering training failed')
                    queue=(datetime.fromisoformat(done['started_at'])-datetime.fromisoformat(done['created_at'])).total_seconds()*1000
                    raw.append({'operation':'training_claim_wait','status':'passed','milliseconds':queue,'job_id':job['job_id']});write_json(output/'requests.json',raw)
                sampler.phase='inference'
                inference_dir=output/'inference';inference_dir.mkdir()
                inference=benchmark(client,api,token,{**protocol['inference'],'model_id':model['model_id']},paths,inference_dir)
                if inference['status']!='completed':raise ValueError('inference benchmark incomplete')
                status='completed'
            except Exception as exc:failure=type(exc).__name__+': '+str(exc)
        resources=sampler.summary()
        if resources['samples']<2 or resources['errors']:status='partial' if status=='completed' else status
        report={'status':status,'error':failure,'protocol_sha256':digest,'output':str(output),'resource_summary':resources,
            'operations':{op:{**describe([r['milliseconds'] for r in raw if r['operation']==op and r['status'] in {'passed','completed'}]),
                'attempted':sum(r['operation']==op for r in raw),
                'failed':sum(r['status'] not in {'passed','completed'} for r in raw if r['operation']==op),
                'failure_rate':sum(r['status'] not in {'passed','completed'} for r in raw if r['operation']==op)/sum(r['operation']==op for r in raw)} for op in sorted({r['operation'] for r in raw})},
            'notes':['Import completion includes upload, worker queue and polling delay (0.25 s).',
                     'Annotation measurements save existing public labels on disposable engineering copies.',
                     'Training claim wait excludes GPU waiting; inference reports GPU waiting separately.',
                     'Three import/training samples give descriptive p95 only; no population tail claim.',
                     'Cold inference is cold model handle, not reboot or process cold start.']}
        write_json(output/'summary.json',report);print(json.dumps({'status':status,'output':str(output),'error':failure}))
        return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())

"""Create an idempotent train-only derivative and verify common val/test content.

Reads BYSJ_USER/BYSJ_PASSWORD. Output contains no credentials. Processing is
asynchronous; repeating the exact request reuses its existing job and version.
"""
import argparse
import os
from pathlib import Path
import httpx
from run_experiments import config_hash, request_data, wait_job, write_json


def snapshot(client, api, headers, version):
    assets=[]; page=1
    while True:
        value=request_data(client,'GET',api+f"/datasets/{version['dataset_id']}/assets",
            headers=headers,params={'version_id':version['version_id'],'page':page,'page_size':200})
        assets.extend(value['items'])
        if len(assets)>=value['total']:break
        if not value['items']:raise ValueError('incomplete asset pagination')
        page+=1
    output={};groups={}
    for split in ('train','val','test'):
        selected=[a for a in assets if a['split']==split]
        if not selected or len(selected)!=version['split_counts'][split]:raise ValueError('split count mismatch')
        groups[split]=set(a['source_group'] for a in selected)
        records=sorted([{k:a[k] for k in ('relative_path','sha256','source_group','annotations')} for a in selected],key=lambda a:a['relative_path'])
        output[split]={'count':len(selected),'sha256':config_hash(records),'source_groups':sorted(groups[split])}
    if groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test']:
        raise ValueError('source groups cross splits')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:18004')
    parser.add_argument('--parent-version',required=True)
    parser.add_argument('--tile-size',type=int,default=320)
    parser.add_argument('--overlap',type=float,default=.2)
    parser.add_argument('--min-visibility',type=float,default=.5)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();api=args.base_url.rstrip('/')+'/api/v1'
    with httpx.Client(timeout=120,trust_env=False) as client:
        token=request_data(client,'POST',api+'/auth/login',json={'username':os.environ['BYSJ_USER'],'password':os.environ['BYSJ_PASSWORD']})['token']
        headers={'Authorization':'Bearer '+token}
        parent=request_data(client,'GET',api+'/dataset-versions/'+args.parent_version,headers=headers)
        if parent['state']!='frozen':raise ValueError('parent must be frozen')
        before=snapshot(client,api,headers,parent)
        body={'parent_version_id':args.parent_version,'tile_size':args.tile_size,'overlap':args.overlap,'min_visibility':args.min_visibility}
        key='train-tiles-'+config_hash(body)
        job=request_data(client,'POST',api+f"/datasets/{parent['dataset_id']}/train-tile-versions",headers={**headers,'Idempotency-Key':key},json=body)
        report={'status':'pending','request':body,'parent':parent,'parent_splits':before,'job_id':job['job_id']}
        write_json(args.output,report)
        job=wait_job(client,api,token,job['job_id'],1800);report['job']=job
        if job['status']!='completed':
            report['status']=job['status'];write_json(args.output,report);raise ValueError('processing did not complete')
        child=request_data(client,'GET',api+'/dataset-versions/'+job['result']['version_id'],headers=headers)
        after=snapshot(client,api,headers,child)
        if child['parent_version_id']!=parent['version_id'] or child['state']!='frozen':raise ValueError('invalid derived version lineage')
        for split in ('val','test'):
            if before[split]!=after[split]:raise ValueError('original '+split+' snapshot changed')
        if before['train']['source_groups']!=after['train']['source_groups']:raise ValueError('train source groups changed')
        report.update(status='verified',child=child,child_splits=after,
            evaluation_space='whole original images; ordinary whole-image inference; no tiled test inference',
            budget_note='Equal epochs/batch/model/seeds; tile count changes optimizer steps and runtime. Report both.')
        write_json(args.output,report)
        print({'status':report['status'],'child_version':child['version_id'],'split_counts':child['split_counts']})


if __name__=='__main__':main()

"""Upload a verified prepared dataset through v1 and prove the split is unchanged.

Credentials: BYSJ_USER/BYSJ_PASSWORD. The output is an exclusive resumable record;
an interrupted submission with no returned job ID requires inspection, not retry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile
import httpx
from run_experiments import request_data, wait_job, write_json


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def checked(root, relative):
    target=(root/relative).resolve();target.relative_to(root.resolve())
    if not target.is_file():raise ValueError('missing prepared file')
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','source-repository','config','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--base-url',default='http://127.0.0.1:18000')
    args=parser.parse_args();config=json.loads(args.config.read_text(encoding='utf-8'))
    manifest=json.loads((args.prepared/'manifest.json').read_text(encoding='utf-8'))
    revision=subprocess.check_output(['git','-C',str(args.source_repository),'rev-parse','HEAD'],text=True).strip()
    if revision!=config['source_revision']:raise ValueError('source revision mismatch')
    if subprocess.check_output(['git','-C',str(args.source_repository),'status','--porcelain']):raise ValueError('source repository has unreviewed changes')
    license_hash=sha(args.source_repository/'LICENSE')
    if manifest['class_map']!=config['class_map']:raise ValueError('prepared class map mismatch')
    entries={a['relative_path']:a for a in manifest['assets']}
    if len(entries)!=len(manifest['assets']):raise ValueError('duplicate manifest paths')
    files=[]
    for relative,item in entries.items():
        image=checked(args.prepared/'images',relative)
        if sha(image)!=item['sha256']:raise ValueError('prepared image hash mismatch')
        label=checked(args.prepared/'labels',str(Path(relative).with_suffix('.txt')))
        actual=sorted(tuple(float(v) for v in line.split()) for line in label.read_text().splitlines() if line.strip())
        expected=sorted((float(a['class_id']),*(float(a[k]) for k in ('cx','cy','w','h'))) for a in item['annotations'])
        if actual!=expected:raise ValueError('prepared labels differ from manifest')
        files.extend(((image,relative),(label,str(Path(relative).with_suffix('.txt')))))
    identity={'source_revision':revision,'license_sha256':license_hash,'prepared_manifest_file_sha256':sha(args.prepared/'manifest.json'),
              'expected_split_hash':manifest['split_manifest_hash'],'asset_count':len(entries)}
    if args.output.exists():
        state=json.loads(args.output.read_text(encoding='utf-8'))
        if state['identity']!=identity:raise ValueError('existing upload record has a different input')
        if state['status']=='verified':print(json.dumps({'status':'verified','version_id':state['version_id'],'reused':True}));return
        if state['status']=='submitting_freeze' and not state.get('freeze_job'):
            raise ValueError('ambiguous freeze submission; inspect server before any retry')
        if not state.get('import_job'):raise ValueError('ambiguous prior submission; inspect server before any retry')
    else:
        state={'identity':identity,'status':'submitting'}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream:json.dump(state,stream)
    api=args.base_url.rstrip('/')+'/api/v1'
    with httpx.Client(timeout=180) as client:
        token=request_data(client,'POST',api+'/auth/login',json={'username':os.environ['BYSJ_USER'],'password':os.environ['BYSJ_PASSWORD']})['token']
        headers={'Authorization':'Bearer '+token}
        if not state.get('import_job'):
            with tempfile.TemporaryFile() as archive:
                with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
                    for path,relative in files:bundle.write(path,relative)
                archive.seek(0)
                job=request_data(client,'POST',api+'/datasets/import',headers=headers,
                    data={'name':'BCCD fixed original-image split','source_url':config['source_repository'],
                          'source_version':revision,'license':config['source_license_ref'],'class_map':json.dumps(config['class_map'])},
                    files={'files':('bccd-verified.zip',archive,'application/zip')})
            state.update(import_job=job['job_id'],status='importing');write_json(args.output,state)
        job=wait_job(client,api,token,state['import_job'],600)
        if job['status']!='completed':raise ValueError('import did not complete: '+job['status'])
        state.update(dataset_id=job['result']['dataset_id'],version_id=job['result']['version_id'])
        if not state.get('freeze_job'):
            state['status']='submitting_freeze';write_json(args.output,state)
            job=request_data(client,'POST',api+'/datasets/'+state['dataset_id']+'/versions',headers=headers,
                json={'base_version_id':state['version_id'],'split_seed':manifest['split_seed'],'ratios':[manifest['ratios'][s] for s in ('train','val','test')]})
            state['freeze_job']=job['job_id'];write_json(args.output,state)
        job=wait_job(client,api,token,state['freeze_job'],600)
        if job['status']!='completed':raise ValueError('freeze did not complete: '+job['status'])
        version=request_data(client,'GET',api+'/dataset-versions/'+state['version_id'],headers=headers)
        if version['split_manifest_hash']!=manifest['split_manifest_hash']:raise ValueError('server split differs from prepared split')
        seen={};page=1
        while True:
            response=request_data(client,'GET',api+'/datasets/'+state['dataset_id']+'/assets',headers=headers,
                params={'version_id':state['version_id'],'page':page,'page_size':100})
            for a in response['items']:
                original=entries[a['relative_path']]
                if any(a[k]!=original[k] for k in ('sha256','source_group','split')):raise ValueError('asset lineage mismatch')
                seen[a['relative_path']]=a['asset_id']
            if len(seen)>=response['total']:break
            page+=1
        if set(seen)!=set(entries):raise ValueError('server asset set mismatch')
        state.update(status='verified',version=version,assets=seen);write_json(args.output,state)
        print(json.dumps({'status':'verified','dataset_id':state['dataset_id'],'version_id':state['version_id'],'split_counts':version['split_counts'],'split_manifest_hash':version['split_manifest_hash']}))


if __name__=='__main__':main()

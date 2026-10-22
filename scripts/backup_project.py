"""Consistent backup of one explicitly named local bysj Compose project.

Requires no pending/running/cancelling jobs and no GPU holder. Stops only this
project's running API/workers/frontend, retains its database, and restarts the
same containers in finally. Dumps and file archives contain private data: keep
the output outside Git. No credentials or row contents are printed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone


def docker(*args, **kwargs):
    return subprocess.check_output(['docker',*args],**kwargs)


def inspect(name):return json.loads(docker('inspect',name))[0]


def sql(db, statement):
    # SQL is passed as a separate shell positional argument, never interpolated.
    return docker('exec',db,'sh','-c','psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -v ON_ERROR_STOP=1 -c "$1"','sh',statement).decode().strip()


def row_counts(db):
    names=sql(db,"SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename").splitlines()
    if any(not re.fullmatch('[a-z_][a-z0-9_]*',name) for name in names):raise ValueError('unexpected table identifier')
    return {name:int(sql(db,f'SELECT count(*) FROM "{name}"')) for name in names}


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


PACK_VOLUME = r'''
import hashlib,json,pathlib,tarfile,sys
root=pathlib.Path('/source');kind=sys.argv[1];records=[]
with tarfile.open('/backup/'+kind+'.tar.gz','w:gz',compresslevel=1) as archive:
 for path in sorted(root.rglob('*')):
  if path.is_symlink():raise ValueError('controlled backup volume contains unexpected symlink')
  relative=path.relative_to(root).as_posix()
  if path.is_file():
   with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
   records.append({'path':relative,'size':path.stat().st_size,'sha256':digest})
  elif not path.is_dir():raise ValueError('unexpected special file')
  archive.add(path,arcname=relative,recursive=False)
pathlib.Path('/backup/'+kind+'-files.json').write_text(json.dumps(records,sort_keys=True),encoding='utf-8')
print(json.dumps({'volume':kind,'files':len(records),'bytes':sum(r['size'] for r in records)}))
'''


def backup(project, output, helper_image):
    if not re.fullmatch(r'bysj-[a-z0-9-]+',project):raise ValueError('explicit isolated bysj-* project required')
    ids=docker('ps','-aq','--filter','label=com.docker.compose.project='+project).decode().splitlines()
    containers={}
    for identity in ids:
        item=inspect(identity);labels=item['Config']['Labels']
        if labels.get('com.docker.compose.project')!=project:raise ValueError('project ownership changed')
        containers[labels['com.docker.compose.service']]=item
    if not {'api','db','worker','dataset-worker','inference-worker'}<=containers.keys():raise ValueError('incomplete project inventory')
    db=containers['db']['Id'];api=containers['api']
    active="SELECT count(*) FROM job_tasks WHERE status IN ('pending','running','cancelling')"
    if int(sql(db,active)):raise ValueError('in-flight jobs exist; wait or explicitly resolve before backup')
    if int(sql(db,"SELECT count(*) FROM resource_leases WHERE resource='gpu:0' AND COALESCE(holder,'')<>''")):
        raise ValueError('GPU holder/quarantine must be resolved before backup')
    mounts={m['Destination']:m for m in api['Mounts']}
    volumes={}
    for kind in ('data','models','artifacts'):
        mount=mounts['/'+kind]
        if mount['Type']!='volume':raise ValueError('expected named project volumes')
        volume=json.loads(docker('volume','inspect',mount['Name']))[0]
        if (volume.get('Labels') or {}).get('com.docker.compose.project')!=project:raise ValueError('volume does not belong to selected project')
        volumes[kind]=mount['Name']
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    stopped=[];report={'status':'started','project':project,'created_at':datetime.now(timezone.utc).isoformat(),
        'images':{role:c['Image'] for role,c in containers.items()},'volumes':volumes,'archives':{}}
    (output/'backup.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    try:
        for role in ('frontend','api','worker','dataset-worker','inference-worker'):
            item=containers.get(role)
            if item and item['State']['Running']:
                docker('stop','--time','30',item['Id']);stopped.append(item['Id'])
        if int(sql(db,active)):raise ValueError('submission raced with quiescing; refusing an inconsistent backup')
        report['row_counts']=row_counts(db)
        report['alembic_revision']=sql(db,'SELECT version_num FROM alembic_version')
        dump=output/'database.dump'
        with dump.open('wb') as stream:
            subprocess.run(['docker','exec',db,'sh','-c','pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --format=custom --no-owner --no-acl'],stdout=stream,check=True)
        report['database_sha256']=sha(dump)
        for kind,volume in volumes.items():
            result=docker('run','--rm','--network','none','--mount',f'type=volume,source={volume},target=/source,readonly',
                '--mount',f'type=bind,source={output},target=/backup','--entrypoint','python',helper_image,'-c',PACK_VOLUME,kind)
            report['archives'][kind]={**json.loads(result),'archive_sha256':sha(output/(kind+'.tar.gz')),
                                      'files_manifest_sha256':sha(output/(kind+'-files.json'))}
        if row_counts(db)!=report['row_counts']:raise ValueError('database changed while writers were stopped')
        report['status']='complete'
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        failures=[]
        for identity in stopped:
            try:docker('start',identity)
            except Exception:failures.append(identity[:12])
        report['restart_failures']=failures
        if failures:report['status']='restart_required'
        (output/'backup.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if report['status']!='complete':raise RuntimeError('backup requires service restart inspection')
    return {'status':report['status'],'output':str(output),'tables':len(report['row_counts']),'volumes':list(report['archives'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--helper-image',required=True)
    args=parser.parse_args();print(json.dumps(backup(args.project,args.output,args.helper_image),indent=2))

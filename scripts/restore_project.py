"""Restore a verified backup into a brand-new local bysj project, data only.

Creates isolated volumes and PostgreSQL. Does not start API/GPU workers until
the operator has verified runtime images and stopped any competing GPU owner.
Private .env and recovery state live in an ignored output directory. Existing
project resources or output folders are never overwritten.
"""
import argparse
import json
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import time
from backup_project import docker, inspect, row_counts, sha, sql
from audit_database_links import audit_container

RESTORE_VOLUME = r'''
import hashlib,json,pathlib,tarfile,sys
kind=sys.argv[1];root=pathlib.Path('/target')
if any(root.iterdir()):raise ValueError('restore volume is not empty')
with tarfile.open('/backup/'+kind+'.tar.gz','r:gz') as archive:
 for item in archive.getmembers():
  path=pathlib.PurePosixPath(item.name)
  if path.is_absolute() or '..' in path.parts or not (item.isfile() or item.isdir()):raise ValueError('unsafe backup member')
 archive.extractall(root,filter='data')
expected=json.loads(pathlib.Path('/backup/'+kind+'-files.json').read_text())
actual={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()}
if set(actual)!={entry['path'] for entry in expected}:raise ValueError('restored file set differs')
for entry in expected:
 path=actual[entry['path']]
 with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 if digest!=entry['sha256'] or path.stat().st_size!=entry['size']:raise ValueError('restored hash differs')
print(json.dumps({'volume':kind,'verified_files':len(actual)}))
'''


def restore(args):
    project=args.project
    if not re.fullmatch('bysj-[a-z0-9-]+',project):raise ValueError('explicit bysj-* project required')
    if docker('ps','-aq','--filter','label=com.docker.compose.project='+project).strip():raise ValueError('destination project already exists')
    names={project+'-'+kind for kind in ('db-data','data','models','artifacts')}
    if names & set(docker('volume','ls','-q').decode().splitlines()):raise ValueError('destination volumes already exist')
    backup=args.backup.resolve();manifest=json.loads((backup/'backup.json').read_text(encoding='utf-8'))
    if manifest['status']!='complete' or manifest['project']==project:raise ValueError('a complete independent source backup is required')
    if sha(backup/'database.dump')!=manifest['database_sha256']:raise ValueError('database backup checksum mismatch')
    for kind,entry in manifest['archives'].items():
        if kind not in {'data','models','artifacts'}:raise ValueError('unknown archive')
        if sha(backup/(kind+'.tar.gz'))!=entry['archive_sha256'] or sha(backup/(kind+'-files.json'))!=entry['files_manifest_sha256']:
            raise ValueError('file backup checksum mismatch')
    if set(manifest['archives'])!={'data','models','artifacts'}:raise ValueError('incomplete backup volumes')
    image=json.loads(docker('image','inspect',args.api_image))[0]
    code=next((v.partition('=')[2] for v in image['Config']['Env'] if v.startswith('BYSJ_CODE_SHA=')),None)
    if not code or not re.fullmatch('[a-f0-9]{40}',code):raise ValueError('clean API image must declare exact commit')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    for name in ('compose.yml','compose.formal.yml','compose.gpu.yml'):shutil.copy2(root/'deploy'/name,output/name)
    values={}
    for line in args.source_env.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        key,separator,value=line.partition('=')
        if separator:values[key]=value
    values.update(APP_ENV='production',BYSJ_VOLUME_PREFIX=project,DB_PASSWORD=secrets.token_hex(24),
        JWT_SECRET_KEY=secrets.token_urlsafe(48),DB_PORT=str(args.db_port),API_PORT=str(args.api_port),
        FRONTEND_PORT=str(args.frontend_port),PUBLIC_ORIGIN=f'http://127.0.0.1:{args.frontend_port}',
        BYSJ_CODE_SHA=code,BYSJ_API_IMAGE=args.api_image,BYSJ_WORKER_IMAGE=args.worker_image,
        BYSJ_FRONTEND_IMAGE=args.frontend_image,INLINE_JOBS='false',ML_BACKEND='ultralytics',
        REQUIRED_WORKER_ROLES='["dataset","trainer","inference"]',DATA_ROOT='/data',MODEL_ROOT='/models',ARTIFACT_ROOT='/artifacts')
    (output/'.env').write_text(''.join(k+'='+v+'\n' for k,v in values.items()),encoding='utf-8')
    command=['docker','compose','--env-file',str(output/'.env'),'-p',project,'-f',str(output/'compose.yml'),'-f',str(output/'compose.formal.yml')]
    report={'status':'restoring','project':project,'source_project':manifest['project'],'backup':str(backup),
            'api_image':args.api_image,'expected_worker_image':args.worker_image,'services_started':['db'],'file_checks':{}}
    try:
        subprocess.run(command+['up','-d','--no-build','db'],check=True)
        db=project+'-db-1';deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if inspect(db)['State'].get('Health',{}).get('Status')=='healthy':break
            time.sleep(2)
        else:raise TimeoutError('new PostgreSQL did not become healthy')
        if row_counts(db):raise ValueError('new database has unexpected public tables')
        for kind in ('data','models','artifacts'):
            volume=project+'-'+kind
            # `compose up db` creates only its referenced database volume.
            # These names were proven absent before project creation above.
            docker('volume','create','--label','com.docker.compose.project='+project,
                   '--label','com.docker.compose.volume='+kind,volume)
            owner=json.loads(docker('volume','inspect',volume))[0]
            if owner['Labels']['com.docker.compose.project']!=project:raise ValueError('destination volume ownership mismatch')
            result=docker('run','--rm','--network','none','--mount',f'type=volume,source={volume},target=/target',
                '--mount',f'type=bind,source={backup},target=/backup,readonly','--entrypoint','python',args.api_image,'-c',RESTORE_VOLUME,kind)
            report['file_checks'][kind]=json.loads(result)
        with (backup/'database.dump').open('rb') as stream:
            subprocess.run(['docker','exec','-i',db,'sh','-c','pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --exit-on-error --no-owner --no-acl'],stdin=stream,check=True)
        report['row_counts']=row_counts(db)
        if report['row_counts']!=manifest['row_counts']:raise ValueError('restored row counts differ')
        if sql(db,'SELECT version_num FROM alembic_version')!=manifest['alembic_revision']:raise ValueError('restored schema revision differs')
        if int(sql(db,"SELECT count(*) FROM job_tasks WHERE status IN ('pending','running','cancelling')")):
            raise ValueError('restore has active jobs; explicit recovery required before service startup')
        if int(sql(db,"SELECT count(*) FROM pg_constraint WHERE contype='f' AND NOT convalidated")):
            raise ValueError('restored foreign key constraint not validated')
        report['reference_audit']=audit_container(db,project)
        if report['reference_audit']['status']!='passed':
            raise ValueError('restored logical references contain orphans')
        report['status']='data_verified_runtime_acceptance_pending'
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        (output/'restore.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'project':project,'output':str(output),'tables':len(report['row_counts']),'file_checks':report['file_checks']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('backup','output','source-env'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('project','api-image','worker-image','frontend-image'):parser.add_argument('--'+name,required=True)
    for name in ('db-port','api-port','frontend-port'):parser.add_argument('--'+name,type=int,required=True)
    restore(parser.parse_args())

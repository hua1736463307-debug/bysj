"""Read-only logical reference audit for the v1 database (no row contents).

Most v1 relations are enforced in services, not SQL foreign keys. An empty
unvalidated-FK query alone cannot prove that references survived a restore.
This checks the explicit column relations below in one repeatable-read snapshot.
It does not prove JSON references, authorization, files, or semantic equivalence.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from backup_project import inspect, sql


RELATIONS = [
    ('auth_sessions','user_id','users'),
    ('dataset_resources','owner_user_id','users'),
    ('dataset_versions','dataset_id','dataset_resources'),
    ('dataset_versions','parent_version_id','dataset_versions'),
    ('dataset_versions','created_by_user_id','users'),
    ('dataset_assets','dataset_version_id','dataset_versions'),
    ('dataset_assets','root_asset_id','dataset_assets'),
    ('dataset_assets','original_asset_id','dataset_assets'),
    ('dataset_imports','owner_id','users'),
    ('dataset_imports','job_id','job_tasks'),
    ('dataset_imports','dataset_id','dataset_resources'),
    ('annotation_drafts','owner_user_id','users'),
    ('annotation_drafts','dataset_id','dataset_resources'),
    ('annotation_drafts','source_version_id','dataset_versions'),
    ('annotation_draft_assets','draft_id','annotation_drafts'),
    ('job_tasks','owner_user_id','users'),
    ('job_events','task_id','job_tasks'),
    ('model_registry','owner_id','users'),
    ('model_registry','task_id','job_tasks'),
    ('model_registry','training_job_id','job_tasks'),
    ('model_registry','dataset_version_id','dataset_versions'),
    ('evaluation_runs','owner_id','users'),
    ('evaluation_runs','job_id','job_tasks'),
    ('evaluation_runs','model_id','model_registry'),
    ('evaluation_runs','dataset_version_id','dataset_versions'),
    ('inference_runs','owner_id','users'),
    ('inference_runs','job_id','job_tasks'),
    ('inference_runs','model_id','model_registry'),
    ('inference_items','run_id','inference_runs'),
    ('inference_items','asset_id','dataset_assets'),
    ('inference_items','artifact_id','artifact_records'),
    ('inference_predictions','item_id','inference_items'),
    ('review_records','owner_id','users'),
    ('review_records','reviewer_id','users'),
    ('review_records','decided_by','users'),
    ('review_records','inference_item_id','inference_items'),
    ('review_records','image_asset_id','dataset_assets'),
    ('review_items','review_id','review_records'),
    ('review_items','prediction_id','inference_predictions'),
    ('feedback_exports','job_id','job_tasks'),
    ('feedback_exports','parent_version_id','dataset_versions'),
    ('feedback_exports','child_version_id','dataset_versions'),
    ('experiment_runs','owner_id','users'),
    ('experiment_runs','training_job_id','job_tasks'),
    ('experiment_runs','model_id','model_registry'),
    ('experiment_runs','dataset_version_id','dataset_versions'),
    ('artifact_records','owner_id','users'),
    ('resource_leases','job_id','job_tasks'),
]


def orphan_query(source, column, target):
    if not all(re.fullmatch('[a-z_][a-z0-9_]*', value) for value in (source,column,target)):
        raise ValueError('invalid fixed schema identifier')
    return (f'SELECT count(*) FROM "{source}" s LEFT JOIN "{target}" t '
            f'ON s."{column}"=t.id WHERE s."{column}" IS NOT NULL AND t.id IS NULL')


def audit_sql():
    rows = [f"SELECT '{source}.{column}->{target}.id' AS relation, ({orphan_query(source,column,target)}) AS orphans"
            for source,column,target in RELATIONS]
    # One read-only snapshot covers all checks even if a live worker commits.
    return ("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY; "
            "WITH links AS ("+' UNION ALL '.join(rows)+") "
            "SELECT json_build_object('relations', (SELECT json_object_agg(relation,orphans) FROM links), "
            "'declared_foreign_keys', (SELECT count(*) FROM pg_constraint WHERE contype='f' AND connamespace='public'::regnamespace), "
            "'unvalidated_foreign_keys', (SELECT count(*) FROM pg_constraint WHERE contype='f' AND connamespace='public'::regnamespace AND NOT convalidated)); COMMIT;")


def audit_container(db_container, project):
    item=inspect(db_container)
    labels=item['Config'].get('Labels') or {}
    if not re.fullmatch('bysj-[a-z0-9-]+',project) or labels.get('com.docker.compose.project')!=project or labels.get('com.docker.compose.service')!='db':
        raise ValueError('database must belong to the explicitly selected bysj project')
    output=sql(db_container,audit_sql())
    reports=[json.loads(line) for line in output.splitlines() if line.startswith('{')]
    if len(reports)!=1:raise ValueError('missing database audit result')
    result=reports[0]
    result.update(project=project,created_at=datetime.now(timezone.utc).isoformat(),
        status='passed' if not any(result['relations'].values()) and not result['unvalidated_foreign_keys'] else 'failed',
        scope='48 explicit scalar reference relations; non-null values only; includes soft-deleted parents; excludes JSON/polymorphic references, file content, permissions and semantic lineage')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True);parser.add_argument('--db-container',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit_container(args.db_container,args.project)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'relations':len(result['relations']),
        'declared_foreign_keys':result['declared_foreign_keys'],'findings':{k:v for k,v in result['relations'].items() if v}}))
    raise SystemExit(0 if result['status']=='passed' else 1)

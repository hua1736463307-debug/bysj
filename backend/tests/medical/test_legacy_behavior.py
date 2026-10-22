"""Same-input comparisons against legacy pure functions, without old startup.

Exact split assignments may differ: both algorithms must retain source groups.
Unsafe box inputs intentionally adopt the stricter medical boundary.
"""
import ast
from collections import Counter
import importlib.util
import math
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace
import types
import zipfile

import pytest
from app.core.errors import ApiError
from app.domains.annotation.service import _validate_box
from app.domains.datasets.split_service import assign_groups_to_splits
from app.domains.datasets.import_service import safe_extract_zip
from app.domains import job_service

LEGACY = Path(__file__).parents[2]
sys.path.insert(0,str(LEGACY.parent/'scripts'))
from legacy_schema import read_legacy_source


def legacy_functions(filename, names):
    tree=ast.parse(read_legacy_source('backend/'+filename))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(selected)==len(names)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,*selected],type_ignores=[]))
    env={'Path':Path,'random':random,'re':re,'math':math,'Counter':Counter,'SPLITS':('train','val','test')}
    exec(compile(module,filename,'exec'),env)
    return env


def old_archive_module():
    module=types.ModuleType('legacy_archive_compare');sys.modules[module.__name__]=module
    exec(compile(read_legacy_source('backend/utils/archive_utils.py'),'legacy/archive_utils.py','exec'),module.__dict__)
    return module


@pytest.mark.parametrize('seed',[0,42,43])
def test_same_groups_stay_disjoint_in_both_splitters(seed):
    old=legacy_functions('dataset_versioning.py',{'_grouped_three_way_split','_asset_source_group','_annotation_class_id'})
    assets=[SimpleNamespace(source_group=f'g{i}',relative_path=f'{i}-{j}.jpg',basename=f'{i}-{j}.jpg',annotations_json=[{'class_id':0}]) for i in range(12) for j in range(2)]
    before=old['_grouped_three_way_split'](assets,train_ratio=.8,test_ratio=.1,seed=seed,major_class_ids=[0])
    groups={f'g{i}':[{'relative_path':f'{i}-{j}.jpg'} for j in range(2)] for i in range(12)}
    after=assign_groups_to_splits(groups,seed=seed)
    assert set(before)==set(after)==set(groups)
    assert set(before.values())==set(after.values())=={'train','val','test'}
    assert after==assign_groups_to_splits(groups,seed=seed)


def test_same_valid_bbox_exports_identical_coordinates():
    old=legacy_functions('dataset_versioning.py',{'annotation_to_yolo_line','_annotation_class_id'})
    before=old['annotation_to_yolo_line']({'class_id':0,'x':.5,'y':.4,'width':.2,'height':.3})
    box=_validate_box({'class_id':0,'cx':.5,'cy':.4,'w':.2,'h':.3},{'0':'RBC'})
    after=f"{box['class_id']} {box['cx']:.6f} {box['cy']:.6f} {box['w']:.6f} {box['h']:.6f}"
    assert before==after


@pytest.mark.parametrize('box',[{'cx':.1,'cy':.5,'w':.8,'h':.2},{'cx':.5,'cy':.5,'w':float('nan'),'h':.2}])
def test_medical_bbox_boundary_is_intentionally_stricter(box):
    old=legacy_functions('dataset_versioning.py',{'annotation_to_yolo_line','_annotation_class_id'})
    # Legacy formatter emitted even off-image/nonfinite values; new API rejects.
    assert old['annotation_to_yolo_line']({'class_id':0,'x':box['cx'],'y':box['cy'],'width':box['w'],'height':box['h']})
    with pytest.raises(ApiError):_validate_box({'class_id':0,**box},{'0':'RBC'})


def test_job_state_transitions_match_legacy_contract():
    tree=ast.parse(read_legacy_source('backend/job_store.py'))
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALLOWED_TRANSITIONS' for t in n.targets))
    env={};exec(compile(ast.Module(body=[assignment],type_ignores=[]),'job_store.py','exec'),env)
    assert env['ALLOWED_TRANSITIONS']==job_service.ALLOWED_TRANSITIONS


@pytest.mark.parametrize('names',[['safe/a.txt'],['../outside.txt'],['same.txt','same.txt'],['C:/outside.txt']])
def test_same_zip_inputs_obey_both_boundaries(tmp_path,settings,names):
    old=old_archive_module();archive=tmp_path/'input.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for name in names:z.writestr(name,b'content')
    def before():return old.safe_extract_zip(archive,tmp_path/'old',old.ArchiveLimits(disk_safety_margin_bytes=0))
    def after():return safe_extract_zip(archive,tmp_path/'new',settings)
    if names==['safe/a.txt']:
        before();after();assert (tmp_path/'old/safe/a.txt').read_bytes()==(tmp_path/'new/safe/a.txt').read_bytes()
    else:
        with pytest.raises(old.ArchiveSafetyError):before()
        with pytest.raises(ApiError):after()
        assert not (tmp_path/'outside.txt').exists()


def test_same_revision_sequence_accepts_current_and_rejects_stale(client, researcher_headers):
    # Execute the actual old handler's revision guard AST, excluding all HTTP
    # startup and unrelated route behavior. This is a comparison, not a rewrite
    # of its predicate in the test.
    from fastapi import HTTPException
    import support
    tree = ast.parse(read_legacy_source('backend/yolo_api.py'))
    guards = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for index, statement in enumerate(node.body):
                if (isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'expected_revision' for t in statement.targets)
                    and ast.unparse(statement.value) == "request.get('revision')"):
                    guards.append(ast.Module(body=node.body[index:index+2], type_ignores=[]))
    assert len(guards) == 1
    code = compile(ast.fix_missing_locations(guards[0]), 'legacy_revision_guard', 'exec')
    dataset, version = support.make_dataset(client, researcher_headers, count=3)
    asset = client.get(f'/api/v1/datasets/{dataset}/assets?version_id={version}', headers=researcher_headers).json()['data']['items'][0]['asset_id']
    revision = 0
    for expected in [0, 0, 1, 1, 2]:
        old_status = 200
        try:
            exec(code, {'request': {'revision': expected}, 'draft': None if revision == 0 else SimpleNamespace(revision=revision), 'HTTPException': HTTPException})
        except HTTPException as exc:
            old_status = exc.status_code
        response = client.put(f'/api/v1/annotations/{asset}', headers=researcher_headers, json={'expected_revision': expected, 'boxes': []})
        assert response.status_code == old_status
        if old_status == 200:
            revision += 1
            assert response.json()['data']['revision'] == revision


def test_same_weight_retains_hash_and_training_version_lineage(db, settings, tmp_path, researcher):
    from datetime import datetime
    import hashlib
    from uuid import uuid4
    from app.db.models import ModelRegistry
    from app.domains.models.service import register_trained_model, verify_model_file
    old = legacy_functions('model_registry.py', {'sha256_file', 'register_model_file'})
    old.update(hashlib=hashlib, ModelRegistry=ModelRegistry, canonical_model_path=lambda p:p.resolve(),
               SUPPORTED_MODEL_SUFFIXES={'.pt'}, uuid4=uuid4, datetime=datetime)
    weight = tmp_path / 'source.pt'; weight.write_bytes(b'synthetic-weight-comparison-no-pickle-load')
    job, version = str(uuid4()), str(uuid4())
    before = old['register_model_file'](db, weight, source='training', task_id=job, dataset_version_id=version, created_by=researcher.username)
    after = register_trained_model(db, settings, job_id=job, version_id=version, owner_id=researcher.id,
        owner_username=researcher.username, name='comparison', best_path=weight, class_map={'0':'RBC'}, metrics={}, backend='dummy')
    assert before.sha256 == after.sha256 == hashlib.sha256(weight.read_bytes()).hexdigest()
    assert before.task_id == after.task_id == after.training_job_id == job
    assert before.dataset_version_id == after.dataset_version_id == version
    assert before.id != after.id and Path(before.path) != Path(after.path)
    assert verify_model_file(after, settings).read_bytes() == weight.read_bytes()

from pathlib import Path
import json
import pytest
import support
from app.db.models import DatasetAsset, DatasetVersion


def prepare(client,headers):
    dataset,version=support.make_dataset(client,headers,count=12)
    frozen=support.freeze(client,headers,dataset,version_id=version).json()['data']
    assert frozen['status']=='completed'
    return dataset,version


def test_train_only_tiles_preserve_original_test_and_val_and_retry_identity(client,researcher_headers,db):
    dataset,parent=prepare(client,researcher_headers)
    before=db.query(DatasetAsset).filter_by(dataset_version_id=parent).all()
    body={'parent_version_id':parent,'tile_size':64,'overlap':.25,'min_visibility':.5}
    headers={**researcher_headers,'Idempotency-Key':'same-processing'}
    response=client.post(f'/api/v1/datasets/{dataset}/train-tile-versions',headers=headers,json=body)
    assert response.status_code==202,response.text
    job=response.json()['data'];assert job['status']=='completed',job
    child=job['result']['version_id'];after=db.query(DatasetAsset).filter_by(dataset_version_id=child).all()
    original={a.id:a for a in before};child_version=db.get(DatasetVersion,child)
    assert child_version.parent_version_id==parent and child_version.state=='frozen'
    assert child_version.metadata_json['group_level']=='original_image'
    assert child_version.metadata_json['processing']['evaluation_space']=='original_image'
    assert child_version.split_config_json['group_level']=='original_image'
    assert len([a for a in after if a.split=='train'])>len([a for a in before if a.split=='train'])
    parent_version=db.get(DatasetVersion,parent)
    for asset in after:
        root=original[asset.original_asset_id]
        assert asset.split==root.split and asset.source_group==root.source_group
        if asset.split in {'val','test'}:
            assert (asset.relative_path,asset.content_hash,asset.annotations_json)==(root.relative_path,root.content_hash,root.annotations_json)
            assert (Path(child_version.root_path)/asset.label_path).read_bytes()==(Path(parent_version.root_path)/root.label_path).read_bytes()
        else:
            assert asset.transform_json['kind']=='train_tile'
            for box in asset.annotations_json:
                assert box['w']>0 and box['h']>0
                assert box['cx']-box['w']/2>=-1e-6 and box['cx']+box['w']/2<=1+1e-6
                assert box['cy']-box['h']/2>=-1e-6 and box['cy']+box['h']/2<=1+1e-6
    retry=client.post(f'/api/v1/datasets/{dataset}/train-tile-versions',headers=headers,json=body).json()['data']
    assert retry['job_id']==job['job_id'] and retry['result']['version_id']==child
    conflict=client.post(f'/api/v1/datasets/{dataset}/train-tile-versions',headers=headers,json={**body,'tile_size':96})
    assert conflict.status_code==409


def test_processing_rejects_unfrozen_and_excessive_derivative_count(client,researcher_headers,settings,db):
    dataset,parent=support.make_dataset(client,researcher_headers,count=12)
    url=f'/api/v1/datasets/{dataset}/train-tile-versions';headers={**researcher_headers,'Idempotency-Key':'limits'}
    body={'parent_version_id':parent,'tile_size':64,'overlap':.75}
    assert client.post(url,headers=headers,json=body).status_code==409
    support.freeze(client,researcher_headers,dataset,version_id=parent)
    original=settings.max_zip_files;settings.max_zip_files=20
    try:
        response=client.post(url,headers=headers,json=body).json()['data']
        assert response['status']=='failed' and response['error']['code']=='processing_limit'
        assert db.query(DatasetVersion).filter_by(parent_version_id=parent).count()==0
    finally:settings.max_zip_files=original


def test_processing_failure_never_publishes_a_frozen_child(client,researcher_headers,db,monkeypatch):
    dataset,parent=prepare(client,researcher_headers)
    def fail(*a,**k):raise OSError('simulated image read failure')
    monkeypatch.setattr('app.domains.datasets.processing_service.Image.open',fail)
    response=client.post(f'/api/v1/datasets/{dataset}/train-tile-versions',headers={**researcher_headers,'Idempotency-Key':'failure'},
                         json={'parent_version_id':parent}).json()['data']
    assert response['status']=='failed'
    assert db.query(DatasetVersion).filter_by(parent_version_id=parent).count()==0


def test_tiles_preserve_image_cluster_declaration(client,researcher_headers,db):
    files=[];groups={}
    for i in range(9):
        name=f'g{i}.jpg'
        files.append((name,support.image_bytes((i*17,i*23,i*29))))
        files.append((f'g{i}.txt',support.yolo_txt([(c,.5,.5,.3,.3) for c in (0,1,2)]).encode()))
        groups[name]='cluster-pair' if i<2 else f'cluster-{i}'
    job=support.import_dataset(client,researcher_headers,files=files,
        form={'group_level':'image_cluster','source_groups':json.dumps(groups)}).json()['data']
    assert job['status']=='completed',job
    dataset=job['result']['dataset_id'];parent=job['result']['version_id']
    assert support.freeze(client,researcher_headers,dataset,version_id=parent).json()['data']['status']=='completed'
    response=client.post(f'/api/v1/datasets/{dataset}/train-tile-versions',
        headers={**researcher_headers,'Idempotency-Key':'cluster-tiles'},
        json={'parent_version_id':parent,'tile_size':64,'overlap':.25,'min_visibility':.5})
    job=response.json()['data'];assert job['status']=='completed',job
    child=db.get(DatasetVersion,job['result']['version_id'])
    assert child.metadata_json['group_level']=='image_cluster'
    assert child.split_config_json['group_level']=='image_cluster'
    # evaluation_space is the coordinate space of evaluation, not the group level.
    assert child.metadata_json['processing']['evaluation_space']=='original_image'
    manifest=json.loads((Path(child.root_path)/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['group_level']=='image_cluster'

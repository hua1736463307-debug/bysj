import json
from pathlib import Path

import pytest
import support


def inputs():
    files=[];groups={}
    for i in range(9):
        name=f'{i}.jpg';files.append((name,support.image_bytes((i*25,i*19,i*13))))
        files.append((f'{i}.txt',support.yolo_txt([(c,.5,.5,.3,.3) for c in (0,1,2)]).encode()))
        groups[name]='cluster-near-duplicate' if i<2 else f'cluster-{i}'
    return files,groups


def _import(client,headers,*,group_level='image_cluster',groups=None):
    files,default_groups=inputs()
    form={'group_level':group_level}
    selected=groups if groups is not None else (default_groups if group_level=='image_cluster' else None)
    if selected is not None:form['source_groups']=json.dumps(selected)
    job=support.import_dataset(client,headers,files=files,form=form).json()['data']
    assert job['status']=='completed',job
    return job['result']['dataset_id'],job['result']['version_id']


def _version(db,version_id):
    from app.db.models import DatasetVersion
    db.expire_all()
    row=db.get(DatasetVersion,version_id)
    assert row is not None
    return row


def _set_metadata(db,version_id,metadata):
    from app.db.models import DatasetVersion
    db.query(DatasetVersion).filter(DatasetVersion.id==version_id).update({'metadata_json':metadata},synchronize_session=False)
    db.commit()
    db.expire_all()


def test_explicit_similar_image_group_stays_in_one_split(client,researcher_headers,db):
    files,groups=inputs()
    response=support.import_dataset(client,researcher_headers,files=files,form={'group_level':'image_cluster','source_groups':json.dumps(groups)})
    job=response.json()['data'];assert job['status']=='completed',job
    dataset=job['result']['dataset_id'];version=job['result']['version_id']
    frozen=support.freeze(client,researcher_headers,dataset,version_id=version).json()['data']
    assert frozen['status']=='completed',frozen
    rows=client.get(f'/api/v1/datasets/{dataset}/assets?version_id={version}',headers=researcher_headers).json()['data']['items']
    pair=[r for r in rows if r['relative_path'] in {'0.jpg','1.jpg'}]
    assert len(pair)==2 and len({r['split'] for r in pair})==1
    assert {r['source_group'] for r in pair}=={'cluster-near-duplicate'}

    # The declaration must survive import -> freeze in every place a reader
    # can look: resource, assets, version metadata, API split_config, manifest.
    from app.db.models import DatasetAsset, DatasetResource
    resource=db.get(DatasetResource,dataset)
    assert resource.group_level=='image_cluster'
    assets=db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id==version).all()
    assert assets and {a.group_level for a in assets}=={'image_cluster'}
    row=_version(db,version)
    assert row.metadata_json['group_level']=='image_cluster'
    detail=client.get(f'/api/v1/dataset-versions/{version}',headers=researcher_headers).json()['data']
    assert detail['split_config']['group_level']=='image_cluster'
    manifest=json.loads((Path(row.root_path)/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['group_level']=='image_cluster'


def test_legacy_metadata_without_level_freezes_as_original_image(client,researcher_headers,db):
    dataset,version=_import(client,researcher_headers,group_level='original_image')
    _set_metadata(db,version,{'source':{'name':'legacy-import'}})
    job=support.freeze(client,researcher_headers,dataset,version_id=version).json()['data']
    assert job['status']=='completed',job
    row=_version(db,version)
    assert row.metadata_json['group_level']=='original_image'
    assert row.split_config_json['group_level']=='original_image'
    manifest=json.loads((Path(row.root_path)/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['group_level']=='original_image'


def test_nested_level_is_used_and_normalized_to_top(client,researcher_headers,db):
    dataset,version=_import(client,researcher_headers)
    _set_metadata(db,version,{'source':{'group_level':'image_cluster'}})
    job=support.freeze(client,researcher_headers,dataset,version_id=version).json()['data']
    assert job['status']=='completed',job
    row=_version(db,version)
    assert row.metadata_json['group_level']=='image_cluster'
    assert row.split_config_json['group_level']=='image_cluster'
    manifest=json.loads((Path(row.root_path)/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['group_level']=='image_cluster'


def test_conflicting_declarations_are_rejected_without_state_change(client,researcher_headers,db):
    dataset,version=_import(client,researcher_headers)
    _set_metadata(db,version,{'group_level':'original_image','source':{'group_level':'image_cluster'}})
    job=support.freeze(client,researcher_headers,dataset,version_id=version).json()['data']
    assert job['status']=='failed',job
    assert job['error']['code']=='group_level_conflict'
    assert _version(db,version).state=='draft'


def test_unknown_level_is_rejected(client,researcher_headers,db):
    dataset,version=_import(client,researcher_headers)
    _set_metadata(db,version,{'group_level':'patient'})
    job=support.freeze(client,researcher_headers,dataset,version_id=version).json()['data']
    assert job['status']=='failed',job
    assert job['error']['code']=='invalid_group_level'
    assert _version(db,version).state=='draft'


@pytest.mark.parametrize('change',['missing','extra','blank','original_level','duplicate_content'])
def test_invalid_group_contract_never_publishes_dataset(client,researcher_headers,change,db):
    from app.db.models import DatasetResource
    files,groups=inputs();level='image_cluster'
    if change=='missing':groups.pop('0.jpg')
    if change=='extra':groups['not-uploaded.jpg']='extra'
    if change=='blank':groups['0.jpg']=' '
    if change=='original_level':level='original_image'
    if change=='duplicate_content':
        files[2]=('1.jpg',files[0][1]);groups['1.jpg']='different-group'
    before=db.query(DatasetResource).count()
    result=support.import_dataset(client,researcher_headers,files=files,form={'group_level':level,'source_groups':json.dumps(groups)}).json()['data']
    assert result['status']=='failed'
    assert db.query(DatasetResource).count()==before


def test_non_object_source_groups_rejected_at_api(client,researcher_headers):
    files,_=inputs()
    response=support.import_dataset(client,researcher_headers,files=files,form={'group_level':'image_cluster','source_groups':'[]'})
    assert response.status_code==422 and response.json()['code']=='invalid_source_groups'

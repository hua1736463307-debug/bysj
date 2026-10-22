import json
import pytest
import support


def inputs():
    files=[];groups={}
    for i in range(9):
        name=f'{i}.jpg';files.append((name,support.image_bytes((i*25,i*19,i*13))))
        files.append((f'{i}.txt',support.yolo_txt([(c,.5,.5,.3,.3) for c in (0,1,2)]).encode()))
        groups[name]='cluster-near-duplicate' if i<2 else f'cluster-{i}'
    return files,groups


def test_explicit_similar_image_group_stays_in_one_split(client,researcher_headers):
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

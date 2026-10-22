import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
from benchmark_workflow import selected_training_inputs


def prepared(tmp_path,count=100):
    (tmp_path/'images').mkdir();(tmp_path/'labels').mkdir()
    assets=[]
    for index in range(count+2):
        name=f'{index:03d}.png';payload=f'image-{index}'.encode()
        (tmp_path/'images'/name).write_bytes(payload)
        (tmp_path/'labels'/f'{index:03d}.txt').write_text('0 0.5 0.5 0.2 0.2\n')
        assets.append({'relative_path':name,'sha256':hashlib.sha256(payload).hexdigest(),
            'split':'train' if index<count else 'test','source_group':name,
            'annotations':[dict(class_id=0,cx=.5,cy=.5,w=.2,h=.2)]})
    (tmp_path/'manifest.json').write_text(json.dumps({'assets':assets,'class_map':{'0':'cell'}}))


def test_only_one_hundred_training_originals_enter_performance_archive(tmp_path):
    prepared(tmp_path)
    _,selected,paths,archive=selected_training_inputs(tmp_path)
    assert len(paths)==len(selected)==100 and all(a['split']=='train' for a in selected)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        assert len(bundle.namelist())==200
        assert '100.png' not in bundle.namelist() and '101.txt' not in bundle.namelist()


@pytest.mark.parametrize('kind',['short','image','label'])
def test_unverifiable_input_is_rejected_before_submitting_jobs(tmp_path,kind):
    prepared(tmp_path,99 if kind=='short' else 100)
    if kind=='image':(tmp_path/'images/001.png').write_bytes(b'changed')
    if kind=='label':(tmp_path/'labels/001.txt').write_text('0 0.1 0.1 0.1 0.1\n')
    with pytest.raises(ValueError):selected_training_inputs(tmp_path)

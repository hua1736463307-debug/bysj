import hashlib,json
from pathlib import Path
import sys
import pytest
from PIL import Image,PngImagePlugin
sys.path.insert(0,str(Path(__file__).parents[1]))
from audit_image_similarity import audit


def dataset(tmp_path):
    root=tmp_path/'images';root.mkdir();assets=[]
    for i,color in enumerate(('red','red','blue')):
        path=root/f'{i}.png';meta=PngImagePlugin.PngInfo();meta.add_text('note',str(i))
        Image.new('RGB',(20,20),color).save(path,pnginfo=meta)
        assets.append({'relative_path':path.name,'split':('train','test','val')[i],'source_group':str(i),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'width':20,'height':20})
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'assets':assets}))
    return manifest


def test_different_encoded_files_with_identical_pixels_are_detected(tmp_path):
    result=audit(dataset(tmp_path))
    assert result['images_verified']==3 and result['pairs_examined']==3
    assert result['decoded_identical_cross_split']==1
    assert result['image_fingerprints'][0]['sha256']!=result['image_fingerprints'][1]['sha256']
    # Flat colors share dHash but differ in decoded identity: candidates only.
    assert len(result['candidate_pairs'])==2
    assert all(p['human_decision'] is None for p in result['candidate_pairs'])


def test_changed_input_is_rejected_before_reporting(tmp_path):
    manifest=dataset(tmp_path);(tmp_path/'images/0.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash changed'):audit(manifest)


def test_manifest_cannot_read_outside_images_root(tmp_path):
    manifest=dataset(tmp_path);data=json.loads(manifest.read_text());data['assets'][0]['relative_path']='../outside.png'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):audit(manifest)

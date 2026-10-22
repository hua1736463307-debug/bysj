"""Post-hoc, read-only duplicate screening; never relabel or change splits.

Exact decoded RGB identity is separate from 64-bit horizontal dHash proximity.
The latter is only an advisory candidate list, not proof of image identity or
absence of leakage. It cannot establish patient/slide independence. Thresholds
are explicit and the input image SHA256 values must match the frozen manifest.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from PIL import Image


def features(path):
    with Image.open(path) as image:
        rgb=image.convert('RGB')
        prefix=f'RGB:{rgb.width}:{rgb.height}:'.encode()
        exact=hashlib.sha256(prefix+rgb.tobytes()).hexdigest()
        gray=list(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS).tobytes())
        bits=0
        for y in range(8):
            for x in range(8):bits=(bits<<1)|int(gray[y*9+x]>gray[y*9+x+1])
        return exact,bits,rgb.size


def audit(manifest_path, distance=6):
    if type(distance) is not int or not 0<=distance<=64:raise ValueError('distance must be in 0..64')
    manifest_path=Path(manifest_path)
    raw=manifest_path.read_bytes();manifest=json.loads(raw)
    assets=manifest.get('assets') or []
    if not 1<=len(assets)<=1000:raise ValueError('screening requires 1..1000 images')
    root=(manifest_path.parent/'images').resolve();rows=[];seen=set()
    for item in sorted(assets,key=lambda row:row['relative_path']):
        relative=item['relative_path']
        if relative in seen:raise ValueError('duplicate manifest path')
        seen.add(relative)
        if item.get('split') not in {'train','val','test'}:raise ValueError('missing split')
        path=(root/relative).resolve();path.relative_to(root)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('input hash changed')
        exact,dhash,size=features(path)
        if size!=(item['width'],item['height']):raise ValueError('input dimensions changed')
        rows.append({'path':relative,'split':item['split'],'source_group':item['source_group'],
                     'sha256':item['sha256'],'decoded_rgb_sha256':exact,'dhash64':dhash})
    exact_pairs=[];candidates=[]
    for a,b in itertools.combinations(rows,2):
        common={'a':a['path'],'a_split':a['split'],'b':b['path'],'b_split':b['split'],'cross_split':a['split']!=b['split']}
        if a['decoded_rgb_sha256']==b['decoded_rgb_sha256']:
            exact_pairs.append(common)
        else:
            hamming=(a['dhash64']^b['dhash64']).bit_count()
            if hamming<=distance:candidates.append({**common,'hamming_distance':hamming,'human_decision':None})
    return {'status':'screening_complete_not_human_adjudicated','created_at':datetime.now(timezone.utc).isoformat(),
        'timing':'post-hoc: E1 and first E2 seed test were already evaluated; no split or experiment parameters changed',
        'manifest_file_sha256':hashlib.sha256(raw).hexdigest(),'split_manifest_hash':manifest.get('split_manifest_hash'),
        'images_verified':len(rows),'pairs_examined':len(rows)*(len(rows)-1)//2,
        'method':{'exact':'SHA256 of dimensions and decoded RGB bytes; no EXIF rotation applied',
            'candidate':'64 horizontal adjacent-pixel comparisons of LANCZOS 9x8 grayscale; Hamming distance <= threshold',
            'threshold':distance},
        'decoded_identical_pairs':exact_pairs,'decoded_identical_cross_split':sum(p['cross_split'] for p in exact_pairs),
        'candidate_pairs':candidates,'candidate_cross_split':sum(p['cross_split'] for p in candidates),
        'candidate_split_pairs':dict(Counter('/'.join(sorted((p['a_split'],p['b_split']))) for p in candidates)),
        'limitations':['Perceptual proximity is not proof of duplication; no candidates does not prove absence.',
            'Rotation/crop/stain changes and patient/slide provenance are not resolved. Human decisions remain empty.',
            'Do not silently repair the already-used test split; any later adjudication and rerun require a new disclosed protocol.'],
        'image_fingerprints':[{**row,'dhash64':f"{row['dhash64']:016x}"} for row in rows]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--distance',type=int,default=6)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('output exists; preserve historical screening')
    result=audit(args.manifest,args.distance);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','images_verified','pairs_examined','decoded_identical_cross_split','candidate_cross_split','candidate_split_pairs')}))

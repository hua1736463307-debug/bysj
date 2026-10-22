"""Prepare a new conservative image-cluster import; never modify old snapshots.

All exact and perceptual candidate edges from the frozen audit are co-grouped,
irrespective of old split or model metrics. This is a development data revision,
not human adjudication and not an unseen confirmatory test set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def grouped_sources(assets,audit):
    names={a['relative_path'] for a in assets}
    if len(names)!=len(assets):raise ValueError('duplicate image path')
    parent={name:name for name in names}
    def find(name):
        while parent[name]!=name:
            parent[name]=parent[parent[name]];name=parent[name]
        return name
    def union(a,b):
        if a not in parent or b not in parent:raise ValueError('audit refers to unknown image')
        left,right=find(a),find(b);parent[max(left,right)]=min(left,right)
    original={}
    for asset in assets:
        name=asset['relative_path'];group=asset['source_group']
        if group in original:union(name,original[group])
        else:original[group]=name
    for pair in audit['decoded_identical_pairs']+audit['candidate_pairs']:union(pair['a'],pair['b'])
    components={}
    for name in sorted(names):components.setdefault(find(name),[]).append(name)
    lookup={a['relative_path']:a for a in assets};result={}
    for members in components.values():
        if len(members)==1:group=lookup[members[0]]['source_group']
        else:group='cluster:'+hashlib.sha256(json.dumps([(n,lookup[n]['sha256']) for n in members],separators=(',',':')).encode()).hexdigest()[:32]
        for name in members:result[name]=group
    return result


def prepare(manifest_path,audit_path,output):
    raw=manifest_path.read_bytes();manifest=json.loads(raw);audit_raw=audit_path.read_bytes();audit=json.loads(audit_raw)
    if audit['manifest_file_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('audit refers to a different manifest')
    assets=manifest['assets'];groups=grouped_sources(assets,audit)
    root=manifest_path.parent.resolve();files=[]
    for asset in assets:
        relative=Path(asset['relative_path']);image=(root/'images'/relative).resolve();image.relative_to(root/'images')
        label=(root/'labels'/relative.with_suffix('.txt')).resolve();label.relative_to(root/'labels')
        if hashlib.sha256(image.read_bytes()).hexdigest()!=asset['sha256']:raise ValueError('source image changed')
        if not label.is_file():raise ValueError('source label missing')
        files.append((image,relative.as_posix(),label,relative.with_suffix('.txt').as_posix()))
    output.mkdir(parents=True,exist_ok=False)
    archive=output/'import.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for image,name,label,label_name in files:bundle.write(image,name);bundle.write(label,label_name)
    mapping=output/'source_groups.json';mapping.write_text(json.dumps(groups,sort_keys=True,indent=2),encoding='utf-8')
    report={'status':'prepared_not_imported','purpose':'post-hoc conservative image-cluster development revision',
        'images':len(assets),'groups':len(set(groups.values())),'class_map':manifest['class_map'],
        'source_manifest_sha256':hashlib.sha256(raw).hexdigest(),'audit_sha256':hashlib.sha256(audit_raw).hexdigest(),
        'group_map_sha256':hashlib.sha256(mapping.read_bytes()).hexdigest(),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'group_level':'image_cluster','fixed_split_seed':42,'clinical_or_unseen_test_claim':False,
        'label_sha256':{name:hashlib.sha256(label.read_bytes()).hexdigest() for _,_,label,name in files}}
    (output/'PREPARATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return {k:report[k] for k in ('status','images','groups','group_map_sha256','archive_sha256')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','audit','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.manifest,args.audit,args.output)))

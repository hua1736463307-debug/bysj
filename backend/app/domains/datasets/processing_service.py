"""Train-only tile derivatives with immutable original-image val/test snapshots.

E2 evaluates ordinary whole-image predictions on the common original test set;
this service does not substitute tile-level metrics for original-image metrics.
"""
import json
from pathlib import Path
import shutil
import uuid
from PIL import Image
from app.core.errors import ApiError
from app.core.storage import atomic_write_bytes, ensure_dir, normalize_relative, resolve_under, sha256_file, remove_tree
from app.db.base import utcnow
from app.db.models import DatasetAsset, DatasetVersion
from app.domains.datasets.split_service import build_manifest
from app.domains.job_service import canonical_hash
from app.ml.geometry import cxcywh_norm_to_xyxy, xyxy_to_cxcywh_norm
from app.ml.tiling import plan_tiles, visibility, clip_box_to_tile


def result(version):
    return {'version_id':version.id,'parent_version_id':version.parent_version_id,'state':version.state,
            'manifest_sha256':version.manifest_sha256,'split_manifest_hash':version.split_manifest_hash,
            'split_counts':version.split_counts_json,'asset_count':version.asset_count,'processing':version.metadata_json['processing']}


def process_train_tiles(ctx):
    payload=ctx.job.input_json;db=ctx.db;settings=ctx.settings
    parent=db.get(DatasetVersion,payload['parent_version_id'])
    if parent is None or parent.deleted_at is not None or parent.state!='frozen':
        raise ApiError(409,'parent_not_frozen','切片只接受已冻结父版本')
    child_id=str(uuid.uuid5(uuid.NAMESPACE_URL,'bysj:train-tiles:'+ctx.job.id))
    fingerprint=canonical_hash({'payload':payload,'parent_manifest':parent.manifest_sha256})
    existing=db.get(DatasetVersion,child_id)
    if existing:
        if existing.state!='frozen' or existing.metadata_json.get('processing_hash')!=fingerprint:
            raise ApiError(409,'processing_conflict','已存在不同的处理版本')
        return result(existing)
    assets=db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id==parent.id,DatasetAsset.deleted_at.is_(None)).order_by(DatasetAsset.relative_path).all()
    parent_root=Path(parent.root_path).resolve();parent_root.relative_to(settings.data_root.resolve())
    tiles={a.id:plan_tiles(a.width,a.height,payload['tile_size'],payload['overlap']) for a in assets if a.split=='train'}
    derived_count=sum(len(items) for items in tiles.values())+sum(a.split!='train' for a in assets)
    estimated_bytes=sum(t.width*t.height*3 for items in tiles.values() for t in items)+sum(a.size for a in assets if a.split!='train')
    if derived_count>settings.max_zip_files or estimated_bytes>settings.max_extract_bytes:
        raise ApiError(422,'processing_limit','切片规模超出配置上限，请增大切片尺寸或减小重叠')
    if shutil.disk_usage(settings.data_root).free<estimated_bytes+64*1024*1024:
        raise ApiError(507,'insufficient_storage','磁盘空间不足以创建处理版本')
    stage_base=ensure_dir(settings.data_root/'staging'/'processing'/ctx.job.id)
    stage=stage_base/str(ctx.attempt)
    stage.mkdir(exist_ok=False)
    final=resolve_under(settings.data_root,f'datasets/{parent.dataset_id}/{child_id}')
    entries=[];lineage={};fixed={}

    def append(relative, source_asset, annotations, width, height, transform):
        path=resolve_under(stage/'images',relative,must_exist=True)
        label=normalize_relative(relative).with_suffix('.txt').as_posix()
        atomic_write_bytes(resolve_under(stage/'labels',label),''.join(f"{a['class_id']} {a['cx']:.6f} {a['cy']:.6f} {a['w']:.6f} {a['h']:.6f}\n" for a in annotations).encode())
        entries.append({'relative_path':relative,'sha256':sha256_file(path),'size':path.stat().st_size,
            'width':width,'height':height,'source_group':source_asset.source_group,'root_asset':source_asset.relative_path,'annotations':annotations})
        fixed[relative]=source_asset.split
        lineage[relative]=(source_asset,transform)

    try:
        for index,asset in enumerate(assets):
            ctx.check_cancelled()
            source=resolve_under(parent_root/'images',asset.relative_path,must_exist=True)
            if sha256_file(source)!=asset.content_hash:raise ApiError(409,'source_tampered','父版本图像摘要改变')
            if asset.split in {'val','test'}:
                target=resolve_under(stage/'images',asset.relative_path);ensure_dir(target.parent);shutil.copy2(source,target)
                append(asset.relative_path,asset,asset.annotations_json or [],asset.width,asset.height,{'kind':'identity'})
                source_label=resolve_under(parent_root,asset.label_path,must_exist=True)
                shutil.copy2(source_label,resolve_under(stage/'labels',normalize_relative(asset.relative_path).with_suffix('.txt').as_posix()))
            elif asset.split=='train':
                with Image.open(source) as original:
                    image=original.convert('RGB')
                    for tile in tiles[asset.id]:
                        ctx.check_cancelled()
                        relative=f'train-tiles/{asset.id}/{tile.index:04d}.png'
                        target=resolve_under(stage/'images',relative);ensure_dir(target.parent)
                        image.crop((tile.x1,tile.y1,tile.x2,tile.y2)).save(target,format='PNG')
                        annotations=[]
                        for annotation in asset.annotations_json or []:
                            box=cxcywh_norm_to_xyxy([annotation[k] for k in ('cx','cy','w','h')],image.width,image.height)
                            if visibility(box,tile)<payload['min_visibility']:continue
                            clipped=clip_box_to_tile(box,tile)
                            if clipped is None:continue
                            local=[clipped[0]-tile.x1,clipped[1]-tile.y1,clipped[2]-tile.x1,clipped[3]-tile.y1]
                            coords=xyxy_to_cxcywh_norm(local,tile.width,tile.height)
                            annotations.append({'class_id':annotation['class_id'],**dict(zip(('cx','cy','w','h'),coords))})
                        append(relative,asset,annotations,tile.width,tile.height,
                            {'kind':'train_tile','origin_xyxy':[tile.x1,tile.y1,tile.x2,tile.y2],'parent_asset_id':asset.id})
            else:raise ApiError(409,'invalid_parent_split','父版本存在未划分资产')
            ctx.progress(.8*(index+1)/len(assets),'processing',f'处理 {index+1}/{len(assets)} 张来源图')
        config=parent.split_config_json
        ratios=config['ratios'];seed=int(config['seed'])
        manifest,stats=build_manifest(entries,class_map=parent.class_map_json,seed=seed,
            ratios=[ratios[k] for k in ('train','val','test')],dataset_id=parent.dataset_id,version_id=child_id,
            parent_version_id=parent.id,group_level=config.get('group_level','original_image'),fixed_splits=fixed)
        for item in manifest['assets']:
            if item['split']=='train':
                label=resolve_under(stage/'labels',normalize_relative(item['relative_path']).with_suffix('.txt').as_posix())
                atomic_write_bytes(label,''.join(f"{a['class_id']} {a['cx']:.6f} {a['cy']:.6f} {a['w']:.6f} {a['h']:.6f}\n" for a in item['annotations']).encode())
        atomic_write_bytes(stage/'manifest.json',json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2).encode())
        atomic_write_bytes(stage/'processing.json',json.dumps({'fingerprint':fingerprint,'job_id':ctx.job.id},sort_keys=True).encode())
        # Only the current attempt can publish/register. A crash after rename is
        # recoverable by byte-for-byte verification against regenerated output.
        ctx.lock_for_write()
        if final.exists():
            expected={p.relative_to(stage).as_posix():sha256_file(p) for p in stage.rglob('*') if p.is_file()}
            actual={p.relative_to(final).as_posix():sha256_file(p) for p in final.rglob('*') if p.is_file()}
            if expected!=actual:raise ApiError(409,'processing_artifact_conflict','已有产物不匹配，保留待检查')
        else:
            ensure_dir(final.parent);stage.rename(final)
        child=DatasetVersion(id=child_id,dataset_id=parent.dataset_id,dataset_role=parent.dataset_role,
            parent_version_id=parent.id,status='ready',state='frozen',frozen_at=utcnow(),root_path=str(final),
            manifest_path=str(final/'manifest.json'),class_map_json=parent.class_map_json,
            split_config_json={'seed':seed,'ratios':manifest['ratios'],'group_level':manifest['group_level']},
            metadata_json={'processing_hash':fingerprint,'processing':{**payload,'evaluation_space':'original_image','transformed_splits':['train']},'job_id':ctx.job.id},
            content_hash=stats['manifest_sha256'],manifest_sha256=stats['manifest_sha256'],
            split_manifest_hash=stats['split_manifest_hash'],class_map_hash=stats['class_map_hash'],
            split_counts_json=stats['split_counts'],asset_count=stats['asset_count'],
            created_by=ctx.job.owner_username,created_by_user_id=ctx.job.owner_user_id)
        db.add(child)
        for item in manifest['assets']:
            parent_asset,transform=lineage[item['relative_path']]
            db.add(DatasetAsset(dataset_version_id=child_id,relative_path=item['relative_path'],basename=Path(item['relative_path']).name,
                size=item['size'],width=item['width'],height=item['height'],content_hash=item['sha256'],
                annotations_json=parent_asset.annotations_json if item['split'] in {'val','test'} else item['annotations'],
                label_path='labels/'+normalize_relative(item['relative_path']).with_suffix('.txt').as_posix(),source_group=item['source_group'],
                split=item['split'],group_level=parent_asset.group_level,root_asset_id=parent_asset.root_asset_id or parent_asset.id,
                original_asset_id=parent_asset.id,transform_json=transform))
        db.commit()
        return result(child)
    except Exception:
        db.rollback();raise
    finally:
        if stage.exists():remove_tree(stage,allowed_root=stage_base)

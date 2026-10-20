import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from .features import encode_coco17, FEATURE_SCHEMA

MAX_BYTES = 64 * 1024 * 1024
MAX_ROWS = 50000
SPLITS = ('train', 'validation', 'test')


def dataset_root():
    return Path(os.getenv('POSE_TRAINING_DATASETS_DIR', str(Path(os.getenv('DEEPVISION_DATA_DIR', '/app/data')) / 'floor_lying_datasets'))).resolve()


def run_root():
    return Path(os.getenv('POSE_TRAINING_RUNS_DIR', str(Path(os.getenv('DEEPVISION_DATA_DIR', '/app/data')) / 'pose_training_runs'))).resolve()


def release_lock():
    return run_root().parent / 'temp/pose-training-release.lock'


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def resolve_dataset(dataset_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', dataset_id):
        raise ValueError('数据集 ID 不合法')
    root = dataset_root()
    folder = (root / dataset_id).resolve()
    if folder.parent != root or not folder.is_dir():
        raise ValueError('受管姿态数据集不存在')
    manifest = folder / 'pose-dataset.json'
    if manifest.is_file():
        if manifest.stat().st_size > 65536:
            raise ValueError('数据集说明过大')
        config = json.loads(manifest.read_text(encoding='utf-8'))
        if not isinstance(config, dict) or not isinstance(config.get('index', 'pose/index.jsonl'), str):
            raise ValueError('数据集说明格式不合法')
        mode = config.get('label_mode')
        if mode not in ('posture', 'source_proxy'):
            raise ValueError('label_mode 只能是 posture 或 source_proxy')
        relative = Path(config.get('index', 'pose/index.jsonl'))
        if relative.is_absolute():
            raise ValueError('索引必须是数据集内相对路径')
        index = (folder / relative).resolve()
        if not index.is_relative_to(folder):
            raise ValueError('索引越过数据集目录')
        name = str(config.get('name') or dataset_id)[:200]
    else:
        # Existing prepared CAUCAFall layout; source labels remain explicitly provisional.
        mode, index, name = 'source_proxy', folder / 'pose/proxy-pose.jsonl', dataset_id
    if not index.is_file() or not index.resolve().is_relative_to(folder):
        raise ValueError('姿态索引不存在')
    if not 0 < index.stat().st_size <= MAX_BYTES:
        raise ValueError('姿态索引大小超限或为空')
    return dict(id=dataset_id, name=name, label_mode=mode, index=index, folder=folder)


def catalog():
    root = dataset_root()
    if not root.is_dir():
        return []
    result = []
    for folder in sorted(root.iterdir()):
        if len(result) >= 200:
            break
        try:
            d = resolve_dataset(folder.name)
            result.append(dict(id=d['id'], name=d['name'], label_mode=d['label_mode'], index_bytes=d['index'].stat().st_size))
        except (ValueError, OSError, json.JSONDecodeError):
            continue
    return result


def load_dataset(dataset_id, expected_sha=None, check_cancel=lambda: None):
    d = resolve_dataset(dataset_id)
    with d['index'].open('rb') as f:
        raw = f.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('姿态索引大小超限')
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha and digest != expected_sha:
        raise ValueError('数据集索引已变化，请重新预检')
    rows, groups, instances = [], {}, set()
    for line in raw.decode('utf-8').splitlines():
        if not line.strip():
            continue
        if len(rows) >= MAX_ROWS:
            raise ValueError('样本数量超限')
        if len(rows) % 100 == 0:
            check_cancel()
        r = json.loads(line)
        if not isinstance(r, dict):
            raise ValueError('每行必须是逐人样本对象')
        if r.get('split') not in SPLITS:
            raise ValueError('需要固定 train/validation/test 划分')
        instance = (str(r.get('image_id') or ''), str(r.get('person_id') or ''))
        if not all(instance) or instance in instances:
            raise ValueError('缺少或重复逐人样本 ID')
        instances.add(instance)
        for key in ('subject_id', 'clip_id', 'image_id', 'image_sha256', 'pixel_sha256'):
            value = str(r.get(key) or '')
            if not value:
                raise ValueError(f'缺少分组字段 {key}')
            if key.endswith('sha256') and not re.fullmatch('[a-f0-9]{64}', value):
                raise ValueError('图片哈希不合法')
            group = (key, value)
            if group in groups and groups[group] != r['split']:
                raise ValueError(f'训练/验证/测试泄漏: {key}')
            groups[group] = r['split']
        if d['label_mode'] == 'source_proxy':
            if r.get('annotation_status') != 'source_proxy_only' or type(r.get('source_class')) is not int or r['source_class'] not in (0, 1):
                raise ValueError('来源标签契约不合法')
            label = r['source_class']
        else:
            if r.get('annotation_status') != 'verified' or r.get('posture_label') not in ('lying', 'not_lying'):
                raise ValueError('需要已审核的 lying/not_lying 标签')
            label = int(r['posture_label'] == 'lying')
        box = np.asarray(r.get('bbox'), dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]-box[:2]) <= 0:
            raise ValueError('人物框不合法')
        p = r.get('keypoints')
        if p is not None:
            a = np.asarray(p, dtype=float)
            if a.shape != (17, 3) or not np.isfinite(a).all() or ((a[:, 2] < 0) | (a[:, 2] > 1)).any():
                raise ValueError('COCO17 关键点或置信度不合法')
        encoded = encode_coco17(p, box)
        usable = bool(encoded['sufficient'] and r.get('pose_usable') is not False)
        rows.append(dict(image_id=instance[0], person_id=instance[1], split=r['split'], label=label,
                         vector=encoded['vector'], usable=usable, reason=encoded['reason']))
    summary = {}
    for split in SPLITS:
        part = [r for r in rows if r['split'] == split]
        counts = {str(c): sum(r['label'] == c and r['usable'] for r in part) for c in (0, 1)}
        if not part or (split != 'test' and min(counts.values()) < 2):
            raise ValueError(f'{split} 样本不足；训练和验证的每类至少需要 2 个可用样本')
        summary[split] = dict(total=len(part), usable=sum(r['usable'] for r in part), usable_classes=counts)
    info = dict(dataset_id=dataset_id, name=d['name'], label_mode=d['label_mode'], index_sha256=digest,
                feature_schema=FEATURE_SCHEMA, splits=summary, production_compatible=False,
                labels=['nofall', 'fall'] if d['label_mode']=='source_proxy' else ['not_lying', 'lying'],
                scope='分组探索实验；不具备完整融合模型或现场验收资格')
    return rows, info

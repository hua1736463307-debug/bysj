"""Draft-owned vocabularies without changing source dataset slots or labels."""
import copy
from functools import lru_cache
from pathlib import Path

SCOPE = "model_session"


def isolated_draft(draft):
    return bool(draft and (getattr(draft, "class_context_json", None) or {}).get("scope") == SCOPE)


def isolated_version(version):
    return (version.metadata_json or {}).get("annotation_class_scope") == SCOPE


def session_base_annotations(version, annotations):
    if isolated_version(version) and not (version.metadata_json or {}).get("annotation_session_has_owned_labels"):
        return []
    return annotations or []


def manifest_frontend_annotations(annotations):
    """Convert indexed YOLO centers to the page/transform API's normalized corners."""
    from dataset_versioning import annotation_to_yolo_line
    shapes = []
    for annotation in annotations or []:
        parts = annotation_to_yolo_line(annotation).split()
        class_id, values = int(parts[0]), [float(value) for value in parts[1:]]
        if len(values) == 4:
            x, y, width, height = values
            shapes.append({'class': class_id, 'type': 'bbox', 'x': x-width/2,
                           'y': y-height/2, 'width': width, 'height': height})
        elif len(values) >= 6 and len(values) % 2 == 0:
            shapes.append({'class': class_id, 'type': 'polygon', 'points': values})
        else:
            raise ValueError("当前标注会话的增强标签格式无效")
    return shapes


def normalize_class_context(classes, context=None, previous=None):
    context = context or {}
    previous = previous or {}
    if not isinstance(context, dict):
        raise ValueError("标注会话类别配置必须是对象")
    inactive = context.get("inactive_class_ids", previous.get("inactive_class_ids", []))
    if not isinstance(inactive, list) or any(type(value) is not int or value < 0 or value >= len(classes) for value in inactive):
        raise ValueError("标注会话停用类别编号无效")
    inactive = sorted(set(inactive))
    if set(previous.get("inactive_class_ids") or []) - set(inactive):
        raise ValueError("当前会话中已停用的类别不能重新启用")
    if not classes or len(inactive) >= len(classes):
        raise ValueError("标注会话至少保留一个启用类别")
    labels = context.get("labels_zh", previous.get("labels_zh", {}))
    if not isinstance(labels, dict) or any(not isinstance(value, str) or len(value) > 80 for value in labels.values()):
        raise ValueError("类别中文名称格式无效")
    return {"scope": SCOPE, "version": 1,
            **{key: previous[key] for key in ("initialization_mode", "initialization_note") if key in previous},
            "native_classes": list(previous.get("native_classes") or classes),
            "inactive_class_ids": inactive,
            "labels_zh": {name: value.strip() for name, value in labels.items() if name in classes and value.strip()}}


class DraftDatasetView:
    """Read-only class projection; SQLAlchemy source rows are never modified."""
    def __init__(self, source, draft):
        self._source = source
        classes = list(draft.class_names_json or [])
        context = draft.class_context_json or {}
        self.class_map_json = {str(index): name for index, name in enumerate(classes)}
        self.metadata_json = {**copy.deepcopy(source.metadata_json or {}),
            "annotation_class_scope": SCOPE,
            "annotation_session_has_owned_labels": bool(
                str(source.dataset_id) == str(draft.temporary_dataset_id or "")
                and (source.metadata_json or {}).get("source_type") == "annotation_temporary"
                and str((source.metadata_json or {}).get("source_dataset_id") or "") == str(draft.dataset_id)),
            "annotation_classes_source": SCOPE,
            "annotation_classes": classes,
            "annotation_class_bindings": copy.deepcopy(draft.class_bindings_json or []),
            "annotation_class_labels_zh": copy.deepcopy(context.get("labels_zh") or {}),
            "annotation_inactive_class_ids": list(context.get("inactive_class_ids") or []),
            "annotation_model_path": draft.model_path,
            "annotation_model_name": draft.model_name,
            "annotation_model_native_classes": list(context.get("native_classes") or classes),
            "annotation_model": {"path": draft.model_path, "name": draft.model_name,
                                 "geometry": draft.annotation_geometry},
            "annotation_geometry": draft.annotation_geometry}

    def __getattr__(self, name):
        return getattr(self._source, name)

    def to_dict(self):
        return {**self._source.to_dict(), "class_map": copy.deepcopy(self.class_map_json),
                "metadata": copy.deepcopy(self.metadata_json)}


def draft_dataset_view(source, draft):
    return DraftDatasetView(source, draft) if isolated_draft(draft) else source


def validate_session_annotation_ids(annotations, classes):
    for annotation in annotations:
        value = annotation.get("page_class_id", annotation.get("class", annotation.get("class_id")))
        if value is None and annotation.get("yolo_line"):
            value = str(annotation['yolo_line']).split()[0]
        try:
            index = int(value)
            if isinstance(value, bool) or (isinstance(value, float) and value != index):
                raise ValueError()
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("标注缺少当前会话的有效类别编号") from exc
        if not 0 <= index < len(classes):
            raise ValueError("标注包含不属于当前会话的类别编号")


@lru_cache(maxsize=32)
def _checkpoint_classes(path, size, modified_ns):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = (checkpoint.get("ema") or checkpoint.get("model")) if isinstance(checkpoint, dict) else checkpoint
    names = getattr(model, "names", None)
    if names is None and isinstance(checkpoint, dict):
        names = checkpoint.get("names")
    if isinstance(names, dict):
        keys = sorted(names, key=lambda key: int(key))
        if [int(key) for key in keys] != list(range(len(keys))):
            raise ValueError("模型原生类别编号不连续，无法安全初始化标注")
        names = [names[key] for key in keys]
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError("所选模型未提供类别表，无法初始化新标注")
    result = tuple(str(name).strip() for name in names)
    if any(not name for name in result) or len(set(name.casefold() for name in result)) != len(result):
        raise ValueError("所选模型类别名称为空或重复")
    return result


def checkpoint_class_names(path):
    path = Path(path)
    stat = path.stat()
    return list(_checkpoint_classes(str(path), stat.st_size, stat.st_mtime_ns))

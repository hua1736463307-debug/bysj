"""Explicit class-name equivalence without changing dataset or model names."""

import re
from collections.abc import Mapping


def annotation_class_key(value):
    key = re.sub(r"\s+", " ", str(value or "").strip()).casefold()
    return "mobilephone" if key == "cell phone" else key


def class_name_ids(names, inactive_ids=()):
    """Index names by meaning, rejecting ambiguous aliases on either side."""
    items = names.items() if isinstance(names, Mapping) else enumerate(names)
    result = {}
    for class_id, name in items:
        if class_id in inactive_ids:
            continue
        key = annotation_class_key(name)
        if key in result:
            raise ValueError(f"类别名称或别名存在歧义，无法一一映射: {name}")
        result[key] = class_id
    return result


def routing_page_class_ids(names, bindings):
    """Resolve reserved alias slots only through one explicit page binding."""
    groups = {}
    for index, name in enumerate(names):
        groups.setdefault(annotation_class_key(name), []).append(index)
    bound = {item["page_index"] for item in bindings}
    ignored = set()
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        chosen = bound.intersection(indices)
        if len(chosen) != 1:
            raise ValueError(f"类别名称或别名存在歧义，缺少唯一启用类别绑定: {key}")
        ignored.update(set(indices) - chosen)
    return class_name_ids(names, ignored)


def resolve_model_page_classes(page_classes, model_classes, inactive_ids):
    """Keep stable dataset slots for retired or equivalent model vocabularies."""
    page_ids = class_name_ids(page_classes, inactive_ids)
    model_ids = class_name_ids(model_classes)
    if not inactive_ids and page_ids.keys() != model_ids.keys():
        # Preserve the existing import behavior for a different vocabulary.
        return list(model_classes)
    known_keys = {annotation_class_key(name) for name in page_classes}
    unknown = [name for name in model_classes if annotation_class_key(name) not in known_keys]
    missing = [name for index, name in enumerate(page_classes)
               if index not in inactive_ids and annotation_class_key(name) not in model_ids]
    if unknown or missing:
        raise ValueError(f"模型类别与数据集无法一一映射；未知模型类别: {unknown}；缺少启用类别: {missing}")
    return list(page_classes)

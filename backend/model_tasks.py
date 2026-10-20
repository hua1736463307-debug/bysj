"""Shared task classification; YOLO pose variants belong to pose modules."""
import re

def registry_task(record):
    metadata = record.metadata_json or {}
    task = metadata.get('task') or metadata.get('task_type')
    text = ' '.join(str(v or '') for v in [getattr(record,'name',''), record.path, metadata.get('base_model'), metadata.get('display_name')]).lower()
    if task == 'pose' or re.search(r'yolo[^\s/]*[-_ ]?pose|yolo\s*pose', text): return 'pose'
    return task

"""Bounded inference for the models behind selected user algorithms."""
import copy
from monitoring_policy import CLASSES


def _category(detection):
    label = str(detection.get('class') or detection.get('class_name') or '').casefold()
    return next((key for key, aliases in CLASSES.items() if label in aliases), label)


def _box(detection):
    value = detection.get('bbox') or {}
    return [value.get(k, 0) for k in ('x1','y1','x2','y2')] if isinstance(value, dict) else value


def _iou(a, b):
    a, b = _box(a), _box(b)
    if len(a) != 4 or len(b) != 4: return 0
    intersection = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0, min(a[3],b[3])-max(a[1],b[1]))
    union = max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-intersection
    return intersection / union if union else 0


async def infer_selected_models(scheduler, image, models, confidence, iou, should_stop=lambda: False,
                                *, enabled_events=None, revision=None):
    if hasattr(scheduler, "infer_monitoring"):
        if should_stop():
            return None
        result = await scheduler.infer_monitoring(image, models, confidence, iou, tracking=True,
                                                  enabled_events=enabled_events or [], revision=revision)
        outputs = result.get("results", [])
        candidates = [d for item in outputs for d in item.get("detections", [])[:256]]
        candidates.sort(key=lambda d: float(d.get('confidence') or 0), reverse=True)
        detections = []
        for detection in candidates:
            if not any(_category(existing) == _category(detection) and _iou(existing, detection) > iou
                       for existing in detections):
                detections.append(copy.deepcopy(detection))
        return {"detections": detections, "detections_count": len(detections),
                "inference_time": sum(float(item.get("inference_time") or 0) for item in outputs),
                "timestamp": __import__('time').time()}
    models = list(dict.fromkeys(models))
    if not 1 <= len(models) <= 3: raise RuntimeError('算法配置无效')
    results = []
    for model in models:
        if should_stop(): return None
        result = await scheduler.infer(image, model, confidence, iou)
        if result.get('gpu_waiting'): return result
        if result.get('error'): raise RuntimeError('算法推理失败，请联系管理员检查服务')
        results.append(result)
    if len(results) == 1: return results[0]
    # Keep full recognition input for behavior rules; UI filtering happens afterwards.
    candidates = [d for result in results for d in result.get('detections', [])[:256]]
    candidates.sort(key=lambda d: float(d.get('confidence') or 0), reverse=True)
    detections = []
    for detection in candidates:
        if not any(_category(d) == _category(detection) and _iou(d, detection) > iou for d in detections):
            detections.append(copy.deepcopy(detection))
    return {**results[0], 'detections':detections, 'detections_count':len(detections),
            'inference_time':sum(float(r.get('inference_time') or 0) for r in results)}

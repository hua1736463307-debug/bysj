"""Versioned static COCO-17 features. No history, tracking or temporal encoder."""
import math
import numpy as np

FEATURE_SCHEMA = 'coco17_static_v1'
FEATURE_SIZE = 120
BONES = ((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
         (11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4))


def encode_coco17(points, bbox, confidence=.25, min_keypoints=8):
    vector = np.zeros(FEATURE_SIZE, dtype=np.float32)
    result = dict(vector=vector, mask=[False]*17, keypoints_available=0,
                  sufficient=False, reason='INVALID_KEYPOINTS', schema=FEATURE_SCHEMA)
    try:
        p = np.asarray(points, dtype=np.float64)
        b = np.asarray(bbox, dtype=np.float64)
    except (TypeError, ValueError):
        return result
    if p.shape != (17,3) or b.shape != (4,) or not np.isfinite(b).all():
        return result
    w,h = b[2:]-b[:2]
    if w <= 0 or h <= 0:
        return result
    padding = max(w,h)*.1
    mask = (np.isfinite(p).all(axis=1) & (p[:,2]>=confidence) & (p[:,2]<=1)
            & (p[:,0]>=b[0]-padding) & (p[:,0]<=b[2]+padding)
            & (p[:,1]>=b[1]-padding) & (p[:,1]<=b[3]+padding))
    count = int(mask.sum())
    result.update(mask=mask.tolist(), keypoints_available=count,
                  quality=float(count/17), reason='INSUFFICIENT_KEYPOINTS')
    if count < min_keypoints or not mask[[5,6,11,12]].all():
        return result
    shoulder = p[[5,6],:2].mean(axis=0)
    hip = p[[11,12],:2].mean(axis=0)
    scale = float(np.linalg.norm(shoulder-hip))
    if scale < max(1., .02*math.hypot(w,h)):
        result['reason'] = 'DEGENERATE_TORSO'
        return result
    normalized = np.zeros((17,4), dtype=np.float32)
    normalized[mask,:2] = np.clip((p[mask,:2]-hip)/scale,-8,8)
    normalized[mask,2] = p[mask,2]
    normalized[:,3] = mask
    geometry = []
    for a,bone in BONES:
        delta = normalized[bone,:2]-normalized[a,:2] if mask[a] and mask[bone] else np.zeros(2)
        geometry.extend([float(delta[0]),float(delta[1]),float(np.linalg.norm(delta))])
    quality = [count/17, float(p[mask,2].mean()),
               abs(float(mask[[5,7,9,11,13,15]].mean()-mask[[6,8,10,12,14,16]].mean())),
               scale/math.hypot(w,h)]
    vector[:] = np.concatenate([normalized.ravel(),geometry,quality])
    result.update(sufficient=True, reason='POSE_FEATURES_AVAILABLE')
    return result

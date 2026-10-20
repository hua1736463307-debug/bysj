"""Reject inconsistent YuNet landmarks without requiring frontal symmetry.

Predicted landmarks do not carry visibility scores. This is a geometry guard,
not proof of facial visibility or a measured yaw angle. Validate camera samples
before relying on it to exclude back-of-head false positives.
"""
import numpy as np

POLICY = 'yunet_landmark_structure_v1'


def has_face_structure(row):
    """Accept asymmetric front/profile topology; reject missing/collapsed points.

YuNet order: right eye, left eye, nose, right mouth, left mouth. Distances use
native model coordinates and the face's own axis, independent of crop margins,
image resolution, translation and in-plane rotation. Hidden-side landmarks may
converge in a profile; the nose must still establish a non-degenerate outline.
    """
    try:
        row = np.asarray(row, dtype=np.float64)
        if row.shape != (15,) or not np.isfinite(row).all():
            return False
        x, y, width, height = row[:4]
        if width <= 0 or height <= 0:
            return False
        points = row[4:14].reshape(5, 2)
        relative = (points - [x, y]) / [width, height]
        if np.any(relative < -.15) or np.any(relative > 1.15):
            return False
        size = max(width, height)
        eyes = (points[0] + points[1]) / 2
        mouth = (points[3] + points[4]) / 2
        axis = (mouth - eyes) / size
        length = float(np.linalg.norm(axis))
        if not .18 <= length <= .8:
            return False
        down = axis / length
        across = np.array([down[1], -down[0]])
        spans = []
        for first, second in ((0, 1), (3, 4)):
            pair = (points[second] - points[first]) / size
            span = float(pair @ across)
            # Beyond-profile reversal/inconsistent semantic ordering is invalid.
            # Near-zero span remains valid for an edge-on side face.
            if not -.03 <= span <= .9 or abs(float(pair @ down)) > .18:
                return False
            spans.append(max(0., span))
        nose = (points[2] - eyes) / size
        depth = float(nose @ down) / length
        offset = abs(float(nose @ across))
        if not .05 <= depth <= .95 or offset > .45:
            return False
        # Frontal faces have eye/mouth span; profiles can instead have a nose
        # protruding from converged eye/mouth points. Do not require both eyes.
        return max(*spans, offset) >= .06
    except (TypeError, ValueError, OverflowError):
        return False

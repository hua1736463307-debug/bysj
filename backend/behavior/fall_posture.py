"""Current-frame fallen posture; insufficient evidence never becomes an alarm."""
import math

DEFAULT_FALL_POSTURE = {'keypointConfidence':.25,'torsoAngle':55.0,'aspectRatio':1.2}

def fallen_posture(detection, settings):
    points=detection.get('keypoints');bbox=detection.get('bbox')
    if not isinstance(points,list) or len(points)!=17 or not isinstance(bbox,(list,tuple)) or len(bbox)!=4:return {'valid':False,'fallen':False}
    if not all(type(v) in (int,float) and math.isfinite(v) for v in bbox):return {'valid':False,'fallen':False}
    width=bbox[2]-bbox[0];height=bbox[3]-bbox[1]
    if width<=0 or height<=0:return {'valid':False,'fallen':False}
    anchors=[];padding=max(width,height)*.1
    for index in [5,6,11,12]:
        p=points[index]
        if not isinstance(p,(list,tuple)) or len(p)!=3 or not all(type(v) in (int,float) and math.isfinite(v) for v in p):return {'valid':False,'fallen':False}
        if not (0<=p[2]<=1 and p[2]>=settings['keypointConfidence']) or (p[0]==0 and p[1]==0):return {'valid':False,'fallen':False}
        if not (bbox[0]-padding<=p[0]<=bbox[2]+padding and bbox[1]-padding<=p[1]<=bbox[3]+padding):return {'valid':False,'fallen':False}
        anchors.append(p)
    shoulders=[(anchors[0][i]+anchors[1][i])/2 for i in (0,1)];hips=[(anchors[2][i]+anchors[3][i])/2 for i in (0,1)]
    dx=abs(hips[0]-shoulders[0]);dy=abs(hips[1]-shoulders[1])
    if math.hypot(dx,dy)<max(1.,.05*math.hypot(width,height)):return {'valid':False,'fallen':False}
    angle=math.degrees(math.atan2(dx,dy));aspect=width/height
    return {'valid':True,'fallen':angle>=settings['torsoAngle'] and aspect>=settings['aspectRatio'],'torsoAngle':round(angle,3),'aspectRatio':round(aspect,3)}

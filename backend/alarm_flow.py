"""V3 same-frame alarm execution. Failed/incomplete recognition never implies absence."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import re
import time
import uuid

class FlowError(ValueError):
    pass

def require(ok, message):
    if not ok: raise FlowError(message)

def number(value, low, high):
    return type(value) in (float, int) and math.isfinite(value) and low <= value <= high

def ident(value):
    return isinstance(value, str) and bool(re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', value)) and value not in ('__proto__','constructor','prototype')

def target(value):
    return isinstance(value,str) and 0 < len(value.strip()) <= 80 and not any(ord(c)<32 for c in value) and value not in ('__proto__','constructor','prototype')

def face_capture_policy(config):
    if 'alarmPolicy' in config:
        policy=config['alarmPolicy']
        require(isinstance(policy,str) and policy in ('require_face','manual_review','ignore'),'人脸告警策略无效')
        return policy
    require(type(config.get('affectAlarm')) is bool,'人脸告警策略无效')
    return 'manual_review' if config['affectAlarm'] else 'ignore'

def validate_flow(flow):
    require(isinstance(flow,dict) and flow.get('schemaVersion')==3 and flow.get('mode')=='preview' and flow.get('executionEnabled') is False, '仅支持 V3 配置')
    require(len(json.dumps(flow,ensure_ascii=False,allow_nan=False)) <= 180000,'流程配置过大')
    a=flow.get('alarm',{})
    require(ident(a.get('id')) and isinstance(a.get('name'),str) and 0<len(a['name'].strip())<=80 and isinstance(a.get('description'),str) and len(a['description'])<=240,'告警信息无效')
    d=flow.get('defaults',{}); out=d.get('output',{})
    require(d.get('input')=='original_frame' and d.get('failurePolicy')=='no_alarm' and out.get('format')=='png' and out.get('source')=='original_frame' and out.get('collectAllResults') is True and type(out.get('saveCrop')) is bool,'默认输入输出配置无效')
    nodes=flow.get('nodes');edges=flow.get('edges')
    require(isinstance(nodes,list) and 2<=len(nodes)<=24 and isinstance(edges,list) and len(edges)==len(nodes)-1,'流程应为 2–24 个模块的完整单线')
    by={};rules=set()
    for n in nodes:
        require(isinstance(n,dict) and ident(n.get('id')) and n['id'] not in by and n.get('type') in ('recognize','pose','condition','behavior_condition','face_capture') and isinstance(n.get('config'),dict),'模块标识或类型无效')
        by[n['id']]=n
        p=n.get('position',{});require(number(p.get('x'),0,10000) and number(p.get('y'),0,10000),'模块坐标无效')
        c=n['config']
        if n['type'] in ('recognize','pose'):
            ts=c.get('targets');require(isinstance(ts,list) and 1<=len(ts)<=80 and all(target(t) for t in ts) and len(set(ts))==len(ts),'识别类别无效')
            require(isinstance(c.get('thresholds'),dict) and all(number(c['thresholds'].get(t),.01,.99) for t in ts),'识别阈值无效')
            require(isinstance(c.get('modelRef'),str) and len(c['modelRef'])<=36 and c['modelRef'],'每个识别模块必须选择注册模型')
            require(isinstance(c.get('modelClasses'),list) and all(target(t) for t in c['modelClasses']),'模型类别元数据无效')
            require(c.get('range') in ('original_frame','target_roi'),'识别范围无效')
            crop=c.get('crop',{});pre=c.get('preprocess',{});post=c.get('postprocess',{})
            require(crop.get('clamp') is True and all(type(crop.get(k)) is int and 0<=crop[k]<=300 for k in ('left','right','top','bottom')),'裁剪参数无效')
            require(type(pre.get('imgsz')) is int and pre['imgsz'] in (320,640,960,1280,1920) and pre.get('keepRatio') is True,'预处理参数无效')
            require(type(post.get('mergeDuplicates')) is bool and number(post.get('iou'),.1,.9),'去重参数无效')
        elif n['type']=='face_capture':
            require(set(c) in ({'sourceNodeId','affectAlarm','threshold','minSize'},{'sourceNodeId','alarmPolicy','threshold','minSize'}) and ident(c.get('sourceNodeId')) and number(c.get('threshold'),.01,.99) and type(c.get('minSize')) is int and 20<=c['minSize']<=300,'人脸捕捉规则无效')
            face_capture_policy(c)
        else:
            rs=c.get('rules');require(isinstance(rs,list) and 1<=len(rs)<=80,'条件规则无效')
            seen=set()
            for r in rs:
                require(isinstance(r,dict) and ident(r.get('id')) and r['id'] not in rules and target(r.get('target')) and r['target'] not in seen and r.get('sourceNodeId')==c.get('sourceNodeId'),'规则标识或引用无效')
                rules.add(r['id']);seen.add(r['target'])
                require(isinstance(r.get('name'),str) and 0<len(r['name'].strip())<=80 and type(r.get('minimumCount')) is int and 1<=r['minimumCount']<=10,'规则参数无效')
                for side in ('present','absent'):
                    v=r.get(side,{})
                    require(v.get('state') in ('alarm','no_alarm') and isinstance(v.get('text'),str) and 0<len(v['text'].strip())<=120,'规则输出无效')
        if n['type']=='behavior_condition':validate_behavior_condition(c)
    require(sum(n['type']=='face_capture' for n in nodes)<=1,'每条流程最多一个人脸捕捉模块')
    outgoing={};incoming={};eid=set()
    for e in edges:
        require(isinstance(e,dict) and ident(e.get('id')) and e['id'] not in eid and e.get('source') in by and e.get('target') in by and e['source']!=e['target'] and e['source'] not in outgoing and e['target'] not in incoming,'连线无效或分叉')
        eid.add(e['id']);outgoing[e['source']]=e['target'];incoming[e['target']]=e['source']
    starts=[id for id in by if id not in incoming];require(len(starts)==1,'流程应为完整单线')
    ordered=[];prior={};cursor=starts[0]
    while cursor:
        require(cursor not in prior,'流程不能循环');n=by[cursor];c=n['config']
        if n['type']=='face_capture':
            source=prior.get(c['sourceNodeId'])
            require(source and source['type'] in ('recognize','pose') and 'person' in source['config']['targets'],'人脸捕捉必须引用前序人员识别模块')
            require(not any(p['type'] in ('condition','behavior_condition') for p in prior.values()),'人脸捕捉应放在判断模块之前')
        elif n['type'] in ('condition','behavior_condition') or c['range']=='target_roi':
            source=prior.get(c.get('sourceNodeId'));require(source and source['type'] in ('recognize','pose'),'必须引用前序识别模块')
            if n['type'] in ('recognize','pose'): require(c.get('sourceTarget') in source['config']['targets'],'裁剪依据类别无效')
            elif n['type']=='condition': require({r['target'] for r in c['rules']}==set(source['config']['targets']),'条件规则须覆盖引用模块的类别')
            else: require(source['type']=='pose' and 'person' in source['config']['targets'],'倒地行为必须引用前序姿态识别的人员结果')
        ordered.append(n);prior[cursor]=n;cursor=outgoing.get(cursor)
    require(len(ordered)==len(nodes) and ordered[0]['type'] in ('recognize','pose') and ordered[-1]['type'] in ('condition','behavior_condition'),'流程应从识别开始，以判断结束')
    return ordered

def flow_model_ids(item):
    if item.get('flow'):
        return list(dict.fromkeys(n['config']['modelRef'] for n in validate_flow(item['flow']) if n['type'] in ('recognize','pose')))
    return [item['model_id']] if item.get('model_id') else []

def flow_model_specs(item):
    if item.get('flow'):
        return list(dict.fromkeys((n['config']['modelRef'], 'pose' if n['type']=='pose' else 'detect') for n in validate_flow(item['flow']) if n['type'] in ('recognize','pose')))
    return [(item['model_id'],'detect')] if item.get('model_id') else []

def validate_pose_result(raw,width,height):
    poses=raw.get('poses')
    require(isinstance(poses,list) and len(poses)==len(raw['detections']), '姿态结果缺少关键点或与人员框不匹配')
    result=[]
    for detection,pose in zip(raw['detections'],poses):
        require(isinstance(pose,dict), '姿态结果无效')
        actual=box(pose.get('bbox'),width,height);expected=box(detection.get('bbox'),width,height)
        require(all(abs(a-b)<.01 for a,b in zip(actual,expected)), '姿态关键点关联框不匹配')
        points=pose.get('keypoints')
        require(isinstance(points,list) and 1<=len(points)<=128, '姿态关键点无效')
        for point in points:
            require(isinstance(point,(list,tuple)) and len(point) in (2,3) and number(point[0],0,width) and number(point[1],0,height) and (len(point)==2 or number(point[2],0,1)), '姿态关键点坐标或置信度无效')
        require(any((p[0] or p[1]) and (len(p)==2 or p[2]>0) for p in points), '没有有效姿态关键点')
        result.append(points)
    return result

def restore_keypoints(points,left,top):
    # The worker uses [0,0] for missing points; preserve that marker.
    return [[p[0]+left,p[1]+top,*p[2:]] if p[0] or p[1] else list(p) for p in points]

def iou(a,b):
    area=lambda r:max(0,r[2]-r[0])*max(0,r[3]-r[1])
    intersection=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
    union=area(a)+area(b)-intersection
    return intersection/union if union>0 else 0

def box(value,width,height):
    if isinstance(value,dict): value=[value.get(k) for k in ('x1','y1','x2','y2')]
    require(isinstance(value,(list,tuple)) and len(value)==4 and all(number(v,0,max(width,height)*2) for v in value),'识别框无效')
    value=[max(0,min(width,value[0])),max(0,min(height,value[1])),max(0,min(width,value[2])),max(0,min(height,value[3]))]
    require(value[0]<value[2] and value[1]<value[3],'识别框为空')
    return value

def contains(region,detection):
    x=(detection[0]+detection[2])/2;y=(detection[1]+detection[3])/2
    return region[0]<=x<=region[2] and region[1]<=y<=region[3]

def execute_flow(flow,image,infer,*,frame_id=None,face_gate=None,face_capture_factory=None):
    """infer(RGB, config)-> one service result; coordinates returned in original pixels."""
    ordered=validate_flow(flow);height,width=image.shape[:2];results={};decisions=[];inference_calls=0
    frame_id=frame_id or uuid.uuid4().hex
    all_detections=[];capture_gate=None;capture_config=None;capture_people=[]
    for n in ordered:
        c=n['config']
        if n['type']=='face_capture':
            from face_capture.alarm_gate import FacePresenceGate
            factory=face_capture_factory or (lambda config:FacePresenceGate(image,frame_id,threshold=config['threshold'],min_size=config['minSize']))
            capture_gate=factory(c);capture_config={**c,'node_id':n['id']}
            capture_people=[d['bbox'] for e in results.get(c['sourceNodeId'],[]) if e['success'] and not e.get('ambiguous') for d in e['detections'] if d['class']=='person']
            # Every person in the same frame participates in ownership checks.
            capture_gate.observe_people(d['bbox'] for d in all_detections if d['class']=='person')
            continue
        if n['type']=='behavior_condition':
            entities=results.get(c['sourceNodeId'],[])
            if face_gate is not None:
                entities=[e for e in entities if admit_face(e,face_gate)]
            judged=judge_fallen(c,entities,frame_id,n['id'])
            if capture_gate is not None:
                for decision in judged:
                    if decision['alarm'] and not attach_face_capture(decision['entity'],capture_gate,capture_config,capture_people):
                        decision.update(alarm=False,text='未检出合格人脸，不满足告警条件')
            decisions.extend(judged)
            continue
        if n['type']=='condition':
            for entity in results.get(c['sourceNodeId'],[]):
                if face_gate is not None and not admit_face(entity,face_gate):
                    for r in c['rules']:
                        decisions.append({'alarm':False,'rule':r,'entity':entity,'count':None,'text':'同一人员未通过人脸检出判断，不执行告警条件','frameId':frame_id,'nodeId':n['id']})
                    continue
                for r in c['rules']:
                    count=sum(d['class']==r['target'] for d in entity.get('detections',[]))
                    side='present' if count>=r['minimumCount'] else 'absent' if count==0 else None
                    outcome=r.get(side,{})
                    alarm=entity['success'] and not entity.get('ambiguous') and outcome.get('state')=='alarm'
                    text=outcome.get('text','无告警')
                    if alarm and capture_gate is not None and not attach_face_capture(entity,capture_gate,capture_config,capture_people):
                        alarm=False;text='未检出合格人脸，不满足告警条件'
                    decisions.append({'alarm':bool(alarm),'rule':r,'entity':entity,'count':count,'text':text,'frameId':frame_id,'nodeId':n['id']})
            continue
        rois=[]
        if c['range']=='original_frame': rois=[{'id':'frame','bbox':[0,0,width,height],'success':True}]
        else:
            for parent in results.get(c['sourceNodeId'],[]):
                if parent['success'] and not parent.get('ambiguous'):
                    for i,d in enumerate(parent['detections']):
                        if d['class']==c['sourceTarget']:
                            person_bbox=d['bbox'] if c['sourceTarget']=='person' else parent.get('person_bbox')
                            rois.append({'id':f'{parent["id"]}:{n["id"]}:{i}','bbox':d['bbox'],'success':True,'parentId':parent['id'],'sourceConfidence':d['confidence'],'person_bbox':person_bbox})
        results[n['id']]=[]
        if not rois: continue
        # Do not truncate regions then infer absence on a partial population.
        if len(rois)>32:
            results[n['id']]=[{'id':'overflow','success':False,'detections':[]}];continue
        regions=[]
        for r in rois:
            b=r['bbox'];pad=c['crop'] if c['range']=='target_roi' else dict(left=0,right=0,top=0,bottom=0)
            regions.append([max(0,math.floor(b[0]-pad['left'])),max(0,math.floor(b[1]-pad['top'])),min(width,math.ceil(b[2]+pad['right'])),min(height,math.ceil(b[3]+pad['bottom']))])
        for r,region in zip(rois,regions):
            entry={**r,'bbox':region,'success':False,'detections':[],'frameId':frame_id}
            try:
                x1,y1,x2,y2=region;require(x2>x1 and y2>y1,'裁剪区域无效')
                inference_calls+=1;require(inference_calls<=128,'单帧流程识别次数超出上限')
                raw=infer(image[y1:y2,x1:x2],c)
                require(isinstance(raw,dict) and raw.get('model_id')==c['modelRef'] and isinstance(raw.get('class_names'),list) and set(c['targets'])<=set(raw['class_names']),'模型没有成功返回所需类别')
                require(isinstance(raw.get('detections'),list) and len(raw['detections'])<=256,'检测返回无效或超出上限')
                pose_points = validate_pose_result(raw,x2-x1,y2-y1) if n['type']=='pose' else None
                candidates=[]
                for index,d in enumerate(raw['detections']):
                    if d.get('class') not in c['targets']: continue
                    confidence=d.get('confidence');require(number(confidence,0,1),'检测置信度无效')
                    if confidence<c['thresholds'][d['class']]: continue
                    b=box(d.get('bbox'),x2-x1,y2-y1);b=[b[0]+x1,b[1]+y1,b[2]+x1,b[3]+y1]
                    extra = {'keypoints':restore_keypoints(pose_points[index],x1,y1)} if pose_points is not None else {}
                    candidates.append({**d,**extra,'bbox':b,'nodeId':n['id'],'entityId':r['id'],'frameId':frame_id})
                candidates.sort(key=lambda d:d['confidence'],reverse=True)
                for d in candidates:
                    if c['postprocess']['mergeDuplicates'] and any(p['class']==d['class'] and iou(p['bbox'],d['bbox'])>c['postprocess']['iou'] for p in entry['detections']):continue
                    if c['range']=='target_roi' and sum(contains(other,d['bbox']) for other in regions)>1:entry['ambiguous']=True
                    entry['detections'].append(d)
                entry['success']=True
            except Exception:
                entry['success']=False;entry['detections']=[]
            results[n['id']].append(entry);all_detections.extend(entry['detections'])
        # Shared overlapping ROIs cannot reliably establish absence either.
        for entry in results[n['id']]:
            if any(other.get('ambiguous') and iou(entry['bbox'],other['bbox'])>0 for other in results[n['id']]):entry['ambiguous']=True
        if face_gate is not None:
            face_gate.observe_people(d['bbox'] for e in results[n['id']] if e['success'] for d in e['detections'] if d['class']=='person')
        if capture_gate is not None:
            capture_gate.observe_people(d['bbox'] for e in results[n['id']] if e['success'] for d in e['detections'] if d['class']=='person')
    return {'decisions':decisions,'detections':all_detections,'modules':results,'frameId':frame_id}

def admit_face(entity,face_gate):
    if not entity.get('success') or entity.get('ambiguous'):
        entity['face_gate']={'passed':False,'reason':'recognition_failed_or_ambiguous'}
        return False
    entity['face_gate']=face_gate.check(entity.get('person_bbox'))
    return entity['face_gate']['passed']


def attach_face_capture(entity,gate,config,people):
    person=entity.get('person_bbox')
    if person is not None and any(iou(person,p)>.85 for p in people):
        result=gate.check(person)
    else:
        result={'passed':False,'reason':'no_person_lineage'}
    failed=result.get('reason') not in ('face_detected','no_unambiguous_face')
    policy=face_capture_policy(config)
    entity['face_capture']={**result,'module_id':config['node_id'],'alarm_policy':policy,'affect_alarm':policy!='ignore',
        'status':'detected' if result.get('passed') else 'failed' if failed else 'missing',
        'requires_manual_confirmation':bool(policy=='manual_review' and not result.get('passed')),
        'comparison_by':'business_system','threshold':config['threshold'],'min_size':config['minSize']}
    return policy!='require_face' or bool(result.get('passed'))


def process_flows(entries,image,*,session_key,context,timestamp=None,infer=None,evidence_factory=None,face_image=None,face_detector=None):
    timestamp=float(timestamp if timestamp is not None else time.time());events=[];detections=[]
    if infer is None:
        from inference_service.client import inference_client
        def infer(img,c):
            return inference_client.infer_flow_sync(img,c,revision=context['inference_revision'])
    from face_capture.alarm_gate import FacePresenceGate,requires_face
    frame=str(context.get('_monitor_source_sequence') or f'{timestamp:.9f}')
    shared_gate=FacePresenceGate(image if face_image is None else face_image,frame,detector=face_detector)
    for item in entries:
        if not item.get('enabled') or not item.get('flow'): continue
        try:
            capture=any(n.get('type')=='face_capture' for n in item['flow'].get('nodes',[]))
            factory=lambda c:FacePresenceGate(image if face_image is None else face_image,frame,detector=face_detector,threshold=c['threshold'],min_size=c['minSize'])
            run=execute_flow(item['flow'],image,infer,frame_id=frame,face_gate=shared_gate if requires_face(item) and not capture else None,face_capture_factory=factory)
            detections.extend(run['detections'])
            for decision in run['decisions']:
                if not decision['alarm']:continue
                r=decision['rule'];entity=decision['entity']
                # Stable per sampled frame / camera / rule / region, retry-safe persistence.
                key=f'{session_key}:{context.get("run_id","")}:{frame}:{item["code"]}:{r["id"]}:{entity["id"]}'
                if requires_face(item) or capture:
                    capture_time=float(context.get('capture_timestamp',timestamp))
                    require(math.isfinite(capture_time),'取帧时间无效')
                    # A gated frame must not overwrite a historical ungated event,
                    # or another frame whose per-process ordinal reset after restart.
                    key+=f':face_required_v1:{context.get("_monitor_capture_sequence","")}:{capture_time:.9f}'
                if capture:
                    key+=':face_capture_v1:'+hashlib.sha256(json.dumps(item['flow'],sort_keys=True).encode()).hexdigest()
                event_id=str(uuid.uuid5(uuid.NAMESPACE_URL,key))
                scores=[d['confidence'] for d in entity['detections'] if d['class']==r['target']]
                event={'event_id':event_id,'event_type':item['code'],'state':'confirmed','should_alert':True,'started_at':timestamp,'confirmed_at':timestamp,'alerted_at':timestamp,'ended_at':timestamp,
                       'persons':[],'rule_score':None if decision.get('behaviorType') else max(scores) if scores else None,'confidence':max(scores) if scores else None,
                       'model_id':next(n['config']['modelRef'] for n in item['flow']['nodes'] if n['id']==r['sourceNodeId']),
                       'reason':{'flow_version':3,'algorithm_name':item['name'],'rule_name':r['name'],'rule_id':r['id'],'target':r['target'],'result_text':decision['text'],'count':decision['count'],'roi_bbox':entity['bbox'],'frame_id':frame,'flow_hash':hashlib.sha256(json.dumps(item['flow'],sort_keys=True).encode()).hexdigest()},
                       'final_decision':{'accepted':True,'reason':'V3 detection succeeded and alarm condition matched'},'decision':{'flow':{'schemaVersion':3,'ruleId':r['id'],'count':decision['count'],'frameId':frame}},'evidence_status':'pending'}
                if capture:
                    event['reason']['face_capture']=copy.deepcopy(entity['face_capture'])
                    event['decision']['face_capture']=copy.deepcopy(entity['face_capture'])
                elif requires_face(item):
                    gate=entity.get('face_gate') or {}
                    if not gate.get('passed'):continue
                    event['reason'].update(face_gate=copy.deepcopy(gate),algorithm_category=item.get('category') or ('object' if item['code']=='phone_use' else 'violation'),
                                           **{k:gate[k] for k in ('face_bbox','face_crop_bbox','face_confidence','face_model_sha256') if k in gate})
                    event['decision']['face_gate']=copy.deepcopy(gate)
                if decision.get('behaviorType'):
                    event['reason'].update(behavior_type=decision['behaviorType'],fallen=True,posture=decision['posture'],confidence_source='person_detection')
                    event['decision']['flow'].update(behaviorType=decision['behaviorType'],fallen=True,posture=decision['posture'])
                    if decision['posture'].get('model_id'):
                        event['confidence']=decision['posture']['probability']
                        event['reason']['confidence_source']='static_pose_classifier'
                if evidence_factory is None:
                    from behavior.evidence import BehaviorEvidenceRecorder
                    evidence=BehaviorEvidenceRecorder(context.get('source_id',session_key),{'independent_snapshot_alerts':True,'evidence':{'snapshot':True}})
                else:evidence=evidence_factory()
                event.update(evidence.confirm(event,face_image if capture and face_image is not None else image))
                if item['flow']['defaults']['output']['saveCrop'] and event.get('snapshot_path'):
                    import cv2,pathlib
                    x1,y1,x2,y2=map(int,entity['bbox']);crop_path=pathlib.Path(event['snapshot_path']).with_name('crop.png')
                    if cv2.imwrite(str(crop_path),cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_RGB2BGR)):event['reason']['crop_path']=str(crop_path)
                events.append(event)
        except Exception:
            # Keep other independent flows operational; failures are never absence.
            continue
    return {'action_events':events,'active_events':events,'detections':detections,'face_gate':shared_gate.stats()}

def validate_behavior_condition(c):
    p=c.get('posture',{});rs=c['rules'];r=rs[0]
    require(c.get('behaviorType')=='fall_down' and len(rs)==1 and r['target']=='person' and r['minimumCount']==1 and r['present']['state']=='alarm' and r['absent']['state']=='no_alarm','倒地行为必须配置倒地告警、未倒地无告警')
    require(isinstance(p,dict) and number(p.get('keypointConfidence'),.05,.99) and number(p.get('torsoAngle'),45,85) and number(p.get('aspectRatio'),1,3),'倒地姿态参数无效')
    if 'classifier' in c:
        from pose_training.inference import validate_spec
        validate_spec(c['classifier'])

def judge_fallen(c,entries,frame_id,node_id):
    from behavior.fall_posture import fallen_posture
    rule=c['rules'][0];decisions=[]
    for entry in entries:
        if not entry['success'] or entry.get('ambiguous'):continue
        for index,d in enumerate(entry.get('detections',[])):
            if d.get('class')!='person':continue
            if c.get('classifier'):
                from pose_training.inference import predict_person
                posture=predict_person(d,c['classifier'])
            else:posture=fallen_posture(d,c['posture'])
            alarm=posture['valid'] and posture['fallen']
            entity={**entry,'id':f'{entry["id"]}:{node_id}:person:{index}','bbox':d['bbox'],'person_bbox':d['bbox'],'detections':[d]}
            decisions.append({'alarm':bool(alarm),'rule':rule,'entity':entity,'count':1 if alarm else 0,'text':rule['present' if alarm else 'absent']['text'] if posture['valid'] else '无有效倒地姿态','frameId':frame_id,'nodeId':node_id,'behaviorType':'fall_down','posture':posture})
    return decisions

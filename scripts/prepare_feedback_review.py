"""Prepare a real training-side review packet without creating human approvals.

Fixed sample: first 12 train originals by relative path. Fixed model: E1 seed 42.
The packet includes existing source annotations and actual model predictions as
separate evidence; neither is labelled a human decision. Ambiguous submissions
are never automatically repeated.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import httpx
from run_experiments import request_data,wait_job,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:18004')
    parser.add_argument('--parent-version',required=True);parser.add_argument('--model-id',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    api=args.base_url.rstrip('/')+'/api/v1';output=args.output;output.mkdir(parents=True,exist_ok=True)
    identity={'parent_version':args.parent_version,'model_id':args.model_id,'selection':'first 12 sorted train originals','confidence':.25,'imgsz':640}
    state_path=output/'state.json'
    if state_path.exists():
        state=json.loads(state_path.read_text(encoding='utf-8'))
        if state['identity']!=identity:raise ValueError('existing packet uses different inputs')
        if state['status']=='prepared_pending_human':print(json.dumps({'status':state['status'],'output':str(output),'run_id':state['run_id']}));return
        if state['status']=='submitting' and not state.get('job_id'):raise ValueError('ambiguous submission: inspect server before retry')
    else:
        state={'identity':identity,'status':'preparing'}
        with state_path.open('x',encoding='utf-8') as stream:json.dump(state,stream)
    with httpx.Client(timeout=120,trust_env=False) as client:
        token=request_data(client,'POST',api+'/auth/login',json={'username':os.environ['BYSJ_USER'],'password':os.environ['BYSJ_PASSWORD']})['token'];headers={'Authorization':'Bearer '+token}
        version=request_data(client,'GET',api+'/dataset-versions/'+args.parent_version,headers=headers)
        if version['state']!='frozen':raise ValueError('frozen original version required')
        model=request_data(client,'GET',api+'/models/'+args.model_id,headers=headers)
        assets=[];page=1
        while True:
            data=request_data(client,'GET',api+f"/datasets/{version['dataset_id']}/assets",headers=headers,params={'version_id':args.parent_version,'page':page,'page_size':200})
            assets.extend(data['items'])
            if len(assets)>=data['total']:break
            if not data['items']:raise ValueError('incomplete pagination')
            page+=1
        selected=sorted([a for a in assets if a['split']=='train'],key=lambda a:a['relative_path'])[:12]
        if len(selected)!=12:raise ValueError('12 train originals required')
        if not state.get('job_id'):
            state['status']='submitting';write_json(state_path,state)
            run=request_data(client,'POST',api+'/inference/images',headers=headers,data={'model_id':args.model_id,'asset_ids':','.join(str(a['asset_id']) for a in selected),'params':json.dumps({'confidence':.25,'imgsz':640,'device':'0'})})
            state.update(job_id=run['job']['job_id'],run_id=run['run_id'],status='running');write_json(state_path,state)
        job=wait_job(client,api,token,state['job_id'],600)
        if job['status']!='completed':raise ValueError('review inference did not complete')
        run=request_data(client,'GET',api+'/inference/runs/'+state['run_id'],headers=headers,params={'page_size':100})
        if run['status']!='completed' or len(run['items'])!=12 or {a['asset_id'] for a in selected}!={v['asset_id'] for v in run['items']}:raise ValueError('review evidence set differs')
        by_id={a['asset_id']:a for a in selected};records=[];rows=[]
        for item in run['items']:
            original=by_id[item['asset_id']];record={'asset':original,'inference':item,'human_decision':None}
            if item.get('artifact_id'):
                response=client.get(api+'/artifacts/'+item['artifact_id'],headers=headers);response.raise_for_status()
                suffix='.png' if 'png' in response.headers.get('content-type','') else '.jpg'
                filename=f"asset-{item['asset_id']}-prediction{suffix}";(output/filename).write_bytes(response.content)
                record.update(overlay=filename,overlay_sha256=hashlib.sha256(response.content).hexdigest())
            records.append(record)
            for prediction in item['predictions']:
                rows.append({'asset_id':item['asset_id'],'item_id':item['item_id'],'prediction_id':prediction['id'],
                    'model_class':prediction['class_name'],'model_confidence':prediction['confidence'],
                    'model_bbox_xyxy':json.dumps(prediction['bbox_xyxy']),'reviewer':'','verdict':'','corrected_class':'','corrected_cxcywh':'','reason':''})
        write_json(output/'evidence.json',{'status':'pending_human','selection':identity,'version':version,'model':{'id':model['model_id'],'sha256':model['sha256']},'records':records})
        if rows:
            with (output/'review-worksheet.csv').open('w',newline='',encoding='utf-8-sig') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        instructions=['# E3 真实人工复核待办','',
            '**状态：待人工审核；没有自动创建、提交或批准任何复核。**','',
            f"推理运行：`{state['run_id']}`；父版本：`{args.parent_version}`；模型：`{args.model_id}`（固定 E1 seed=42）。",'',
            '样本仅为 train 原图按文件名排序前 12 张，未按 test 表现挑选。evidence.json 分别保存原始来源标注与模型预测，不能把它们当成人工纠错结果。','',
            '1. 打开本机平台的“推理验证”，刷新推理记录并选择上述运行 ID 前缀。','2. 逐图检查预测叠图和原图；对每项记录正确/误检/无法判定，漏检以原图归一化坐标补录。填写实际审核人、依据和日期。','3. 确认后由授权审核者在页面提交、批准；未审核项不批准。保留原预测与修改记录。','4. 只选择这些训练侧审核记录，使用指定父版本生成子版本；再次核对共同 val/test 图像和标注 hash。','5. E3 尚未形成正式质量结论；人工记录完成后另行冻结协议。既有 E1/E2 test 已被使用，后续报告应披露登记时间与测试使用历史，不能称未见 test 的初始确认实验。','',
            'review-worksheet.csv 中人工字段刻意留空。漏检可在 UI 添加；不得用脚本批量填 correct/approved 来解除本项阻塞。','']
        (output/'README.md').write_text('\n'.join(instructions),encoding='utf-8')
        state['status']='prepared_pending_human';write_json(state_path,state)
        print(json.dumps({'status':state['status'],'output':str(output),'run_id':state['run_id'],'images':12,'predictions':len(rows)}))


if __name__=='__main__':main()

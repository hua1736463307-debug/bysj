import {test} from 'node:test'
import assert from 'node:assert/strict'
import {poseTaskActive,poseCanEvaluate,mergePoseEvents} from './poseTraining.mjs'
test('active and terminal states remain authoritative',()=>{
  for(const status of ['pending','running','cancelling','interrupted']) assert.equal(poseTaskActive({status}),true)
  for(const status of ['completed','failed','cancelled']) assert.equal(poseTaskActive({status}),false)
})
test('only completed training with a validation threshold can request held-out evaluation',()=>{
  assert.equal(poseCanEvaluate({type:'pose_training',status:'completed',result:{threshold:.7}}),true)
  for(const task of [{type:'pose_training',status:'running',result:{threshold:.7}},{type:'pose_evaluation',status:'completed',result:{threshold:.7}},{type:'pose_training',status:'completed',result:{threshold:null}}]) assert.equal(poseCanEvaluate(task),false)
})
test('event retries deduplicate and retain bounded chronological logs',()=>{
  const first=Array.from({length:210},(_,sequence)=>({sequence,message:'old'}))
  const result=mergePoseEvents(first,[{sequence:209,message:'new'},{sequence:211,message:'next'}])
  assert.equal(result.length,200);assert.equal(result.at(-1).sequence,211);assert.equal(result.at(-2).message,'new')
})

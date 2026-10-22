import test from 'node:test'
import assert from 'node:assert/strict'
import {editor,deferred} from './draft-harness.mjs'

test('simultaneous flushes submit only one CAS write',async()=>{
  const gate=deferred(), e=editor({save:()=>gate.promise});await e.draft.load(1);e.draft.scheduleSave([{id:1}]);const a=e.draft.flush(),b=e.draft.flush();assert.equal(e.calls.length,1);gate.resolve({revision:1,annotations:[{id:1}]});await Promise.all([a,b]);assert.equal(e.draft.dirty.value,false)
})
test('edit during in-flight save drains at the next revision',async()=>{
  const gate=deferred();let count=0;const e=editor({save:async(id,p)=>++count===1?gate.promise:{revision:2,annotations:p.boxes}});await e.draft.load(1);e.draft.scheduleSave([{id:'first'}]);const saving=e.draft.flush();e.draft.scheduleSave([{id:'newer'}]);gate.resolve({revision:1,annotations:[{id:'first'}]});await saving;assert.equal(e.calls.length,2);assert.equal(e.calls[1].expectedRevision,1);assert.equal(e.draft.annotations.value[0].id,'newer');assert.equal(e.draft.dirty.value,false)
})
test('failed save retains latest pending edits and retries',async()=>{
  let count=0;const e=editor({save:async(id,p)=>{if(++count===1)throw new Error('network');return {revision:1,annotations:p.boxes}}});await e.draft.load(1);e.draft.scheduleSave([{id:2}]);await assert.rejects(e.draft.flush());assert.equal(e.draft.dirty.value,true);assert.ok(e.storage.size);await e.draft.flush();assert.equal(e.draft.dirty.value,false);assert.equal(e.storage.size,0)
})
test('conflict preserves local data and does not overwrite server revision',async()=>{
  const e=editor({save:async()=>{throw Object.assign(new Error('conflict'),{code:'revision_conflict',data:{server_revision:3}})}});await e.draft.load(1);e.draft.scheduleSave([{id:4}]);await assert.rejects(e.draft.flush());assert.equal(e.draft.conflict.value.serverRevision,3);assert.equal(e.draft.annotations.value[0].id,4);await assert.rejects(e.draft.setImage(2));assert.equal(e.calls.length,1);await e.draft.load(1,{discard:true});assert.equal(e.draft.dirty.value,false)
})
test('switch image flushes all queued changes before requesting next image',async()=>{
  const gate=deferred(),e=editor({save:()=>gate.promise});await e.draft.load(1);e.draft.scheduleSave([{id:5}]);const switching=e.draft.setImage(2);assert.equal(e.calls[0].id,1);assert.equal(e.draft.annotations.value[0].id,5);gate.resolve({revision:1,annotations:[{id:5}]});await switching;assert.equal(e.draft.annotations.value.length,0)
})
test('unmount preserves unsaved snapshot for refresh recovery',async()=>{
  const e=editor();await e.draft.load(1);e.draft.scheduleSave([{id:'recover'}]);e.unmount();const next=editor({storage:e.storage});await next.draft.load(1);assert.equal(next.draft.dirty.value,true);assert.equal(next.draft.annotations.value[0].id,'recover')
})

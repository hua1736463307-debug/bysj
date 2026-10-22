import test from 'node:test'
import assert from 'node:assert/strict'
import {editor,deferred} from './draft-harness.mjs'

test('late response from old asset cannot overwrite current asset',async()=>{
  const first=deferred(),second=deferred(),e=editor({get:id=>id===1?first.promise:second.promise});const a=e.draft.load(1),b=e.draft.load(2);second.resolve({revision:2,annotations:[{id:'second'}],editable:true});await b;first.resolve({revision:1,annotations:[{id:'first'}],editable:true});await a;assert.equal(e.draft.annotations.value[0].id,'second');assert.equal(e.draft.revision.value,2)
})
test('local drafts never leak across owners or assets',async()=>{
  const storage=new Map(),a=editor({storage,owner:1});await a.draft.load(1);a.draft.scheduleSave([{id:'private'}]);const otherOwner=editor({storage,owner:2});await otherOwner.draft.load(1);assert.equal(otherOwner.draft.annotations.value.length,0);const otherAsset=editor({storage,owner:1});await otherAsset.draft.load(2);assert.equal(otherAsset.draft.annotations.value.length,0)
})
test('restored stale revision becomes an explicit conflict',async()=>{
  const storage=new Map([['bysj:draft:1:8',JSON.stringify({revision:0,boxes:[{id:'local'}]})]]);const e=editor({storage,get:async()=>({revision:4,annotations:[],editable:true})});await e.draft.load(8);assert.equal(e.draft.conflict.value.serverRevision,4);await assert.rejects(e.draft.flush());assert.equal(e.calls.length,0)
})
test('frozen assets ignore local draft overlays',async()=>{
  const storage=new Map([['bysj:draft:1:8',JSON.stringify({revision:0,boxes:[{id:'stale'}]})]]);const e=editor({storage,get:async()=>({revision:4,annotations:[{id:'frozen'}],editable:false})});await e.draft.load(8);assert.equal(e.draft.annotations.value[0].id,'frozen');assert.equal(e.draft.dirty.value,false)
})

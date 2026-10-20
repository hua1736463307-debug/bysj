import {test} from 'node:test'
import assert from 'node:assert/strict'
import {webcrypto} from 'node:crypto'
import {readFileSync} from 'node:fs'
import vm from 'node:vm'
import {createRequire} from 'node:module'
import {createPoseRequestId} from './poseTraining.mjs'
const require=createRequire(import.meta.url)
const {parse}=require('@babel/parser')
const {parse:parseSfc}=require('@vue/compiler-sfc')
const source=parseSfc(readFileSync(new URL('../components/PoseTrainingPanel.vue',import.meta.url),'utf8')).descriptor.scriptSetup.content
const node=parse(source,{sourceType:'module'}).program.body.find(n=>n.type==='FunctionDeclaration' && n.id.name==='submit')
const submitSource=source.slice(node.start,node.end)
const httpCrypto={getRandomValues:array=>webcrypto.getRandomValues(array)}
const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
function harness(options={}) {
  const storage=new Map(),requests=[],messages=[]
  const context={submitting:{value:false},checked:{value:{index_sha256:'a'.repeat(64)}},form:{name:'test',dataset_id:'caucafall',epochs:100,batch_size:8},submissionKey:'pending',storageKey:'active',
    createPoseRequestId:()=>createPoseRequestId(options.crypto ?? httpCrypto),
    sessionStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>{if(options.storageFailure) throw new Error('storage blocked');storage.set(k,v)},removeItem:k=>storage.delete(k)},
    localStorage:{setItem:()=>{}},
    api:{post:async(path,body)=>{requests.push({path,body});if(options.failPost)throw Object.assign(new Error('network failed'),{isAxiosError:true});return {task:{id:'task-1'},reused:false}}},
    show:async()=>{},refresh:async()=>{},ElMessage:{success:m=>messages.push(m),error:m=>messages.push(m)}}
  vm.createContext(context);vm.runInContext(submitSource,context)
  return {context,storage,requests,messages,submit:()=>context.submit()}
}
test('HTTP fallback generates unique RFC4122 v4 IDs without randomUUID',()=>{
  const ids=new Set(Array.from({length:256},()=>createPoseRequestId(httpCrypto)))
  assert.equal(ids.size,256);for(const id of ids)assert.match(id,uuid)
  assert.equal(createPoseRequestId({randomUUID:()=> 'native-id'}),'native-id')
})
test('actual component submission succeeds in an HTTP crypto environment',async()=>{
  const h=harness();await h.submit()
  assert.equal(h.requests.length,1);assert.match(h.requests[0].body.request_id,uuid)
  assert.equal(h.requests[0].body.epochs,100);assert.equal(h.requests[0].body.batch_size,8)
  assert.equal(h.context.submitting.value,false);assert.equal(h.storage.size,0)
  assert.deepEqual(h.messages,['姿态训练已排队'])
})
test('transport retry preserves request ID across component remounts',async()=>{
  const options={failPost:true},h=harness(options);await h.submit()
  assert.equal(h.context.submitting.value,false);assert.equal(h.messages.length,0)
  const saved=h.storage.get('pending'),next=harness();next.storage.set('pending',saved);await next.submit()
  assert.equal(h.requests[0].body.request_id,next.requests[0].body.request_id)
})
test('crypto preparation failure clears loading, reports error, and permits retry',async()=>{
  const options={crypto:{}},h=harness(options);await h.submit()
  assert.equal(h.context.submitting.value,false);assert.equal(h.requests.length,0)
  assert.match(h.messages[0],/浏览器不支持/)
  options.crypto=httpCrypto;await h.submit();assert.equal(h.requests.length,1)
})
test('storage preparation failure clears loading and reports error before POST',async()=>{
  const options={storageFailure:true},h=harness(options);await h.submit()
  assert.equal(h.context.submitting.value,false);assert.equal(h.requests.length,0)
  assert.equal(h.messages[0],'storage blocked')
  options.storageFailure=false;await h.submit();assert.equal(h.requests.length,1)
})
test('changing training parameters creates a new idempotency request',async()=>{
  const options={failPost:true},h=harness(options);await h.submit()
  h.context.form.epochs=30;await h.submit()
  assert.notEqual(h.requests[0].body.request_id,h.requests[1].body.request_id)
})

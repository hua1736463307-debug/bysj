import vm from 'node:vm'
import { readFileSync } from 'node:fs'
export const deferred = () => { let resolve, reject; const promise = new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject} }
export function editor({ get, save, storage=new Map(), owner=1 }={}) {
  const source=readFileSync(new URL('../src/composables/useAnnotationDraft.js',import.meta.url),'utf8').replace(/^import .*$/gm,'').replace('export function','function')
  const calls=[], unmount=[]; let serverRevision=0
  const context={ ref:value=>({value}), onBeforeUnmount:fn=>unmount.push(fn), AbortController,
    getCurrentUser:()=>({id:owner}), localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},
    window:{addEventListener(){},removeEventListener(){}},setTimeout:()=>1,clearTimeout(){},
    getAnnotations:get || (async id=>({revision:serverRevision,annotations:[],editable:true,asset:{asset_id:id}})),
    saveAnnotations:async(id,payload)=>{calls.push({id,...JSON.parse(JSON.stringify(payload))});return save ? save(id,payload) : {revision:++serverRevision,annotations:payload.boxes}}
  }
  vm.createContext(context);vm.runInContext(source+'\nthis.draft=useAnnotationDraft()',context)
  return {draft:context.draft,calls,storage,unmount:()=>unmount.forEach(fn=>fn())}
}

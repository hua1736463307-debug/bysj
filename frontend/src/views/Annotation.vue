<template>
  <main class="medical-page"><h1>数据标注</h1><p>在图像上拖动绘制框；保存使用原图归一化坐标。切图前会等待保存完成。</p>
    <div class="toolbar"><label>数据集 <select aria-label="标注数据集" :value="datasetId" :disabled="busy" @change="act(()=>selectDataset($event.target.value))"><option v-for="d in datasets" :key="d.dataset_id" :value="d.dataset_id">{{ d.name }}</option></select></label><label>版本 <select aria-label="标注版本" :value="version?.version_id" :disabled="busy" @change="act(()=>selectVersion($event.target.value))"><option v-for="v in versions" :key="v.version_id" :value="v.version_id">{{v.state}} · {{v.version_id.slice(0,8)}}</option></select></label></div>
    <p v-if="error || draft.lastError.value" class="error" role="alert">{{ error || draft.lastError.value?.message }}</p>
    <section v-if="draft.conflict.value" role="alert" class="job-card"><strong>保存冲突，本地草稿已保留。</strong><p>服务器 revision：{{draft.conflict.value.serverRevision}}。先下载本地草稿，再加载服务器版本进行人工合并。</p><el-button @click="exportDraft">下载本地草稿</el-button><el-button @click="act(reloadServer)">放弃本地修改并重载服务器</el-button></section>
    <div class="toolbar"><el-button :disabled="page<=1 || busy" @click="act(()=>goPage(page-1))">上一页</el-button><span>第 {{ page }} / {{ Math.max(1,Math.ceil(total/8)) }} 页 · {{total}} 图</span><el-button :disabled="page*8>=total || busy" @click="act(()=>goPage(page+1))">下一页</el-button><el-button v-for="a in assets" :key="a.asset_id" :type="a.asset_id===current?.asset_id?'primary':''" :disabled="busy" @click="act(()=>openAsset(a))">{{a.basename}}</el-button></div>
    <template v-if="current"><div class="toolbar"><strong>{{current.basename}}</strong><label>类别 <select aria-label="标注类别" v-model.number="classId"><option v-for="(name,id) in classMap" :key="id" :value="Number(id)">{{name}}</option></select></label><el-button :disabled="!editable" @click="undo">撤销</el-button><el-button :disabled="!editable" @click="redo">重做</el-button><el-button :disabled="!editable || busy" @click="act(()=>draft.flush())">保存标注</el-button><span>{{ !editable ? '版本只读' : draft.saving.value ? '正在保存' : draft.dirty.value ? '本地草稿待保存' : '已保存' }} · revision {{draft.revision.value}}</span></div>
      <div class="annotation-layout"><div class="image-board" @pointerdown="begin" @pointermove="move" @pointerup="end" @pointercancel="drag=null">
        <ControlledImage :asset-id="current.asset_id" :alt="current.basename" />
        <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-label="标注画布"><g v-for="(box,i) in boxes" :key="box.client_id || i"><rect :x="(box.cx-box.w/2)*1000" :y="(box.cy-box.h/2)*1000" :width="box.w*1000" :height="box.h*1000" /><text :x="(box.cx-box.w/2)*1000+3" :y="(box.cy-box.h/2)*1000+22">{{classMap[box.class_id]}}</text></g><rect v-if="drag" class="drawing" :x="Math.min(drag.x,drag.ex)*1000" :y="Math.min(drag.y,drag.ey)*1000" :width="Math.abs(drag.ex-drag.x)*1000" :height="Math.abs(drag.ey-drag.y)*1000" /></svg>
      </div><div><p>{{boxes.length}} 个标注框</p><div v-for="(box,i) in boxes" :key="i" class="toolbar"><span>{{i+1}} · {{classMap[box.class_id]}} · {{box.w.toFixed(3)}} × {{box.h.toFixed(3)}}</span><el-button :disabled="!editable" @click="removeBox(i)">删除框 {{i+1}}</el-button></div></div></div>
    </template>
  </main>
</template>
<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, onBeforeRouteLeave } from 'vue-router'
import { ElMessageBox } from 'element-plus'
import { listDatasets, listVersions, listAssets } from '../api/datasets'
import { useAnnotationDraft } from '../composables/useAnnotationDraft'
import { createBoundedHistory } from '../utils/annotationHistory.mjs'
import { useTaskAction } from '../composables/useTaskAction'
import { saveBlob } from '../utils/download'
import ControlledImage from '../components/medical/ControlledImage.vue'
const route=useRoute(),draft=useAnnotationDraft(),history=createBoundedHistory(50),{error,busy,act}=useTaskAction()
const datasets=ref([]),versions=ref([]),datasetId=ref(''),version=ref(null),assets=ref([]),current=ref(null),page=ref(1),total=ref(0),classMap=ref({}),classId=ref(0),drag=ref(null)
const boxes=computed(()=>draft.annotations.value),editable=computed(()=>version.value?.state==='draft' && !draft.conflict.value && !busy.value)
const clone=value=>JSON.parse(JSON.stringify(value))
async function selectDataset(id){await draft.flush();datasetId.value=id;versions.value=(await listVersions(id)).items;await selectVersion(route.query.version && versions.value.some(v=>v.version_id===route.query.version)?route.query.version:versions.value[0]?.version_id)}
async function selectVersion(id){await draft.flush();version.value=versions.value.find(v=>v.version_id===id);current.value=null;assets.value=[];if(version.value)await goPage(1)}
async function goPage(next){await draft.flush();const data=await listAssets(datasetId.value,{version_id:version.value.version_id,page:next,page_size:8});assets.value=data.items;page.value=next;total.value=data.total;if(data.items.length)await openAsset(data.items[0]);else current.value=null}
async function openAsset(asset){const data=await draft.setImage(asset.asset_id);if(!data)return;current.value=asset;classMap.value=data.class_map;classId.value=Number(Object.keys(data.class_map)[0]||0);history.reset(asset.asset_id)}
function edit(next){if(!editable.value)return;history.push(current.value.asset_id,clone(boxes.value));draft.scheduleSave(next)}
function removeBox(i){const next=clone(boxes.value);next.splice(i,1);edit(next)}
function undo(){if(!editable.value)return;const next=history.undo(current.value.asset_id,clone(boxes.value));if(next)draft.scheduleSave(next)}
function redo(){if(!editable.value)return;const next=history.redo(current.value.asset_id,clone(boxes.value));if(next)draft.scheduleSave(next)}
function point(event){const r=event.currentTarget.getBoundingClientRect();return{x:Math.max(0,Math.min(1,(event.clientX-r.left)/r.width)),y:Math.max(0,Math.min(1,(event.clientY-r.top)/r.height))}}
function begin(event){if(!editable.value || event.button!==0)return;const p=point(event);drag.value={...p,ex:p.x,ey:p.y};event.currentTarget.setPointerCapture(event.pointerId)}
function move(event){if(!drag.value)return;const p=point(event);drag.value.ex=p.x;drag.value.ey=p.y}
function end(event){if(!drag.value)return;move(event);const d=drag.value;drag.value=null;const w=Math.abs(d.ex-d.x),h=Math.abs(d.ey-d.y);if(w<.002||h<.002)return;edit([...clone(boxes.value),{client_id:crypto.randomUUID(),class_id:classId.value,cx:(d.x+d.ex)/2,cy:(d.y+d.ey)/2,w,h,source:'manual'}])}
function exportDraft(){saveBlob({data:new Blob([JSON.stringify({asset_id:current.value.asset_id,boxes:boxes.value},null,2)],{type:'application/json'})},'annotation-draft.json')}
async function reloadServer(){await ElMessageBox.confirm('丢弃本地未保存修改并加载服务器标注？','处理冲突');await draft.load(current.value.asset_id,{discard:true});history.reset(current.value.asset_id)}
onBeforeRouteLeave(async()=>{try{await draft.flush();return true}catch(e){error.value=e.message;return false}})
onMounted(()=>act(async()=>{const all=[];let p=1,data;do{data=await listDatasets({page:p++,page_size:100});all.push(...data.items)}while(all.length<data.total);datasets.value=all;if(all.length)await selectDataset(all.some(d=>d.dataset_id===route.query.dataset)?route.query.dataset:all[0].dataset_id)}))
</script>
<style scoped>
.annotation-layout{display:grid;grid-template-columns:minmax(400px,3fr) minmax(260px,1fr);gap:24px;align-items:start}.image-board{position:relative;display:inline-block;touch-action:none;user-select:none;line-height:0}.image-board :deep(img){width:100%;height:auto;display:block;pointer-events:none}.image-board svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.image-board rect{fill:rgba(255,185,0,.08);stroke:#ffcd38;stroke-width:3;vector-effect:non-scaling-stroke}.image-board text{fill:#fff;font-size:20px;paint-order:stroke;stroke:#222;stroke-width:3px}.image-board .drawing{stroke:#21e0d0}@media(max-width:800px){.annotation-layout{grid-template-columns:1fr}}
</style>

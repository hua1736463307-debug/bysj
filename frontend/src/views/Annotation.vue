<template>
  <main class="medical-page"><h1>数据标注</h1><p>在图像上拖动绘制框；保存使用原图归一化坐标。切图前会等待保存完成。</p>
    <div class="toolbar"><label>数据集 <select aria-label="标注数据集" :value="datasetId" :disabled="busy" @change="act(()=>selectDataset($event.target.value))"><option v-for="d in datasets" :key="d.dataset_id" :value="d.dataset_id">{{ d.name }}</option></select></label><label>版本 <select aria-label="标注版本" :value="version?.version_id" :disabled="busy" @change="act(()=>selectVersion($event.target.value))"><option v-for="v in versions" :key="v.version_id" :value="v.version_id">{{v.state}} · {{v.version_id.slice(0,8)}}</option></select></label></div>
    <el-alert v-if="version?.state === 'frozen'" title="当前为冻结实验版本，创建标注副本后即可正常画框、删除与保存；原版本保持不变。" type="info" :closable="false" /><div v-if="version?.state === 'frozen'" class="toolbar"><el-button type="primary" :loading="busy" @click="act(forkVersion)">创建可编辑标注版本</el-button></div>
    <div v-if="version?.state === 'frozen'" class="toolbar"><el-button :disabled="!current" @click="openAi(true)">AI 标注当前图片</el-button><el-button @click="openAi(false)">AI 标注整个版本</el-button></div>
    <DerivativeDialog v-model:visible="aiVisible" operation="preannotate" :version-ids="version ? [version.version_id] : []" :asset-ids="aiAssets" @submitted="aiJob = $event" />
    <JobProgress :job-id="aiJob" @terminal="act(() => aiFinished($event))" />
    <el-alert v-if="version?.requires_annotation_review" title="此版本含 AI 预标注，请检查所有图片并修正错误，再由复核员确认。" type="warning" :closable="false" />
    <el-button v-if="version?.requires_annotation_review && canReview" :disabled="busy" @click="confirmReview">确认已完成人工检查</el-button>
    <p v-if="error || draft.lastError.value" class="error" role="alert">{{ error || draft.lastError.value?.message }}</p>
    <section v-if="draft.conflict.value" role="alert" class="job-card"><strong>保存冲突，本地草稿已保留。</strong><p>服务器 revision：{{draft.conflict.value.serverRevision}}。先下载本地草稿，再加载服务器版本进行人工合并。</p><el-button @click="exportDraft">下载本地草稿</el-button><el-button @click="act(reloadServer)">放弃本地修改并重载服务器</el-button></section>
    <div class="toolbar"><el-button :disabled="page<=1 || busy" @click="act(()=>goPage(page-1))">上一页</el-button><span>第 {{ page }} / {{ Math.max(1,Math.ceil(total/8)) }} 页 · {{total}} 图</span><el-button :disabled="page*8>=total || busy" @click="act(()=>goPage(page+1))">下一页</el-button><el-button v-for="a in assets" :key="a.asset_id" :type="a.asset_id===current?.asset_id?'primary':''" :disabled="busy" @click="act(()=>openAsset(a))">{{a.basename}}</el-button></div>
    <template v-if="current"><div class="toolbar"><strong>{{current.basename}}</strong><label>类别 <select aria-label="标注类别" v-model.number="classId"><option v-for="(name,id) in classMap" :key="id" :value="Number(id)">{{name}}</option></select></label><el-button :disabled="!editable" @click="undo">撤销</el-button><el-button :disabled="!editable" @click="redo">重做</el-button><el-button :disabled="!editable || busy" @click="act(()=>draft.flush())">保存标注</el-button><span>{{ !editable ? '冻结版本（可创建标注副本）' : draft.saving.value ? '正在保存' : draft.dirty.value ? '本地草稿待保存' : '已保存' }} · revision {{draft.revision.value}}</span></div>
      <div class="toolbar">
        <el-button :disabled="!editable || polygonPoints.length > 0" @click="openClasses">编辑类别</el-button>
        <label>工具 <select aria-label="标注工具" v-model="mode" @change="polygonPoints = []"><option value="draw">绘制矩形</option><option value="polygon">绘制多边形</option><option value="select">选择 / 移动 / 调整大小</option></select></label>
        <el-button v-if="mode === 'polygon'" :disabled="polygonPoints.length < 3 || !editable" @click="finishPolygon">完成多边形</el-button>
        <el-button v-if="polygonPoints.length" @click="polygonPoints = []">取消多边形</el-button>
        <el-button :disabled="zoom <= .25" @click="zoom = Math.max(.25, zoom - .25)">缩小</el-button>
        <span>{{ Math.round(zoom * 100) }}%</span>
        <el-button :disabled="zoom >= 4" @click="zoom = Math.min(4, zoom + .25)">放大</el-button>
        <el-button @click="zoom = 1">适应窗口</el-button>
        <label v-if="selectedIndex >= 0">选中框类别 <select :value="boxes[selectedIndex]?.class_id" :disabled="!editable" @change="changeClass(Number($event.target.value))"><option v-for="(name,id) in classMap" :key="id" :value="Number(id)">{{ name }}</option></select></label>
      </div>
      <p v-if="mode === 'polygon'">逐点点击后完成多边形。区域顶点会保存并可导出 YOLO 分割标签，当前检测训练使用区域外接框。</p>
      <div class="annotation-layout"><div class="image-viewport"><div class="image-board" tabindex="0" :style="{ width: (zoom * 100) + '%' }" @keydown="canvasKey" @pointerdown="begin" @pointermove="move" @pointerup="end" @pointercancel="drag=null">
        <ControlledImage :asset-id="current.asset_id" :alt="current.basename" />
        <svg class="polygon-layer" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-label="多边形轮廓"><polygon v-for="(box,index) in visibleBoxes.filter(item => item.polygon)" :key="index" :points="box.polygon.map(([x,y]) => [x*1000,y*1000].join(',')).join(' ')" /><polyline v-if="polygonPoints.length" :points="polygonPoints.map(([x,y]) => [x*1000,y*1000].join(',')).join(' ')" /></svg>
        <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-label="标注画布"><g v-for="(box,i) in visibleBoxes" :key="box.client_id || i"><rect :class="{ selected: selectedIndex === i }" :x="(box.cx-box.w/2)*1000" :y="(box.cy-box.h/2)*1000" :width="box.w*1000" :height="box.h*1000" /><text :x="(box.cx-box.w/2)*1000+3" :y="(box.cy-box.h/2)*1000+22">{{classMap[box.class_id]}}</text><rect v-if="selectedIndex === i" class="resize-handle" :x="(box.cx+box.w/2)*1000-8" :y="(box.cy+box.h/2)*1000-8" width="16" height="16" /></g><rect v-if="drag?.kind === 'draw'" class="drawing" :x="Math.min(drag.x,drag.ex)*1000" :y="Math.min(drag.y,drag.ey)*1000" :width="Math.abs(drag.ex-drag.x)*1000" :height="Math.abs(drag.ey-drag.y)*1000" /></svg>
      </div></div><div><p>{{boxes.length}} 个标注框。选择工具下拖动框体移动，拖动右下角调整大小。</p><div v-for="(box,i) in boxes" :key="i" class="toolbar"><el-button @click="selectedIndex = i; mode = 'select'">{{i+1}} · {{classMap[box.class_id]}}</el-button><span>{{box.w.toFixed(3)}} × {{box.h.toFixed(3)}}</span><el-button :disabled="!editable" @click="removeBox(i)">删除框 {{i+1}}</el-button></div></div></div>
    </template>
    <el-dialog v-model="classesVisible" title="编辑类别" width="min(600px, 95vw)">
      <div v-for="(item,index) in classEntries" :key="item.id" class="toolbar"><span>ID {{ item.id }}</span><el-input v-model="item.name" maxlength="64" style="width:260px" /><el-button @click="classEntries.splice(index, 1)">移除</el-button></div>
      <el-button @click="addClass">新增类别</el-button>
      <el-alert v-if="error" :title="error" type="error" :closable="false" />
      <template #footer><el-button :disabled="busy" @click="classesVisible = false">取消</el-button><el-button type="primary" :loading="busy" @click="act(saveClasses)">保存类别</el-button></template>
    </el-dialog>
  </main>
</template>
<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { useRoute, useRouter, onBeforeRouteLeave } from 'vue-router'
import { ElMessageBox } from 'element-plus'
import { listDatasets, listVersions, listAssets, createAnnotationVersion, confirmAnnotationReview, updateClasses } from '../api/datasets'
import { getCurrentUser } from '../api/client'
import { useAnnotationDraft } from '../composables/useAnnotationDraft'
import { createBoundedHistory } from '../utils/annotationHistory.mjs'
import { useTaskAction } from '../composables/useTaskAction'
import { saveBlob } from '../utils/download'
import ControlledImage from '../components/medical/ControlledImage.vue'
import { hitBox, moveBox, resizeBox } from '../utils/boxGeometry.mjs'
import DerivativeDialog from '../components/medical/DerivativeDialog.vue'
import JobProgress from '../components/medical/JobProgress.vue'
const route=useRoute(),draft=useAnnotationDraft(),history=createBoundedHistory(50),{error,busy,act}=useTaskAction()
const router = useRouter()
const datasets=ref([]),versions=ref([]),datasetId=ref(''),version=ref(null),assets=ref([]),current=ref(null),page=ref(1),total=ref(0),classMap=ref({}),classId=ref(0),drag=ref(null)
const boxes=computed(()=>draft.annotations.value),editable=computed(()=>['draft','failed'].includes(version.value?.state) && !draft.conflict.value && !busy.value)
const clone=value=>JSON.parse(JSON.stringify(value))
const zoom = ref(1), mode = ref('draw'), selectedIndex = ref(-1)
const polygonPoints = ref([])
function warnPendingPolygon(event) {
  if (polygonPoints.value.length) { event.preventDefault(); event.returnValue = '' }
}
onMounted(() => window.addEventListener('beforeunload', warnPendingPolygon))
onBeforeUnmount(() => window.removeEventListener('beforeunload', warnPendingPolygon))
const classesVisible = ref(false), classEntries = ref([])
function openClasses() { classEntries.value = Object.entries(classMap.value).map(([id, name]) => ({ id, name })); classesVisible.value = true }
function addClass() {
  const id = String(Math.max(-1, ...Object.keys(classMap.value).map(Number), ...classEntries.value.map(item => Number(item.id))) + 1)
  classEntries.value.push({ id, name: '' })
}
async function saveClasses() {
  requireFinishedPolygon()
  await draft.flush()
  await updateClasses(version.value.version_id, classMap.value, Object.fromEntries(classEntries.value.map(item => [item.id, item.name])))
  classesVisible.value = false
  await openAsset(current.value)
}
function finishPolygon() {
  if (!editable.value || polygonPoints.value.length < 3) return
  const points = polygonPoints.value
  const xs = points.map(point => point[0]), ys = points.map(point => point[1])
  const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys)
  edit([...clone(boxes.value), { client_id: crypto.randomUUID(), class_id: classId.value,
    cx: (left + right) / 2, cy: (top + bottom) / 2, w: right - left, h: bottom - top, polygon: clone(points), source: 'manual' }])
  polygonPoints.value = []
}
const aiVisible = ref(false), aiAssets = ref([]), aiJob = ref('')
const canReview = ['admin', 'reviewer'].includes(getCurrentUser()?.role)
async function confirmReview() {
  try { await ElMessageBox.confirm('确认已人工检查此版本所有预标注并修正错误？此操作将记录复核身份。', '确认人工检查') }
  catch { return }
  await act(async () => {
    await draft.flush()
    version.value = await confirmAnnotationReview(version.value.version_id)
  })
}
function openAi(single) { aiAssets.value = single ? [current.value.asset_id] : []; aiVisible.value = true }
async function aiFinished(job) {
  if (job.status !== 'completed') return
  versions.value = (await listVersions(datasetId.value)).items
  await selectVersion(job.result.version_id)
}
const visibleBoxes = computed(() => {
  if (!drag.value?.preview) return boxes.value
  return boxes.value.map((box, index) => index === drag.value.index ? drag.value.preview : box)
})
async function forkVersion(){const child=await createAnnotationVersion(datasetId.value,version.value.version_id);versions.value=(await listVersions(datasetId.value)).items;await selectVersion(child.version_id)}
function requireFinishedPolygon() { if (polygonPoints.value.length) throw new Error('请先完成或取消正在绘制的多边形') }
async function selectDataset(id){requireFinishedPolygon();await draft.flush();datasetId.value=id;versions.value=(await listVersions(id)).items;await selectVersion(route.query.version && versions.value.some(v=>v.version_id===route.query.version)?route.query.version:versions.value[0]?.version_id)}
async function selectVersion(id){requireFinishedPolygon();await draft.flush();version.value=versions.value.find(v=>v.version_id===id);current.value=null;assets.value=[];if(version.value){await goPage(1);await router.replace({path:route.path,query:{...route.query,dataset:datasetId.value,version:id}})}}
async function goPage(next){requireFinishedPolygon();await draft.flush();const data=await listAssets(datasetId.value,{version_id:version.value.version_id,page:next,page_size:8});assets.value=data.items;page.value=next;total.value=data.total;if(data.items.length)await openAsset(data.items[0]);else current.value=null}
async function openAsset(asset){requireFinishedPolygon();const data=await draft.setImage(asset.asset_id);if(!data)return;drag.value=null;selectedIndex.value=-1;zoom.value=1;current.value=asset;classMap.value=data.class_map;classId.value=Number(Object.keys(data.class_map)[0]||0);history.reset(asset.asset_id)}
function edit(next){if(!editable.value)return;history.push(current.value.asset_id,clone(boxes.value));draft.scheduleSave(next)}
function removeBox(i){const next=clone(boxes.value);next.splice(i,1);selectedIndex.value=-1;edit(next)}
function changeClass(id) {
  const next = clone(boxes.value)
  if (next[selectedIndex.value]) { next[selectedIndex.value].class_id = id; edit(next) }
}
function undo(){if(!editable.value)return;const next=history.undo(current.value.asset_id,clone(boxes.value));if(next)draft.scheduleSave(next)}
function redo(){if(!editable.value)return;const next=history.redo(current.value.asset_id,clone(boxes.value));if(next)draft.scheduleSave(next)}
function point(event){const r=event.currentTarget.getBoundingClientRect();return{x:Math.max(0,Math.min(1,(event.clientX-r.left)/r.width)),y:Math.max(0,Math.min(1,(event.clientY-r.top)/r.height))}}
function begin(event) {
  if (!editable.value || event.button !== 0) return
  event.currentTarget.focus()
  const p = point(event)
  if (mode.value === 'polygon') {
    if (polygonPoints.value.length < 256) polygonPoints.value.push([p.x, p.y])
    return
  }
  if (mode.value === 'select') {
    const selected = boxes.value[selectedIndex.value]
    const resize = selected && Math.abs(p.x - selected.cx - selected.w / 2) < .02 && Math.abs(p.y - selected.cy - selected.h / 2) < .02
    const index = resize ? selectedIndex.value : hitBox(boxes.value, p)
    selectedIndex.value = index
    if (index < 0) return
    drag.value = { kind: resize ? 'resize' : 'move', ...p, index, original: clone(boxes.value[index]) }
  } else {
    selectedIndex.value = -1
    drag.value = { kind: 'draw', ...p, ex: p.x, ey: p.y }
  }
  event.currentTarget.setPointerCapture(event.pointerId)
}
function move(event) {
  const d = drag.value
  if (!d) return
  const p = point(event)
  d.ex = p.x
  d.ey = p.y
  if (d.kind === 'move') d.preview = moveBox(d.original, p.x - d.x, p.y - d.y)
  if (d.kind === 'resize') d.preview = resizeBox(d.original, p.x, p.y)
}
function end(event) {
  if (!drag.value) return
  move(event)
  const d = drag.value
  drag.value = null
  if (d.kind !== 'draw') {
    if (JSON.stringify(d.preview) === JSON.stringify(d.original)) return
    const next = clone(boxes.value)
    next[d.index] = d.preview
    edit(next)
    return
  }
  const w = Math.abs(d.ex - d.x), h = Math.abs(d.ey - d.y)
  if (w < .002 || h < .002) return
  edit([...clone(boxes.value), { client_id: crypto.randomUUID(), class_id: classId.value, cx: (d.x + d.ex) / 2, cy: (d.y + d.ey) / 2, w, h, source: 'manual' }])
}
function canvasKey(event) {
  const key = event.key.toLowerCase()
  if (key === 'escape') { polygonPoints.value = []; drag.value = null; return }
  if ((event.ctrlKey || event.metaKey) && ['z', 'y'].includes(key)) {
    event.preventDefault()
    if (polygonPoints.value.length && key === 'z') polygonPoints.value.pop()
    else if (key === 'y' || event.shiftKey) redo()
    else undo()
  }
  if (key === 'delete' && editable.value && selectedIndex.value >= 0) {
    event.preventDefault()
    removeBox(selectedIndex.value)
  }
}
function exportDraft(){saveBlob({data:new Blob([JSON.stringify({asset_id:current.value.asset_id,boxes:boxes.value},null,2)],{type:'application/json'})},'annotation-draft.json')}
async function reloadServer(){await ElMessageBox.confirm('丢弃本地未保存修改并加载服务器标注？','处理冲突');await draft.load(current.value.asset_id,{discard:true});history.reset(current.value.asset_id)}
onBeforeRouteLeave(async()=>{try{requireFinishedPolygon();await draft.flush();return true}catch(e){error.value=e.message;return false}})
onMounted(()=>act(async()=>{const all=[];let p=1,data;do{data=await listDatasets({task:'detect',page:p++,page_size:100});all.push(...data.items)}while(all.length<data.total);datasets.value=all.filter(d=>d.task_type==='detect');if(all.length)await selectDataset(all.some(d=>d.dataset_id===route.query.dataset)?route.query.dataset:all[0].dataset_id)}))
</script>
<style scoped>
.image-viewport { overflow: auto; max-height: 75vh; min-width: 0; background: #182536; }
.image-board .selected { stroke: #21e0d0; }
.image-board .resize-handle { fill: #21e0d0; stroke: white; }
.polygon-layer polygon { fill: rgba(33,224,208,.2); stroke: #21e0d0; stroke-width: 3; }
.polygon-layer polyline { fill: none; stroke: #21e0d0; stroke-width: 3; }
.annotation-layout{display:grid;grid-template-columns:minmax(400px,3fr) minmax(260px,1fr);gap:24px;align-items:start}.image-board{position:relative;display:inline-block;touch-action:none;user-select:none;line-height:0}.image-board :deep(img){width:100%;height:auto;display:block;pointer-events:none}.image-board svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.image-board rect{fill:rgba(255,185,0,.08);stroke:#ffcd38;stroke-width:3;vector-effect:non-scaling-stroke}.image-board text{fill:#fff;font-size:20px;paint-order:stroke;stroke:#222;stroke-width:3px}.image-board .drawing{stroke:#21e0d0}@media(max-width:800px){.annotation-layout{grid-template-columns:1fr}}
</style>

<template>
  <section class="pose-training">
    <el-alert type="info" :closable="false" title="使用已提取的 COCO-17 关键点训练单帧姿态分类器；不微调关键点检测模型。实验权重不会自动启用到告警。" />
    <el-alert v-if="!worker.available" type="warning" :closable="false" title="姿态训练 worker 未就绪，暂不能提交任务。" />
    <el-alert v-else-if="worker.gpu_status !== 'ready'" type="warning" :closable="false" title="GPU 暂未就绪或显存不足；可以提交排队，恢复后自动开始，不回退 CPU。" />
    <el-form label-width="120px" class="pose-form" @submit.prevent>
      <el-form-item label="训练名称"><el-input v-model.trim="form.name" maxlength="100" /></el-form-item>
      <el-form-item label="姿态数据集">
        <el-select v-model="form.dataset_id" placeholder="请选择已准备关键点的受管数据集" @change="invalidatePreflight">
          <el-option v-for="d in datasets" :key="d.id" :value="d.id" :label="`${d.name}（${d.label_mode === 'source_proxy' ? '来源标签探索' : '已审核姿态标签'}）`" />
        </el-select>
        <el-button :loading="checking" :disabled="!form.dataset_id" @click="preflight">校验数据</el-button>
      </el-form-item>
      <el-form-item label="训练参数">
        <label>轮次 <el-input-number v-model="form.epochs" :min="1" :max="100" /></label>
        <label>批次 <el-input-number v-model="form.batch_size" :min="1" :max="256" /></label>
        <label>学习率 <el-input-number v-model="form.learning_rate" :min="0.000001" :max="0.1" :step="0.001" :precision="6" /></label>
        <label>验证误报上限 <el-input-number v-model="form.max_false_positive_rate" :min="0" :max="1" :step="0.01" :precision="2" /></label>
      </el-form-item>
      <el-alert v-if="checked?.label_mode === 'source_proxy'" type="warning" :closable="false" title="当前 fall/nofall 来源标签不等于人工卧倒/支撑真值；同摄像机人员分组结果不能用于完整融合告警验收。" />
      <el-table v-if="checked" :data="splitRows" size="small">
        <el-table-column prop="split" label="固定集合" /><el-table-column prop="total" label="总样本" />
        <el-table-column prop="usable" label="关键点可用" /><el-table-column prop="positive" label="可用正类" /><el-table-column prop="negative" label="可用负类" />
      </el-table>
      <el-form-item><el-button type="primary" :loading="submitting" :disabled="!checked || !worker.available" @click="submit">提交姿态训练</el-button><span>GPU CUDA:0；独立队列；资源占用时等待；测试集不参与调参。</span></el-form-item>
    </el-form>
    <div class="task-toolbar"><el-button @click="refresh">刷新任务</el-button><span>共 {{ total }} 项</span></div>
    <el-table :data="tasks" row-key="id">
      <el-table-column label="名称 / 类型" min-width="190"><template #default="{ row }">{{ row.input?.name || row.input?.training_task_id }}<br>{{ row.type === 'pose_training' ? '姿态训练' : '锁定测试' }}</template></el-table-column>
      <el-table-column prop="owner_username" label="创建人" width="110" />
      <el-table-column label="状态" width="110"><template #default="{ row }">{{ poseStatusText(row.status) }}</template></el-table-column>
      <el-table-column label="进度" min-width="160"><template #default="{ row }"><el-progress :percentage="Math.round(row.progress || 0)" />{{ row.message }}</template></el-table-column>
      <el-table-column label="操作" min-width="280"><template #default="{ row }">
        <el-button link type="primary" @click="show(row)">详情 / 日志</el-button>
        <el-button v-if="poseTaskActive(row)" link type="danger" @click="cancel(row)">取消</el-button>
        <el-button v-if="poseCanEvaluate(row)" link type="primary" @click="evaluate(row)">锁定测试</el-button>
        <el-button v-if="row.status === 'completed' && row.type === 'pose_training'" link type="primary" @click="imageTask=row;imageVisible=true">本地图片测试 / 告警启用</el-button>
        <el-button v-if="row.status === 'completed'" link @click="download(row, 'report')">报告</el-button>
        <el-button v-if="row.status === 'completed' && row.type === 'pose_training'" link @click="download(row, 'weights')">实验权重</el-button>
        <el-button v-if="row.status === 'completed' && row.type === 'pose_evaluation'" link @click="download(row, 'predictions')">逐图结果</el-button>
      </template></el-table-column>
    </el-table>
    <el-pagination v-model:current-page="page" :page-size="20" :total="total" layout="prev, pager, next" @current-change="refresh" />
    <el-dialog v-model="detailsVisible" title="姿态任务详情" width="80%">
      <p>执行设备：{{ selected?.result?.execution?.device_name || (selected?.status === 'completed' ? '历史任务（未记录设备）' : 'GPU CUDA:0（等待任务执行）') }}；实际设备、耗时与显存记录在任务结果及报告中。</p>
      <p>关键点不足产生的拒判计入召回率分母；coverage 表示关键点覆盖率，并非识别准确率。</p>
      <p v-if="selected?.error_detail" class="error">{{ selected.error_detail }}</p>
      <pre>{{ JSON.stringify(selected?.result || {}, null, 2) }}</pre>
      <pre class="pose-events">{{ events.map(e => `${e.sequence} ${e.message}`).join('\n') }}</pre>
    </el-dialog>
    <el-dialog v-model="imageVisible" title="本地图片测试与倒地告警" width="92%" destroy-on-close :close-on-click-modal="false">
      <PoseImageTestPanel v-if="imageVisible && imageTask" :key="imageTask.id" :task-id="imageTask.id" />
    </el-dialog>
  </section>
</template>

<script setup>
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api/api.js'
import PoseImageTestPanel from './PoseImageTestPanel.vue'
import { poseTaskActive, poseCanEvaluate, mergePoseEvents, poseStatusText, createPoseRequestId } from '../utils/poseTraining.mjs'
const datasets=ref([]), worker=ref({available:false}), checked=ref(null), checking=ref(false), submitting=ref(false)
const tasks=ref([]), total=ref(0), page=ref(1), selected=ref(null), detailsVisible=ref(false), events=ref([])
const imageTask=ref(null),imageVisible=ref(false)
const form=reactive({name:'单帧姿态实验',dataset_id:'',epochs:30,batch_size:32,learning_rate:.001,max_false_positive_rate:.05,seed:20261018})
const splitRows=computed(()=>Object.entries(checked.value?.splits || {}).map(([split,s])=>({split,...s,positive:s.usable_classes['1'],negative:s.usable_classes['0']})))
let stopped=false, timer, polling=false, selectedPolling=false, failures=0, cursor=0, checkingVersion=0
const storageKey='pose-training-active-task'
const submissionKey='pose-training-pending-submission'
function invalidatePreflight() { checkingVersion++;checked.value=null;checking.value=false }
async function preflight() {
  const version=++checkingVersion, id=form.dataset_id;checked.value=null;checking.value=true
  try { const r=await api.get(`/pose-training/datasets/${encodeURIComponent(id)}/preflight`);if(version===checkingVersion && id===form.dataset_id) checked.value=r }
  catch { /* shared API reports request failure */ } finally { if(version===checkingVersion) checking.value=false }
}
async function refresh() {
  const r=await api.get('/pose-training/jobs',{params:{page:page.value,page_size:20}})
  if(stopped) return
  tasks.value=r.tasks;total.value=r.total
}
async function submit() {
  if(submitting.value || !checked.value) return
  submitting.value=true
  try {
    const config={...form,index_sha256:checked.value.index_sha256}
    // Preserve the same request ID across transport failures and refreshes.
    let saved
    try { saved=JSON.parse(sessionStorage.getItem(submissionKey) || 'null') } catch { saved=null }
    const fingerprint=JSON.stringify(config)
    const request_id=saved?.fingerprint===fingerprint && saved.request_id ? saved.request_id : createPoseRequestId()
    sessionStorage.setItem(submissionKey,JSON.stringify({fingerprint,request_id}))
    const r=await api.post('/pose-training/jobs',{...config,request_id});sessionStorage.removeItem(submissionKey)
    localStorage.setItem(storageKey,r.task.id);await show(r.task);await refresh();ElMessage.success(r.reused ? '已恢复已有任务' : '姿态训练已排队')
  } catch(error) {
    // Axios failures are reported by the shared API; also surface preparation errors.
    if(!error?.isAxiosError) ElMessage.error(error?.message || '姿态训练提交失败，请重试')
  } finally { submitting.value=false }
}
async function show(task) {
  if(selected.value?.id!==task.id) { events.value=[];cursor=0 }
  selected.value=task;detailsVisible.value=true;localStorage.setItem(storageKey,task.id)
  await pollSelected()
}
async function pollSelected() {
  if(!selected.value || selectedPolling) return
  selectedPolling=true
  const id=selected.value.id, after=cursor
  try {
    const [s,e]=await Promise.all([api.get(`/pose-training/jobs/${id}`),api.get(`/pose-training/jobs/${id}/events`,{params:{after_sequence:after}})])
    if(stopped || selected.value?.id!==id) return
    selected.value=s;events.value=mergePoseEvents(events.value,e.events);cursor=Math.max(cursor,e.next_sequence)
  } catch(e) {
    if(e.response?.status===404 && selected.value?.id===id) { selected.value=null;localStorage.removeItem(storageKey) }
    else throw e
  } finally { selectedPolling=false }
}
async function cancel(task) { try { await api.post(`/pose-training/jobs/${task.id}/cancel`);await refresh() } catch {} }
async function evaluate(task) {
  try {
    await ElMessageBox.confirm('测试将使用验证集确定的阈值。同一训练版本的测试任务只创建一次；查看后请勿据此反复调参。','启动锁定测试')
    const r=await api.post(`/pose-training/jobs/${task.id}/evaluate`);await show(r.task);await refresh()
  } catch {}
}
async function download(task,artifact) {
  try {
    const r=await api.get(`/pose-training/jobs/${task.id}/files/${artifact}`,{responseType:'blob'})
    const url=URL.createObjectURL(r),a=document.createElement('a');a.href=url;a.download=`${task.id}-${artifact}${artifact==='weights' ? '.pt' : artifact==='predictions' ? '.jsonl' : '.json'}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)
  } catch {}
}
async function poll() {
  if(stopped || polling) return
  polling=true
  try {
    const r=await api.get('/pose-training/datasets');if(stopped) return
    datasets.value=r.datasets;worker.value=r.worker
    await refresh();await pollSelected();failures=0
  } catch { failures=Math.min(failures+1,4) } finally {
    polling=false;if(!stopped) timer=setTimeout(poll,Math.min(30000,3000*2**failures))
  }
}
onMounted(()=>{
  const id=localStorage.getItem(storageKey)
  if(/^[a-f0-9-]{36}$/.test(id || '')) selected.value={id}
  poll()
})
onUnmounted(()=>{ stopped=true;clearTimeout(timer);checkingVersion++ })
</script>

<style scoped>
.pose-form { margin-top: 20px; }
.pose-training :deep(.el-button.is-link) { background:transparent !important;box-shadow:none !important; }
.pose-form .el-select { width: min(600px, 100%); }
.pose-form label { margin: 4px 16px 4px 0; }
.pose-form .el-form-item { margin-top: 16px; }
.task-toolbar { display: flex; align-items: center; gap: 16px; margin: 20px 0; }
pre { max-height: 320px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; }
.pose-events { background: #f5f7fa; padding: 12px; }.error { color: #c45656; }
</style>

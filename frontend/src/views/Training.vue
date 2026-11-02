<template><main class="medical-page"><h1>模型训练</h1><p>选择冻结数据版本和可信模型；预检成功后提交到独立训练队列。</p>
  <el-descriptions v-if="resources" border :column="3"><el-descriptions-item label="CPU 核数">{{ resources.cpu_count }}</el-descriptions-item><el-descriptions-item label="可用磁盘">{{ (resources.disk_free_bytes / 1073741824).toFixed(1) }} GB</el-descriptions-item><el-descriptions-item label="GPU 租约">{{ leaseLabels[resources.gpu_lease_state] }}</el-descriptions-item><el-descriptions-item label="排队任务">{{ resources.pending_jobs }}</el-descriptions-item><el-descriptions-item label="Worker">{{ resources.workers }}</el-descriptions-item></el-descriptions>
  <div class="toolbar"><label>训练项目名称 <input v-model="projectName" maxlength="128" /></label></div>
  <DatasetVersionPicker ref="picker" frozen-only @change="selection=$event" />
  <div class="toolbar"><label>初始模型 <select aria-label="训练模型" v-model="modelId"><option value="">请选择</option><option v-for="m in models" :key="m.model_id" :value="m.model_id">{{m.name}}</option></select></label><label>Epochs <input aria-label="Epochs" type="number" v-model.number="params.epochs" min="1" max="1000" /></label><label>Batch <input aria-label="Batch" type="number" v-model.number="params.batch" min="1" max="128" /></label><label>图像尺寸 <input aria-label="图像尺寸" type="number" v-model.number="params.imgsz" min="32" max="2048" /></label><label>Seed <input type="number" v-model.number="params.seed" min="0" /></label><label>设备 <select v-model="params.device"><option value="0">GPU 0</option><option value="cpu">CPU</option></select></label></div>
  <el-collapse><el-collapse-item title="优化器、学习率与早停" name="advanced"><div class="toolbar">
    <label>优化器 <select v-model="params.optimizer"><option>AdamW</option><option>SGD</option></select></label>
    <label>学习率 <input aria-label="学习率" v-model.number="params.lr0" type="number" min="0.000001" max="1" step="0.0001" /></label>
    <label>最终学习率比例 <input v-model.number="params.lrf" type="number" min="0.000001" max="1" step="0.01" /></label>
    <label>权重衰减 <input v-model.number="params.weight_decay" type="number" min="0" max="1" step="0.0001" /></label>
    <label>预热轮数 <input v-model.number="params.warmup_epochs" type="number" min="0" max="100" /></label>
    <label>早停耐心值 <input v-model.number="params.early_stopping.patience" type="number" min="0" max="1000" /></label>
  </div></el-collapse-item></el-collapse>
  <div class="toolbar"><el-button :disabled="busy || !modelId || !selection?.version" @click="act(preflight)">训练预检</el-button><el-button type="primary" :disabled="busy || !report?.ok || preflightKey!==requestKey" @click="act(start)">提交训练</el-button><el-button @click="act(refresh)">刷新任务</el-button></div>
  <p v-if="error" role="alert" class="error">{{error}}</p><div v-if="report"><strong>{{report.ok?'预检通过':'预检未通过'}}</strong><pre>{{JSON.stringify(report,null,2)}}</pre></div>
  <JobProgress :job-id="jobId" @update="updated" @terminal="refresh" />
  <details v-if="events.length"><summary>任务事件</summary><p v-for="e in events" :key="e.sequence">{{e.created_at}} · {{e.message}}</p></details>
  <div class="toolbar"><label>任务名称 / ID <input v-model="jobSearch" /></label><label>训练人 <input v-model="ownerSearch" /></label><label><input v-model="includeArchived" type="checkbox" />包含归档</label><el-button @click="page = 1; act(refresh)">查询任务</el-button></div>
  <el-table :data="jobs" stripe border>
    <el-table-column label="项目 / 训练人" min-width="180"><template #default="{ row }">{{ row.input?.name || '训练任务' }}<p>{{ row.owner }}</p></template></el-table-column>
    <el-table-column prop="job_id" label="训练任务" min-width="200" show-overflow-tooltip />
    <el-table-column label="训练配置" min-width="190"><template #default="{ row }">{{ row.input?.params?.epochs }} 轮 · Batch {{ row.input?.params?.batch }} · {{ row.input?.params?.imgsz }} px</template></el-table-column>
    <el-table-column prop="status" label="状态" width="120" />
    <el-table-column prop="stage" label="阶段" width="130" />
    <el-table-column label="进度" width="150"><template #default="{ row }"><el-progress :percentage="Math.round((row.progress || 0) * 100)" /></template></el-table-column>
    <el-table-column label="操作" min-width="310"><template #default="{ row }"><el-button size="small" @click="jobId = row.job_id; events = []; detail = row">详情</el-button><el-button size="small" :disabled="busy" @click="act(() => reuse(row))">载入配置重训</el-button><el-button v-if="row.archived_at" size="small" :disabled="busy" @click="act(() => restore(row))">恢复</el-button><el-button v-else-if="['completed','failed','cancelled'].includes(row.status)" size="small" :disabled="busy" @click="archive(row)">归档</el-button></template></el-table-column>
  </el-table>
  <el-dialog :model-value="!!detail" title="训练详情" width="min(850px, 95vw)" @close="detail = null"><pre class="training-detail">{{ JSON.stringify(detail, null, 2) }}</pre></el-dialog>
  <el-pagination v-model:current-page="page" :total="total" :page-size="10" layout="prev, pager, next" @current-change="act(refresh)" />
</main></template>
<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import DatasetVersionPicker from '../components/medical/DatasetVersionPicker.vue'
import JobProgress from '../components/medical/JobProgress.vue'
import { listModels } from '../api/models'
import { preflightTraining, startTraining, listJobs, getJobEvents, getResources, archiveJob, restoreJob } from '../api/training'
import { ElMessageBox } from 'element-plus'
import { useTaskAction } from '../composables/useTaskAction'
import { getVersion } from '../api/datasets'
const {error,busy,act}=useTaskAction(),selection=ref(null),models=ref([]),modelId=ref(''),jobs=ref([]),jobId=ref(''),events=ref([]),page=ref(1),total=ref(0),report=ref(null),preflightKey=ref('')
const params=reactive({epochs:1,batch:2,imgsz:160,seed:42,deterministic:true,device:'0',optimizer:'AdamW',lr0:.001,lrf:.01,weight_decay:.0005,warmup_epochs:3,early_stopping:{patience:20,monitor:'val/mAP50-95'}})
const picker = ref(null), detail = ref(null)
const resources = ref(null), projectName = ref('训练任务'), jobSearch = ref(''), ownerSearch = ref(''), includeArchived = ref(false)
const leaseLabels = { available: '当前无租约占用', held: '任务占用中', recovery_required: '租约过期，等待恢复确认' }
async function archive(job) {
  try { await ElMessageBox.confirm('归档此任务？训练血缘和模型记录仍可追溯。', '归档任务') }
  catch { return }
  await act(async () => { await archiveJob(job.job_id); await refresh() })
}
async function restore(job) { await restoreJob(job.job_id); await refresh() }
async function reuse(job) {
  const version = await getVersion(job.input.dataset_version_id)
  await picker.value.select(version.dataset_id, version.version_id)
  Object.assign(params, JSON.parse(JSON.stringify(job.input.params)))
  modelId.value = job.input.model_id || ''
  projectName.value = job.input.name || '训练任务'
  report.value = null
  preflightKey.value = ''
  window.scrollTo({ top: 0, behavior: 'smooth' })
}
const request=computed(()=>({name:projectName.value,dataset_version_id:selection.value?.version?.version_id,model_id:modelId.value,task:'detect',params:{...params}})),requestKey=computed(()=>JSON.stringify(request.value))
async function preflight(){const key=requestKey.value;report.value=await preflightTraining(JSON.parse(key));preflightKey.value=key}
async function start(){jobId.value=(await startTraining(request.value,crypto.randomUUID())).job_id;events.value=[];await refresh()}
async function refresh(){const data=await listJobs({type:'training',page:page.value,page_size:10,search:jobSearch.value,owner:ownerSearch.value,include_archived:includeArchived.value});jobs.value=data.items;total.value=data.total;resources.value=await getResources()}
async function updated(job){try{const data=await getJobEvents(job.job_id,events.value.at(-1)?.sequence||0);if(jobId.value===job.job_id)events.value=[...events.value,...data.items].slice(-200)}catch(e){error.value=e.message}}
onMounted(()=>act(async()=>{models.value=(await listModels({page_size:200,task:'detect'})).items;await refresh()}))
</script>
<style scoped>
.training-detail { white-space: pre-wrap; overflow-wrap: anywhere; }
</style>

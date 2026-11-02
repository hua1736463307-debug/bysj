<template><main class="medical-page"><h1>模型与评估</h1><p>通过文件哈希和训练血缘识别模型；验证集用于开发，测试集评估必须声明用途。</p>
  <fieldset v-if="isAdmin"><legend>登记基础模型</legend><div class="toolbar"><label>名称 <input v-model="baseName" /></label><label>基础权重目录内文件 <input v-model="basePath" placeholder="yolo11n.pt" /></label><label>SHA256 <input v-model="baseHash" size="48" /></label><el-button :disabled="busy" @click="act(register)">登记模型</el-button></div><p class="muted">由管理员预先放入受控基础权重目录，只登记哈希匹配的文件。</p></fieldset>
  <el-button v-if="isAdmin" @click="uploadVisible = true">上传检测模型</el-button>
  <p v-if="error" class="error" role="alert">{{error}}</p>
  <el-dialog v-model="uploadVisible" title="上传检测模型" width="min(650px, 95vw)" :close-on-click-modal="false">
    <el-form label-width="100px">
      <el-form-item label="名称"><el-input v-model="uploadName" /></el-form-item>
      <el-form-item label="模型文件"><input type="file" accept=".pt" @change="uploadFile = $event.target.files[0]" /></el-form-item>
      <el-form-item label="SHA256"><el-input v-model="uploadHash" maxlength="64" placeholder="填写来源文件的 SHA256" /></el-form-item>
      <el-form-item label="类别映射"><el-input v-model="uploadClasses" type="textarea" :rows="3" /></el-form-item>
    </el-form>
    <el-alert v-if="error" :title="error" type="error" :closable="false" />
    <template #footer><el-button :disabled="busy" @click="uploadVisible = false">取消</el-button><el-button type="primary" :loading="busy" :disabled="!uploadFile || !uploadName || uploadHash.length !== 64" @click="act(upload)">上传并校验</el-button></template>
  </el-dialog>
  <el-form inline @submit.prevent="searchModels">
    <el-form-item label="模型名称"><el-input v-model="search" placeholder="搜索模型" clearable /></el-form-item>
    <el-form-item label="任务"><el-select v-model="task" clearable placeholder="全部任务"><el-option label="目标检测" value="detect" /><el-option label="胸片多标签" value="cxr_multilabel" /><el-option label="皮肤镜分类" value="isic_multiclass" /></el-select></el-form-item>
    <el-form-item><el-button native-type="submit" :disabled="busy">搜索</el-button></el-form-item>
  </el-form>
  <div class="model-grid">
    <el-card v-for="m in models" :key="m.model_id">
      <template #header><strong>{{ m.name }}</strong> <el-tag size="small">{{ m.format }}</el-tag></template>
      <el-descriptions :column="1" size="small">
        <el-descriptions-item label="任务">{{ m.task }}</el-descriptions-item>
        <el-descriptions-item label="来源">{{ m.trusted_source }}</el-descriptions-item>
        <el-descriptions-item label="训练人">{{ m.lineage?.created_by }}</el-descriptions-item>
        <el-descriptions-item label="大小">{{ (m.size / 1048576).toFixed(2) }} MB</el-descriptions-item>
      </el-descriptions>
      <div class="model-actions">
        <el-button size="small" @click="detail = m">详情</el-button>
        <el-button size="small" :disabled="busy" @click="act(() => download(m))">下载</el-button>
        <el-button v-if="m.task === 'detect'" size="small" @click="modelId = m.model_id">选择评估</el-button>
        <el-button v-if="m.trusted_source !== 'base'" size="small" :disabled="busy" @click="removeModel(m)">删除</el-button>
      </div>
    </el-card>
  </div>
  <el-empty v-if="!busy && !models.length" description="暂无匹配模型" />
  <el-dialog :model-value="!!detail" title="模型详情与血缘" width="min(750px, 95vw)" @close="detail = null">
    <pre class="model-detail">{{ JSON.stringify(detail, null, 2) }}</pre>
  </el-dialog>
  <el-pagination v-model:current-page="page" :total="total" :page-size="10" layout="prev, pager, next" @current-change="act(refresh)" />
  <h2>独立评估</h2><DatasetVersionPicker frozen-only @change="selection=$event" /><div class="toolbar"><span>所选模型：{{modelId || '尚未选择'}}</span><label>Split <select v-model="split"><option value="val">val</option><option value="test">test</option></select></label><label>用途 <input aria-label="评估用途" v-model="purpose" /></label><label>尺寸 <input type="number" v-model.number="imgsz" min="32" max="2048" /></label><el-button type="primary" :disabled="busy || !modelId || !selection?.version || (split==='test'&&!purpose)" @click="act(evaluate)">开始评估</el-button></div>
  <JobProgress :job-id="jobId" @terminal="act(loadEvaluations)" />
  <table><thead><tr><th>评估 / split</th><th>状态</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50-95</th><th>详情</th></tr></thead><tbody><tr v-for="e in evaluations" :key="e.evaluation_id"><td>{{e.evaluation_id}}<div>{{e.split}}</div></td><td>{{e.status}}</td><td>{{metric(e,'precision')}}</td><td>{{metric(e,'recall')}}</td><td>{{metric(e,'mAP50')}}</td><td>{{metric(e,'mAP50_95')}}</td><td><details><summary>指标与产物</summary><pre>{{JSON.stringify({metrics:e.metrics,artifacts:e.artifacts},null,2)}}</pre></details></td></tr></tbody></table>
</main></template>
<script setup>
import { ref, onMounted } from 'vue'
import { getCurrentUser } from '../api/client'
import { listModels, registerModel, downloadModel, deleteModel, uploadModel } from '../api/models'
import { ElMessageBox } from 'element-plus'
import { createEvaluation, listEvaluations } from '../api/experiments'
import { useTaskAction } from '../composables/useTaskAction'
import { saveBlob } from '../utils/download'
import DatasetVersionPicker from '../components/medical/DatasetVersionPicker.vue'
import JobProgress from '../components/medical/JobProgress.vue'
const {error,busy,act}=useTaskAction(),isAdmin=getCurrentUser()?.role==='admin',models=ref([]),page=ref(1),total=ref(0),modelId=ref(''),selection=ref(null),split=ref('val'),purpose=ref(''),imgsz=ref(160),jobId=ref(''),evaluations=ref([]),baseName=ref('yolo11n'),basePath=ref('yolo11n.pt'),baseHash=ref('')
const search = ref(''), task = ref(''), detail = ref(null)
const uploadVisible = ref(false), uploadName = ref(''), uploadHash = ref(''), uploadFile = ref(null), uploadClasses = ref('{"0":"cell"}')
async function upload() {
  const form = new FormData()
  form.append('file', uploadFile.value)
  form.append('name', uploadName.value)
  form.append('sha256', uploadHash.value)
  form.append('class_map', uploadClasses.value)
  await uploadModel(form)
  uploadVisible.value = false
  await refresh()
}
async function refresh(){const data=await listModels({page:page.value,page_size:10,search:search.value,task:task.value || undefined});models.value=data.items;total.value=data.total}
function searchModels() { page.value = 1; act(refresh) }
async function removeModel(model) {
  try { await ElMessageBox.confirm('删除模型「' + model.name + '」？已被评估或复核引用的模型将拒绝删除。', '删除模型') }
  catch { return }
  await act(async () => {
    await deleteModel(model.model_id)
    if (modelId.value === model.model_id) modelId.value = ''
    if (models.value.length === 1 && page.value > 1) page.value--
    await refresh()
  })
}
async function register(){await registerModel({name:baseName.value,relative_path:basePath.value,sha256:baseHash.value,class_map:{0:'RBC',1:'WBC',2:'Platelets'}});await refresh()}
async function download(m){saveBlob(await downloadModel(m.model_id),`${m.name.replace(/[^a-zA-Z0-9_.-]/g,'_')}.pt`)}
async function evaluate(){const data=await createEvaluation({model_id:modelId.value,dataset_version_id:selection.value.version.version_id,split:split.value,purpose:purpose.value,params:{imgsz:imgsz.value,device:'0'},idempotency_key:crypto.randomUUID()});jobId.value=data.job.job_id;await loadEvaluations()}
async function loadEvaluations(){evaluations.value=(await listEvaluations({page_size:50})).items}
function metric(e,key){const value=e.metrics?.[key.toLowerCase()];return Number.isFinite(value)?value.toFixed(4):'未提供'}
onMounted(()=>act(async()=>{await refresh();await loadEvaluations()}))
</script>
<style scoped>
.model-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 18px; margin: 18px 0; }
.model-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.model-actions .el-button { margin: 0; }
.model-detail { white-space: pre-wrap; overflow-wrap: anywhere; }
.el-select { width: 190px; }
</style>

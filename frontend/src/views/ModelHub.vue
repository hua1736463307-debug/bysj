<template><main class="medical-page"><h1>模型与评估</h1><p>通过文件哈希和训练血缘识别模型；验证集用于开发，测试集评估必须声明用途。</p>
  <fieldset v-if="isAdmin"><legend>登记基础模型</legend><div class="toolbar"><label>名称 <input v-model="baseName" /></label><label>基础权重目录内文件 <input v-model="basePath" placeholder="yolo11n.pt" /></label><label>SHA256 <input v-model="baseHash" size="48" /></label><el-button :disabled="busy" @click="act(register)">登记模型</el-button></div><p class="muted">由管理员预先放入受控基础权重目录，只登记哈希匹配的文件。</p></fieldset>
  <p v-if="error" class="error" role="alert">{{error}}</p>
  <table><thead><tr><th>模型</th><th>来源与血缘</th><th>SHA256</th><th>操作</th></tr></thead><tbody><tr v-for="m in models" :key="m.model_id"><td>{{m.name}}<div class="muted">{{m.model_id}}</div></td><td>{{m.trusted_source}}<div class="muted">版本 {{m.lineage?.dataset_version_id || '基础权重'}}</div></td><td class="hash">{{m.sha256}}</td><td><el-button @click="modelId=m.model_id">选择评估</el-button><el-button @click="act(()=>download(m))">下载</el-button></td></tr></tbody></table>
  <el-pagination v-model:current-page="page" :total="total" :page-size="10" layout="prev, pager, next" @current-change="act(refresh)" />
  <h2>独立评估</h2><DatasetVersionPicker frozen-only @change="selection=$event" /><div class="toolbar"><span>所选模型：{{modelId || '尚未选择'}}</span><label>Split <select v-model="split"><option value="val">val</option><option value="test">test</option></select></label><label>用途 <input aria-label="评估用途" v-model="purpose" /></label><label>尺寸 <input type="number" v-model.number="imgsz" min="32" max="2048" /></label><el-button type="primary" :disabled="busy || !modelId || !selection?.version || (split==='test'&&!purpose)" @click="act(evaluate)">开始评估</el-button></div>
  <JobProgress :job-id="jobId" @terminal="act(loadEvaluations)" />
  <table><thead><tr><th>评估 / split</th><th>状态</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50-95</th><th>详情</th></tr></thead><tbody><tr v-for="e in evaluations" :key="e.evaluation_id"><td>{{e.evaluation_id}}<div>{{e.split}}</div></td><td>{{e.status}}</td><td>{{metric(e,'precision')}}</td><td>{{metric(e,'recall')}}</td><td>{{metric(e,'mAP50')}}</td><td>{{metric(e,'mAP50_95')}}</td><td><details><summary>指标与产物</summary><pre>{{JSON.stringify({metrics:e.metrics,artifacts:e.artifacts},null,2)}}</pre></details></td></tr></tbody></table>
</main></template>
<script setup>
import { ref, onMounted } from 'vue'
import { getCurrentUser } from '../api/client'
import { listModels, registerModel, downloadModel } from '../api/models'
import { createEvaluation, listEvaluations } from '../api/experiments'
import { useTaskAction } from '../composables/useTaskAction'
import { saveBlob } from '../utils/download'
import DatasetVersionPicker from '../components/medical/DatasetVersionPicker.vue'
import JobProgress from '../components/medical/JobProgress.vue'
const {error,busy,act}=useTaskAction(),isAdmin=getCurrentUser()?.role==='admin',models=ref([]),page=ref(1),total=ref(0),modelId=ref(''),selection=ref(null),split=ref('val'),purpose=ref(''),imgsz=ref(160),jobId=ref(''),evaluations=ref([]),baseName=ref('yolo11n'),basePath=ref('yolo11n.pt'),baseHash=ref('')
async function refresh(){const data=await listModels({page:page.value,page_size:10});models.value=data.items;total.value=data.total}
async function register(){await registerModel({name:baseName.value,relative_path:basePath.value,sha256:baseHash.value,class_map:{0:'RBC',1:'WBC',2:'Platelets'}});await refresh()}
async function download(m){saveBlob(await downloadModel(m.model_id),`${m.name.replace(/[^a-zA-Z0-9_.-]/g,'_')}.pt`)}
async function evaluate(){const data=await createEvaluation({model_id:modelId.value,dataset_version_id:selection.value.version.version_id,split:split.value,purpose:purpose.value,params:{imgsz:imgsz.value,device:'0'},idempotency_key:crypto.randomUUID()});jobId.value=data.job.job_id;await loadEvaluations()}
async function loadEvaluations(){evaluations.value=(await listEvaluations({page_size:50})).items}
function metric(e,key){const value=e.metrics?.[key.toLowerCase()];return Number.isFinite(value)?value.toFixed(4):'未提供'}
onMounted(()=>act(async()=>{await refresh();await loadEvaluations()}))
</script>

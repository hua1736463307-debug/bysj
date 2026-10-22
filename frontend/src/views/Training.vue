<template><main class="medical-page"><h1>模型训练</h1><p>选择冻结数据版本和可信模型；预检成功后提交到独立训练队列。</p>
  <DatasetVersionPicker frozen-only @change="selection=$event" />
  <div class="toolbar"><label>初始模型 <select aria-label="训练模型" v-model="modelId"><option value="">请选择</option><option v-for="m in models" :key="m.model_id" :value="m.model_id">{{m.name}}</option></select></label><label>Epochs <input aria-label="Epochs" type="number" v-model.number="params.epochs" min="1" max="1000" /></label><label>Batch <input aria-label="Batch" type="number" v-model.number="params.batch" min="1" max="128" /></label><label>图像尺寸 <input aria-label="图像尺寸" type="number" v-model.number="params.imgsz" min="32" max="2048" /></label><label>Seed <input type="number" v-model.number="params.seed" min="0" /></label><label>设备 <select v-model="params.device"><option value="0">GPU 0</option><option value="cpu">CPU</option></select></label></div>
  <div class="toolbar"><el-button :disabled="busy || !modelId || !selection?.version" @click="act(preflight)">训练预检</el-button><el-button type="primary" :disabled="busy || !report?.ok || preflightKey!==requestKey" @click="act(start)">提交训练</el-button><el-button @click="act(refresh)">刷新任务</el-button></div>
  <p v-if="error" role="alert" class="error">{{error}}</p><div v-if="report"><strong>{{report.ok?'预检通过':'预检未通过'}}</strong><pre>{{JSON.stringify(report,null,2)}}</pre></div>
  <JobProgress :job-id="jobId" @update="updated" @terminal="refresh" />
  <details v-if="events.length"><summary>任务事件</summary><p v-for="e in events" :key="e.sequence">{{e.created_at}} · {{e.message}}</p></details>
  <table><thead><tr><th>任务 ID</th><th>状态</th><th>阶段</th><th>进度</th><th>操作</th></tr></thead><tbody><tr v-for="job in jobs" :key="job.job_id"><td>{{job.job_id}}</td><td>{{job.status}}</td><td>{{job.stage}}</td><td>{{Math.round((job.progress||0)*100)}}%</td><td><el-button @click="jobId=job.job_id;events=[]">查看</el-button></td></tr></tbody></table>
  <el-pagination v-model:current-page="page" :total="total" :page-size="10" layout="prev, pager, next" @current-change="act(refresh)" />
</main></template>
<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import DatasetVersionPicker from '../components/medical/DatasetVersionPicker.vue'
import JobProgress from '../components/medical/JobProgress.vue'
import { listModels } from '../api/models'
import { preflightTraining, startTraining, listJobs, getJobEvents } from '../api/training'
import { useTaskAction } from '../composables/useTaskAction'
const {error,busy,act}=useTaskAction(),selection=ref(null),models=ref([]),modelId=ref(''),jobs=ref([]),jobId=ref(''),events=ref([]),page=ref(1),total=ref(0),report=ref(null),preflightKey=ref('')
const params=reactive({epochs:1,batch:2,imgsz:160,seed:42,deterministic:true,device:'0'})
const request=computed(()=>({dataset_version_id:selection.value?.version?.version_id,model_id:modelId.value,task:'detect',params:{...params}})),requestKey=computed(()=>JSON.stringify(request.value))
async function preflight(){report.value=await preflightTraining(request.value);preflightKey.value=requestKey.value}
async function start(){jobId.value=(await startTraining(request.value,crypto.randomUUID())).job_id;events.value=[];await refresh()}
async function refresh(){const data=await listJobs({type:'training',page:page.value,page_size:10});jobs.value=data.items;total.value=data.total}
async function updated(job){try{const data=await getJobEvents(job.job_id,events.value.at(-1)?.sequence||0);if(jobId.value===job.job_id)events.value=[...events.value,...data.items].slice(-200)}catch(e){error.value=e.message}}
onMounted(()=>act(async()=>{models.value=(await listModels({page_size:200})).items;await refresh()}))
</script>

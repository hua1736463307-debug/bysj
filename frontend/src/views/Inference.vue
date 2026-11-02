<template><main class="medical-page"><h1>推理与人工复核</h1><p>上传单张或多张图片，或选取数据版本中的资产。推理结果会保存，可在刷新后继续查看。</p>
  <div class="toolbar"><label>推理模型 <select aria-label="推理模型" v-model="modelId"><option value="">请选择</option><option v-for="m in models" :key="m.model_id" :value="m.model_id">{{m.name}}</option></select></label><label>置信度 <input type="number" min="0" max="1" step=".05" v-model.number="confidence" /></label><input aria-label="推理文件" type="file" multiple accept="image/*" @change="files=Array.from($event.target.files)" /><el-button :disabled="busy || !modelId || !files.length" @click="act(()=>start(false))">上传图片推理</el-button></div>
  <DatasetVersionPicker @change="changed" /><div class="toolbar"><label v-for="a in assets" :key="a.asset_id"><input type="checkbox" v-model="selectedAssets" :value="a.asset_id" :disabled="!selectedAssets.includes(a.asset_id) && selectedAssets.length>=100" />{{a.basename}}</label><el-button :disabled="busy || !modelId || !selectedAssets.length" @click="act(()=>start(true))">所选资产推理</el-button><span>已选 {{selectedAssets.length}} / 100 张</span></div>
  <el-pagination v-if="selection?.version" class="asset-pagination" v-model:current-page="assetPage" :page-size="20" :total="assetTotal" layout="total, prev, pager, next" @current-change="act(loadAssets)" />
  <p v-if="error" class="error" role="alert">{{error}}</p><JobProgress :job-id="jobId" @terminal="act(()=>openRun(runId))" />
  <div class="toolbar"><el-button @click="act(refreshRuns)">刷新运行记录</el-button><select aria-label="推理记录" v-model="runId" @change="act(()=>openRun(runId))"><option value="">请选择记录</option><option v-for="r in runs" :key="r.run_id" :value="r.run_id">{{r.created_at}} · {{r.status}} · {{r.run_id.slice(0,8)}}</option></select></div>
  <template v-if="run"><div class="toolbar"><strong>{{run.status}} · 完成 {{run.completed}} / {{run.total}} · 失败 {{run.failed}}</strong><el-button @click="act(()=>download('json'))">导出 JSON</el-button><el-button @click="act(()=>download('csv'))">导出 CSV</el-button></div>
    <article v-for="item in run.items" :key="item.item_id" class="job-card"><strong>{{item.status}} · {{item.item_id}}</strong><p v-if="item.error" class="error">{{item.error}}</p><div class="toolbar"><ControlledImage v-if="item.artifact_id" :artifact-id="item.artifact_id" class="preview" /><div><p v-for="p in item.predictions" :key="p.id">{{p.class_name}} {{p.confidence.toFixed(3)}} · {{p.bbox_xyxy.map(v=>v.toFixed(1)).join(', ')}}</p><el-button v-if="item.asset_id && item.status==='completed'" @click="selectReview(item)">人工复核</el-button><p v-else class="muted">资产推理结果可进入数据版本反馈。</p></div></div></article>
    <el-pagination v-model:current-page="itemPage" :page-size="8" :total="run.total" layout="prev, pager, next" @current-change="act(()=>openRun(runId,false))" />
  </template>
  <fieldset v-if="reviewTarget"><legend>人工复核</legend><p>资产 {{reviewTarget.asset_id}}；逐项确认预测，或填写漏检框（原图归一化坐标）。</p><div v-for="p in reviewTarget.predictions" :key="p.id" class="toolbar"><span>{{p.class_name}} {{p.confidence.toFixed(3)}}</span><select v-model="verdicts[p.id]"><option value="correct">正确</option><option value="false_positive">误检</option><option value="unreviewable">无法判定</option></select></div>
    <div class="toolbar"><label><input type="checkbox" v-model="missed" />补充漏检框</label><label>类别 <input type="number" min="0" v-model.number="missedBox.class_id" /></label><label v-for="key in ['cx','cy','w','h']" :key="key">{{key}} <input type="number" min="0" max="1" step=".01" v-model.number="missedBox[key]" /></label></div><label>复核说明 <input aria-label="复核说明" v-model="reason" size="60" /></label><el-button :disabled="busy" @click="act(submit)">提交人工复核</el-button>
  </fieldset>
  <h2>复核记录与回流</h2><el-button @click="act(refreshReviews)">刷新复核</el-button><table><thead><tr><th>选择</th><th>记录 / 资产</th><th>状态</th><th>说明</th><th>操作</th></tr></thead><tbody><tr v-for="r in reviews" :key="r.review_id"><td><input type="checkbox" :disabled="r.state!=='approved'" v-model="selectedReviews" :value="r.review_id" /></td><td>{{r.review_id}} / {{r.image_asset_id}}</td><td>{{r.state}}</td><td>{{r.reason}}</td><td><el-button v-if="canReview && r.state==='submitted'" @click="act(()=>decide(r,true))">批准</el-button><el-button v-if="canReview && r.state==='submitted'" @click="act(()=>decide(r,false))">驳回</el-button></td></tr></tbody></table>
  <p class="muted">回流以当前上方所选版本为父版本；仅使用训练侧资产，固定验证/测试来源组。</p><el-button :disabled="busy || !selection?.version || !selectedReviews.length" @click="act(feedback)">生成反馈子版本</el-button>
</main></template>
<script setup>
import { ref, reactive, onMounted } from 'vue'
import { listModels } from '../api/models'
import { listAssets } from '../api/datasets'
import { createInference,listRuns,getRun,exportRun } from '../api/inference'
import { createReview,submitReview,listReviews,decideReview,exportVersion } from '../api/reviews'
import { getCurrentUser } from '../api/client'
import { useTaskAction } from '../composables/useTaskAction'
import { saveBlob } from '../utils/download'
import DatasetVersionPicker from '../components/medical/DatasetVersionPicker.vue'
import ControlledImage from '../components/medical/ControlledImage.vue'
import JobProgress from '../components/medical/JobProgress.vue'
const {error,busy,act}=useTaskAction(),models=ref([]),modelId=ref(''),confidence=ref(.25),files=ref([]),selection=ref(null),assets=ref([]),selectedAssets=ref([]),runs=ref([]),runId=ref(''),run=ref(null),jobId=ref(''),itemPage=ref(1),reviewTarget=ref(null),verdicts=reactive({}),reason=ref(''),missed=ref(false),missedBox=reactive({class_id:0,cx:.5,cy:.5,w:.1,h:.1}),reviews=ref([]),selectedReviews=ref([]),canReview=['admin','reviewer'].includes(getCurrentUser()?.role)
const assetPage=ref(1),assetTotal=ref(0);let assetGeneration=0
async function changed(data){++assetGeneration;selection.value=data;assets.value=[];selectedAssets.value=[];assetPage.value=1;assetTotal.value=0;if(data.version)await act(loadAssets)}
async function loadAssets(){const data=selection.value;if(!data?.version)return;const ticket=++assetGeneration;const result=await listAssets(data.datasetId,{version_id:data.version.version_id,page:assetPage.value,page_size:20});if(ticket!==assetGeneration)return;assets.value=result.items;assetTotal.value=result.total}
async function start(useAssets){const data=await createInference({modelId:modelId.value,files:useAssets?[]:files.value,assetIds:useAssets?selectedAssets.value:[],params:{confidence:confidence.value,imgsz:640,device:'0'}});runId.value=data.run_id;jobId.value=data.job.job_id;await openRun(data.run_id);await refreshRuns()}
async function refreshRuns(){runs.value=(await listRuns({page_size:100})).items}
async function openRun(id,reset=true){if(!id)return;if(reset)itemPage.value=1;run.value=await getRun(id,{page:itemPage.value,page_size:8});runId.value=id;await refreshRuns()}
async function download(fmt){saveBlob(await exportRun(runId.value,fmt),`${runId.value}.${fmt}`)}
function selectReview(item){reviewTarget.value=item;for(const key of Object.keys(verdicts))delete verdicts[key];for(const p of item.predictions)verdicts[p.id]='correct';reason.value='';missed.value=false}
async function submit(){const items=reviewTarget.value.predictions.map(p=>({prediction_id:p.id,verdict:verdicts[p.id]}));if(missed.value)items.push({verdict:'missed_object',corrected:{...missedBox}});if(!items.length)throw new Error('请至少填写一条复核结论或漏检框');const review=await createReview({inference_item_id:reviewTarget.value.item_id,image_asset_id:reviewTarget.value.asset_id,reason:reason.value,items});await submitReview(review.review_id);reviewTarget.value=null;await refreshReviews()}
async function refreshReviews(){reviews.value=(await listReviews({page_size:100})).items}
async function decide(review,approve){await decideReview(review.review_id,approve,approve?'人工检查通过':'需要进一步修正');await refreshReviews()}
async function feedback(){const job=await exportVersion(selection.value.version.version_id,selectedReviews.value,crypto.randomUUID());jobId.value=job.job_id}
onMounted(()=>act(async()=>{models.value=(await listModels({page_size:200,task:'detect'})).items;await refreshRuns();await refreshReviews()}))
</script>

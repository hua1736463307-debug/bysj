<template>
  <main class="medical-page"><h1>数据版本与划分</h1>
    <DatasetVersionPicker ref="picker" @change="change" />
    <div class="toolbar"><label>导出格式 <select aria-label="导出格式" v-model="exportFormat"><option value="yolo-txt">YOLO 检测标签</option><option value="yolo-seg">YOLO 多边形标签</option><option value="yolo-dataset">完整检测训练集（图片与标签）</option><option value="yolo-seg-dataset">完整多边形训练集（图片与标签）</option><option value="json">JSON 完整标注</option></select></label></div>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <template v-if="version"><div class="toolbar"><strong>{{ version.state }} · {{ version.asset_count }} 图</strong><label>划分种子 <input type="number" min="0" v-model.number="seed" :disabled="version.preserve_parent_splits" /></label><el-tag v-if="version.preserve_parent_splits">沿用父版本划分</el-tag><el-button type="primary" :disabled="busy || version.requires_annotation_review || !['draft','failed'].includes(version.state)" @click="act(freeze)">冻结版本</el-button><el-button :disabled="busy" @click="act(exportLabels)">{{ exportFormat === 'json' ? '导出 JSON' : exportFormat.endsWith('dataset') ? '导出训练数据集' : '导出标签 ZIP' }}</el-button><el-button v-if="artifactId" @click="act(download)">下载导出文件</el-button></div>
      <el-alert v-if="version.requires_annotation_review" title="AI 预标注需先在标注页面完成人工检查并确认。" type="warning" :closable="false" />
      <p>train / val / test：{{ version.split_counts.train ?? '待划分' }} / {{ version.split_counts.val ?? '待划分' }} / {{ version.split_counts.test ?? '待划分' }}</p>
      <p class="hash">Manifest SHA256：{{ version.manifest_sha256 || '冻结后生成' }}</p>
      <fieldset v-if="version.state==='frozen'"><legend>创建训练集切片版本</legend>
        <p>仅对训练图像切片；验证集、测试集保留原图和原标注。原版本保持可用。</p>
        <div class="toolbar"><label>切片边长 <input aria-label="切片边长" type="number" min="64" max="2048" v-model.number="tileSize" /></label>
          <label>重叠比例 <input aria-label="重叠比例" type="number" min="0" max="0.75" step="0.05" v-model.number="overlap" /></label>
          <label>标注最小可见比例 <input aria-label="标注最小可见比例" type="number" min="0.01" max="1" step="0.05" v-model.number="minVisibility" /></label>
          <el-button :disabled="busy || processing" @click="act(tileTrain)">创建切片版本</el-button></div>
      </fieldset>
      <JobProgress :job-id="jobId" @terminal="finished" />
      <table><thead><tr><th>图像</th><th>尺寸</th><th>来源组</th><th>split</th><th>标签数</th><th>操作</th></tr></thead><tbody><tr v-for="a in assets" :key="a.asset_id"><td>{{ a.relative_path }}</td><td>{{ a.width }} × {{ a.height }}</td><td>{{ a.source_group }}</td><td>{{ a.split || '未划分' }}</td><td>{{ a.annotations.length }}</td><td><el-button :disabled="version.state==='frozen'" @click="act(()=>remove(a))">删除图像</el-button></td></tr></tbody></table>
      <el-pagination v-model:current-page="page" :page-size="8" :total="total" layout="total, prev, pager, next" @current-change="act(loadAssets)" />
    </template>
  </main>
</template>
<script setup>
import { ref } from 'vue'
import { ElMessageBox } from 'element-plus'
import DatasetVersionPicker from '../components/medical/DatasetVersionPicker.vue'
import JobProgress from '../components/medical/JobProgress.vue'
import { listAssets, createVersion, createTrainTiles, deleteAsset } from '../api/datasets'
import { exportAnnotations } from '../api/annotations'
import { getArtifact } from '../api/inference'
import { saveBlob } from '../utils/download'
import { useTaskAction } from '../composables/useTaskAction'
const {error,busy,act}=useTaskAction(),picker=ref(null),version=ref(null),datasetId=ref(''),assets=ref([]),page=ref(1),total=ref(0),seed=ref(42),jobId=ref(''),artifactId=ref('')
const tileSize=ref(320),overlap=ref(.2),minVisibility=ref(.5),processing=ref(false)
const exportFormat=ref('yolo-txt'),artifactFormat=ref('yolo-txt')
const tileRequests=new Map()
let finishedJob = ''
async function change(data){datasetId.value=data.datasetId;version.value=data.version;seed.value=data.version?.split_config?.seed ?? 42;page.value=1;assets.value=[];if(data.version)await act(loadAssets)}
async function loadAssets(){const data=await listAssets(datasetId.value,{version_id:version.value.version_id,page:page.value,page_size:8});assets.value=data.items;total.value=data.total}
async function freeze(){jobId.value=(await createVersion(datasetId.value,{base_version_id:version.value.version_id,split_seed:seed.value,ratios:[.8,.1,.1]})).job_id}
async function tileTrain(){
  const payload={parent_version_id:version.value.version_id,tile_size:tileSize.value,overlap:overlap.value,min_visibility:minVisibility.value}
  const signature=JSON.stringify([datasetId.value,payload])
  if(!tileRequests.has(signature))tileRequests.set(signature,crypto.randomUUID())
  const job=await createTrainTiles(datasetId.value,payload,tileRequests.get(signature))
  jobId.value=job.job_id;processing.value=!['completed','failed','cancelled','interrupted'].includes(job.status)
}
async function exportLabels(){artifactId.value='';jobId.value=(await exportAnnotations(datasetId.value,version.value.version_id,exportFormat.value)).job_id}
async function finished(job) {
  processing.value = false
  if (job.status !== 'completed' || finishedJob === job.job_id) return
  // Refresh temporarily unmounts JobProgress. Handle its remount notification once.
  finishedJob = job.job_id
  artifactId.value = job.result?.artifact_id || ''
  artifactFormat.value = job.result?.format || 'yolo-txt'
  await picker.value.refresh()
}
async function download(){saveBlob(await getArtifact(artifactId.value),artifactFormat.value==='json'?'annotations.json':artifactFormat.value.endsWith('dataset')?'dataset.zip':'labels.zip')}
async function remove(asset){await ElMessageBox.confirm(`删除图像 ${asset.relative_path}？`,'删除图像');await deleteAsset(asset.asset_id);await loadAssets()}
</script>

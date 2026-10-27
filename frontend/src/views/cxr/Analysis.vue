<template>
  <main class="medical-page">
    <h1>病例分析</h1>
    <p class="muted">
      一图一条分析，不把同患者多图擅自融合。输出五项完整结果；"接近阈值"是工程提示，
      不是医学不确定性估计；未超过阈值不等于排除该征象或其他疾病。
    </p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>

    <section class="toolbar">
      <label>模型
        <select v-model="modelId">
          <option value="">请选择</option>
          <option v-for="m in models" :key="m.model_id" :value="m.model_id">
            {{ m.name }} · {{ m.backend || 'unknown' }}
          </option>
        </select>
      </label>
      <label>数据集
        <select v-model="datasetId" @change="act(loadVersions)">
          <option value="">请选择</option>
          <option v-for="d in datasets" :key="d.dataset_id" :value="d.dataset_id">{{ d.name }}</option>
        </select>
      </label>
      <label>版本
        <select v-model="versionId" @change="act(loadAssets)">
          <option value="">请选择</option>
          <option v-for="v in versions" :key="v.version_id" :value="v.version_id">
            {{ v.state }} · {{ v.asset_count }} 图
          </option>
        </select>
      </label>
      <el-button :disabled="busy || !modelId || !selected.length" @click="act(runBatch)">分析所选图（{{ selected.length }}）</el-button>
    </section>

    <section v-if="versionId">
      <h2>受控资产</h2>
      <table>
        <thead><tr><th>选择</th><th>路径（伪标识）</th><th>尺寸</th><th>split</th></tr></thead>
        <tbody>
          <tr v-for="asset in assets" :key="asset.asset_id">
            <td><input type="checkbox" :value="asset.asset_id" v-model="selected" /></td>
            <td class="hash">{{ asset.relative_path }}</td>
            <td>{{ asset.width }}×{{ asset.height }}</td>
            <td>{{ asset.split || '—' }}</td>
          </tr>
        </tbody>
      </table>
    </section>

    <fieldset>
      <legend>或上传 JPG/PNG 胸片（≤ 100 张）</legend>
      <div class="toolbar">
        <input type="file" accept=".jpg,.jpeg,.png" multiple @change="pickFiles" />
        <el-button :disabled="busy || !modelId || !files.length" @click="act(runUpload)">上传并分析（{{ files.length }}）</el-button>
      </div>
      <p class="muted">上传图不会写入任何数据集版本，也不参与训练。</p>
    </fieldset>

    <JobProgress :job-id="jobId" @terminal="act(loadBatch)" />
    <p v-if="batch" class="muted">
      批次 {{ batch.batch.batch_id.slice(0, 8) }} · {{ batch.batch.status }} ·
      完成 {{ batch.batch.completed }} / 失败 {{ batch.batch.failed }}
    </p>

    <section v-for="item in batch?.items || []" :key="item.analysis_id" class="analysis-card">
      <header>
        <strong>分析 {{ item.analysis_id.slice(0, 8) }}</strong>
        <span class="muted">{{ item.status }} · 后端 {{ item.model.backend }} · 校准 {{ item.calibration.status }}</span>
        <span class="muted">阈值来源 {{ item.calibration.threshold_source }}</span>
      </header>
      <div class="analysis-body">
        <div class="analysis-image">
          <img v-if="images[item.analysis_id]" :src="images[item.analysis_id]" alt="分析输入影像" />
          <span v-else class="muted">图像读取中…</span>
          <p v-if="(item.quality?.flags || []).length" class="error">质量提示：{{ item.quality.flags.join('、') }}</p>
        </div>
        <div>
          <table>
            <thead><tr><th>观察项</th><th>原始评分</th><th>校准评分</th><th>阈值</th><th>提示</th><th>解释</th></tr></thead>
            <tbody>
              <tr v-for="label in item.labels" :key="label.label_key">
                <td>{{ label.display_name }}</td>
                <td>{{ fmt(label.raw_score) }}</td>
                <td>{{ fmt(label.calibrated_score) }}</td>
                <td>{{ fmt(label.threshold) }}</td>
                <td :class="label.flag === 'above_threshold' ? 'warn' : 'muted'">{{ flagText(label.flag) }}</td>
                <td>
                  <el-button size="small" :disabled="busy" @click="act(() => explain(item.analysis_id, label.label_key))">
                    关注区域
                  </el-button>
                  <img v-if="explanations[item.analysis_id]?.[label.label_key]" :src="explanations[item.analysis_id][label.label_key]" alt="模型关注区域" class="thumb" />
                </td>
              </tr>
            </tbody>
          </table>
          <p class="muted">{{ item.explanation?.score_semantics }}</p>
          <el-button :disabled="busy || item.status !== 'completed'" @click="act(() => startReview(item))">发起人工复核</el-button>
          <span v-if="reviewIds[item.analysis_id]" class="muted">复核草稿：{{ reviewIds[item.analysis_id].slice(0, 8) }}</span>
        </div>
      </div>
    </section>
  </main>
</template>
<script setup>
import { onMounted, reactive, ref } from 'vue'
import { api } from '../../api/client'
import {
  LABEL_FLAG_TEXT, createAnalysisBatch, createExplanation, createReview, getBatch, listCxrDatasets,
  listCxrModels, uploadAnalysisBatch
} from '../../api/cxr'
import { listAssets, listVersions } from '../../api/datasets'
import { useTaskAction } from '../../composables/useTaskAction'
import JobProgress from '../../components/medical/JobProgress.vue'

const { error, busy, act } = useTaskAction()
const models = ref([]), datasets = ref([]), versions = ref([]), assets = ref([])
const modelId = ref(''), datasetId = ref(''), versionId = ref(''), selected = ref([]), files = ref([])
const jobId = ref(''), batch = ref(null)
const images = reactive({}), explanations = reactive({}), reviewIds = reactive({})

function fmt(value) { return Number.isFinite(value) ? Number(value).toFixed(4) : '—' }
function flagText(flag) { return LABEL_FLAG_TEXT[flag] || flag }
function pickFiles(event) { files.value = Array.from(event.target.files || []) }

async function loadVersions() {
  versions.value = []; versionId.value = ''; assets.value = []; selected.value = []
  if (!datasetId.value) return
  versions.value = (await listVersions(datasetId.value)).items
}
async function loadAssets() {
  assets.value = []; selected.value = []
  if (!versionId.value) return
  assets.value = (await listAssets(datasetId.value, { version_id: versionId.value, page_size: 100 })).items
}
async function runBatch() {
  const job = await createAnalysisBatch({ model_id: modelId.value, asset_ids: selected.value }, crypto.randomUUID())
  jobId.value = job.job_id
}
async function runUpload() {
  const job = await uploadAnalysisBatch({ modelId: modelId.value, files: files.value, idempotencyKey: crypto.randomUUID() })
  jobId.value = job.job_id
}
async function loadBatch() {
  const batchId = jobId.value ? null : batch.value?.batch.batch_id
  const target = batchId || (await resolveBatchId())
  if (!target) return
  batch.value = await getBatch(target, { page_size: 100 })
  for (const item of batch.value.items) {
    if (item.status === 'completed' && !images[item.analysis_id]) loadImage(item.analysis_id)
  }
}
async function resolveBatchId() {
  const list = await api.get('/cxr/analysis-batches', { params: { page_size: 1 } })
  return list.items?.[0]?.batch_id || null
}
async function loadImage(analysisId) {
  try {
    const response = await api.get(`/cxr/analyses/${analysisId}/image`, { responseType: 'blob' })
    images[analysisId] = URL.createObjectURL(response.data)
  } catch (e) { /* 图像读取失败不影响评分展示 */ }
}
async function explain(analysisId, labelKey) {
  const job = await createExplanation(analysisId, labelKey)
  jobId.value = job.job_id
  const analysis = await api.get(`/cxr/analyses/${analysisId}`)
  const artifactId = analysis.explanation?.artifacts?.[labelKey]?.artifact_id
  if (artifactId) {
    const response = await api.get(`/artifacts/${artifactId}`, { responseType: 'blob' })
    explanations[analysisId] = { ...(explanations[analysisId] || {}), [labelKey]: URL.createObjectURL(response.data) }
  }
}
async function startReview(item) {
  const review = await createReview({ analysis_id: item.analysis_id, items: [] })
  reviewIds[item.analysis_id] = review.review_id
}
onMounted(() => act(async () => {
  models.value = (await listCxrModels({ page_size: 50 })).items.filter(m => m.task === 'cxr_multilabel')
  datasets.value = (await listCxrDatasets({ page_size: 50 })).items
}))
</script>
<style scoped>
.analysis-card { border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; margin: 12px 0; }
.analysis-card header { display: flex; gap: 16px; align-items: baseline; margin-bottom: 8px; }
.analysis-body { display: grid; grid-template-columns: 240px 1fr; gap: 16px; }
.analysis-image img { width: 100%; border: 1px solid #e5e7eb; }
.thumb { width: 96px; margin-left: 8px; vertical-align: middle; }
.warn { color: #b45309; }
</style>

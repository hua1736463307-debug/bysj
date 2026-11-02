<template>
  <section class="isic-workspace">
    <el-card>
      <template #header><h2>皮肤镜七分类研究</h2></template>
      <el-alert title="研究用途，分析结果需人工复核。按病变分组划分数据，指标资格以服务端校验为准。" type="info" :closable="false" />
      <p><el-tag v-for="label in schema.labels || []" :key="label.key" style="margin:8px">{{ label.display_name }}</el-tag></p>
      <el-alert v-if="error" :title="error" type="error" :closable="false" />
      <el-tabs v-model="tab">
        <el-tab-pane label="数据管理" name="data">
          <el-form v-if="isAdmin" inline>
            <el-form-item label="已配置来源 ID"><el-input v-model="sourceId" /></el-form-item>
            <el-form-item label="来源集"><el-select v-model="sourceSplit"><el-option v-for="split in ['train','val','test']" :key="split" :value="split" :label="split" /></el-select></el-form-item>
            <el-form-item label="授权说明"><el-input v-model="licenseNote" /></el-form-item>
            <el-form-item><el-button :disabled="busy || !sourceId" @click="act(importData)">导入来源</el-button></el-form-item>
          </el-form>
          <el-table :data="datasets" stripe>
            <el-table-column prop="name" label="数据集" min-width="180" />
            <el-table-column label="版本" min-width="160"><template #default="{ row }">{{ row.latest_version?.state }} · {{ row.latest_version?.asset_count }} 图</template></el-table-column>
            <el-table-column label="划分统计" min-width="250"><template #default="{ row }">{{ row.latest_version?.split_counts }}</template></el-table-column>
            <el-table-column label="操作" width="200"><template #default="{ row }"><el-button :disabled="busy || row.latest_version?.state === 'frozen'" @click="act(() => freeze(row))">冻结</el-button><el-button :disabled="busy" @click="act(() => showSummary(row))">详情</el-button></template></el-table-column>
          </el-table>
          <el-pagination v-model:current-page="page" :total="total" :page-size="20" layout="total, prev, pager, next" @current-change="act(loadDatasets)" />
        </el-tab-pane>
        <el-tab-pane label="训练与评估" name="training">
          <el-form label-width="130px">
            <el-form-item label="冻结版本"><el-select v-model="versionId"><el-option v-for="dataset in datasets.filter(item => item.latest_version?.state === 'frozen')" :key="dataset.dataset_id" :label="dataset.name" :value="dataset.latest_version.version_id" /></el-select></el-form-item>
            <el-form-item label="初始骨干"><el-select v-model="baseModelId" clearable><el-option v-for="model in backbones" :key="model.model_id" :label="model.name" :value="model.model_id" /></el-select></el-form-item>
            <el-form-item label="训练轮数"><el-input-number v-model="epochs" :min="1" :max="50" /></el-form-item>
            <el-form-item label="批次大小"><el-input-number v-model="batch" :min="1" :max="128" /></el-form-item>
            <el-form-item label="学习率"><el-input-number v-model="lr" :min=".000001" :max=".1" :step=".0001" :precision="6" /></el-form-item>
            <el-form-item label="实验用途"><el-radio-group v-model="purpose"><el-radio label="smoke">工程验证</el-radio><el-radio label="formal">正式协议</el-radio></el-radio-group></el-form-item>
            <template v-if="purpose === 'formal'"><el-form-item label="协议 ID"><el-input v-model="protocolId" /></el-form-item><el-form-item label="协议 SHA256"><el-input v-model="protocolHash" /></el-form-item></template>
            <el-form-item><el-button :disabled="busy || !versionId" @click="act(preflight)">训练预检</el-button><el-button type="primary" :disabled="busy || !report?.ok || checkedRequest !== requestKey" @click="act(train)">提交训练</el-button></el-form-item>
          </el-form>
          <el-table v-if="report" :data="report.checks"><el-table-column prop="name" label="检查项" /><el-table-column prop="status" label="状态" width="90" /><el-table-column prop="detail" label="说明" /></el-table>
          <el-divider>独立评估</el-divider>
          <el-form inline>
            <el-form-item label="分类模型"><el-select v-model="evaluationModel"><el-option v-for="model in models" :key="model.model_id" :label="model.name" :value="model.model_id" /></el-select></el-form-item>
            <el-form-item label="评估集"><el-select v-model="evaluationSplit"><el-option v-for="split in ['val','calibration','test']" :key="split" :value="split" :label="split" /></el-select></el-form-item>
            <el-form-item label="评估目的"><el-input v-model="evaluationPurpose" /></el-form-item>
            <el-form-item><el-button :disabled="busy || !evaluationModel || !versionId || (evaluationSplit === 'test' && !evaluationPurpose)" @click="act(evaluate)">创建评估</el-button></el-form-item>
          </el-form>
        </el-tab-pane>
        <el-tab-pane label="模型与结果" name="results">
          <el-table :data="models"><el-table-column prop="name" label="模型" /><el-table-column label="研究资格"><template #default="{ row }"><el-tag :type="row.paper_eligibility?.paper_eligible ? 'success' : 'warning'">{{ row.paper_eligibility?.paper_eligible ? '满足论文资格门禁' : '未满足论文资格门禁' }}</el-tag><p>{{ row.paper_eligibility?.reasons?.join('、') }}</p></template></el-table-column></el-table>
          <el-table :data="evaluations"><el-table-column prop="model_name" label="模型" /><el-table-column prop="split" label="评估集" /><el-table-column prop="status" label="状态" /><el-table-column label="指标"><template #default="{ row }"><el-button @click="detail = row">查看指标与资格</el-button></template></el-table-column></el-table>
        </el-tab-pane>
      </el-tabs>
      <JobProgress :job-id="jobId" @terminal="act(refresh)" />
      <el-button :disabled="busy" @click="act(refresh)">刷新</el-button>
    </el-card>
    <el-dialog :model-value="!!detail" title="详细记录" width="min(900px, 95vw)" @close="detail = null"><pre>{{ JSON.stringify(detail, null, 2) }}</pre></el-dialog>
  </section>
</template>
<script setup>
import { computed, onMounted, ref } from 'vue'
import { api, getCurrentUser } from '../api/client'
import JobProgress from '../components/medical/JobProgress.vue'
import { useTaskAction } from '../composables/useTaskAction'
const { error, busy, act } = useTaskAction()
const isAdmin = getCurrentUser()?.role === 'admin'
const tab = ref('data'), schema = ref({}), datasets = ref([]), page = ref(1), total = ref(0)
const models = ref([]), backbones = ref([]), evaluations = ref([]), detail = ref(null), jobId = ref('')
const sourceId = ref(''), sourceSplit = ref('train'), licenseNote = ref('')
const versionId = ref(''), baseModelId = ref(''), epochs = ref(1), batch = ref(16), lr = ref(.0003)
const purpose = ref('smoke'), protocolId = ref(''), protocolHash = ref(''), report = ref(null), checkedRequest = ref('')
const evaluationModel = ref(''), evaluationSplit = ref('val'), evaluationPurpose = ref('')
const request = computed(() => ({
  dataset_version_id: versionId.value, base_model_id: baseModelId.value || null,
  params: { epochs: epochs.value, batch: batch.value, lr: lr.value, seed: 42, device: '0' },
  experiment_purpose: purpose.value,
  protocol_id: purpose.value === 'formal' ? protocolId.value : null,
  protocol_sha256: purpose.value === 'formal' ? protocolHash.value : null,
}))
const requestKey = computed(() => JSON.stringify(request.value))
const postJob = (path, body) => api.post('/isic/' + path, body, { headers: { 'Idempotency-Key': crypto.randomUUID() } })
async function loadDatasets() {
  const data = await api.get('/isic/datasets', { params: { page: page.value, page_size: 20 } })
  datasets.value = data.items
  total.value = data.total
}
async function refresh() {
  await loadDatasets()
  const [modelData, backboneData, evaluationData] = await Promise.all([
    api.get('/isic/models', { params: { page_size: 200 } }),
    api.get('/isic/models', { params: { task: 'isic_backbone', page_size: 200 } }),
    api.get('/isic/evaluations', { params: { page_size: 100 } }),
  ])
  models.value = modelData.items
  backbones.value = backboneData.items
  evaluations.value = evaluationData.items
}
async function importData() { jobId.value = (await postJob('datasets/import', { source_id: sourceId.value, split: sourceSplit.value, license_note: licenseNote.value })).job_id }
async function freeze(dataset) { jobId.value = (await postJob('datasets/' + dataset.dataset_id + '/versions', { split_seed: 42 })).job_id }
async function showSummary(dataset) { detail.value = await api.get('/isic/datasets/' + dataset.dataset_id + '/versions/' + dataset.latest_version.version_id + '/summary') }
async function preflight() { const key = requestKey.value; report.value = await api.post('/isic/training/preflight', JSON.parse(key)); checkedRequest.value = key }
async function train() { jobId.value = (await postJob('training/jobs', request.value)).job_id }
async function evaluate() { jobId.value = (await postJob('evaluations', { model_id: evaluationModel.value, dataset_version_id: versionId.value, split: evaluationSplit.value, purpose: evaluationPurpose.value || null })).job_id }
onMounted(() => act(async () => { schema.value = await api.get('/isic/schema'); await refresh() }))
</script>
<style scoped>
.isic-workspace pre { white-space: pre-wrap; overflow-wrap: anywhere; }
.isic-workspace h2 { margin: 0; font-size: 20px; }
</style>

<template>
  <main class="medical-page">
    <h1>胸片训练与校准</h1>
    <p class="muted">
      DenseNet121 + 五路 sigmoid（禁止 softmax）。masked BCE 只监督明确 positive/negative；
      温度与阈值只在 calibration 集拟合，test 集需管理员 + 预登记目的。
    </p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>

    <section class="toolbar">
      <label>已冻结版本
        <select v-model="versionId">
          <option value="">请选择</option>
          <option v-for="v in frozenVersions" :key="v.version_id" :value="v.version_id">
            {{ v.version_id.slice(0, 8) }} · {{ v.asset_count }} 图
          </option>
        </select>
      </label>
      <label>初始骨干（cxr_backbone）
        <select v-model="baseModelId">
          <option value="">请选择</option>
          <option v-for="m in backbones" :key="m.model_id" :value="m.model_id">{{ m.name }}</option>
        </select>
      </label>
      <label>类别权重
        <select v-model="classWeight">
          <option value="none">A0 无权重</option>
          <option value="pos_weight">A1 pos_weight</option>
        </select>
      </label>
      <label>种子 <input type="number" v-model.number="seed" min="0" /></label>
      <label>epoch <input type="number" v-model.number="epochs" min="1" max="20" /></label>
      <label>batch <input type="number" v-model.number="batch" min="1" max="64" /></label>
      <el-button :disabled="busy" @click="act(runPreflight)">运行预检</el-button>
      <el-button type="primary" :disabled="busy || !versionId || !baseModelId" @click="act(runTraining)">提交训练</el-button>
    </section>

    <section v-if="preflight">
      <h2>预检结果：{{ preflight.ok ? '通过' : '未通过' }}</h2>
      <table>
        <thead><tr><th>检查项</th><th>状态</th><th>说明</th></tr></thead>
        <tbody>
          <tr v-for="check in preflight.checks" :key="check.name">
            <td class="hash">{{ check.name }}</td>
            <td :class="check.status === 'fail' ? 'error' : 'muted'">{{ check.status }}</td>
            <td><pre>{{ typeof check.detail === 'string' ? check.detail : JSON.stringify(check.detail) }}</pre></td>
          </tr>
        </tbody>
      </table>
      <p class="muted">后端：{{ preflight.backend }}；dummy 只验证契约，不产生研究指标。</p>
    </section>

    <JobProgress :job-id="jobId" @terminal="act(refreshModels)" />

    <h2>胸片模型</h2>
    <table>
      <thead><tr><th>模型</th><th>任务</th><th>后端</th><th>SHA256</th><th>操作</th></tr></thead>
      <tbody>
        <tr v-for="m in models" :key="m.model_id">
          <td>{{ m.name }}<div class="muted">{{ m.model_id.slice(0, 8) }}</div></td>
          <td>{{ m.task }}</td>
          <td>{{ m.backend || '—' }}<div v-if="m.research_valid === false" class="muted">非研究有效</div></td>
          <td class="hash">{{ m.sha256 }}</td>
          <td>
            <el-button :disabled="busy || m.task !== 'cxr_multilabel'" @click="act(() => calibrate(m))">拟合校准</el-button>
            <el-button :disabled="busy || m.task !== 'cxr_multilabel'" @click="act(() => evaluateTest(m))">test 评价</el-button>
          </td>
        </tr>
      </tbody>
    </table>
    <p v-if="testPurpose" class="muted">test 目的：{{ testPurpose }}</p>

    <h2>评价记录</h2>
    <table>
      <thead><tr><th>记录</th><th>split / 动作</th><th>状态</th><th>macro AUROC</th><th>阈值</th></tr></thead>
      <tbody>
        <tr v-for="e in evaluations" :key="e.evaluation_id">
          <td class="muted">{{ e.evaluation_id.slice(0, 8) }}</td>
          <td>{{ e.split }} / {{ e.action }}</td>
          <td>{{ e.status }}</td>
          <td>{{ e.metrics?.macro_auroc?.value ?? (e.metrics?.macro_auroc?.reason || '未定义') }}</td>
          <td>{{ thresholdText(e.metrics?.thresholds) }}</td>
        </tr>
      </tbody>
    </table>
  </main>
</template>
<script setup>
import { computed, onMounted, ref } from 'vue'
import {
  createEvaluation, createTrainingJob, listCxrDatasets, listCxrModels, listEvaluations, preflightTraining
} from '../../api/cxr'
import { listVersions } from '../../api/datasets'
import { useTaskAction } from '../../composables/useTaskAction'
import JobProgress from '../../components/medical/JobProgress.vue'

const { error, busy, act } = useTaskAction()
const datasets = ref([]), versions = ref([]), versionId = ref('')
const models = ref([]), evaluations = ref([]), preflight = ref(null), jobId = ref('')
const baseModelId = ref(''), classWeight = ref('none'), seed = ref(42), epochs = ref(20), batch = ref(8)
const testPurpose = ref('')

const backbones = computed(() => models.value.filter(m => m.task === 'cxr_backbone'))
const frozenVersions = computed(() => versions.value.filter(v => v.state === 'frozen'))

function thresholdText(thresholds) {
  if (!thresholds) return '—'
  return Object.values(thresholds).map(v => Number(v).toFixed(2)).join(' / ')
}
function params() {
  return {
    epochs: epochs.value, batch: batch.value, image_size: 224, lr: 0.0001,
    weight_decay: 0.0001, patience: 5, seed: seed.value, class_weight: classWeight.value,
    num_workers: 0, amp: false, device: '0'
  }
}
async function refreshModels() {
  models.value = (await listCxrModels({ page_size: 50 })).items
  evaluations.value = (await listEvaluations({ page_size: 50 })).items
}
async function runPreflight() {
  preflight.value = await preflightTraining({
    dataset_version_id: versionId.value, base_model_id: baseModelId.value || null, params: params()
  })
}
async function runTraining() {
  const job = await createTrainingJob({
    dataset_version_id: versionId.value, base_model_id: baseModelId.value, params: params()
  }, crypto.randomUUID())
  jobId.value = job.job_id
}
async function calibrate(model) {
  const job = await createEvaluation({
    model_id: model.model_id, dataset_version_id: versionId.value, split: 'calibration',
    action: 'calibrate', purpose: '温度缩放与阈值拟合'
  }, crypto.randomUUID())
  jobId.value = job.job_id
}
async function evaluateTest(model) {
  const purpose = window.prompt('test 集读取需要预登记目的（写入评价记录）：', '冻结协议后的首次 test 评价')
  if (!purpose) return
  testPurpose.value = purpose
  const job = await createEvaluation({
    model_id: model.model_id, dataset_version_id: versionId.value, split: 'test',
    action: 'evaluate', purpose
  }, crypto.randomUUID())
  jobId.value = job.job_id
}
onMounted(() => act(async () => {
  datasets.value = (await listCxrDatasets({ page_size: 50 })).items
  if (datasets.value.length) versions.value = (await listVersions(datasets.value[0].dataset_id)).items
  await refreshModels()
}))
</script>

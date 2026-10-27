<template>
  <main class="medical-page">
    <h1>胸片数据与患者隔离</h1>
    <p class="muted">
      导入只接受管理员配置的只读来源 ID；图像复制到受控目录，患者与检查只保存伪标识。
      四集按患者 component 做确定性 hash 划分，同患者/同源重复不跨集。
    </p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>

    <fieldset v-if="isAdmin">
      <legend>从已配置来源导入</legend>
      <div class="toolbar">
        <label>来源 ID <input v-model="sourceId" placeholder="例如 chexpert-local" /></label>
        <label>CSV 文件 <input v-model="sourceCsv" placeholder="train.csv" /></label>
        <label>最多患者 <input type="number" v-model.number="maxPatients" min="1" max="10000" /></label>
        <label>名称 <input v-model="name" placeholder="可选" /></label>
        <label>许可记录 <input v-model="licenseNote" placeholder="条款记录引用（可选）" /></label>
        <el-button type="primary" :disabled="busy || !sourceId" @click="act(runImport)">提交导入任务</el-button>
      </div>
      <JobProgress :job-id="jobId" @terminal="act(refresh)" />
    </fieldset>

    <table>
      <thead><tr><th>数据集</th><th>任务</th><th>最新版本</th><th>四集</th><th>操作</th></tr></thead>
      <tbody>
        <tr v-for="item in items" :key="item.dataset_id">
          <td>{{ item.name }}<div class="muted">{{ item.dataset_code }}</div></td>
          <td>{{ item.task_type }}<div class="muted">{{ item.group_level }}</div></td>
          <td>
            <span v-if="item.latest_version">{{ item.latest_version.state }} · {{ item.latest_version.asset_count }} 图</span>
            <span v-else class="muted">尚无版本</span>
          </td>
          <td>
            <span v-if="item.latest_version">{{ splitText(item.latest_version.split_counts) }}</span>
            <span v-else class="muted">—</span>
          </td>
          <td>
            <el-button
              v-if="item.latest_version && item.latest_version.state !== 'frozen'"
              :disabled="busy" @click="act(() => runFreeze(item))">冻结四集</el-button>
            <el-button :disabled="busy || !item.latest_version" @click="act(() => loadSummary(item))">查看摘要</el-button>
          </td>
        </tr>
      </tbody>
    </table>
    <el-pagination v-model:current-page="page" :total="total" :page-size="10" layout="prev, pager, next" @current-change="act(refresh)" />

    <section v-if="summary">
      <h2>版本摘要 · {{ summary.version.version_id.slice(0, 8) }}</h2>
      <table>
        <thead><tr><th>集合</th><th>图数</th><th>患者</th><th>component</th></tr></thead>
        <tbody>
          <tr v-for="split in splits" :key="split">
            <td>{{ split }}</td>
            <td>{{ summary.counts?.[split] ?? 0 }}</td>
            <td>{{ summary.patient_counts?.[split] ?? 0 }}</td>
            <td>{{ summary.component_counts?.[split] ?? 0 }}</td>
          </tr>
        </tbody>
      </table>
      <h3>逐标签四态覆盖（train）</h3>
      <table>
        <thead><tr><th>标签</th><th>positive</th><th>negative</th><th>uncertain</th><th>unmentioned</th></tr></thead>
        <tbody>
          <tr v-for="(counts, key) in summary.label_coverage?.train || {}" :key="key">
            <td class="hash">{{ key }}</td>
            <td>{{ counts.positive }}</td><td>{{ counts.negative }}</td>
            <td>{{ counts.uncertain }}</td><td>{{ counts.unmentioned }}</td>
          </tr>
        </tbody>
      </table>
      <h3>重复候选与训练门禁</h3>
      <p class="muted">重复候选：{{ (summary.duplicate_candidates || []).length }} 组；已按 component 合并，不删除样本。</p>
      <p v-if="(summary.training_gate || []).length" class="error">
        训练门禁未通过：{{ summary.training_gate.map(g => `${g.label_key} ${g.problem}`).join('；') }}
      </p>
      <p v-else class="muted">train 每个标签均有明确正例与负例。</p>
    </section>
  </main>
</template>
<script setup>
import { ref, onMounted } from 'vue'
import { getCurrentUser } from '../../api/client'
import { freezeCxrVersion, getVersionSummary, importCxrDataset, listCxrDatasets } from '../../api/cxr'
import { useTaskAction } from '../../composables/useTaskAction'
import JobProgress from '../../components/medical/JobProgress.vue'

const { error, busy, act } = useTaskAction()
const isAdmin = getCurrentUser()?.role === 'admin'
const splits = ['train', 'val', 'calibration', 'test']
const items = ref([]), total = ref(0), page = ref(1), jobId = ref(''), summary = ref(null)
const sourceId = ref(''), sourceCsv = ref('train.csv'), maxPatients = ref(10000), name = ref(''), licenseNote = ref('')

function splitText(counts) {
  if (!counts) return '—'
  return splits.map(s => `${s}:${counts[s] ?? 0}`).join(' ')
}
async function refresh() {
  const data = await listCxrDatasets({ page: page.value, page_size: 10 })
  items.value = data.items; total.value = data.total
}
async function runImport() {
  const job = await importCxrDataset({
    source_id: sourceId.value,
    source_csv: sourceCsv.value,
    name: name.value,
    license_note: licenseNote.value || null,
    max_patients: maxPatients.value,
    dry_run: false
  }, crypto.randomUUID())
  jobId.value = job.job_id
}
async function runFreeze(item) {
  const job = await freezeCxrVersion(item.dataset_id, { split_seed: 42, ratios: [0.7, 0.1, 0.1, 0.1] }, crypto.randomUUID())
  jobId.value = job.job_id
}
async function loadSummary(item) {
  summary.value = await getVersionSummary(item.dataset_id, item.latest_version.version_id)
}
onMounted(() => act(refresh))
</script>

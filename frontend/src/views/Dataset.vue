<template>
  <section class="dataset-page">
    <el-card class="query-card">
      <el-form inline @submit.prevent="searchDatasets">
        <el-form-item label="数据集名称"><el-input v-model="query.search" placeholder="请输入数据集名称" clearable /></el-form-item>
        <el-form-item label="上传人"><el-input v-model="query.owner" placeholder="请输入上传人" clearable /></el-form-item>
        <el-form-item><el-button native-type="submit" :disabled="busy">搜索</el-button><el-button :disabled="busy" @click="resetQuery">重置</el-button></el-form-item>
        <el-form-item><el-button type="primary" @click="uploadVisible = true">上传数据集</el-button></el-form-item>
        <el-form-item><el-button :disabled="selectedVersions.length < 2" @click="openDerivative('merge', selectedVersions)">合并所选数据集</el-button></el-form-item>
      </el-form>
    </el-card>
    <el-alert v-if="error" :title="error" type="error" :closable="false" role="alert" />
    <JobProgress :job-id="jobId" @terminal="act(refresh)" />
    <div v-loading="busy" class="dataset-grid">
      <el-card v-for="dataset in datasets" :key="dataset.dataset_id">
        <template #header><div class="card-heading"><el-checkbox :model-value="selectedVersions.includes(dataset.latest_version?.version_id)" :disabled="dataset.latest_version?.state !== 'frozen'" @change="toggleSelection(dataset.latest_version.version_id, $event)" /><h2>{{ dataset.name }}</h2><el-tag>{{ dataset.latest_version?.state || '无版本' }}</el-tag></div></template>
        <el-descriptions :column="1" size="small">
          <el-descriptions-item label="图片数量">{{ dataset.latest_version?.asset_count ?? 0 }}</el-descriptions-item>
          <el-descriptions-item label="上传人">{{ dataset.owner }}</el-descriptions-item>
          <el-descriptions-item label="创建时间">{{ dataset.created_at }}</el-descriptions-item>
          <el-descriptions-item label="授权说明">{{ dataset.source?.license || '未填写' }}</el-descriptions-item>
        </el-descriptions>
        <div class="card-actions">
          <el-button :disabled="busy || !dataset.latest_version" @click="act(() => openDetails(dataset))">详情</el-button>
          <el-button :disabled="busy || !dataset.latest_version" @click="annotate(dataset)">标注</el-button>
          <el-button :disabled="busy || !dataset.latest_version" @click="manageVersions(dataset)">版本管理</el-button>
          <el-button :disabled="busy" @click="remove(dataset)">删除</el-button>
          <el-button :disabled="busy || dataset.latest_version?.state !== 'frozen'" @click="openDerivative('augment', [dataset.latest_version.version_id])">数据增强</el-button>
          <el-button :disabled="busy || dataset.latest_version?.state !== 'frozen'" @click="openDerivative('preannotate', [dataset.latest_version.version_id])">AI 预标注</el-button>
        </div>
      </el-card>
    </div>
    <el-empty v-if="!busy && !datasets.length" description="暂无匹配的数据集" />
    <el-pagination v-model:current-page="page" :page-size="12" :total="total" :disabled="busy" layout="total, prev, pager, next" @current-change="act(refresh)" />
    <el-dialog v-model="uploadVisible" title="上传数据集" width="min(640px, 95vw)" :close-on-click-modal="false">
      <el-form label-width="100px">
        <el-form-item label="名称"><el-input v-model="name" aria-label="数据集名称" /></el-form-item>
        <el-form-item label="来源 URL"><el-input v-model="sourceUrl" /></el-form-item>
        <el-form-item label="授权说明"><el-input v-model="license" /></el-form-item>
        <el-form-item label="类别映射"><el-input v-model="classMap" aria-label="类别映射" type="textarea" :rows="3" /></el-form-item>
        <el-form-item label="来源分组"><el-select v-model="groupLevel"><el-option label="原始图像" value="original_image" /><el-option label="显式图像组" value="image_cluster" /></el-select></el-form-item>
        <el-form-item v-if="groupLevel === 'image_cluster'" label="分组 JSON"><el-input v-model="sourceGroups" type="textarea" :rows="3" placeholder="完整路径到组 ID 的映射，可填入导出归档中的 source_groups.json" /></el-form-item>
        <el-form-item label="导入文件"><input aria-label="导入文件" type="file" multiple accept=".zip,.jpg,.jpeg,.png,.bmp,.txt,.yaml,.json" @change="files = Array.from($event.target.files)" /></el-form-item>
      </el-form>
      <p>选择多张图片及同名 YOLO 标签，或上传 ZIP。类别映射使用 JSON 对象，归档已有 class_map.json 时可留空。重导入派生训练集时，请选择显式图像组并填入归档中的 source_groups.json。</p>
      <el-alert v-if="error" :title="error" type="error" :closable="false" />
      <template #footer><el-button :disabled="busy" @click="uploadVisible = false">取消</el-button><el-button type="primary" :loading="busy" :disabled="!files.length" @click="act(upload)">开始导入</el-button></template>
    </el-dialog>
    <el-dialog v-model="detailVisible" :title="(selected?.name || '') + ' · 图片详情'" width="min(1100px, 95vw)">
      <el-alert v-if="error" :title="error" type="error" :closable="false" />
      <el-table :data="assets" v-loading="busy">
        <el-table-column label="图片" width="170"><template #default="{ row }"><ControlledImage :asset-id="row.asset_id" :alt="row.basename" class="asset-preview" /></template></el-table-column>
        <el-table-column prop="basename" label="文件名" min-width="170" />
        <el-table-column label="尺寸" width="140"><template #default="{ row }">{{ row.width }} × {{ row.height }}</template></el-table-column>
        <el-table-column label="大小" width="120"><template #default="{ row }">{{ (row.size / 1024).toFixed(1) }} KB</template></el-table-column>
        <el-table-column prop="source_group" label="来源组" min-width="140" />
        <el-table-column label="标注数" width="90"><template #default="{ row }">{{ row.annotations?.length || 0 }}</template></el-table-column>
      </el-table>
      <el-pagination v-model:current-page="assetPage" :total="assetTotal" :page-size="8" :disabled="busy" layout="total, prev, pager, next" @current-change="act(loadAssets)" />
    </el-dialog>
    <DerivativeDialog v-model:visible="derivativeVisible" :operation="derivativeOperation" :version-ids="derivativeVersions" @submitted="jobId = $event" />
  </section>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessageBox } from 'element-plus'
import { listDatasets, listAssets, importDataset, deleteDataset } from '../api/datasets'
import { useTaskAction } from '../composables/useTaskAction'
import JobProgress from '../components/medical/JobProgress.vue'
import ControlledImage from '../components/medical/ControlledImage.vue'
import DerivativeDialog from '../components/medical/DerivativeDialog.vue'

const router = useRouter()
const { error, busy, act } = useTaskAction()
const query = reactive({ search: '', owner: '' })
const datasets = ref([]), total = ref(0), page = ref(1)
const name = ref(''), sourceUrl = ref(''), license = ref(''), files = ref([]), jobId = ref('')
const classMap = ref('{"0":"RBC","1":"WBC","2":"Platelets"}')
const groupLevel = ref('original_image'), sourceGroups = ref('{}')
const uploadVisible = ref(false), detailVisible = ref(false), selected = ref(null)
const assets = ref([]), assetPage = ref(1), assetTotal = ref(0)
const selectedVersions = ref([]), derivativeVisible = ref(false), derivativeOperation = ref('merge'), derivativeVersions = ref([])
function toggleSelection(id, selected) {
  selectedVersions.value = selected ? [...new Set([...selectedVersions.value, id])] : selectedVersions.value.filter(value => value !== id)
}
function openDerivative(operation, versions) {
  derivativeOperation.value = operation
  derivativeVersions.value = [...versions]
  derivativeVisible.value = true
}

async function refresh() {
  const data = await listDatasets({ ...query, task: 'detect', page: page.value, page_size: 12 })
  datasets.value = data.items
  total.value = data.total
}
function searchDatasets() { page.value = 1; act(refresh) }
function resetQuery() { query.search = ''; query.owner = ''; searchDatasets() }
function annotate(dataset) {
  router.push({ path: '/dataset-annotation', query: { dataset: dataset.dataset_id, version: dataset.latest_version?.version_id } })
}
function manageVersions(dataset) {
  router.push({ path: '/train-dataset', query: { dataset: dataset.dataset_id, version: dataset.latest_version?.version_id } })
}
async function openDetails(dataset) {
  selected.value = dataset
  assetPage.value = 1
  assets.value = []
  detailVisible.value = true
  await loadAssets()
}
async function loadAssets() {
  const data = await listAssets(selected.value.dataset_id, { version_id: selected.value.latest_version.version_id, page: assetPage.value, page_size: 8 })
  assets.value = data.items
  assetTotal.value = data.total
}
async function upload() {
  const classes = classMap.value.trim() ? JSON.parse(classMap.value) : null
  if (classes !== null && (Array.isArray(classes) || typeof classes !== 'object')) throw new Error('类别映射必须是 JSON 对象')
  const form = new FormData()
  for (const file of files.value) form.append('files', file)
  form.append('name', name.value || '新数据集')
  form.append('source_url', sourceUrl.value)
  form.append('license', license.value)
  if (classes) form.append('class_map', JSON.stringify(classes))
  form.append('group_level', groupLevel.value)
  if (groupLevel.value === 'image_cluster') form.append('source_groups', sourceGroups.value)
  jobId.value = (await importDataset(form)).job_id
  uploadVisible.value = false
}
async function remove(dataset) {
  try { await ElMessageBox.confirm('删除数据集「' + dataset.name + '」？被引用的数据集会拒绝删除。', '删除数据集') }
  catch { return }
  await act(async () => {
    await deleteDataset(dataset.dataset_id)
    if (datasets.value.length === 1 && page.value > 1) page.value--
    await refresh()
  })
}
onMounted(() => act(refresh))
</script>

<style scoped>
.query-card, .el-alert { margin-bottom: 18px; }
.dataset-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 18px; margin: 20px 0; }
.card-heading { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.card-heading h2 { font-size: 17px; margin: 0; overflow-wrap: anywhere; }
.card-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.card-actions .el-button { margin: 0; }
.asset-preview { max-width: 140px; max-height: 100px; object-fit: contain; }
.el-pagination { margin-top: 18px; }
</style>

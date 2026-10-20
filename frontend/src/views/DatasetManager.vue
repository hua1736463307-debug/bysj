<template>
  <div class="dataset-page train-dataset-page">
    <div class="filter-panel">
      <el-form class="filter-form" inline @submit.prevent>
        <el-form-item label="数据集名称">
          <el-input v-model="filters.name" class="dataset-name-input" clearable placeholder="请输入数据集名称" @keyup.enter="handleSearch" />
        </el-form-item>
        <el-form-item label="生成人">
          <el-select v-model="filters.creator" class="creator-select" clearable filterable placeholder="请选择生成人">
            <el-option v-for="creator in creatorOptions" :key="creator" :label="creator" :value="creator" />
          </el-select>
        </el-form-item>
        <el-form-item class="filter-actions">
          <el-button type="primary" class="toolbar-button" @click="handleSearch">搜索</el-button>
          <el-button class="toolbar-button reset-button" @click="resetFilters">重置</el-button>
        </el-form-item>
      </el-form>
    </div>

    <div class="dataset-content" v-loading="loading">
      <div v-if="datasetList.length" class="dataset-grid">
        <article v-for="dataset in datasetList" :key="dataset.id || dataset.name" class="dataset-card">
          <div class="dataset-cover">
            <div class="cover-preview">
              <img v-if="getDatasetCover(dataset)" :src="getDatasetCover(dataset)" :alt="datasetTitle(dataset)" />
              <div v-else class="cover-placeholder">
                <el-icon><PictureRounded /></el-icon>
                <span>暂无预览</span>
              </div>
            </div>
            <h3>{{ datasetTitle(dataset) }}</h3>
          </div>
          <div class="dataset-meta">
            <div class="meta-row">
              <span>图片数：</span><strong>{{ formatImageCount(dataset) }}</strong><span>张</span>
              <span class="meta-gap">大小：</span><strong>{{ formatSizeMB(dataset) }} M</strong>
            </div>
            <div class="meta-row">
              <span>生成时间：</span><strong>{{ formatDate(dataset.created_at || dataset.createdAt || dataset.generated_at) }}</strong>
              <span class="meta-gap">生成人：</span><strong>{{ datasetCreator(dataset) }}</strong>
            </div>
          </div>
          <div class="dataset-actions">
            <el-button plain @click="openDatasetDetail(dataset)">详情</el-button>
            <el-button v-if="dataset.can_delete" type="danger" plain @click="deleteDataset(dataset)">删除</el-button>
          </div>
        </article>
      </div>
      <el-empty v-else class="empty-state" description="暂无训练数据集" />
      <div v-if="total > pageSize" class="pagination-wrap">
        <el-pagination v-model:current-page="currentPage" v-model:page-size="pageSize" :total="total" :page-sizes="[12, 20, 40, 80]" layout="total, sizes, prev, pager, next, jumper" @current-change="handlePageChange" @size-change="handlePageSizeChange" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { PictureRounded } from '@element-plus/icons-vue'
import { authorizedFetch, deleteTrainingDataset, getTrainingDatasets } from '../utils/api'

const loading = ref(false)
const router = useRouter()
const datasetList = ref([])
const total = ref(0)
const creatorOptions = ref([])
const currentPage = ref(1)
const pageSize = ref(12)
const datasetCoverObjectUrls = new Set()
const loadingDatasetCovers = new WeakSet()
let datasetCoverGeneration = 0

const filters = ref({
  name: '',
  creator: ''
})

const datasetTitle = (dataset) => {
  return String(dataset.name || dataset.id || dataset.original_filename || '未命名数据集')
}

const datasetImageCount = (dataset) => {
  const value = dataset.image_count ?? dataset.imageCount ?? dataset.images ?? 0
  return Array.isArray(value) ? value.length : Number(value) || 0
}

const datasetCreator = (dataset) => {
  return dataset.creator || dataset.uploader || dataset.upload_user || dataset.created_by || dataset.owner || '管理员'
}

const getAnnotationModelBinding = (dataset) => {
  const nested = dataset?.annotation_model
  if (nested && typeof nested === 'object') {
    const path = String(nested.path || nested.model_path || '').trim()
    if (path) {
      return {
        path,
        name: String(nested.name || nested.model_name || path).trim() || path,
        geometry: String(nested.geometry || dataset.annotation_geometry || '').trim().toLowerCase()
      }
    }
  }
  const path = String(dataset?.annotation_model_path || '').trim()
  if (!path) return null
  return {
    path,
    name: String(dataset.annotation_model_name || path).trim() || path,
    geometry: String(dataset.annotation_geometry || '').trim().toLowerCase()
  }
}

const getDatasetCover = (dataset) => {
  return dataset?._coverPreviewUrl || ''
}

const clearDatasetCoverUrls = () => {
  datasetCoverGeneration += 1
  datasetCoverObjectUrls.forEach(url => URL.revokeObjectURL(url))
  datasetCoverObjectUrls.clear()
}

const hydrateDatasetCovers = async (datasets) => {
  const generation = datasetCoverGeneration
  const queue = datasets.filter(dataset => {
    if (!dataset?.cover_url || dataset._coverPreviewUrl || loadingDatasetCovers.has(dataset)) return false
    loadingDatasetCovers.add(dataset)
    return true
  })
  const workerCount = Math.min(6, queue.length)

  await Promise.all(Array.from({ length: workerCount }, async () => {
    while (queue.length) {
      const dataset = queue.shift()
      try {
        const response = await authorizedFetch(dataset.cover_url)
        if (!response.ok) continue
        const objectUrl = URL.createObjectURL(await response.blob())
        if (generation !== datasetCoverGeneration) {
          URL.revokeObjectURL(objectUrl)
          continue
        }
        datasetCoverObjectUrls.add(objectUrl)
        dataset._coverPreviewUrl = objectUrl
      } catch (error) {
        console.warn('训练数据集预览图加载失败:', datasetTitle(dataset), error)
      } finally {
        loadingDatasetCovers.delete(dataset)
      }
    }
  }))
}

watch(datasetList, datasets => {
  void hydrateDatasetCovers(datasets)
})

const handleSearch = () => {
  currentPage.value = 1
  void loadDatasets()
}

const resetFilters = () => {
  filters.value.name = ''
  filters.value.creator = ''
  currentPage.value = 1
  void loadDatasets()
}

const handlePageChange = () => {
  window.scrollTo({ top: 0, behavior: 'smooth' })
  void loadDatasets()
}

const handlePageSizeChange = () => {
  currentPage.value = 1
  void loadDatasets()
}

const loadDatasets = async () => {
  loading.value = true
  try {
    const data = await getTrainingDatasets({
      page: currentPage.value,
      page_size: pageSize.value,
      name: filters.value.name.trim(),
      creator: filters.value.creator
    })
    clearDatasetCoverUrls()
    datasetList.value = Array.isArray(data.datasets) ? data.datasets : []
    total.value = Number(data.total || data.total_datasets || 0)
    creatorOptions.value = Array.isArray(data.creator_options)
      ? data.creator_options
      : creatorOptions.value
  } catch (error) {
    console.error('加载训练数据集列表失败:', error)
    ElMessage.error('加载训练数据集列表失败')
  } finally {
    loading.value = false
  }
}

const deleteDataset = async (dataset) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除训练数据集 "${datasetTitle(dataset)}" 吗？此操作不可恢复。`,
      '删除确认',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
  } catch {
    return
  }

  try {
    await deleteTrainingDataset(dataset.id)
    await loadDatasets()
    ElMessage.success('训练数据集已删除')
  } catch (error) {
    console.error('删除训练数据集失败:', error)
    ElMessage.error('删除训练数据集失败')
  }
}

const openDatasetDetail = (dataset) => {
  const datasetId = String(dataset?.id || '').trim()
  if (!datasetId) {
    ElMessage.warning('训练数据集缺少有效 ID')
    return
  }

  const binding = getAnnotationModelBinding(dataset)
  router.push({
    path: '/annotation',
    query: {
      datasetId,
      datasetName: datasetTitle(dataset),
      datasetRole: 'training',
      ...(binding?.path ? {
        model: binding.path,
        modelName: binding.name,
        annotationGeometry: binding.geometry || undefined
      } : {})
    }
  })
}

const formatImageCount = (dataset) => {
  return datasetImageCount(dataset).toLocaleString('zh-CN')
}

const formatSizeMB = (dataset) => {
  const directSizeMB = dataset.size_mb ?? dataset.sizeMB
  if (directSizeMB !== undefined && directSizeMB !== null) {
    return Number(directSizeMB).toFixed(1)
  }

  const directSizeGB = dataset.size_gb ?? dataset.sizeGB
  if (directSizeGB !== undefined && directSizeGB !== null) {
    return (Number(directSizeGB) * 1024).toFixed(1)
  }

  const bytes = Number(dataset.size) || 0
  return bytes ? (bytes / 1024 / 1024).toFixed(1) : '0.0'
}

const formatDate = (date) => {
  if (!date) return '-'
  if (typeof date === 'string' && /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(date)) {
    return date
  }
  const parsed = new Date(date)
  if (Number.isNaN(parsed.getTime())) return '-'
  const pad = (value) => String(value).padStart(2, '0')
  return `${parsed.getFullYear()}-${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())} ${pad(parsed.getHours())}:${pad(parsed.getMinutes())}:${pad(parsed.getSeconds())}`
}

onMounted(loadDatasets)
onBeforeUnmount(clearDatasetCoverUrls)
</script>

<style scoped>
.dataset-page {
  min-height: 100%;
  padding: 20px !important;
  background: #f5f7fb !important;
  color: #1f2937;
}

.filter-panel {
  margin: 0 0 18px !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.filter-form {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}

.filter-form :deep(.el-form-item) {
  margin: 0;
}

.filter-form :deep(.el-form-item__label) {
  height: 40px;
  align-items: center;
  color: #1f2937 !important;
  font-size: 18px;
  font-weight: 700 !important;
  line-height: 40px;
}

.dataset-name-input {
  width: min(345px, 32vw);
}

.creator-select {
  width: 185px;
}

.filter-form :deep(.el-input__wrapper),
.filter-form :deep(.el-select__wrapper) {
  min-height: 40px;
  border-radius: 4px !important;
  background: #fff;
  box-shadow: 0 0 0 1px #dfe5ee inset !important;
}

.filter-form :deep(.el-input__inner),
.filter-form :deep(.el-select__placeholder),
.filter-form :deep(.el-select__selected-item) {
  font-size: 15px;
}

.toolbar-button {
  min-width: 100px;
  height: 36px;
  border-radius: 4px !important;
  font-size: 14px;
  font-weight: 400;
}

.reset-button {
  margin-left: 10px !important;
}

.dataset-content {
  min-height: 420px;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.dataset-grid {
  display: grid !important;
  grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
  gap: 18px !important;
}

.dataset-card {
  overflow: hidden !important;
  border: 0 !important;
  border-radius: 8px !important;
  background: #fff !important;
  box-shadow: 0 6px 18px rgba(23, 35, 54, 0.05) !important;
}

.dataset-cover {
  position: relative;
  height: 99px;
  display: grid;
  grid-template-columns: 150px minmax(0, 1fr);
  align-items: center;
  gap: 18px;
  padding: 17px 20px;
  border-bottom: 1px solid #edf1f6;
  background: #fff !important;
}

.cover-preview {
  width: 94px;
  height: 70px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-left: 1px solid #111827;
  border-right: 1px solid #111827;
  border-radius: 0 !important;
  background: #f4f6fa !important;
}

.cover-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.cover-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: #c8ced8;
  font-size: 11px;
  background: #f4f6fa !important;
}

.cover-placeholder .el-icon {
  font-size: 20px;
}

.dataset-cover h3 {
  margin: 0 24px 0 0;
  color: #132238;
  font-size: clamp(18px, 1.12vw, 22px);
  font-weight: 800;
  text-align: right;
  line-height: 1.3;
  word-break: break-word;
}

.dataset-meta {
  min-height: 110px;
  padding: 0 20px 12px;
  overflow: hidden;
}

.meta-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
  color: #687384;
  font-size: clamp(14px, 0.9vw, 16px);
  line-height: 1.4;
}

.meta-row strong {
  color: #253244;
  font-weight: 500;
}

.meta-gap {
  margin-left: 18px;
}

.dataset-card .dataset-actions {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 12px 20px;
  border-top: 1px solid #edf1f6;
}

.dataset-card .dataset-actions :deep(.el-button) {
  min-width: 96px;
  height: 36px;
  border-radius: 4px !important;
  font-size: 14px;
  letter-spacing: 0.12em;
}

.dataset-card .dataset-actions :deep(.el-button:last-child) {
  margin-left: auto;
}

.empty-state {
  min-height: 360px;
  border-radius: 8px;
  background: #fff;
}

.pagination-wrap {
  display: flex;
  justify-content: flex-end;
  padding: 22px 0 0;
}

@media (max-width: 1500px) {
  .dataset-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  }

  .dataset-name-input {
    width: 360px;
  }
}

@media (max-width: 980px) {
  .dataset-grid {
    grid-template-columns: 1fr !important;
  }
}

@media (max-width: 768px) {
  .dataset-page {
    padding: 16px !important;
  }

  .filter-form {
    display: block;
  }

  .filter-form :deep(.el-form-item) {
    display: block;
    margin-bottom: 14px;
  }

  .dataset-name-input,
  .creator-select,
  .toolbar-button {
    width: 100%;
  }

  .reset-button {
    margin: 10px 0 0 !important;
  }

  .dataset-cover {
    height: auto;
    grid-template-columns: 1fr;
    text-align: left;
  }

  .cover-preview {
    width: 128px;
  }

  .dataset-cover h3 {
    margin-right: 28px;
    text-align: left;
    font-size: 20px;
  }

  .dataset-card .dataset-actions {
    gap: 10px;
    flex-wrap: wrap;
  }

  .dataset-card .dataset-actions :deep(.el-button),
  .dataset-card .dataset-actions :deep(.el-button:last-child) {
    flex: 1;
    min-width: 100px;
    margin-left: 0;
  }
}
</style>

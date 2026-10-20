<template>
  <div class="app-container model-hub-page">
    <div class="page-header">
      <div class="page-header__copy">
        <h1>模型中心 (Model Hub)</h1>
        <p>统一管理 PyTorch 权重与 ONNX 推理模型。</p>
      </div>
      <el-button v-if="modelUploadEnabled && isAdmin" type="primary" :icon="Upload" @click="openUploadDialog">
        上传模型
      </el-button>
    </div>

    <div v-if="!loading && models.length > 0" class="model-toolbar">
      <div class="format-tabs" role="tablist" aria-label="模型筛选">
        <button
          v-for="option in formatOptions"
          :key="option.value"
          type="button"
          class="format-tab"
          :class="{ 'is-active': formatFilter === option.value }"
          role="tab"
          :aria-selected="formatFilter === option.value"
          @click="formatFilter = option.value"
        >
          <span>{{ option.label }}</span>
          <span class="format-tab__count">{{ option.count }}</span>
        </button>
      </div>
      <span class="model-toolbar__summary">当前显示 {{ filteredModels.length }} 个模型</span>
    </div>

    <div v-if="loading" class="page-state">
      <el-icon class="is-loading"><Loading /></el-icon>
      <span>正在加载模型...</span>
    </div>

    <div v-else-if="filteredModels.length === 0" class="page-state empty-state">
      <el-icon><Box /></el-icon>
      <strong>{{ models.length === 0 ? '模型仓库为空' : '没有匹配的模型' }}</strong>
      <span>{{ models.length === 0 ? '当前没有已注册模型。' : '请切换模型筛选条件。' }}</span>
    </div>

    <div v-else class="model-grid">
      <div
        v-for="model in filteredModels"
        :key="model.id || model.path || model.name"
        class="model-card"
        :class="[getModelAccentClass(model), { 'is-protected': isProtectedModel(model) }]"
      >
        <div class="model-card__header">
          <div class="model-format-mark" :class="`is-${getModelFormat(model)}`">
            {{ getModelFormat(model).toUpperCase() }}
          </div>
          <div class="tag-list">
            <span class="tag" :class="getModelTagClass(model.type)">{{ getModelTypeName(model.type) }}</span>
            <span v-if="model.is_latest" class="tag tag-latest">最新</span>
            <span v-if="isProtectedModel(model)" class="tag tag-protected"><el-icon><Warning /></el-icon>受保护</span>
          </div>
        </div>

        <div class="model-card__body">
          <div class="model-name">{{ model.display_name || model.name }}</div>
          <div class="model-desc">{{ model.type === 'pretrained' ? '系统预置的基础训练权重' : (model.description || '已注册模型文件') }}</div>

          <div class="model-facts">
            <div class="model-fact">
              <span class="model-fact__label">文件格式</span>
              <strong>{{ getModelFormat(model).toUpperCase() }}</strong>
            </div>
            <div class="model-fact">
              <span class="model-fact__label">文件大小</span>
              <strong>{{ formatBytes(model.size) }}</strong>
            </div>
            <div class="model-fact">
              <span class="model-fact__label">训练轮次</span>
              <strong :title="formatTrainingEpochs(model.epochs)">{{ formatTrainingEpochs(model.epochs) }}</strong>
            </div>
            <div class="model-fact">
              <span class="model-fact__label">更新时间</span>
              <strong>{{ formatDateShort(model.modified_time) }}</strong>
            </div>
          </div>

          <div class="model-status" :class="getModelStatusClass(model)">
            <span class="model-status__dot"></span>
            <span>{{ getModelStatusText(model) }}</span>
            <span v-if="model.weight_type" class="best-mark"><el-icon><Medal /></el-icon>{{ model.weight_type }}</span>
          </div>
        </div>

        <div
          class="model-card__footer"
          :class="{ 'has-convert': canConvertToOnnx(model) }"
        >
          <el-button
            v-if="canConvertToOnnx(model)"
            class="convert-button"
            type="primary"
            :loading="isModelConverting(model)"
            @click="convertToOnnx(model)"
          >
            {{ isModelConverting(model) ? '转换中...' : '转换 ONNX' }}
          </el-button>
          <el-button
            v-if="model.type === 'trained'"
            plain
            :icon="Download"
            :loading="downloadingModelId === model.id"
            title="下载训练模型"
            @click="downloadModel(model)"
          >
            下载
          </el-button>
          <el-button
            v-if="isAdmin"
            class="model-delete-button"
            plain
            :icon="Delete"
            :disabled="isProtectedModel(model)"
            :title="isProtectedModel(model) ? '基础模型受保护，不允许删除' : '删除模型'"
            @click="confirmDelete(model)"
          >
            删除
          </el-button>
        </div>
      </div>
    </div>

    <el-dialog
      v-model="showUploadDialog"
      width="720px"
      append-to-body
      class="model-upload-dialog"
      :show-close="false"
      :close-on-click-modal="!uploading"
      @closed="resetUploadForm"
    >
      <div class="upload-dialog__content">
        <h2>上传模型</h2>
        <p class="upload-dialog__desc">上传您训练好的YOLO模型文件（.pt格式）</p>

        <div class="upload-dialog__label">选择模型文件 (.pt) *</div>

        <div class="upload-file-box">
          <input
            ref="modelFileInput"
            class="upload-file-input"
            type="file"
            accept=".pt"
            @change="handleModelFileChange"
          >
          <button type="button" class="file-select-btn" :disabled="uploading" @click="triggerFileSelect">
            选择文件
          </button>
          <span class="file-name-text">{{ selectedFileName || '未选择文件' }}</span>
        </div>

        <div class="upload-dialog__tip">
          仅支持PyTorch模型文件（.pt格式），将保持原始文件名
        </div>

        <div v-if="uploading" class="upload-progress">
          <el-icon class="is-loading"><Loading /></el-icon>
          正在上传...
        </div>

        <div class="upload-dialog__footer">
          <el-button :disabled="uploading" @click="closeUploadDialog">取消</el-button>
          <el-button type="primary" :loading="uploading" :disabled="!selectedModelFile" @click="submitModelUpload">
            {{ uploading ? '上传中...' : '确认上传' }}
          </el-button>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Box, Delete, Download, Loading, Medal, Upload, Warning } from '@element-plus/icons-vue'
import { authorizedFetch, convertModelToOnnx, getJob, getStoredUser } from '../utils/api'

const loading = ref(true)
const showUploadDialog = ref(false)
const uploading = ref(false)
const models = ref([])
const downloadingModelId = ref('')
const formatFilter = ref('all')
const convertingModelIds = ref(new Set())
const selectedModelFile = ref(null)
const selectedFileName = ref('')
const modelFileInput = ref(null)
const modelUploadEnabled = true
const isAdmin = computed(() => getStoredUser()?.role === 'admin')

const fetchModels = async () => {
  try {
    loading.value = true
    const response = await fetch('/api/models')
    if (!response.ok) throw new Error('无法获取模型列表')
    const data = await response.json()
    models.value = data.models || []
  } catch (error) {
    console.error('获取模型失败:', error)
    ElMessage.error('获取模型列表失败！')
  } finally {
    loading.value = false
  }
}

const isProtectedModel = (model) => String(model?.type || '').toLowerCase() === 'pretrained'

const getModelTypeName = (type) => ({
  pretrained: '预训练',
  trained: '训练模型',
  uploaded: '自定义'
})[type] || type

const getModelTagClass = (type) => ({
  pretrained: 'tag-pretrained',
  trained: 'tag-trained',
  uploaded: 'tag-uploaded'
})[type] || 'tag-uploaded'

const getModelFormat = (model) => String(model?.format || 'pt').toLowerCase()

const filteredModels = computed(() => {
  if (formatFilter.value === 'all') return models.value
  if (['pretrained', 'trained'].includes(formatFilter.value)) {
    return models.value.filter(model => model?.type === formatFilter.value)
  }
  return models.value.filter(model => getModelFormat(model) === formatFilter.value)
})

const formatOptions = computed(() => [
  { value: 'all', label: '全部模型', count: models.value.length },
  { value: 'pt', label: 'PyTorch', count: models.value.filter(model => getModelFormat(model) === 'pt').length },
  { value: 'onnx', label: 'ONNX', count: models.value.filter(model => getModelFormat(model) === 'onnx').length },
  { value: 'pretrained', label: '预训练模型', count: models.value.filter(model => model?.type === 'pretrained').length },
  { value: 'trained', label: '训练模型', count: models.value.filter(model => model?.type === 'trained').length }
])

const hasOnnxSibling = (model) => getModelFormat(model) === 'pt' && Boolean(model?.onnx_available)

const canConvertToOnnx = (model) => (
  isAdmin.value
  && model?.type === 'trained'
  && getModelFormat(model) === 'pt'
  && !hasOnnxSibling(model)
)

const isModelConverting = (model) => Boolean(model?.id && convertingModelIds.value.has(model.id))

const setModelConverting = (modelId, converting) => {
  const next = new Set(convertingModelIds.value)
  if (converting) next.add(modelId)
  else next.delete(modelId)
  convertingModelIds.value = next
}

const getModelAccentClass = (model) => `model-card--${getModelFormat(model)}`

const getModelStatusClass = (model) => {
  if (getModelFormat(model) === 'onnx') return 'is-onnx'
  if (hasOnnxSibling(model)) return 'is-ready'
  return 'is-source'
}

const getModelStatusText = (model) => {
  if (getModelFormat(model) === 'onnx') return 'ONNX 推理模型'
  if (hasOnnxSibling(model)) return '已生成 ONNX 版本'
  if (model?.type === 'trained') return '可转换为 ONNX'
  return 'PyTorch 训练权重'
}

const sleep = (milliseconds) => new Promise(resolve => window.setTimeout(resolve, milliseconds))

const waitForModelConversion = async (taskId) => {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const task = await getJob(taskId)
    if (['completed', 'failed', 'cancelled'].includes(task?.status)) return task
    if (task?.status === 'interrupted') throw new Error(task.message || '服务重启导致转换中断，请重试')
    await sleep(Math.min(3000, 800 + attempt * 50))
  }
  throw new Error('ONNX 转换等待超时，请到任务状态中查看结果')
}

const convertToOnnx = async (model) => {
  if (!canConvertToOnnx(model) || isModelConverting(model)) return
  setModelConverting(model.id, true)
  try {
    const response = await convertModelToOnnx(model.id)
    if (response?.already_exists) {
      ElMessage.info('该模型已存在 ONNX 文件')
    } else if (response?.task_id) {
      const task = await waitForModelConversion(response.task_id)
      if (task?.status !== 'completed') throw new Error(task?.error_detail || task?.message || 'ONNX 转换失败')
      ElMessage.success('ONNX 模型转换完成')
    }
    await fetchModels()
  } catch (error) {
    console.error('转换 ONNX 失败:', error)
    ElMessage.error(`转换失败: ${error?.response?.data?.detail || error?.message || '未知错误'}`)
  } finally {
    setModelConverting(model.id, false)
  }
}

const openUploadDialog = () => {
  resetUploadForm()
  showUploadDialog.value = true
}

const triggerFileSelect = () => modelFileInput.value?.click()

const clearNativeFileValue = () => {
  if (modelFileInput.value) modelFileInput.value.value = ''
}

const resetUploadForm = () => {
  selectedModelFile.value = null
  selectedFileName.value = ''
  clearNativeFileValue()
}

const handleModelFileChange = (event) => {
  const file = event.target.files?.[0]
  if (!file) {
    resetUploadForm()
    return
  }
  if (!file.name.toLowerCase().endsWith('.pt')) {
    ElMessage.error('请选择 .pt 格式的模型文件')
    resetUploadForm()
    return
  }
  selectedModelFile.value = file
  selectedFileName.value = file.name
}

const closeUploadDialog = () => {
  if (uploading.value) return
  showUploadDialog.value = false
}

const submitModelUpload = async () => {
  if (!selectedModelFile.value) {
    ElMessage.warning('请先选择模型文件')
    return
  }

  const formData = new FormData()
  formData.append('file', selectedModelFile.value)
  formData.append('name', selectedModelFile.value.name)

  try {
    uploading.value = true
    const response = await authorizedFetch('/api/models/upload', {
      method: 'POST',
      body: formData
    })
    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || '上传失败')
    }
    const result = await response.json()
    ElMessage.success(result.overwritten ? `模型文件已更新：${result.filename}` : `模型上传成功：${result.filename}`)
    showUploadDialog.value = false
    await fetchModels()
  } catch (error) {
    console.error('上传模型失败:', error)
    ElMessage.error(`上传失败: ${error.message}`)
  } finally {
    uploading.value = false
  }
}

const downloadModel = async (model) => {
  if (!model?.id) {
    ElMessage.error('模型信息缺少 ID，无法下载')
    return
  }

  try {
    downloadingModelId.value = model.id
    const response = await authorizedFetch(`/api/models/${encodeURIComponent(model.id)}/download`)
    if (!response.ok) {
      let detail = '下载失败'
      try {
        const errorData = await response.json()
        detail = errorData.detail || detail
      } catch {
        // 保留默认错误提示。
      }
      throw new Error(detail)
    }

    const blob = await response.blob()
    if (!blob.size) throw new Error('下载文件为空')

    const weightName = String(model.name || 'model.pt').split('/').pop() || 'model.pt'
    const prefix = String(model.experiment_name || '').trim()
    const filename = `${prefix ? `${prefix}_` : ''}${weightName}`.replace(/[\\/:*?"<>|]+/g, '_')
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename.toLowerCase().endsWith('.pt') ? filename : `${filename}.pt`
    link.style.display = 'none'
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  } catch (error) {
    console.error('下载模型失败:', error)
    ElMessage.error(`下载失败: ${error.message}`)
  } finally {
    downloadingModelId.value = ''
  }
}

const confirmDelete = async (model) => {
  if (isProtectedModel(model)) {
    ElMessage.warning('基础预训练模型受保护，不允许删除！')
    return
  }

  const modelName = model.display_name || model.name
  try {
    await ElMessageBox.confirm(
      `确定要删除模型 “${modelName}” 吗？此操作不可恢复。`,
      '删除模型',
      { confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning' }
    )
  } catch {
    return
  }

  try {
    const modelRef = model.id || model.path || model.name
    const response = await authorizedFetch(`/api/models/${encodeURIComponent(modelRef)}`, {
      method: 'DELETE'
    })
    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || '删除失败')
    }
    ElMessage.success('模型删除成功！')
    await fetchModels()
  } catch (error) {
    console.error('删除模型失败:', error)
    ElMessage.error(`删除失败: ${error.message}`)
  }
}

const formatTrainingEpochs = (epochs) => {
  if (!epochs) return '未记录'
  const value = String(epochs).trim()
  if (!value) return '未记录'
  if (value.includes('轮')) return value.replace(/^\(|\)$/g, '')
  return `${value} 轮`
}

const formatBytes = (bytes, decimals = 2) => {
  const value = Number(bytes) || 0
  if (value === 0) return '0 Bytes'
  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
  const index = Math.floor(Math.log(value) / Math.log(k))
  return `${parseFloat((value / Math.pow(k, index)).toFixed(dm))} ${sizes[index]}`
}

const formatDate = (timestamp) => {
  if (!timestamp) return 'N/A'
  const date = new Date(Number(timestamp) * 1000)
  if (Number.isNaN(date.getTime())) return 'N/A'
  return date.toLocaleString('zh-CN', { hour12: false })
}

const formatDateShort = (timestamp) => {
  if (!timestamp) return '未知'
  const date = new Date(Number(timestamp) * 1000)
  if (Number.isNaN(date.getTime())) return '未知'
  return date.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' })
}

onMounted(fetchModels)
</script>

<style scoped>
.model-hub-page {
  min-height: calc(100vh - 84px);
  background:
    radial-gradient(circle at 100% 0, rgba(59, 130, 246, 0.08), transparent 34%),
    #f6f8fc !important;
}

.model-hub-page .page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin: 0 0 18px !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.model-hub-page .page-header h1 {
  margin: 0 0 8px !important;
  font-size: clamp(24px, 2.2vw, 30px) !important;
  line-height: 1.1 !important;
  font-weight: 700 !important;
  color: #303133;
}

.model-hub-page .page-header p {
  margin: 0 !important;
  font-size: 14px !important;
  color: #909399 !important;
}

.model-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 20px;
  padding: 8px;
  border: 1px solid #e7ebf3;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.88);
  box-shadow: 0 8px 28px rgba(15, 23, 42, 0.04);
}

.format-tabs {
  display: flex;
  align-items: center;
  gap: 6px;
}

.format-tab {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 36px;
  padding: 0 14px;
  border: 0;
  border-radius: 10px;
  background: transparent;
  color: #64748b;
  font: inherit;
  cursor: pointer;
  transition: background 0.18s ease, color 0.18s ease, box-shadow 0.18s ease;
}

.format-tab:hover {
  color: #2563eb;
  background: #f1f5ff;
}

.format-tab.is-active {
  color: #fff;
  background: linear-gradient(135deg, #2563eb, #3b82f6);
  box-shadow: 0 6px 14px rgba(37, 99, 235, 0.24);
}

.format-tab__count {
  min-width: 20px;
  padding: 2px 6px;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.16);
  font-size: 12px;
  line-height: 16px;
  text-align: center;
}

.format-tab.is-active .format-tab__count {
  background: rgba(255, 255, 255, 0.2);
}

.model-toolbar__summary {
  padding-right: 10px;
  font-size: 13px;
  color: #94a3b8;
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
  gap: clamp(16px, 1.8vw, 24px);
  align-items: stretch;
}

.model-hub-page .model-card {
  position: relative;
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 420px;
  overflow: hidden;
  background: #fff !important;
  border: 1px solid #e6eaf2 !important;
  border-radius: 18px !important;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06) !important;
  transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.model-hub-page .model-card::before {
  position: absolute;
  inset: 0 0 auto;
  height: 4px;
  background: linear-gradient(90deg, #2563eb, #60a5fa);
  content: '';
}

.model-hub-page .model-card--onnx::before {
  background: linear-gradient(90deg, #7c3aed, #c084fc);
}

.model-hub-page .model-card:hover {
  transform: translateY(-4px);
  border-color: #bfdbfe !important;
  box-shadow: 0 18px 42px rgba(30, 64, 175, 0.12) !important;
}

.model-hub-page .model-card.is-protected {
  border-color: #fed7aa !important;
}

.model-hub-page .page-header :deep(.el-button--primary),
.model-hub-page .model-card__footer :deep(.el-button) {
  height: 35.3333px;
  padding: 10px 20px;
  font-size: 14px;
  font-weight: 400;
}

.model-hub-page .page-header :deep(.el-button--primary) {
  min-width: 120.2292px;
  background: #1890ff !important;
  border-color: #1890ff !important;
}

.model-hub-page .model-card__footer :deep(.el-button) {
  min-width: 0;
  margin-left: 0;
}

.model-hub-page .model-card__footer :deep(.el-button.is-disabled) {
  color: #c0c4cc !important;
  border-color: #e6ebf5 !important;
}

.model-card__header,
.model-card__footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 20px;
}

.model-card__header {
  min-height: 72px;
  align-items: flex-start;
}

.model-card__body {
  flex: 1;
  min-height: 0;
  padding: 0 20px 20px;
}

.model-card__footer {
  flex-wrap: wrap;
  min-height: 68px;
  justify-content: flex-end;
  gap: 8px;
  border-top: 1px solid #eef2f7;
  background: #fbfcfe;
}

.model-card__footer.has-convert {
  display: grid;
  grid-template-columns: minmax(120px, 1fr) auto auto;
}

.model-card__footer .model-delete-button {
  color: #64748b;
}

.model-card__footer .convert-button {
  width: 100%;
  margin-right: 0;
}

.model-format-mark {
  display: grid;
  width: 48px;
  height: 48px;
  flex: 0 0 48px;
  place-items: center;
  border-radius: 14px;
  background: linear-gradient(135deg, #dbeafe, #eff6ff);
  color: #2563eb;
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 0.04em;
}

.model-format-mark.is-onnx {
  background: linear-gradient(135deg, #ede9fe, #faf5ff);
  color: #7c3aed;
}

.tag-list {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  line-height: 1;
}

.tag-pretrained {
  color: #409eff;
  background: #ecf5ff;
}

.tag-trained {
  color: #67c23a;
  background: #f0f9eb;
}

.tag-uploaded {
  color: #8b5cf6;
  background: #f4edff;
}

.tag-latest {
  color: #e83e8c;
  background: #fdf0f6;
}

.tag-protected {
  color: #f59a23;
  background: #fff6e8;
}

.model-name {
  display: -webkit-box;
  height: 51px;
  min-height: 51px;
  margin: 2px 0 8px;
  font-size: 18px;
  line-height: 1.4;
  font-weight: 750;
  color: #172033;
  overflow: hidden;
  overflow-wrap: anywhere;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.model-desc {
  height: 40px;
  min-height: 40px;
  margin-bottom: 18px;
  font-size: 13px;
  line-height: 20px;
  color: #7b879b;
  overflow: hidden;
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.model-facts {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}

.model-fact {
  min-width: 0;
  padding: 11px 12px;
  border: 1px solid #edf0f5;
  border-radius: 12px;
  background: #f8fafc;
}

.model-fact__label {
  display: block;
  margin-bottom: 5px;
  font-size: 11px;
  color: #94a3b8;
}

.model-fact strong {
  display: block;
  overflow: hidden;
  color: #334155;
  font-size: 13px;
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-status {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 36px;
  padding: 8px 12px;
  border-radius: 11px;
  background: #eff6ff;
  color: #2563eb;
  font-size: 12px;
  font-weight: 600;
}

.model-status.is-onnx {
  background: #f5f3ff;
  color: #7c3aed;
}

.model-status.is-ready {
  background: #ecfdf5;
  color: #059669;
}

.model-status__dot {
  width: 7px;
  height: 7px;
  flex: 0 0 7px;
  border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 0 4px color-mix(in srgb, currentColor 12%, transparent);
}

.model-status .best-mark {
  margin-left: auto;
  color: inherit;
}

.model-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 12px;
  font-size: 14px;
  color: #909399;
}

.model-size {
  margin-bottom: 0;
  justify-content: space-between;
}

.model-meta:not(.model-size) span {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-size__value,
.best-mark {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.best-mark {
  color: #909399;
}

.page-state {
  min-height: 280px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: #909399;
  text-align: center;
}

.page-state .el-icon {
  font-size: 42px;
}

.empty-state strong {
  color: #303133;
  font-size: 18px;
}

.upload-dialog__content {
  padding: 6px 6px 0;
}

.upload-dialog__content h2 {
  margin: 0 0 18px;
  font-size: 24px;
  font-weight: 700;
  color: #303133;
}

.upload-dialog__desc {
  margin: 0 0 34px;
  font-size: 16px;
  color: #606266;
}

.upload-dialog__label {
  margin-bottom: 14px;
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

.upload-file-box {
  display: flex;
  align-items: center;
  min-height: 60px;
  padding: 0 14px;
  border: 1px solid #dcdfe6;
  border-radius: 6px;
  background: #fff;
}

.upload-file-input {
  display: none;
}

.file-select-btn {
  height: 34px;
  padding: 0 16px;
  border: 1px solid #c0c4cc;
  border-radius: 4px;
  background: #fff;
  font-size: 14px;
  color: #303133;
  cursor: pointer;
}

.file-select-btn:disabled {
  color: #c0c4cc;
  cursor: not-allowed;
}

.file-name-text {
  margin-left: 12px;
  font-size: 14px;
  color: #606266;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.upload-dialog__tip {
  margin-top: 12px;
  font-size: 14px;
  color: #909399;
}

.upload-progress {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-top: 18px;
  color: #409eff;
}

.upload-dialog__footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 36px;
}

@media screen and (max-width: 768px) {
  .model-hub-page .page-header {
    flex-direction: column;
    gap: 16px;
  }

  .model-hub-page .page-header :deep(.el-button--primary) {
    width: 100%;
  }

  .model-toolbar {
    align-items: stretch;
    flex-direction: column;
  }

  .format-tabs {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .format-tab {
    justify-content: center;
    padding: 0 8px;
  }

  .model-toolbar__summary {
    padding: 0 8px 4px;
  }

  .model-grid {
    grid-template-columns: 1fr;
  }

  .model-hub-page .model-card {
    min-height: 0;
  }

  .model-card__footer :deep(.el-button) {
    flex: 1 1 auto;
  }

  .model-card__footer.has-convert {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .model-card__footer.has-convert .convert-button {
    grid-column: 1 / -1;
  }

  .upload-file-box {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
    padding: 14px;
  }

  .file-name-text {
    margin-left: 0;
  }
}
</style>

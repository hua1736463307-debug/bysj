<!-- 
  模型推理视图
  支持选择已训练模型，对上传图片进行目标检测测试，包含单图及批量推理功能
-->
<template>
  <div class="inference-page">
    <div v-if="!embedded" class="page-header">
      <h1>模型推理</h1>
      <p>使用训练好的模型进行目标检测</p>
    </div>
    
    <div class="page-content">
      <!-- 批量推理模式 -->
      <div class="batch-mode">
        <el-row :gutter="20">
        <!-- 左侧配置面板 -->
        <el-col :xs="24" :lg="8">
          <el-card class="config-panel">
            <template #header>
              <span>推理配置</span>
            </template>
            
            <el-form :model="inferenceConfig" label-width="100px">
              <el-form-item label="选择模型">
                <div class="model-selector-container">
                  <el-input v-if="isLocked" :model-value="lockedModel?.display_name || '当前训练模型'" disabled />
                  <el-select v-else
                    v-model="inferenceConfig.modelPath" 
                    placeholder="选择模型" 
                    filterable
                    class="model-selector"
                    popper-class="model-dropdown"
                  >
                    <el-option-group label="预训练模型">
                      <el-option 
                        v-for="model in pretrainedModels"
                        :key="model.name"
                        :label="model.display_name || model.name"
                        :value="model.path || model.name"
                        class="simple-model-option"
                      >
                        <div class="simple-option-content">
                          <span class="model-name-simple">{{ model.display_name || model.name }}</span>
                          <span class="model-size-simple">{{ formatFileSize(model.size) }}</span>
                        </div>
                      </el-option>
                    </el-option-group>
                  <el-option-group label="训练模型" v-if="trainedModels.length > 0">
                    <el-option 
                      v-for="model in trainedModels"
                      :key="model.path"
                      :label="model.display_name"
                      :value="model.path"
                      class="custom-model-option"
                    >
                      <div class="model-option-content">
                        <div class="model-main-info">
                          <div class="model-name">
                            {{ model.display_name }}
                            <el-tag v-if="model.is_latest" type="success" size="small" class="latest-tag">
                              最新
                            </el-tag>
                          </div>
                          <div class="model-meta">
                            {{ model.dataset }}{{ model.epochs }} • {{ formatTime(model.modified_time) }}
                          </div>
                        </div>
                        <span class="model-size">
                          {{ formatFileSize(model.size) }}
                        </span>
                      </div>
                    </el-option>
                  </el-option-group>
                  <el-option-group label="自定义模型" v-if="customModels.length > 0">
                    <el-option 
                      v-for="model in customModels"
                      :key="model.path"
                      :label="model.name"
                      :value="model.path"
                    />
                  </el-option-group>
                  </el-select>
                  <el-button v-if="!isLocked"
                    type="text" 
                    size="small" 
                    @click="loadAvailableModels" 
                    class="refresh-btn"
                    :loading="isLoadingModels"
                  >
                    刷新模型列表
                  </el-button>
                </div>
              </el-form-item>
              
              <el-form-item label="置信度阈值">
                <el-slider 
                  v-model="inferenceConfig.confidence"
                  :min="0.1"
                  :max="1"
                  :step="0.05"
                  show-input
                  input-size="small"
                  :disabled="isProcessing"
                />
              </el-form-item>
              
              <el-form-item label="IoU阈值">
                <el-slider 
                  v-model="inferenceConfig.iouThreshold"
                  :min="0.1"
                  :max="1"
                  :step="0.05"
                  show-input
                  input-size="small"
                  :disabled="isProcessing"
                />
              </el-form-item>
              
              <!--<el-form-item label="最大检测数">
                <el-input-number 
                  v-model="inferenceConfig.maxDetections"
                  :min="1"
                  :max="1000"
                  controls-position="right"
                  :disabled="isProcessing"
                />
              </el-form-item>-->
              
              <el-form-item label="上传图片">
                <div class="upload-button-row">
                  <el-upload
                    ref="uploadRef"
                    :show-file-list="false"
                    :before-upload="handleImageUpload"
                    accept="image/*"
                    multiple
                  >
                    <el-button type="primary" :loading="isProcessing">
                      <el-icon><Upload /></el-icon>
                      上传图片并自动推理
                    </el-button>
                  </el-upload>

                  <el-button
                    v-if="results.length > 0"
                    @click="clearOnlyResults"
                    :disabled="isProcessing"
                  >
                    清空输出
                  </el-button>

                  <el-button
                    v-if="uploadedImages.length > 0"
                    @click="clearImages"
                    :disabled="isProcessing"
                  >
                    <el-icon><Delete /></el-icon>
                    清空图片
                  </el-button>
                </div>
              </el-form-item>
            </el-form>

            <div v-if="uploadedImages.length > 0" class="uploaded-panel">
              <div class="uploaded-panel-header">
                <div>
                  <div class="uploaded-panel-title">待推理图片</div>
                  <div class="uploaded-panel-subtitle">
                    已上传 {{ uploadedImages.length }} 张；上传后会自动调用模型推理，无需手动选择图片或点击推理
                  </div>
                </div>
              </div>

              <div class="uploaded-image-list">
                <div
                  v-for="image in uploadedImages"
                  :key="image.id"
                  class="uploaded-image-card"
                >
                  <div class="uploaded-thumb-wrap">
                    <img :src="image.url" :alt="image.name" class="uploaded-thumb" />
                    
                  </div>
                  <div class="uploaded-meta">
                    <div class="uploaded-name" :title="image.name">{{ image.name }}</div>
                    <div class="uploaded-extra">
                      <span>{{ formatFileSize(image.size) }}</span>
                      <el-tag size="small" :type="image.status === 'completed' ? 'success' : image.status === 'failed' ? 'danger' : 'info'">
                        {{ getUploadStatusText(image.status) }}
                      </el-tag>
                    </div>
                  </div>
                  <el-button
                    class="remove-uploaded-btn"
                    text
                    type="danger"
                    size="small"
                    @click.stop="removeUploadedImage(image.id)"
                    :disabled="isProcessing"
                  >
                    移除
                  </el-button>
                </div>
              </div>
            </div>
          </el-card>
          
          <!-- 检测统计 -->
          <el-card class="stats-panel">
            <template #header>
              <span>检测统计</span>
            </template>
            
            <div class="stat-item">
              <span class="stat-label">已处理图片:</span>
              <span class="stat-value">{{ processedCount }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">总检测数:</span>
              <span class="stat-value">{{ totalDetections }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">平均置信度:</span>
              <span class="stat-value">{{ averageConfidence }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">处理时间:</span>
              <span class="stat-value">{{ processingTime }}ms</span>
            </div>
          </el-card>
        </el-col>
        
        <!-- 右侧结果展示 -->
        <el-col :xs="24" :lg="16">
          <el-card class="results-panel">
            <template #header>
              <div class="results-header">
                <span>推理结果</span>
                <div class="header-actions">
                  <el-button-group>
                    <el-button 
                      :type="viewMode === 'grid' ? 'primary' : ''"
                      @click="viewMode = 'grid'"
                    >
                      <el-icon><Grid /></el-icon>
                    </el-button>
                    <el-button 
                      :type="viewMode === 'list' ? 'primary' : ''"
                      @click="viewMode = 'list'"
                    >
                      <el-icon><List /></el-icon>
                    </el-button>
                  </el-button-group>
                  
                  <el-button @click="downloadResults" :disabled="results.length === 0">
                    <el-icon><Download /></el-icon>
                    下载结果
                  </el-button>
                </div>
              </div>
            </template>
            
            <!-- 网格视图 -->
            <div v-if="viewMode === 'grid'" class="grid-view inference-results-grid-view">
              <div v-if="results.length === 0" class="empty-results">
                <el-empty description="暂无推理结果，请上传图片，系统会自动开始检测" />
              </div>
              
              <div
                v-else
                class="results-grid"
                :class="`results-grid--${Math.min(results.length, 4)}`"
              >
                <div
                  v-for="(result, index) in results"
                  :key="index"
                >
                  <div class="result-card" @click="showResultDetail(result)">
                    <div class="result-image">
                      <img :src="result.annotatedImageUrl || result.imageUrl" :alt="result.fileName" />
                      <div class="detection-overlay">
                        <span class="detection-count">{{ result.detections.length }} 个目标</span>
                      </div>
                      <div v-if="result.annotatedImageUrl" class="annotated-badge">
                        <el-tag type="success" size="small">已标注</el-tag>
                      </div>
                    </div>
                    <div class="result-info">
                      <div class="file-name">{{ result.fileName }}</div>
                      <div class="detection-summary">
                        <el-tag 
                          v-for="detection in result.detections.slice(0, 3)" 
                          :key="detection.id"
                          size="small"
                          class="detection-tag"
                        >
                          {{ detection.class }} ({{ (detection.confidence * 100).toFixed(1) }}%)
                        </el-tag>
                        <span v-if="result.detections.length > 3" class="more-text">
                          +{{ result.detections.length - 3 }} 更多
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <!-- 列表视图 -->
            <div v-if="viewMode === 'list'" class="list-view">
              <el-table :data="results" v-if="results.length > 0">
                <el-table-column label="图片" width="120">
                  <template #default="{ row }">
                    <img :src="row.imageUrl" class="table-image" @click="showResultDetail(row)" />
                  </template>
                </el-table-column>
                <el-table-column prop="fileName" label="文件名" />
                <el-table-column label="检测数量" width="100">
                  <template #default="{ row }">
                    {{ row.detections.length }}
                  </template>
                </el-table-column>
                <el-table-column label="主要目标" width="200">
                  <template #default="{ row }">
                    <el-tag 
                      v-for="detection in row.detections.slice(0, 2)" 
                      :key="detection.id"
                      size="small"
                      style="margin-right: 5px;"
                    >
                      {{ detection.class }}
                    </el-tag>
                  </template>
                </el-table-column>
                <el-table-column label="处理时间" width="100">
                  <template #default="{ row }">
                    {{ row.processingTime }}ms
                  </template>
                </el-table-column>
                <el-table-column label="操作" width="120">
                  <template #default="{ row }">
                    <el-button size="small" @click="showResultDetail(row)">
                      <el-icon><View /></el-icon>
                      查看
                    </el-button>
                  </template>
                </el-table-column>
              </el-table>
              
              <el-empty v-else description="暂无推理结果" />
            </div>
          </el-card>
        </el-col>
      </el-row>
      </div>
    </div>
    
    <!-- 结果详情对话框 -->
    <el-dialog 
      v-model="showDetailDialog" 
      :title="`检测结果 - ${selectedResult?.fileName}`"
      width="900px"
      top="5vh"
      append-to-body
    >
      <div v-if="selectedResult" class="result-detail">
        <el-row :gutter="20">
                  <el-col :xs="24" :md="16">
                    <div class="detail-image">
                      <div class="image-tabs">
                        <el-tabs v-model="activeImageTab" type="card">
                          <el-tab-pane label="原始图片" name="original">
                            <img :src="selectedResult.imageUrl" :alt="selectedResult.fileName" />
                          </el-tab-pane>
                          <el-tab-pane label="检测结果" name="annotated" v-if="selectedResult.annotatedImageUrl">
                            <img :src="selectedResult.annotatedImageUrl" :alt="selectedResult.fileName + '_annotated'" />
                          </el-tab-pane>
                        </el-tabs>
                      </div>
                    </div>
                  </el-col>
          
          <el-col :xs="24" :md="8">
            <div class="detail-info">
              <h4>检测详情</h4>
              <div class="detection-list">
                <div 
                  v-for="detection in selectedResult.detections" 
                  :key="detection.id"
                  class="detection-item"
                >
                  <div class="detection-header">
                    <span class="detection-class">{{ detection.class }}</span>
                    <span class="detection-confidence">
                      {{ (detection.confidence * 100).toFixed(1) }}%
                    </span>
                  </div>
                  <div class="detection-coords">
                    <small>
                      位置: ({{ formatCoord(detection.bbox?.x1) }}, {{ formatCoord(detection.bbox?.y1) }}) - ({{ formatCoord(detection.bbox?.x2) }}, {{ formatCoord(detection.bbox?.y2) }})
                      <br/>
                      大小: {{ formatCoord(detection.bbox?.width) }} × {{ formatCoord(detection.bbox?.height) }}
                      <br/>
                      中心: ({{ formatCoord(detection.bbox?.center_x) }}, {{ formatCoord(detection.bbox?.center_y) }})
                    </small>
                  </div>
                </div>
              </div>
            </div>
          </el-col>
        </el-row>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
/**
 * 模型推理视图脚本
 * 提供模型选择、参数配置、单图及批量图片推理功能
 * 支持实时监控和九宫格监控模式集成
 */
import { ref, reactive, computed, onMounted, onUnmounted, watch } from 'vue'
import { ElMessage, ElNotification } from 'element-plus'
import { predictImage, predictBatch, getModels } from '../utils/api'
import { saveAs } from 'file-saver'

const props = defineProps({
  embedded: { type: Boolean, default: false },
  trainingTaskId: { type: String, default: '' },
  lockedModel: { type: Object, default: null }
})
const emit = defineEmits(['busy-change'])
const isLocked = computed(() => Boolean(props.trainingTaskId && props.lockedModel?.available))
const embedded = computed(() => props.embedded)

const stripTrailingSlash = (value) => String(value || '').replace(/\/+$/, '')

const getConfiguredApiBase = () => stripTrailingSlash(
  import.meta.env.VITE_API_BASE_URL ||
  import.meta.env.VITE_BACKEND_API_URL ||
  import.meta.env.VITE_BACKEND_URL ||
  ''
)

const encodeImageUrl = (url) => {
  try {
    return encodeURI(url).replace(/#/g, '%23')
  } catch {
    return url
  }
}

const buildApiStaticUrl = (staticPath) => {
  const path = String(staticPath || '').startsWith('/')
    ? String(staticPath || '')
    : `/${staticPath || ''}`
  const apiPath = path.startsWith('/api/static/') ? path : `/api${path}`
  const configured = getConfiguredApiBase()

  if (!configured) return apiPath
  if (/\/api$/i.test(configured)) return `${configured}${path}`
  return `${configured}${apiPath}`
}

const normalizeAbsoluteImageUrl = (urlText) => {
  if (/^(data:|blob:)/i.test(urlText)) return urlText

  try {
    const url = new URL(urlText)
    const staticIndex = url.pathname.indexOf('/static/')
    if (staticIndex >= 0) {
      return buildApiStaticUrl(`${url.pathname.slice(staticIndex)}${url.search}${url.hash}`)
    }
    return url.toString()
  } catch {
    return urlText
  }
}

// --- UI 状态控制 ---
const isProcessing = ref(false)              // 是否正在执行推理
const isBusy = computed(() => isProcessing.value || pendingAutoInferenceIds.value.length > 0)
const viewMode = ref('grid')                 // 结果展示模式：grid, list
const showDetailDialog = ref(false)          // 是否显示详情弹窗
const activeImageTab = ref('original')       // 详情弹窗当前选中的标签页

// --- 业务数据 ---
const selectedResult = ref(null)             // 当前选中的推理结果详情
const uploadedImages = ref([])               // 已上传的图片列表
const results = ref([])                      // 当前推理结果列表；每次触发推理前会清空
// --- 自动推理队列 ---
// 用于解决 multiple 上传时 before-upload 会逐张触发的问题
let autoInferenceTimer = null
const pendingAutoInferenceIds = ref([])

// --- 推理配置 ---
const inferenceConfig = reactive({
  modelPath: 'yolov8n.pt',                   // 使用的模型路径
  confidence: 0.25,                          // 置信度阈值
  iouThreshold: 0.7,                         // IoU 阈值
  maxDetections: 100                         // 最大检测数量
})

// --- 模型列表数据 ---
const pretrainedModels = ref([])             // 预训练模型列表
const trainedModels = ref([])                // 已训练模型列表
const customModels = ref([])                 // 自定义上传模型列表
const isLoadingModels = ref(false)           // 是否正在加载模型列表

// --- 推理结果标准化工具 ---

const isAbsoluteUrl = (url) => /^(https?:|data:|blob:)/i.test(String(url || ''))
const IMAGE_URL_PATTERN = /\.(jpe?g|png|bmp|webp|gif)(\?.*)?$/i

/**
 * 判断后端返回值是否像一个真正可访问的图片地址。
 * 重点过滤 2010_006324.jpg_annotated 这类“文件名标识”，否则浏览器会请求
 * http://localhost:8000/2010_006324.jpg_annotated 并显示 alt 文本。
 */
const isUsableImageUrlValue = (value) => {
  if (!value) return false

  const text = String(value).trim()
  if (!text || text === 'null' || text === 'undefined') return false
  if (text.includes('[object Object]')) return false
  if (isAbsoluteUrl(text)) return true
  if (text.startsWith('/static/') || text.startsWith('static/')) return true
  if (text.includes('/static/')) return true

  // 纯文件名不具备访问路径，即使带 .jpg 也不能直接作为后端 URL 使用。
  if (!text.includes('/') && !text.includes('\\')) return false

  return IMAGE_URL_PATTERN.test(text)
}

/**
 * 从后端返回的字符串或对象中提取真正的图片 URL。
 * 兼容：
 * - "/static/predictions/xxx.jpg"
 * - { url: "/static/predictions/xxx.jpg", file_name: "xxx.jpg" }
 * - { path: "..." } / { file_path: "..." } / { src: "..." }
 */
const extractUrlValue = (value) => {
  if (!value) return null

  if (typeof value === 'string') {
    return value
  }

  if (typeof value === 'object') {
    return value.url
      ?? value.path
      ?? value.file_path
      ?? value.fileUrl
      ?? value.file_url
      ?? value.src
      ?? value.href
      ?? null
  }

  return null
}

/**
 * 将后端返回的相对路径转换为浏览器可访问 URL。
 * 这里必须先解包对象，否则会出现 http://localhost:8000/[object Object]。
 */
const buildBackendUrl = (url) => {
  const rawValue = extractUrlValue(url)
  if (!isUsableImageUrlValue(rawValue)) return null

  let value = String(rawValue).trim()

  if (isAbsoluteUrl(value)) return encodeImageUrl(normalizeAbsoluteImageUrl(value))

  // 兼容 /app/static/xxx 或 xxx/static/yyy 这种文件系统路径。
  const staticIndex = value.replace(/\\/g, '/').indexOf('/static/')
  if (staticIndex >= 0) {
    value = value.replace(/\\/g, '/').slice(staticIndex)
  }

  if (value.startsWith('static/')) {
    value = `/${value}`
  }

  if (value.startsWith('/api/static/')) {
    return encodeImageUrl(value)
  }

  if (value.startsWith('/static/')) {
    return encodeImageUrl(buildApiStaticUrl(value))
  }

  const configured = getConfiguredApiBase()
  if (configured) {
    return encodeImageUrl(`${configured.replace(/\/api$/i, '')}/${value.replace(/^\//, '')}`)
  }

  return encodeImageUrl(value.startsWith('/') ? value : `/${value}`)
}

const normalizeDisplayFileName = (name, fallback = 'unknown') => {
  const rawName = String(name || fallback)
  return rawName
    .replace(/(\.[A-Za-z0-9]+)_annotated$/i, '$1')
    .replace(/_annotated$/i, '')
}

/**
 * 兼容 axios / fetch / 后端直接返回的多种响应包裹格式
 */
const getPayload = (apiResult) => {
  if (!apiResult) return {}
  if (apiResult.data?.results) return apiResult.data.results
  if (apiResult.results) return apiResult.results
  if (apiResult.data) return apiResult.data
  return apiResult
}

const toNumber = (value, fallback = 0) => {
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue : fallback
}

/**
 * 统一检测框格式，避免模板读取 bbox.x1 / bbox.width 时报 undefined
 */
const normalizeBbox = (bbox = {}) => {
  let x1 = 0
  let y1 = 0
  let x2 = 0
  let y2 = 0

  if (Array.isArray(bbox)) {
    x1 = toNumber(bbox[0])
    y1 = toNumber(bbox[1])
    x2 = toNumber(bbox[2])
    y2 = toNumber(bbox[3])
  } else {
    x1 = toNumber(bbox.x1 ?? bbox.xmin ?? bbox.left ?? bbox.x)
    y1 = toNumber(bbox.y1 ?? bbox.ymin ?? bbox.top ?? bbox.y)

    if (bbox.x2 !== undefined || bbox.xmax !== undefined || bbox.right !== undefined) {
      x2 = toNumber(bbox.x2 ?? bbox.xmax ?? bbox.right)
    } else {
      x2 = x1 + toNumber(bbox.width ?? bbox.w)
    }

    if (bbox.y2 !== undefined || bbox.ymax !== undefined || bbox.bottom !== undefined) {
      y2 = toNumber(bbox.y2 ?? bbox.ymax ?? bbox.bottom)
    } else {
      y2 = y1 + toNumber(bbox.height ?? bbox.h)
    }
  }

  const width = Math.max(0, x2 - x1)
  const height = Math.max(0, y2 - y1)

  return {
    x1,
    y1,
    x2,
    y2,
    width,
    height,
    center_x: x1 + width / 2,
    center_y: y1 + height / 2
  }
}

/**
 * 统一检测结果字段，兼容 class/class_name/name、bbox/box/xyxy 等不同后端格式
 */
const normalizeDetection = (detection = {}, index = 0) => ({
  id: detection.id ?? `${Date.now()}_${index}`,
  class: detection.class ?? detection.class_name ?? detection.name ?? detection.label ?? detection.category ?? String(detection.cls ?? 'unknown'),
  confidence: toNumber(detection.confidence ?? detection.score ?? detection.conf ?? detection.probability),
  bbox: normalizeBbox(detection.bbox ?? detection.box ?? detection.xyxy ?? detection.rect ?? detection.coordinates ?? detection)
})

const extractDetections = (payload) => {
  const candidates = payload.detections
    ?? payload.predictions
    ?? payload.objects
    ?? payload.boxes
    ?? payload.result?.detections
    ?? []

  return Array.isArray(candidates)
    ? candidates.map((item, index) => normalizeDetection(item, index))
    : []
}

const extractAnnotatedImageUrl = (payload) => {
  const candidates = [
    payload.annotated_image_url,
    payload.annotated_image,
    payload.output_image_url,
    payload.output_image,
    payload.result_image_url,
    payload.result_image,
    ...(Array.isArray(payload.annotated_images) ? payload.annotated_images : []),
    ...(Array.isArray(payload.result_images) ? payload.result_images : [])
  ]

  for (const candidate of candidates) {
    const url = buildBackendUrl(candidate)
    if (url) return url
  }

  return null
}

/**
 * 将单张或批量中的一条后端输出转换成页面统一使用的数据结构
 */
const normalizeInferenceResult = (apiResult, file, imageUrl, processingTime = 0) => {
  const payload = getPayload(apiResult)
  const detections = extractDetections(payload)

  return {
    id: apiResult?.prediction_id ?? payload.prediction_id ?? payload.id ?? `${Date.now()}_${file?.name || 'image'}`,
    fileName: normalizeDisplayFileName(payload.original_name ?? payload.file_name ?? payload.filename ?? payload.image_name ?? file?.name ?? 'unknown'),
    imageUrl: buildBackendUrl(payload.original_image_url ?? payload.image_url) ?? imageUrl,
    annotatedImageUrl: extractAnnotatedImageUrl(payload),
    detections,
    processingTime,
    timestamp: new Date(),
    modelUsed: payload.model_used ?? payload.model ?? inferenceConfig.modelPath
  }
}

const formatCoord = (value) => toNumber(value).toFixed(0)

const createImagePreviewUrl = (file) => URL.createObjectURL(file)

const createUploadedImageId = (file) => `${Date.now()}_${Math.random().toString(16).slice(2)}_${file.name}`

const revokePreviewUrls = (images = uploadedImages.value) => {
  images.forEach((image) => {
    if (image.url && String(image.url).startsWith('blob:')) {
      URL.revokeObjectURL(image.url)
    }
  })
}

const getUploadStatusText = (status) => {
  const statusMap = {
    pending: '待推理',
    processing: '推理中',
    completed: '已完成',
    failed: '失败'
  }
  return statusMap[status] || '待推理'
}


const extractBatchItems = (apiResult) => {
  const payload = getPayload(apiResult)
  const candidates = payload.items
    ?? payload.images
    ?? payload.outputs
    ?? payload.predictions
    ?? payload.inference_results
    ?? payload.batch_results
    ?? payload.results
    ?? []

  if (!Array.isArray(candidates)) return []

  const resultImages = Array.isArray(payload.result_images)
    ? payload.result_images
    : Array.isArray(payload.annotated_images)
      ? payload.annotated_images
      : []

  return candidates.map((item, index) => ({
    ...item,
    annotated_image: item.annotated_image
      ?? item.annotated_image_url
      ?? resultImages[index]
      ?? null,
    annotated_image_url: item.annotated_image_url
      ?? item.annotated_image
      ?? resultImages[index]
      ?? null
  }))
}

// --- 计算属性 ---

/**
 * 已处理的图片总数
 */
const processedCount = computed(() => results.value.length)

/**
 * 所有结果中的检测目标总数
 */
const totalDetections = computed(() => 
  results.value.reduce((sum, result) => sum + result.detections.length, 0)
)

/**
 * 所有检测目标的平均置信度
 */
const averageConfidence = computed(() => {
  const allDetections = results.value.flatMap(result => result.detections)
  if (allDetections.length === 0) return '0.0%'
  
  const avgConf = allDetections.reduce((sum, det) => sum + det.confidence, 0) / allDetections.length
  return (avgConf * 100).toFixed(1) + '%'
})

/**
 * 平均推理处理时间（毫秒）
 */
const processingTime = computed(() => {
  if (results.value.length === 0) return 0
  return Math.round(results.value.reduce((sum, result) => sum + result.processingTime, 0) / results.value.length)
})


// --- 业务方法：图片处理 ---

/**
 * 处理图片选择：这里只保存待推理图片，不再自动触发单图推理
 * 这样可以避免 multiple 上传时 isProcessing 抢占，导致只有第一张图被处理。
 * @param {File} file 图片文件对象
 */
const handleImageUpload = (file) => {
  const imageId = createUploadedImageId(file)
  const previewUrl = createImagePreviewUrl(file)

  uploadedImages.value.push({
    id: imageId,
    file,
    url: previewUrl,
    name: file.name,
    size: file.size || 0,
    status: 'pending'
  })

  // 加入自动推理队列
  addPendingAutoInference(imageId)

  // 延迟触发，确保多图上传时能合并成一次批量推理
  scheduleAutoInference()

  ElMessage.success(`已添加图片：${file.name}，即将自动推理`)

  // 阻止 el-upload 默认上传行为，仍然由 predictImage / predictBatch 接口上传并推理
  return false
}

const clearOnlyResults = () => {
  results.value = []
  selectedResult.value = null
  showDetailDialog.value = false
}

const markUploadedStatus = (ids, status) => {
  const idSet = new Set(ids)
  uploadedImages.value = uploadedImages.value.map(image => (
    idSet.has(image.id) ? { ...image, status } : image
  ))
}

const removeUploadedImage = (id) => {
  const image = uploadedImages.value.find(item => item.id === id)
  if (image?.url && String(image.url).startsWith('blob:')) {
    URL.revokeObjectURL(image.url)
  }

  uploadedImages.value = uploadedImages.value.filter(item => item.id !== id)
}

/**
 * 自动推理指定图片列表。
 * 支持：
 * - 单张图片：调用 predictImage
 * - 多张图片：优先调用 predictBatch
 * - 如果批量接口没有返回逐图详情，则自动回退为逐张 predictImage
 */
const runImagesInference = async (targetImages, { clearBeforeRun = true } = {}) => {
  if (isProcessing.value) return false

  const images = Array.isArray(targetImages)
    ? targetImages.filter(Boolean)
    : []

  if (images.length === 0) return false

  isProcessing.value = true

  if (clearBeforeRun) {
    clearOnlyResults()
  }

  const imageIds = images.map(image => image.id)
  markUploadedStatus(imageIds, 'processing')

  const config = {
    confidence: inferenceConfig.confidence,
    iou_threshold: inferenceConfig.iouThreshold,
    max_det: inferenceConfig.maxDetections,
    max_detections: inferenceConfig.maxDetections,
    save_results: true,
    training_task_id: props.trainingTaskId || undefined
  }

  try {
    // 单张图片直接走单图接口
    if (images.length === 1) {
      const uploaded = images[0]
      const startTime = Date.now()

      const result = await predictImage(
        uploaded.file,
        inferenceConfig.modelPath,
        config
      )

      const inferenceResult = normalizeInferenceResult(
        result,
        uploaded.file,
        uploaded.url,
        Date.now() - startTime
      )

      results.value = [inferenceResult]
      markUploadedStatus([uploaded.id], 'completed')

      ElMessage.success(`推理完成，检测到 ${inferenceResult.detections.length} 个目标`)
      return true
    }

    // 多张图片优先走批量接口
    const files = images.map(image => image.file)
    const batchStartTime = Date.now()

    const result = await predictBatch(
      files,
      inferenceConfig.modelPath,
      config
    )

    const batchItems = extractBatchItems(result)

    if (batchItems.length > 0) {
      const normalizedResults = batchItems.map((item, index) => {
        const uploaded = images[index] || {}

        return normalizeInferenceResult(
          item,
          uploaded.file,
          uploaded.url,
          Math.round((Date.now() - batchStartTime) / Math.max(batchItems.length, 1))
        )
      })

      results.value = normalizedResults
      markUploadedStatus(imageIds, 'completed')

      ElMessage.success(`自动批量推理完成，共处理 ${normalizedResults.length} 张图片`)
      return true
    }

    // 批量接口没有返回逐图详情时，自动回退为逐张推理
    const normalizedResults = []

    for (const uploaded of images) {
      const singleStartTime = Date.now()

      const singleResult = await predictImage(
        uploaded.file,
        inferenceConfig.modelPath,
        config
      )

      normalizedResults.push(
        normalizeInferenceResult(
          singleResult,
          uploaded.file,
          uploaded.url,
          Date.now() - singleStartTime
        )
      )
    }

    results.value = normalizedResults
    markUploadedStatus(imageIds, 'completed')

    ElMessage.success(`自动推理完成，共处理 ${normalizedResults.length} 张图片`)
    return true
  } catch (error) {
    markUploadedStatus(imageIds, 'failed')
    ElMessage.error('自动推理失败')
    console.error('自动推理错误:', error)
    return false
  } finally {
    isProcessing.value = false
  }
}

/**
 * 将上传的图片加入自动推理队列。
 */
const addPendingAutoInference = (imageId) => {
  if (!pendingAutoInferenceIds.value.includes(imageId)) {
    pendingAutoInferenceIds.value.push(imageId)
  }
}

/**
 * 延迟触发自动推理。
 * 作用：multiple 多图上传时，把短时间内加入的图片合并成一次批量推理。
 */
const scheduleAutoInference = (delay = 120) => {
  if (autoInferenceTimer) {
    window.clearTimeout(autoInferenceTimer)
  }

  autoInferenceTimer = window.setTimeout(async () => {
    autoInferenceTimer = null
    await runPendingAutoInference()
  }, delay)
}

/**
 * 执行当前等待中的自动推理任务。
 */
const runPendingAutoInference = async () => {
  if (isProcessing.value) {
    scheduleAutoInference(300)
    return
  }

  const pendingIds = [...pendingAutoInferenceIds.value]

  if (pendingIds.length === 0) return

  pendingAutoInferenceIds.value = []

  const pendingIdSet = new Set(pendingIds)
  const imagesToInfer = uploadedImages.value.filter(image => pendingIdSet.has(image.id))

  if (imagesToInfer.length === 0) return

  await runImagesInference(imagesToInfer, { clearBeforeRun: true })

  // 如果推理过程中又上传了新图片，继续执行下一轮
  if (pendingAutoInferenceIds.value.length > 0) {
    scheduleAutoInference(120)
  }
}

/**
 * 清空上传图片及推理结果列表
 */
const clearImages = () => {
  revokePreviewUrls()
  uploadedImages.value = []
  clearOnlyResults()
}

/**
 * 显示推理结果详情弹窗
 * @param {Object} result 推理结果对象
 */
const showResultDetail = (result) => {
  selectedResult.value = result
  activeImageTab.value = result.annotatedImageUrl ? 'annotated' : 'original'
  showDetailDialog.value = true
}

/**
 * 下载所有推理结果为 CSV 文件：按检测框逐行导出，而不是只导出图片级汇总。
 */
const downloadResults = () => {
  if (results.value.length === 0) return

  const escapeCsv = (value) => `"${String(value ?? '').replace(/"/g, '""')}"`
  const rows = [
    [
      'fileName',
      'class',
      'confidence',
      'x1',
      'y1',
      'x2',
      'y2',
      'width',
      'height',
      'center_x',
      'center_y',
      'processingTimeMs',
      'modelUsed',
      'annotatedImageUrl',
      'timestamp'
    ]
  ]

  results.value.forEach((result) => {
    if (!result.detections || result.detections.length === 0) {
      rows.push([
        result.fileName,
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        result.processingTime,
        result.modelUsed,
        result.annotatedImageUrl || '',
        result.timestamp?.toISOString?.() || ''
      ])
      return
    }

    result.detections.forEach((detection) => {
      rows.push([
        result.fileName,
        detection.class,
        detection.confidence.toFixed(6),
        detection.bbox.x1.toFixed(2),
        detection.bbox.y1.toFixed(2),
        detection.bbox.x2.toFixed(2),
        detection.bbox.y2.toFixed(2),
        detection.bbox.width.toFixed(2),
        detection.bbox.height.toFixed(2),
        detection.bbox.center_x.toFixed(2),
        detection.bbox.center_y.toFixed(2),
        result.processingTime,
        result.modelUsed,
        result.annotatedImageUrl || '',
        result.timestamp?.toISOString?.() || ''
      ])
    })
  })

  const csvContent = rows.map(row => row.map(escapeCsv).join(',')).join('\n')
  const blob = new Blob([`\ufeff${csvContent}`], { type: 'text/csv;charset=utf-8' })
  saveAs(blob, `inference_results_${new Date().toISOString().split('T')[0]}.csv`)

  ElMessage.success('结果已下载')
}

// --- 业务方法：模型管理 ---

/**
 * 从后端获取可用模型列表
 */
const loadAvailableModels = async () => {
  try {
    isLoadingModels.value = true
    const modelsData = await getModels()
    
    // 分离并格式化不同类型的模型
    pretrainedModels.value = modelsData.models
      .filter(model => model.type === 'pretrained')
      .map(model => ({
        name: model.name,
        path: model.path,
        size: model.size,
        display_name: model.display_name
      }))
    
    trainedModels.value = modelsData.models
      .filter(model => model.type === 'trained')
      .map(model => ({
        name: model.name,
        path: model.path,
        size: model.size,
        display_name: model.display_name,
        experiment_name: model.experiment_name,
        weight_type: model.weight_type,
        dataset: model.dataset,
        epochs: model.epochs,
        is_latest: model.is_latest,
        modified_time: model.modified_time
      }))
    
    customModels.value = modelsData.models
      .filter(model => model.type === 'uploaded')
      .map(model => ({
        name: model.name,
        path: model.path,
        size: model.size,
        display_name: model.display_name,
        modified_time: model.modified_time
      }))
    
    // 自动选择最新训练的模型
    if (trainedModels.value.length > 0 && trainedModels.value[0].is_latest) {
      inferenceConfig.modelPath = trainedModels.value[0].path
      ElMessage.success(`已自动选择最新训练的模型: ${trainedModels.value[0].display_name}`)
    }
      
  } catch (error) {
    console.error('加载模型列表失败:', error)
    ElMessage.error('加载模型列表失败')
  } finally {
    isLoadingModels.value = false
  }
}

// --- 辅助工具方法 ---

/**
 * 格式化文件大小展示
 * @param {number} bytes 字节数
 */
const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

/**
 * 格式化相对时间展示
 * @param {number} timestamp 秒级时间戳
 */
const formatTime = (timestamp) => {
  const date = new Date(timestamp * 1000)
  const now = new Date()
  const diffMs = now - date
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)
  
  if (diffMins < 60) {
    return `${diffMins}分钟前`
  } else if (diffHours < 24) {
    return `${diffHours}小时前`
  } else if (diffDays < 7) {
    return `${diffDays}天前`
  } else {
    return date.toLocaleDateString('zh-CN')
  }
}

/**
 * 提示实时监控配置
 */
const showRealtimeSourceDialog = () => {
  ElMessage.info('请在实时监控模式中配置视频源，然后在九宫格中使用')
}

// --- 事件监听处理 ---

/**
 * 处理模型训练完成事件，刷新模型列表
 */
const handleModelTrainingCompleted = (event) => {
  const { projectName } = event.detail
  ElNotification({
    title: '新模型可用',
    message: `训练完成！模型 "${projectName}" 已添加到可选列表`,
    type: 'success',
    duration: 5000
  })
  
  loadAvailableModels()
}

// --- 生命周期钩子 ---
watch(isBusy, value => emit('busy-change', value), { immediate: true })

onMounted(() => {
  if (isLocked.value) {
    inferenceConfig.modelPath = props.lockedModel?.model_id || 'locked-training-model'
    return
  }
  loadAvailableModels()
  window.addEventListener('modelTrainingCompleted', handleModelTrainingCompleted)
})

onUnmounted(() => {
  window.removeEventListener('modelTrainingCompleted', handleModelTrainingCompleted)

  if (autoInferenceTimer) {
    window.clearTimeout(autoInferenceTimer)
    autoInferenceTimer = null
  }

  revokePreviewUrls()
})
</script>

<style scoped>
.inference-page {
  padding: 20px;
  background: #f5f7fa;
}

.page-header {
  text-align: center;
  margin-bottom: 30px;
}

.page-header h1 {
  color: #303133;
  margin-bottom: 10px;
}

.page-header p {
  color: #606266;
  font-size: 16px;
}

.mode-selector {
  display: flex;
  justify-content: center;
  margin-bottom: 30px;
}

.realtime-mode,
.batch-mode,
.grid-mode {
  width: 100%;
}

.page-content {
  max-width: 1400px;
  margin: 0 auto;
}

.config-panel,
.stats-panel,
.results-panel {
  margin-bottom: 20px;
}

.stats-panel {
  margin-top: 20px;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 10px;
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
}

.stat-label {
  color: #606266;
  font-size: 14px;
}

.stat-value {
  color: #303133;
  font-weight: bold;
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-actions {
  display: flex;
  gap: 10px;
}

.inference-results-grid-view {
  display: block !important;
  width: 100%;
  min-height: 400px;
}

.results-grid {
  display: grid !important;
  width: 100%;
  grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
  gap: 20px;
}

.results-grid--1 {
  grid-template-columns: minmax(0, 1fr) !important;
}

.results-grid--2 {
  grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
}

.results-grid--3 {
  grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
}

.result-card {
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  overflow: hidden;
  cursor: default;
  transition: all 0.3s ease;
  background: white;
}

.result-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.result-image {
  position: relative;
}

.result-image img {
  width: 100%;
  height: auto;
  display: block;
}

.detection-overlay {
  position: absolute;
  top: 10px;
  right: 10px;
  background: rgba(0,0,0,0.7);
  color: white;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
}

.annotated-badge {
  position: absolute;
  top: 10px;
  left: 10px;
}

.image-tabs {
  width: 100%;
}

.image-tabs img {
  max-width: 100%;
  max-height: 400px;
  object-fit: contain;
  border-radius: 8px;
}

.result-info {
  padding: 15px;
}

.file-name {
  font-weight: bold;
  margin-bottom: 10px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.detection-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  align-items: center;
}

.detection-tag {
  font-size: 11px;
}

.more-text {
  font-size: 12px;
  color: #909399;
}

.table-image {
  width: 80px;
  height: 60px;
  object-fit: cover;
  border-radius: 4px;
  cursor: pointer;
}

.empty-results {
  text-align: center;
  padding: 60px 20px;
}

.result-detail {
  padding: 10px 0;
}

.detail-image {
  text-align: center;
}

.detail-image img {
  max-width: 100%;
  max-height: 400px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.detail-info h4 {
  margin-bottom: 15px;
  color: #303133;
}

.detection-list {
  max-height: 300px;
  overflow-y: auto;
}

.detection-item {
  border: 1px solid #e4e7ed;
  border-radius: 6px;
  padding: 10px;
  margin-bottom: 8px;
  background: #f9f9f9;
}

.detection-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 5px;
}

.detection-class {
  font-weight: bold;
  color: #303133;
}

.detection-confidence {
  color: #67c23a;
  font-weight: bold;
}

.detection-coords {
  color: #909399;
  font-size: 12px;
}


.upload-button-row,

.uploaded-panel {
  margin-top: 6px;
  border-top: 1px solid #ebeef5;
  padding-top: 14px;
}

.uploaded-panel-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
}

.uploaded-panel-title {
  font-weight: 600;
  color: #303133;
  margin-bottom: 4px;
}

.uploaded-panel-subtitle {
  font-size: 12px;
  color: #909399;
  line-height: 1.5;
}

.uploaded-image-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  max-height: 420px;
  overflow-y: auto;
  padding-right: 4px;
}

.uploaded-image-card {
  position: relative;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  background: #fff;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.2s ease;
}

.uploaded-image-card:hover {
  border-color: #409eff;
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.12);
}


.uploaded-thumb-wrap {
  position: relative;
  height: 110px;
  background: #f5f7fa;
}

.uploaded-thumb {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}


.uploaded-meta {
  padding: 8px;
}

.uploaded-name {
  font-size: 13px;
  font-weight: 600;
  color: #303133;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  margin-bottom: 6px;
}

.uploaded-extra {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 6px;
  color: #909399;
  font-size: 12px;
}

.remove-uploaded-btn {
  position: absolute;
  right: 4px;
  top: 4px;
  background: rgba(255, 255, 255, 0.9) !important;
}

@media (max-width: 768px) {
  .inference-page {
    padding: 10px;
  }
  
  .results-header {
    flex-direction: column;
    gap: 10px;
    align-items: stretch;
  }
  
  .header-actions {
    justify-content: space-between;
  }
  
  .result-card {
    margin-bottom: 0;
  }
  
  .results-grid,
  .results-grid--2,
  .results-grid--3 {
    grid-template-columns: minmax(0, 1fr) !important;
    gap: 10px;
  }

  .uploaded-image-list {
    grid-template-columns: 1fr;
  }
}

/* 修复模型选择器遮挡问题 */
.model-selector-container {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
}

.model-selector {
  flex: 1;
  width: 100% !important;
}

.refresh-btn {
  flex-shrink: 0;
  margin-left: 0 !important;
}

/* 下拉框样式 */
:deep(.el-select-dropdown) {
  z-index: 9999 !important;
  max-width: 500px !important;
}

.model-dropdown {
  z-index: 9999 !important;
}

:deep(.el-select-dropdown .el-select-dropdown__item) {
  height: auto !important;
  padding: 8px 12px !important;
  line-height: 1.4 !important;
}

/* 简单模型选项样式 */
.simple-model-option {
  height: auto !important;
  min-height: 36px !important;
  padding: 6px 12px !important;
}

.simple-option-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.model-name-simple {
  font-weight: 500;
  color: #303133;
}

.model-size-simple {
  color: #8492a6;
  font-size: 12px;
  font-weight: 500;
}

/* 复杂模型选项样式 */
.custom-model-option {
  height: auto !important;
  min-height: 60px !important;
  padding: 10px 12px !important;
}

.model-option-content {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  width: 100%;
  gap: 12px;
}

.model-main-info {
  flex: 1;
  min-width: 0;
}

.model-name {
  font-weight: 600;
  font-size: 14px;
  color: #303133;
  margin-bottom: 4px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.model-meta {
  font-size: 11px;
  color: #909399;
  line-height: 1.2;
}

.model-size {
  color: #8492a6;
  font-size: 12px;
  white-space: nowrap;
  font-weight: 500;
}

.latest-tag {
  margin-left: 8px !important;
}

/* 确保下拉框有足够的最大高度和滚动 */
:deep(.el-select-dropdown__wrap) {
  max-height: 350px !important;
  overflow-y: auto !important;
}

:deep(.el-select-dropdown__list) {
  padding: 4px 0 !important;
}

/* 优化选择器的宽度 */
:deep(.el-form-item .el-select) {
  width: 100% !important;
}

/* 修复选项组标题的样式 */
:deep(.el-select-group__title) {
  padding: 8px 12px !important;
  font-weight: 600 !important;
  color: #606266 !important;
  background-color: #f5f7fa !important;
  position: sticky !important;
  top: 0 !important;
  z-index: 10 !important;
}

/* 修复选项悬浮效果 */
:deep(.el-select-dropdown__item:hover) {
  background-color: #f5f7fa !important;
}

/* 修复选中状态 */
:deep(.el-select-dropdown__item.selected) {
  background-color: #ecf5ff !important;
  color: #409eff !important;
}

/* 确保下拉框不被其他元素遮挡 */
:deep(.el-popper) {
  z-index: 9999 !important;
}

/* 修复配置面板的层级 */
.config-panel {
  position: relative;
  z-index: 1;
}
</style>

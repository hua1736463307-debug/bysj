<!-- 
  数据集管理视图
  用于浏览服务器上的数据集目录，支持数据集的导入、导出和基本操作
-->
<template>
  <div class="dataset-page raw-dataset-page">
    <div class="filter-panel">
      <el-form class="filter-form" inline @submit.prevent>
        <el-form-item label="数据集名称">
          <el-input v-model="filters.name" class="dataset-name-input" clearable placeholder="请输入数据集名称" @keyup.enter="handleSearch" />
        </el-form-item>
        <el-form-item label="上传人">
          <el-select v-model="filters.uploader" class="uploader-select" clearable filterable placeholder="请选择上传人">
            <el-option v-for="uploader in uploaderOptions" :key="uploader" :label="uploader" :value="uploader" />
          </el-select>
        </el-form-item>
        <el-form-item class="filter-actions">
          <el-button type="primary" class="toolbar-button" @click="handleSearch">搜索</el-button>
          <el-button class="toolbar-button merge-button" @click="openMergeDialog">合并数据集</el-button>
          <el-button type="primary" class="toolbar-button upload-button" @click="openUploadDialog">上传数据集</el-button>
        </el-form-item>
      </el-form>
    </div>

    <div class="dataset-content" v-loading="loading">
      <div v-if="filteredDatasets.length" class="dataset-grid">
        <article v-for="dataset in filteredDatasets" :key="dataset.id || dataset.name" class="dataset-card">
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
              <span>图片数：</span><strong>{{ datasetImageCount(dataset) }}</strong><span>张</span>
              <span class="meta-gap">大小：</span><strong>{{ formatDatasetSize(dataset.size) }} M</strong>
            </div>
            <div class="meta-row">
              <span>上传时间：</span><strong>{{ formatDate(dataset.created_at || dataset.createdAt) }}</strong>
              <span class="meta-gap">上传人：</span><strong>{{ datasetUploader(dataset) }}</strong>
            </div>
          </div>
          <div class="dataset-actions">
            <el-button class="dataset-action-primary" plain @click="viewDataset(dataset)">详情</el-button>
            <el-button v-if="dataset.can_delete" class="dataset-action-primary" plain @click="annotateDataset(dataset)">标注</el-button>
            <el-button v-if="dataset.can_delete" class="dataset-action-primary" type="warning" plain @click="openDatasetAugmentation(dataset)">数据增强</el-button>
            <el-button v-if="dataset.can_delete" class="dataset-action-delete" type="danger" plain @click="deleteDataset(dataset)">删除</el-button>
          </div>
        </article>
      </div>
      <el-empty v-else class="empty-state" description="暂无数据集" />
      <div v-if="total > pageSize" class="pagination-wrap">
        <el-pagination v-model:current-page="currentPage" v-model:page-size="pageSize" :total="total" :page-sizes="[12, 20, 40, 80]" layout="total, sizes, prev, pager, next, jumper" @current-change="handlePageChange" @size-change="handleSizeChange" />
      </div>
    </div>

    <el-dialog v-model="showDetailDialog" :title="`${selectedDataset ? datasetTitle(selectedDataset) : ''} - 详情列表`" width="76%" class="dataset-detail-dialog" @closed="clearDetailPreviewUrls">
      <el-table v-loading="detailLoading" :data="detailImages" empty-text="暂无图片明细">
        <el-table-column label="图片" width="120" align="center">
          <template #default="{ row }">
            <el-image v-if="imageThumbnailUrl(row)" class="detail-thumbnail" :src="imageThumbnailUrl(row)" fit="cover" />
            <span v-else class="detail-thumbnail-placeholder">暂无预览</span>
          </template>
        </el-table-column>
        <el-table-column label="图片大小" width="130" align="center"><template #default="{ row }">{{ formatFileSize(row.size) }}</template></el-table-column>
        <el-table-column label="上传时间" min-width="190" align="center"><template #default="{ row }">{{ formatDateTime(row.created_at || row.updated_at) }}</template></el-table-column>
        <el-table-column label="上传人" width="130" align="center"><template #default>{{ selectedDataset ? datasetUploader(selectedDataset) : '-' }}</template></el-table-column>
        <el-table-column label="操作" width="120" align="center"><template #default><el-button type="danger" plain size="small" @click="notifyImageDeleteUnavailable">删除</el-button></template></el-table-column>
      </el-table>
      <el-alert v-if="detailError" class="detail-error" :title="detailError" type="warning" :closable="false" show-icon />
      <div v-if="detailTotal" class="detail-pagination">
        <el-pagination v-model:current-page="detailPage" v-model:page-size="detailPageSize" :total="detailTotal" :page-sizes="[10, 20, 50, 100]" layout="total, sizes, prev, pager, next, jumper" @current-change="loadDatasetImages" @size-change="handleDetailSizeChange" />
      </div>
    </el-dialog>

    <el-dialog
      v-model="showAnnotationDraftDialog"
      title="选择标注草稿"
      width="700px"
      class="annotation-draft-dialog"
    >
      <div class="annotation-draft-card-grid">
        <el-card
          v-for="action in annotationDraftActions"
          :key="`${action.mode}-${action.kind}`"
          shadow="hover"
          class="annotation-draft-card"
          :class="{ 'is-disabled': action.disabled }"
          @click="!action.disabled && chooseAnnotationDraftAction(action.kind, action.mode)"
        >
          <div class="annotation-draft-card-title">{{ action.title }}</div>
          <div class="annotation-draft-card-description">{{ action.description }}</div>
          <el-tag v-if="action.exists" size="small" type="success">{{ action.updatedAt ? `已保存 ${formatDate(action.updatedAt)}` : '已有草稿' }}</el-tag>
          <el-tag v-else-if="action.mode === 'resume'" size="small" type="info">暂无可恢复草稿</el-tag>
        </el-card>
      </div>
      <template #footer>
        <el-button @click="showAnnotationDraftDialog = false">取消</el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="showAnnotationModelDialog"
      :title="annotationSelectionKind === 'multimodal' ? '选择多模态模型' : annotationSelectionKind === 'yolo' ? '选择普通 YOLO 模型' : '选择训练模型'"
      width="620px"
      class="annotation-model-dialog"
      :close-on-click-modal="!annotationModelLoading"
    >
      <el-alert
        title="模型类型决定标注方式，后续训练应使用相同类型的标注。"
        type="info"
        :closable="false"
        show-icon
      />
      <div v-loading="annotationModelLoading" class="annotation-model-list">
        <template v-for="group in annotationModelGroups" :key="group.title">
          <div class="annotation-model-group-title">{{ group.title }}（{{ group.items.length }}）</div>
          <el-radio-group v-model="selectedAnnotationModel" class="annotation-model-radio-group">
            <el-radio
              v-for="model in group.items"
              :key="modelIdentity(model) || modelSelectionValue(model)"
              :label="model.path || model.name || model.display_name"
              border
              class="annotation-model-option"
            >
              <div class="annotation-model-option-content">
                <div class="annotation-model-option-title">
                  <span>{{ model.display_name || model.name || model.path }}</span>
                  <el-tag size="small" :type="model.geometry === 'polygon' ? 'success' : 'primary'">
                    {{ model.geometry === 'polygon' ? '多边形' : '矩形框' }}
                  </el-tag>
                </div>
                <div class="annotation-model-option-description">
                  {{ model.description || (model.geometry === 'polygon' ? '分割模型，使用多边形标注' : '检测模型，使用矩形框标注') }}
                </div>
              </div>
            </el-radio>
          </el-radio-group>
        </template>
        <el-empty v-if="!annotationModelLoading && !filteredAnnotationModels.length" description="暂无可用 PT 模型" />
      </div>
      <template #footer>
        <el-button @click="showAnnotationModelDialog = false">取消</el-button>
        <el-button type="primary" :loading="annotationModelSaving" :disabled="!selectedAnnotationModel || annotationModelSaving" @click="confirmAnnotationModel">进入标注</el-button>
      </template>
    </el-dialog>

    <!-- 与数据标注页保持相同的增强选项；此处任务作用域为整个原始数据集。 -->
    <el-dialog
      v-model="showAugmentationDialog"
      title="数据增强"
      width="600px"
      center
      :close-on-click-modal="!isApplyingAugmentation"
      :close-on-press-escape="!isApplyingAugmentation"
    >
      <div class="augmentation-config">
        <el-form :model="augmentationConfig" label-width="120px">
          <el-form-item label="旋转角度">
            <el-checkbox-group v-model="augmentationConfig.rotations" class="augmentation-option-group" :disabled="isApplyingAugmentation">
              <el-checkbox v-for="angle in ROTATION_OPTIONS" :key="angle" :label="angle">{{ angle }}°</el-checkbox>
            </el-checkbox-group>
            <div class="augmentation-help">可多选；每个角度固定生成 1 张</div>
          </el-form-item>

          <el-form-item label="镜像">
            <el-radio-group v-model="augmentationConfig.mirror" class="augmentation-option-group" :disabled="isApplyingAugmentation">
              <el-radio v-for="option in MIRROR_OPTIONS" :key="option.value" :label="option.value">{{ option.label }}</el-radio>
            </el-radio-group>
            <div class="augmentation-help">三选一；不镜像不额外生成图片</div>
          </el-form-item>

          <el-form-item label="亮度">
            <el-checkbox-group v-model="augmentationConfig.brightnessLevels" class="augmentation-option-group" :disabled="isApplyingAugmentation">
              <el-checkbox v-for="option in BRIGHTNESS_OPTIONS" :key="option.value" :label="option.value">{{ option.label }}</el-checkbox>
            </el-checkbox-group>
            <div class="augmentation-help">亮度倍率；每个档位固定生成 1 张</div>
          </el-form-item>

          <el-form-item label="灰度强度">
            <el-checkbox-group v-model="augmentationConfig.grayscaleLevels" class="augmentation-option-group" :disabled="isApplyingAugmentation">
              <el-checkbox v-for="option in GRAYSCALE_OPTIONS" :key="option.value" :label="option.value">{{ option.label }}</el-checkbox>
            </el-checkbox-group>
            <div class="augmentation-help">灰度化强度；30% 为轻度，100% 为完全灰度</div>
          </el-form-item>

          <el-form-item label="对比度">
            <el-checkbox-group v-model="augmentationConfig.contrastLevels" class="augmentation-option-group" :disabled="isApplyingAugmentation">
              <el-checkbox v-for="option in CONTRAST_OPTIONS" :key="option.value" :label="option.value">{{ option.label }}</el-checkbox>
            </el-checkbox-group>
            <div class="augmentation-help">对比度倍率：60% 低、100% 标准、140% 高</div>
          </el-form-item>

          <el-alert
            :title="augmentationOperationCount > 0
              ? `将处理 ${datasetImageCount(augmentationDataset)} 张原始图片，生成约 ${augmentationOperationCount * datasetImageCount(augmentationDataset)} 张增强图片；原始图片始终保留`
              : '请选择至少一种增强操作'"
            :type="augmentationOperationCount > 0 ? 'info' : 'warning'"
            :closable="false"
            show-icon
          />
          <el-progress v-if="isApplyingAugmentation" class="augmentation-progress" :percentage="augmentationTaskProgress" :status="augmentationTaskStatus === 'failed' ? 'exception' : undefined" />
          <div v-if="augmentationTaskMessage" class="augmentation-task-message">{{ augmentationTaskMessage }}</div>
        </el-form>

        <div v-if="augmentationPreview.length > 0" class="augmentation-preview">
          <h4>预览效果（取数据集首张原图）</h4>
          <div class="preview-grid">
            <div v-for="(preview, index) in augmentationPreview" :key="index" class="preview-item">
              <img :src="preview" :alt="`预览${index + 1}`" />
            </div>
          </div>
        </div>
      </div>

      <template #footer>
        <el-button @click="generateAugmentationPreview" :disabled="augmentationOperationCount === 0 || isApplyingAugmentation" :loading="isGeneratingPreview">生成预览</el-button>
        <el-button v-if="!isApplyingAugmentation" @click="showAugmentationDialog = false">取消</el-button>
        <el-button v-else type="danger" plain @click="cancelDatasetAugmentation">取消任务</el-button>
        <el-button type="primary" @click="applyDatasetAugmentation" :disabled="augmentationOperationCount === 0 || isApplyingAugmentation" :loading="isApplyingAugmentation">应用到整个数据集</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="showUploadDialog" title="上传文件" width="560px" class="dataset-upload-dialog" :close-on-click-modal="!uploading" :close-on-press-escape="!uploading">
      <p class="upload-description">上传原始数据集文件（支持压缩包格式）</p>
      <el-form label-position="top" class="upload-form" @submit.prevent>
        <el-form-item label="数据集名称" required>
          <el-input v-model="uploadDatasetName" placeholder="请输入数据集名称" :disabled="uploading" />
        </el-form-item>
        <el-form-item v-if="uploadRoleOptions.length > 1" label="归属角色" required>
          <el-select v-model="uploadOwnerRoleId" placeholder="请选择数据集归属角色" :disabled="uploading">
            <el-option v-for="role in uploadRoleOptions" :key="role.id" :label="`${role.name}（级别 ${role.authorityLevel}）`" :value="role.id" />
          </el-select>
        </el-form-item>
        <el-form-item label="选择数据集文件" required>
          <div class="file-picker-row">
            <el-upload action="#" :auto-upload="false" :show-file-list="false" accept=".zip" :disabled="uploading" :on-change="handleUploadFileChange">
              <el-button :disabled="uploading">选择文件</el-button>
            </el-upload>
            <span class="selected-file-name">{{ selectedUploadFile?.name || '未选择文件' }}</span>
          </div>
          <div class="upload-tip">仅支持上传 <code>.zip</code> 格式文件，将保持原始文件名</div>
        </el-form-item>
      </el-form>

      <div v-if="uploading || uploadProgress > 0 || uploadPhase === 'error'" class="upload-progress-panel">
        <div class="upload-progress-header">
          <div><div class="upload-file-name">{{ uploadFileName || '数据集上传中' }}</div><div class="upload-progress-text">{{ uploadProgressText }}</div></div>
          <el-tag :type="uploadStatus === 'success' ? 'success' : uploadStatus === 'exception' ? 'danger' : 'info'">{{ uploadPhaseText }}</el-tag>
        </div>
        <el-progress :percentage="uploadProgress" :status="uploadStatus" :stroke-width="10" />
        <div class="upload-progress-footer">
          <span>{{ formatFileSize(uploadLoaded) }} / {{ formatFileSize(uploadTotal) }}</span>
          <span v-if="uploadPhase === 'processing'">后端正在解压和统计，已等待 {{ processingSeconds }} 秒</span>
          <el-button v-if="uploading && uploadPhase === 'uploading'" type="danger" link @click="cancelCurrentUpload">取消上传</el-button>
          <el-button v-else-if="uploadPhase === 'error' && lastUploadFile" type="primary" link @click="retryLastUpload">重试</el-button>
        </div>
      </div>

      <template #footer>
        <el-button :disabled="uploading" @click="showUploadDialog = false">取消</el-button>
        <el-button type="primary" :loading="uploading" :disabled="!canConfirmUpload" @click="confirmDatasetUpload">确认上传</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="showMergeDialog" title="合并数据集" width="680px" class="dataset-merge-dialog" :close-on-click-modal="!merging" :close-on-press-escape="!merging">
      <el-alert title="源数据集不会被修改；同内容图片会全部保留，新数据集按来源分目录存储。" type="info" :closable="false" show-icon />
      <el-form label-position="top" class="merge-form" @submit.prevent>
        <el-form-item label="新数据集名称" required>
          <el-input v-model="mergeDatasetName" maxlength="128" show-word-limit placeholder="请输入合并后的数据集名称" :disabled="merging" />
        </el-form-item>
        <el-form-item v-if="uploadRoleOptions.length > 1" label="归属角色" required>
          <el-select v-model="mergeOwnerRoleId" placeholder="请选择新数据集归属角色" :disabled="merging">
            <el-option v-for="role in uploadRoleOptions" :key="role.id" :label="`${role.name}（级别 ${role.authorityLevel}）`" :value="role.id" />
          </el-select>
        </el-form-item>
        <el-form-item label="选择源数据集" required>
          <el-input v-model="mergeKeyword" clearable placeholder="搜索数据集名称" :disabled="merging || mergeCandidateLoading" />
          <div v-loading="mergeCandidateLoading" class="merge-candidate-list">
            <el-checkbox-group v-model="mergeSourceIds">
              <el-checkbox
                v-for="dataset in filteredMergeCandidates"
                :key="dataset.id"
                :label="dataset.id"
                :disabled="datasetImageCount(dataset) <= 0 || (mergeSourceIds.length >= 50 && !mergeSourceIds.includes(dataset.id)) || merging"
                class="merge-candidate-item"
              >
                <span class="merge-candidate-name">{{ datasetTitle(dataset) }}</span>
                <span class="merge-candidate-meta">{{ datasetImageCount(dataset) }} 张 · {{ formatDatasetSize(dataset.size) }} · {{ datasetUploader(dataset) }}</span>
              </el-checkbox>
            </el-checkbox-group>
            <el-empty v-if="!mergeCandidateLoading && !filteredMergeCandidates.length" description="暂无可选数据集" :image-size="72" />
          </div>
          <div class="merge-selection-summary">已选择 {{ mergeSourceIds.length }} 个数据集，共 {{ mergeSelectedImageCount }} 张图片（最多 50 个）</div>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button :disabled="merging" @click="showMergeDialog = false">取消</el-button>
        <el-button type="primary" :loading="merging" :disabled="!canConfirmMerge" @click="confirmDatasetMerge">开始合并</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
/**
 * 数据集管理逻辑
 * 提供数据集的浏览、上传、详情查看、删除、批量操作及简单统计功能
 */
import { ref, reactive, computed, onMounted, onBeforeUnmount } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useRouter } from 'vue-router'
import { modelSupportsTextPrompts, modelAnnotationGeometry } from '../utils/modelCapabilities.mjs'
import {
  getAnnotationDraft,
  getAnnotationDrafts,
  getModels,
  createAnnotationDraft
} from '../utils/api'
import {
  ROTATION_OPTIONS,
  MIRROR_OPTIONS,
  BRIGHTNESS_OPTIONS,
  GRAYSCALE_OPTIONS,
  CONTRAST_OPTIONS,
  buildAugmentationOperations,
  getAugmentedImageDimensions
} from '../utils/imageAugmentation.mjs'

// --- 路由实例 ---
const router = useRouter()

// --- UI 状态控制 ---
const loading = ref(false)              // 数据列表加载状态
const uploadProgress = ref(0)           // 上传进度百分比
const uploadStatus = ref('')            // 上传状态 (success/exception)
const uploading = ref(false)            // 是否正在上传或等待后端处理
const uploadFileName = ref('')          // 当前上传文件名
const uploadLoaded = ref(0)             // 已上传字节数
const uploadTotal = ref(0)              // 总字节数
const uploadPhase = ref('idle')         // idle/uploading/processing/success/error
const uploadProgressText = ref('')      // 上传状态说明
const processingSeconds = ref(0)          // 后端解压/统计等待秒数
let processingTimer = null                // 后端处理阶段计时器
let currentUploadXhr = null
let detailObjectUrls = []
let datasetCoverObjectUrls = []
let datasetCoverGeneration = 0
const lastUploadFile = ref(null)
const showDetailDialog = ref(false)     // 详情对话框可见性
const selectedDataset = ref(null)       // 当前选中的单个数据集（用于详情）
const showUploadDialog = ref(false)     // 上传弹窗可见性
const uploadDatasetName = ref('')
const selectedUploadFile = ref(null)
const uploadOwnerRoleId = ref(null)
const showMergeDialog = ref(false)
const showAugmentationDialog = ref(false)
const augmentationDataset = ref(null)
const augmentationConfig = reactive({
  rotations: [],
  mirror: 'none',
  brightnessLevels: [],
  grayscaleLevels: [],
  contrastLevels: []
})
const augmentationPreview = ref([])
const isGeneratingPreview = ref(false)
const isApplyingAugmentation = ref(false)
const augmentationTaskId = ref('')
const augmentationTaskStatus = ref('')
const augmentationTaskProgress = ref(0)
const augmentationTaskMessage = ref('')
let augmentationPollTimer = null
const augmentationOperationCount = computed(() => buildAugmentationOperations(augmentationConfig).length)
const showAnnotationModelDialog = ref(false)
const showAnnotationDraftDialog = ref(false)
const annotationModelLoading = ref(false)
const annotationModelSaving = ref(false)
const annotationModels = ref([])
const selectedAnnotationModel = ref('')
const pendingAnnotationDataset = ref(null)
const annotationSelectionKind = ref('')
const annotationDraftStatus = ref({ multimodal: { exists: false }, yolo: { exists: false } })
const pendingDraftReplacement = ref(null)
const merging = ref(false)
const mergeCandidateLoading = ref(false)
const mergeDatasetName = ref('')
const mergeOwnerRoleId = ref(null)
const mergeSourceIds = ref([])
const mergeCandidates = ref([])
const mergeKeyword = ref('')
const detailImages = ref([])
const detailLoading = ref(false)
const detailError = ref('')
const detailPage = ref(1)
const detailPageSize = ref(10)
const detailTotal = ref(0)
const filters = ref({
  name: '',
  uploader: ''
})

const filteredAnnotationModels = computed(() => annotationModels.value.filter(model => {
  const format = String(model?.format || String(model?.path || model?.name || '').split('.').pop() || '').toLowerCase()
  if (format !== 'pt') return false
  const supportsTextPrompts = modelSupportsTextPrompts(model)
  if (annotationSelectionKind.value === 'multimodal') return supportsTextPrompts
  if (annotationSelectionKind.value === 'yolo') return !supportsTextPrompts
  return true
}))

const annotationModelSource = (model = {}) => String(
  model?.source || model?.model_source || model?.source_type || model?.type || ''
).trim().toLowerCase().replace(/[\s-]+/g, '_')

const annotationModelGroups = computed(() => [
  {
    title: '预训练模型',
    items: filteredAnnotationModels.value.filter(model => ['pretrained', 'pre_trained'].includes(annotationModelSource(model)))
  },
  {
    title: '训练模型',
    items: filteredAnnotationModels.value.filter(model => ['trained', 'training'].includes(annotationModelSource(model)))
  }
].filter(group => group.items.length))

const annotationDraftActions = computed(() => [
  {
    mode: 'resume',
    kind: 'multimodal',
    title: '恢复多模态标注',
    description: '恢复 YOLOE 模型、类别、几何和上次位置',
    exists: Boolean(annotationDraftStatus.value.multimodal?.exists),
    updatedAt: annotationDraftStatus.value.multimodal?.updated_at,
    disabled: !annotationDraftStatus.value.multimodal?.exists
  },
  {
    mode: 'resume',
    kind: 'yolo',
    title: '恢复普通 YOLO 标注',
    description: '恢复普通 YOLO 模型、类别、几何和上次位置',
    exists: Boolean(annotationDraftStatus.value.yolo?.exists),
    updatedAt: annotationDraftStatus.value.yolo?.updated_at,
    disabled: !annotationDraftStatus.value.yolo?.exists
  },
  {
    mode: 'new',
    kind: 'multimodal',
    title: '新建多模态标注',
    description: '选择 YOLOE 模型开始新的多模态标注',
    exists: Boolean(annotationDraftStatus.value.multimodal?.exists),
    disabled: false
  },
  {
    mode: 'new',
    kind: 'yolo',
    title: '新建普通 YOLO 标注',
    description: '选择非 YOLOE PT 模型开始新的普通标注',
    exists: Boolean(annotationDraftStatus.value.yolo?.exists),
    disabled: false
  }
])

// --- 数据列表 ---
const datasetList = ref([])

const currentUser = (() => {
  try {
    return JSON.parse(localStorage.getItem('user') || '{}')
  } catch {
    return {}
  }
})()
const uploadRoleOptions = computed(() => Array.isArray(currentUser.roles) ? currentUser.roles : [])

const datasetTitle = (dataset) => {
  return String(dataset.name || dataset.id || dataset.original_filename || '未命名数据集')
}

const datasetImageCount = (dataset) => {
  const value = dataset.image_count || dataset.imageCount || dataset.images || 0
  return Array.isArray(value) ? value.length : Number(value) || 0
}

const datasetUploader = (dataset) => {
  return dataset.uploader || dataset.upload_user || dataset.created_by || dataset.owner || '管理员'
}

const getDatasetCover = (dataset) => {
  return dataset?._coverPreviewUrl || ''
}

const uploaderOptions = computed(() => {
  return Array.from(new Set(datasetList.value.map(datasetUploader).filter(Boolean)))
})

const filteredDatasets = computed(() => {
  const nameKeyword = filters.value.name.trim().toLowerCase()
  const uploader = filters.value.uploader

  return datasetList.value.filter((dataset) => {
    const matchName = !nameKeyword || datasetTitle(dataset).toLowerCase().includes(nameKeyword)
    const matchUploader = !uploader || datasetUploader(dataset) === uploader
    return matchName && matchUploader
  })
})

// --- 计算属性：系统统计 ---
const totalDatasets = computed(() => datasetList.value.length)
const totalImages = computed(() => 
  datasetList.value.reduce((sum, dataset) => sum + datasetImageCount(dataset), 0)
)
const totalSize = computed(() => 
  datasetList.value.reduce((sum, dataset) => sum + (Number(dataset.size) || 0), 0)
)

const uploadPhaseText = computed(() => {
  switch (uploadPhase.value) {
    case 'uploading':
      return '上传中'
    case 'processing':
      return '后端解压中'
    case 'success':
      return '已完成'
    case 'error':
      return '失败'
    default:
      return '等待上传'
  }
})

const canConfirmUpload = computed(() => {
  const hasOwnerRole = uploadRoleOptions.value.length === 1 || Boolean(uploadOwnerRoleId.value)
  return Boolean(uploadDatasetName.value.trim() && selectedUploadFile.value && hasOwnerRole && !uploading.value)
})

const filteredMergeCandidates = computed(() => {
  const keyword = mergeKeyword.value.trim().toLowerCase()
  return mergeCandidates.value.filter(dataset => !keyword || datasetTitle(dataset).toLowerCase().includes(keyword))
})

const mergeSelectedImageCount = computed(() => {
  const selectedIds = new Set(mergeSourceIds.value)
  return mergeCandidates.value.reduce(
    (sum, dataset) => sum + (selectedIds.has(dataset.id) ? datasetImageCount(dataset) : 0),
    0
  )
})

const canConfirmMerge = computed(() => {
  const hasOwnerRole = uploadRoleOptions.value.length === 1 || Boolean(mergeOwnerRoleId.value)
  return Boolean(
    mergeDatasetName.value.trim()
    && mergeSourceIds.value.length >= 2
    && mergeSourceIds.value.length <= 50
    && hasOwnerRole
    && !merging.value
    && !mergeCandidateLoading.value
  )
})

// --- 业务方法 ---

const handleSearch = () => {
  currentPage.value = 1
}

const openUploadDialog = () => {
  if (!uploading.value) {
    uploadDatasetName.value = ''
    selectedUploadFile.value = null
    uploadOwnerRoleId.value = uploadRoleOptions.value.length === 1 ? uploadRoleOptions.value[0].id : null
  }
  showUploadDialog.value = true
}

const handleUploadFileChange = (uploadFile) => {
  const file = uploadFile?.raw
  if (!file || !beforeUploadDataset(file)) return
  selectedUploadFile.value = file
  if (!uploadDatasetName.value.trim()) {
    uploadDatasetName.value = file.name.replace(/\.zip$/i, '')
  }
}

const confirmDatasetUpload = async () => {
  if (!canConfirmUpload.value) return
  await handleUploadRequest({ file: selectedUploadFile.value })
}

const loadMergeCandidates = async () => {
  mergeCandidateLoading.value = true
  try {
    const candidates = []
    let page = 1
    let expectedTotal = 0
    do {
      const data = await requestDatasets({ page, page_size: 100 })
      const pageItems = data.datasets || []
      candidates.push(...pageItems)
      expectedTotal = Number(data.total || data.total_datasets || candidates.length)
      if (!pageItems.length || candidates.length >= expectedTotal) break
      page += 1
    } while (page <= 100)
    mergeCandidates.value = candidates
  } catch (error) {
    mergeCandidates.value = []
    ElMessage.error(`加载可合并数据集失败：${error?.message || '未知错误'}`)
  } finally {
    mergeCandidateLoading.value = false
  }
}

const openMergeDialog = async () => {
  mergeDatasetName.value = ''
  mergeSourceIds.value = []
  mergeKeyword.value = ''
  mergeOwnerRoleId.value = uploadRoleOptions.value.length === 1 ? uploadRoleOptions.value[0].id : null
  showMergeDialog.value = true
  await loadMergeCandidates()
}

/**
 * 解析 API 基础地址。
 *
 * 后端 yolo_api.py 暴露的真实接口是 /upload/dataset、/datasets 等。
 * 浏览器访问前端页面时应通过前端 Nginx/Vite 代理访问，即 /api/upload/dataset，
 * 不能直接请求 http://localhost:8000，否则容易被浏览器跨域策略拦截并触发 xhr.onerror。
 */
const getApiBaseUrl = () => {
  const envBaseUrl = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL
  if (envBaseUrl) {
    return String(envBaseUrl).replace(/\/$/, '')
  }

  // 图像训推实验平台前端的 nginx.conf / vite 代理通常都会把 /api 转发到后端。
  return '/api'
}

const buildApiUrl = (path) => {
  const baseUrl = getApiBaseUrl()
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${baseUrl}${normalizedPath}`
}

const getAuthToken = () => {
  if (typeof localStorage === 'undefined') return ''
  return (
    localStorage.getItem('token') ||
    localStorage.getItem('access_token') ||
    localStorage.getItem('auth_token') ||
    ''
  )
}

const getAuthHeaderValue = () => {
  const token = getAuthToken()
  return token ? `Bearer ${token}` : ''
}

const applyAuthHeaderToXhr = (xhr) => {
  const authorization = getAuthHeaderValue()
  if (authorization) {
    xhr.setRequestHeader('Authorization', authorization)
  }
}

const fetchWithAuth = (url, options = {}) => {
  const headers = new Headers(options.headers || {})
  const authorization = getAuthHeaderValue()

  if (authorization && !headers.has('Authorization')) {
    headers.set('Authorization', authorization)
  }

  return fetch(url, {
    ...options,
    headers
  })
}

const parseApiResponse = async (response) => {
  const text = await response.text()
  let data = null

  if (text) {
    try {
      data = JSON.parse(text)
    } catch (error) {
      data = { message: text }
    }
  }

  if (!response.ok) {
    const detail = data?.detail || data?.message || `HTTP ${response.status}`
    throw new Error(detail)
  }

  return data || {}
}

const requestDatasets = async ({ page = 1, page_size = 20 } = {}) => {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(page_size)
  })

  const response = await fetchWithAuth(buildApiUrl(`/datasets?${params.toString()}`), {
    method: 'GET'
  })

  return parseApiResponse(response)
}

const requestDatasetImages = async (datasetId, { page = 1, page_size = 10 } = {}) => {
  const params = new URLSearchParams({ page: String(page), page_size: String(page_size) })
  const response = await fetchWithAuth(buildApiUrl(`/datasets/${encodeURIComponent(datasetId)}/images?${params.toString()}`), {
    method: 'GET'
  })
  return parseApiResponse(response)
}

const requestDeleteDataset = async (datasetId) => {
  const response = await fetchWithAuth(buildApiUrl(`/datasets/${encodeURIComponent(datasetId)}`), {
    method: 'DELETE'
  })

  return parseApiResponse(response)
}

const requestMergeDatasets = async () => {
  const response = await fetchWithAuth(buildApiUrl('/datasets/merge'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_name: mergeDatasetName.value.trim(),
      source_dataset_ids: mergeSourceIds.value,
      owner_role_id: mergeOwnerRoleId.value
    })
  })
  return parseApiResponse(response)
}

const requestStartDatasetAugmentation = async (datasetId) => {
  const response = await fetchWithAuth(buildApiUrl(`/datasets/${encodeURIComponent(datasetId)}/augmentation`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...augmentationConfig,
      create_new_dataset: true,
      output_name: `${datasetTitle(augmentationDataset.value)}_增强`
    })
  })
  return parseApiResponse(response)
}

const requestJobStatus = async (taskId) => {
  const response = await fetchWithAuth(buildApiUrl(`/jobs/${encodeURIComponent(taskId)}`), { method: 'GET' })
  return parseApiResponse(response)
}

const requestCancelJob = async (taskId) => {
  const response = await fetchWithAuth(buildApiUrl(`/jobs/${encodeURIComponent(taskId)}/cancel`), { method: 'POST' })
  return parseApiResponse(response)
}

const resetAugmentationConfig = () => {
  augmentationConfig.rotations = []
  augmentationConfig.mirror = 'none'
  augmentationConfig.brightnessLevels = []
  augmentationConfig.grayscaleLevels = []
  augmentationConfig.contrastLevels = []
  augmentationPreview.value = []
  augmentationTaskId.value = ''
  augmentationTaskStatus.value = ''
  augmentationTaskProgress.value = 0
  augmentationTaskMessage.value = ''
}

const openDatasetAugmentation = (dataset) => {
  if (!dataset?.can_delete || isApplyingAugmentation.value) return
  clearAugmentationPoll()
  resetAugmentationConfig()
  augmentationDataset.value = dataset
  showAugmentationDialog.value = true
}

const fetchImageAsDataUrl = async (source) => {
  const url = /^(https?:|data:|blob:)/i.test(source) || source.startsWith('/api/') ? source : buildApiUrl(source)
  const response = await fetchWithAuth(url)
  if (!response.ok) throw new Error(`预览图片加载失败（HTTP ${response.status}）`)
  const blob = await response.blob()
  return await new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = () => reject(new Error('预览图片读取失败'))
    reader.readAsDataURL(blob)
  })
}

const getAugmentationPreviewSource = async () => {
  if (augmentationDataset.value?._coverPreviewUrl) return augmentationDataset.value._coverPreviewUrl
  const datasetId = augmentationDataset.value?.id || augmentationDataset.value?.name
  const data = await requestDatasetImages(datasetId, { page: 1, page_size: 1 })
  const image = data?.images?.[0]
  const source = image?.thumbnail_url || image?.fetch_url || image?.url
  if (!source) throw new Error('数据集没有可预览图片')
  return await fetchImageAsDataUrl(source)
}

const generateAugmentationPreview = async () => {
  const operations = buildAugmentationOperations(augmentationConfig)
  if (!operations.length) {
    ElMessage.warning('请至少选择一种增强操作')
    return
  }
  isGeneratingPreview.value = true
  try {
    const sourceImageUrl = await getAugmentationPreviewSource()
    augmentationPreview.value = await Promise.all(operations.map(operation => applyImageAugmentation(sourceImageUrl, operation)))
    ElMessage.success(`预览生成完成，共 ${augmentationPreview.value.length} 张`)
  } catch (error) {
    ElMessage.error(`预览生成失败：${error?.message || '未知错误'}`)
  } finally {
    isGeneratingPreview.value = false
  }
}

const clearAugmentationPoll = () => {
  if (augmentationPollTimer) {
    clearTimeout(augmentationPollTimer)
    augmentationPollTimer = null
  }
}

const finishAugmentationTask = async (task) => {
  clearAugmentationPoll()
  isApplyingAugmentation.value = false
  augmentationTaskStatus.value = task.status || ''
  augmentationTaskProgress.value = Number(task.progress) || 0
  augmentationTaskMessage.value = task.message || ''
  if (task.status === 'completed') {
    try {
      await loadDatasets()
    } catch (error) {
      ElMessage.warning(`数据增强已完成，但列表刷新失败：${error?.message || '请手动刷新页面'}`)
    }
    showAugmentationDialog.value = false
    ElMessage.success(`数据增强完成，已生成新数据集：${task.result?.dataset_info?.name || task.result?.dataset_id || '增强数据集'}；原始数据集未修改`)
  } else if (task.status === 'cancelled') {
    ElMessage.info('数据增强任务已取消，原始图片未受影响')
  } else if (task.status === 'failed') {
    ElMessage.error(`数据增强失败：${task.error_detail || task.message || '未知错误'}`)
  }
}

const pollAugmentationTask = async () => {
  if (!augmentationTaskId.value) return
  try {
    const task = await requestJobStatus(augmentationTaskId.value)
    augmentationTaskStatus.value = task.status || ''
    augmentationTaskProgress.value = Number(task.progress) || 0
    augmentationTaskMessage.value = task.message || ''
    if (['completed', 'failed', 'cancelled'].includes(task.status)) {
      await finishAugmentationTask(task)
      return
    }
    augmentationPollTimer = setTimeout(pollAugmentationTask, 1000)
  } catch (error) {
    augmentationTaskMessage.value = `任务状态读取失败：${error?.message || '网络错误'}`
    augmentationPollTimer = setTimeout(pollAugmentationTask, 2000)
  }
}

const applyDatasetAugmentation = async () => {
  const operations = buildAugmentationOperations(augmentationConfig)
  const datasetId = augmentationDataset.value?.id || augmentationDataset.value?.name
  if (!datasetId || !operations.length) {
    ElMessage.warning('请至少选择一种增强操作')
    return
  }
  isApplyingAugmentation.value = true
  augmentationTaskStatus.value = 'pending'
  augmentationTaskProgress.value = 0
  augmentationTaskMessage.value = '正在创建整库增强任务...'
  try {
    const task = await requestStartDatasetAugmentation(datasetId)
    augmentationTaskId.value = task.task_id || task.id
    augmentationTaskStatus.value = task.status || 'pending'
    augmentationTaskMessage.value = task.message || '任务已排队'
    await pollAugmentationTask()
  } catch (error) {
    isApplyingAugmentation.value = false
    augmentationTaskStatus.value = 'failed'
    augmentationTaskMessage.value = error?.message || '任务创建失败'
    ElMessage.error(`数据增强任务创建失败：${augmentationTaskMessage.value}`)
  }
}

const cancelDatasetAugmentation = async () => {
  if (!augmentationTaskId.value) return
  try {
    await requestCancelJob(augmentationTaskId.value)
    augmentationTaskMessage.value = '正在取消任务...'
  } catch (error) {
    ElMessage.error(`取消增强任务失败：${error?.message || '未知错误'}`)
  }
}

const applyImageAugmentation = async (imageUrl, transformParams) => {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      const sourceWidth = img.naturalWidth || img.width
      const sourceHeight = img.naturalHeight || img.height
      const dimensions = getAugmentedImageDimensions(sourceWidth, sourceHeight, transformParams.type === 'rotation' ? transformParams.value : 0)
      canvas.width = dimensions.width
      canvas.height = dimensions.height
      ctx.save()
      const filterParts = []
      if (transformParams.type === 'brightness') filterParts.push(`brightness(${transformParams.value})`)
      if (transformParams.type === 'grayscale') filterParts.push(`grayscale(${transformParams.value})`)
      if (transformParams.type === 'contrast') filterParts.push(`contrast(${transformParams.value})`)
      ctx.filter = filterParts.join(' ') || 'none'
      ctx.translate(dimensions.width / 2, dimensions.height / 2)
      if (transformParams.type === 'rotation') ctx.rotate(transformParams.value * Math.PI / 180)
      ctx.scale(
        transformParams.type === 'mirror' && transformParams.value === 'horizontal' ? -1 : 1,
        transformParams.type === 'mirror' && transformParams.value === 'vertical' ? -1 : 1
      )
      ctx.drawImage(img, -sourceWidth / 2, -sourceHeight / 2, sourceWidth, sourceHeight)
      ctx.restore()
      resolve(canvas.toDataURL('image/png'))
    }
    img.onerror = () => reject(new Error('预览图片解码失败'))
    img.src = imageUrl
  })
}

const confirmDatasetMerge = async () => {
  if (!canConfirmMerge.value) return
  merging.value = true
  try {
    const result = await requestMergeDatasets()
    showMergeDialog.value = false
    currentPage.value = 1
    await loadDatasets()
    ElMessage.success(`数据集合并完成：${result?.dataset_info?.name || mergeDatasetName.value}`)
  } catch (error) {
    ElMessage.error(`数据集合并失败：${error?.message || '未知错误'}`)
  } finally {
    merging.value = false
  }
}

const stopProcessingTimer = () => {
  if (processingTimer) {
    clearInterval(processingTimer)
    processingTimer = null
  }
}

const startProcessingTimer = () => {
  stopProcessingTimer()
  processingSeconds.value = 0
  processingTimer = setInterval(() => {
    processingSeconds.value += 1
    uploadProgressText.value = `文件已上传完成，后端正在解压、校验并统计数据集...已等待 ${processingSeconds.value} 秒`
  }, 1000)
}

/**
 * 使用 XMLHttpRequest 上传数据集。
 * 这里不用原来的 axios 封装，原因是：
 * 1. XMLHttpRequest.upload.onprogress 是浏览器真实上传进度；
 * 2. xhr.timeout = 0 表示不设置前端请求超时；
 * 3. 上传完成后仍会等待后端解压、校验和统计完成。
 * @param {File} file
 */
const uploadDatasetWithRealProgress = (file, ownerRoleId) => {
  return new Promise((resolve, reject) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('dataset_name', uploadDatasetName.value.trim())
    if (ownerRoleId) formData.append('owner_role_id', String(ownerRoleId))

    const xhr = new XMLHttpRequest()
    currentUploadXhr = xhr

    // 关键点：只使用浏览器原生 XHR，不走项目中的 axios 封装，因此不会触发 axios 的 30000ms timeout。
    // 同时 timeout = 0 表示浏览器端不主动超时；上传完成后连接会继续等待后端解压、校验和统计完成。
    xhr.open('POST', buildApiUrl('/upload/dataset'), true)
    applyAuthHeaderToXhr(xhr)
    xhr.timeout = 0
    xhr.responseType = 'text'

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) {
        uploadPhase.value = 'uploading'
        uploadProgressText.value = '浏览器正在上传文件，暂时无法计算总大小'
        return
      }

      uploadLoaded.value = event.loaded
      uploadTotal.value = event.total || file.size

      const percent = Math.floor((event.loaded / event.total) * 100)
      uploadProgress.value = Math.max(0, Math.min(99, percent))
      uploadPhase.value = 'uploading'
      uploadProgressText.value = `正在上传：${uploadProgress.value}%`
    }

    xhr.upload.onload = () => {
      uploadLoaded.value = uploadTotal.value || file.size
      uploadTotal.value = uploadTotal.value || file.size
      uploadProgress.value = 100
      uploadPhase.value = 'processing'
      uploadStatus.value = ''
      uploadProgressText.value = '文件已上传完成，后端正在解压、校验并统计数据集...'
      startProcessingTimer()
    }

    xhr.onload = () => {
      currentUploadXhr = null
      stopProcessingTimer()

      let response = null
      try {
        response = xhr.responseText ? JSON.parse(xhr.responseText) : null
      } catch (error) {
        reject(new Error(`后端返回内容不是有效 JSON：${xhr.responseText || '空响应'}`))
        return
      }

      if (xhr.status >= 200 && xhr.status < 300) {
        uploadLoaded.value = uploadTotal.value || file.size
        uploadTotal.value = uploadTotal.value || file.size
        uploadProgress.value = 100
        resolve(response)
        return
      }

      let detail = response?.detail || response?.message || `HTTP ${xhr.status}`
      if (xhr.status === 413) {
        detail = '上传文件过大，已被前端代理或后端拒绝。请检查 nginx client_max_body_size / 后端上传限制配置。'
      }
      reject(new Error(detail))
    }

    xhr.onerror = () => {
      currentUploadXhr = null
      stopProcessingTimer()
      reject(new Error('网络错误：无法连接后端上传接口。请确认 /api 已转发到后端，并且后端服务正在运行。'))
    }

    xhr.onabort = () => {
      currentUploadXhr = null
      stopProcessingTimer()
      reject(new Error('上传已取消'))
    }

    xhr.ontimeout = () => {
      currentUploadXhr = null
      stopProcessingTimer()
      reject(new Error('上传请求超时。当前页面已关闭浏览器端超时，如果仍看到该错误，请检查是否仍在使用旧版前端或 axios 上传封装。'))
    }

    xhr.send(formData)
  })
}

/**
 * 上传前校验。这里只做校验，不在 before-upload 里执行异步上传。
 * 原因：如果 before-upload 写成 async，Element Plus 会等待 Promise，容易和默认上传生命周期混在一起。
 */
const beforeUploadDataset = (file) => {
  if (uploading.value) {
    ElMessage.warning('已有数据集正在上传，请等待当前上传完成')
    return false
  }

  if (!file.name.toLowerCase().endsWith('.zip')) {
    ElMessage.error('只支持上传 ZIP 格式的数据集')
    return false
  }

  return true
}

/**
 * Element Plus 自定义上传入口。
 * 通过 :http-request 接管上传，彻底绕开默认上传和项目 axios 封装，避免 30000ms timeout。
 */
const handleUploadRequest = async (options) => {
  const file = options.file
  lastUploadFile.value = file

  uploadProgress.value = 0
  uploadStatus.value = ''
  uploading.value = true
  uploadFileName.value = file.name
  uploadLoaded.value = 0
  uploadTotal.value = file.size
  uploadPhase.value = 'uploading'
  uploadProgressText.value = '准备上传数据集...'
  processingSeconds.value = 0
  stopProcessingTimer()

  try {
    const result = await uploadDatasetWithRealProgress(file, uploadOwnerRoleId.value)

    uploadPhase.value = 'processing'
    uploadProgressText.value = '后端处理完成，正在刷新数据集列表...'

    // 必须等后端 /upload/dataset 返回成功后再刷新。
    // 此时 ZIP 已经完成解压、结构修复和 metadata 统计，页面拿到的列表是真实状态。
    await loadDatasets()

    uploadStatus.value = 'success'
    uploadPhase.value = 'success'
    uploadProgress.value = 100
    uploadProgressText.value = `数据集上传完成：${result?.dataset_info?.name || file.name}`
    ElMessage.success('数据集上传成功，列表已刷新')
    options.onSuccess?.(result)

    setTimeout(() => {
      uploadProgress.value = 0
      uploadStatus.value = ''
      uploadFileName.value = ''
      uploadLoaded.value = 0
      uploadTotal.value = 0
      uploadPhase.value = 'idle'
      uploadProgressText.value = ''
      processingSeconds.value = 0
    }, 2500)

    return true

  } catch (error) {
    console.error('数据集上传失败:', error)
    stopProcessingTimer()
    uploadStatus.value = 'exception'
    uploadPhase.value = 'error'
    uploadProgressText.value = error?.message || '数据集上传失败'
    ElMessage.error(`数据集上传失败：${error?.message || '未知错误'}`)
    options.onError?.(error)
    return false
  } finally {
    stopProcessingTimer()
    uploading.value = false
  }
}

const cancelCurrentUpload = () => {
  currentUploadXhr?.abort()
}

const retryLastUpload = () => {
  if (!lastUploadFile.value || uploading.value) return
  handleUploadRequest({ file: lastUploadFile.value })
}

/**
 * 从后端加载所有数据集
 */
const currentPage = ref(1)
const pageSize = ref(20)
const total = ref(0)

const loadDatasets = async () => {
  loading.value = true

  try {
    const data = await requestDatasets({
      page: currentPage.value,
      page_size: pageSize.value
    })

    const datasets = data.datasets || []
    clearDatasetCoverUrls()
    datasetList.value = datasets
    total.value = data.total || data.total_datasets || datasetList.value.length
    await hydrateDatasetCovers(datasets)
  } catch (error) {
    console.error('加载数据集列表失败:', error)

    if (String(error.message).includes('timeout')) {
      ElMessage.error('加载数据集超时：请检查后端 /datasets 是否仍在实时扫描目录')
    } else {
      ElMessage.error('加载数据集列表失败')
    }
  } finally {
    loading.value = false
  }
}

const handlePageChange = (page) => {
  currentPage.value = page
  loadDatasets()
}

const handleSizeChange = (size) => {
  pageSize.value = size
  currentPage.value = 1
  loadDatasets()
}

/**
 * 查看数据集详情
 * @param {Object} dataset 
 */
const loadDatasetImages = async () => {
  if (!selectedDataset.value) return
  detailLoading.value = true
  detailError.value = ''

  try {
    const data = await requestDatasetImages(selectedDataset.value.id || selectedDataset.value.name, {
      page: detailPage.value,
      page_size: detailPageSize.value
    })
    const images = data.images || []
    clearDetailPreviewUrls()
    await hydrateImagePreviews(images)
    detailImages.value = images
    detailTotal.value = Number(data.total) || detailImages.value.length
  } catch (error) {
    detailImages.value = []
    detailTotal.value = 0
    detailError.value = String(error?.message || '').includes('资产索引')
      ? '该数据集尚未建立图片资产索引，请先在数据标注页刷新索引。'
      : `图片明细加载失败：${error?.message || '未知错误'}`
  } finally {
    detailLoading.value = false
  }
}

const clearDetailPreviewUrls = () => {
  detailObjectUrls.forEach(url => URL.revokeObjectURL(url))
  detailObjectUrls = []
}

const clearDatasetCoverUrls = () => {
  datasetCoverGeneration += 1
  datasetCoverObjectUrls.forEach(url => URL.revokeObjectURL(url))
  datasetCoverObjectUrls = []
}

const hydrateDatasetCovers = async (datasets) => {
  const generation = datasetCoverGeneration
  await Promise.all(datasets.map(async (dataset) => {
    const source = dataset?.cover_url
    if (!source) return
    const url = /^(https?:)/i.test(source) || source.startsWith('/api/') ? source : buildApiUrl(source)

    try {
      const response = await fetchWithAuth(url)
      if (!response.ok) return
      const objectUrl = URL.createObjectURL(await response.blob())
      if (generation !== datasetCoverGeneration) {
        URL.revokeObjectURL(objectUrl)
        return
      }
      datasetCoverObjectUrls.push(objectUrl)
      dataset._coverPreviewUrl = objectUrl
    } catch (error) {
      console.warn('数据集预览图加载失败:', datasetTitle(dataset), error)
    }
  }))
}

const hydrateImagePreviews = async (images) => {
  await Promise.all(images.map(async (image) => {
    const source = image?.thumbnail_url || image?.fetch_url || image?.url
    if (!source) return
    const url = /^(https?:)/i.test(source) || source.startsWith('/api/') ? source : buildApiUrl(source)

    try {
      const response = await fetchWithAuth(url)
      if (!response.ok) return
      const objectUrl = URL.createObjectURL(await response.blob())
      detailObjectUrls.push(objectUrl)
      image.preview_url = objectUrl
    } catch (error) {
      console.warn('缩略图加载失败:', image?.filename || image?.name, error)
    }
  }))
}

const handleDetailSizeChange = () => {
  detailPage.value = 1
  loadDatasetImages()
}

const viewDataset = (dataset) => {
  selectedDataset.value = dataset
  detailPage.value = 1
  detailImages.value = []
  detailTotal.value = 0
  detailError.value = ''
  showDetailDialog.value = true
  loadDatasetImages()
}

const imageThumbnailUrl = (image) => {
  const value = image?.preview_url || ''
  if (!value) return ''
  if (/^(https?:|data:|blob:)/i.test(value)) return value
  if (value.startsWith('/api/')) return value
  return buildApiUrl(value)
}

const notifyImageDeleteUnavailable = () => {
  ElMessage.warning('当前后端未提供数据集图片删除接口，请在数据标注页管理图片')
}

const staticAnnotationModels = [
  {
    path: 'yoloe-26m-seg.pt',
    name: 'yoloe-26m-seg.pt',
    display_name: 'YOLOE-26M-Seg',
    geometry: 'polygon',
    description: '分割模型，使用多边形标注'
  },
  {
    path: 'yolo26m.pt',
    name: 'yolo26m.pt',
    display_name: 'YOLO26M',
    geometry: 'rect',
    description: '检测模型，使用矩形框标注'
  }
]

const normalizeModelReference = (value) => {
  const normalized = String(value || '').trim().replace(/\\/g, '/').toLowerCase()
  if (!normalized) return ''
  const basename = normalized.split('/').filter(Boolean).pop() || normalized
  return basename.replace(/\.(pt|onnx|engine|pth)$/i, '')
}

const normalizeModelPath = (value) => String(value || '').trim().replace(/\\/g, '/').toLowerCase()
const staticAnnotationModelIdentities = new Set(
  staticAnnotationModels.map(model => normalizeModelReference(model.path)).filter(Boolean)
)

const modelSelectionValue = (model) => model?.path || model?.name || model?.display_name || ''

/**
 * API records use container paths while the built-in fallbacks use bare names.
 * Only canonicalize known annotation-model aliases; distinct trained `best.pt`
 * files must continue to remain separate choices.
 */
const modelIdentity = (model = {}) => {
  const references = [model.path, model.name, model.display_name]
    .map(normalizeModelReference)
    .filter(Boolean)
  const knownIdentity = references.find(reference => staticAnnotationModelIdentities.has(reference))
  if (knownIdentity) return `annotation:${knownIdentity}`
  const rawReference = [model.path, model.name, model.display_name]
    .map(normalizeModelPath)
    .find(Boolean)
  return rawReference ? `path:${rawReference}` : ''
}

const modelGeometry = (model) => {
  return modelAnnotationGeometry(model)
}

const mergeAnnotationModels = (models = []) => {
  const merged = [...(Array.isArray(models) ? models : []), ...staticAnnotationModels]
  const seen = new Set()
  annotationModels.value = merged.filter(model => {
    const key = modelIdentity(model)
    if (!key || seen.has(key)) return false
    seen.add(key)
    return true
  }).map(model => ({ ...model, geometry: modelGeometry(model) }))
}

const loadAnnotationModels = async () => {
  annotationModelLoading.value = true
  try {
    const response = await getModels()
    mergeAnnotationModels(response?.models)
  } catch (error) {
    console.warn('加载标注模型列表失败，使用内置模型列表', error)
    mergeAnnotationModels([])
  } finally {
    annotationModelLoading.value = false
  }
}

const openAnnotationModelDialog = async (dataset, kind = '') => {
  pendingAnnotationDataset.value = dataset
  annotationSelectionKind.value = kind
  selectedAnnotationModel.value = ''
  showAnnotationModelDialog.value = true
  await loadAnnotationModels()
  const bound = dataset?.annotation_model || (dataset?.annotation_model_path ? {
    path: dataset.annotation_model_path,
    name: dataset.annotation_model_name,
    geometry: dataset.annotation_geometry
  } : null)
  const boundPath = String(bound?.path || '').trim()
  const boundIdentity = modelIdentity(bound || {})
  let boundModel = boundIdentity
    ? annotationModels.value.find(item => modelIdentity(item) === boundIdentity)
    : null
  if (boundPath && !boundModel) {
    boundModel = {
      path: boundPath,
      name: bound.name || boundPath,
      display_name: bound.name || boundPath,
      geometry: bound.geometry || 'rect',
      description: '最近一次使用的标注模型，可在本次进入标注时重新选择'
    }
    annotationModels.value.unshift(boundModel)
  }
  if (boundPath && boundModel && filteredAnnotationModels.value.some(item => modelIdentity(item) === modelIdentity(boundModel))) {
    selectedAnnotationModel.value = modelSelectionValue(boundModel || bound)
  } else if (!selectedAnnotationModel.value && filteredAnnotationModels.value.length > 0) {
    selectedAnnotationModel.value = modelSelectionValue(filteredAnnotationModels.value[0])
  }
}

const chooseAnnotationDraftAction = async (kind, mode) => {
  const dataset = pendingAnnotationDataset.value
  if (!dataset) return
  const datasetId = dataset.id || dataset.name
  if (mode === 'resume') {
    try {
      const response = await getAnnotationDraft(datasetId, kind)
      showAnnotationDraftDialog.value = false
      router.push({
        path: '/annotation',
        query: {
          datasetId,
          datasetName: datasetTitle(dataset),
          datasetRole: 'original',
          draftKind: kind,
          draftId: response.id,
          draftRevision: response.revision,
          draftPage: response.last_page,
          draftImagePath: response.last_image_path,
          model: response.model_path,
          modelName: response.model_name,
          annotationGeometry: response.annotation_geometry
        }
      })
    } catch (error) {
      ElMessage.error(error?.response?.data?.detail || '读取标注草稿失败')
    }
    return
  }

  pendingDraftReplacement.value = null
  if (annotationDraftStatus.value[kind]?.exists) {
    try {
      await ElMessageBox.confirm(
        '同类型已有临时草稿，继续新建会覆盖它；另一类型草稿不会受影响。',
        '覆盖确认',
        { type: 'warning', confirmButtonText: '覆盖并新建', cancelButtonText: '取消' }
      )
      const existing = await getAnnotationDraft(datasetId, kind)
      pendingDraftReplacement.value = { id: existing.id, revision: existing.revision }
    } catch (error) {
      if (error !== 'cancel' && error !== 'close') ElMessage.error(error?.response?.data?.detail || '覆盖旧草稿失败')
      return
    }
  }
  showAnnotationDraftDialog.value = false
  await openAnnotationModelDialog(dataset, kind)
}

const confirmAnnotationModel = async () => {
  const dataset = pendingAnnotationDataset.value
  const model = filteredAnnotationModels.value.find(item =>
    modelSelectionValue(item) === selectedAnnotationModel.value
  ) || annotationModels.value.find(item =>
    modelIdentity(item) === modelIdentity({ path: selectedAnnotationModel.value })
  )
  if (!dataset || !model) {
    ElMessage.warning('请选择训练模型')
    return
  }

  annotationModelSaving.value = true
  try {
    const kind = annotationSelectionKind.value || (modelSupportsTextPrompts(model) ? 'multimodal' : 'yolo')
    const response = await createAnnotationDraft(dataset.id || dataset.name, kind, {
      ...(pendingDraftReplacement.value ? {
        replace_draft_id: pendingDraftReplacement.value.id,
        replace_revision: pendingDraftReplacement.value.revision
      } : {}),
      model_path: model.path || model.name || model.display_name,
      model_name: model.display_name || model.name || model.path,
      annotation_geometry: model.geometry,
      model_source: model.source || model.type || '',
      annotation_kind: annotationSelectionKind.value || (model.supports_text_prompts === true ? 'multimodal' : 'yolo'),
      trained_class_names: model.trained_class_names || model.class_names || model.names || []
    })
    if (response?.warning) ElMessage.warning(response.warning)
    if (response?.class_context?.initialization_note) ElMessage.info(response.class_context.initialization_note)
    showAnnotationModelDialog.value = false
    router.push({
      path: '/annotation',
      query: {
        datasetId: dataset.id || dataset.name,
        datasetName: datasetTitle(dataset),
        datasetRole: 'original',
        model: response.model_path,
        modelName: response.model_name,
        annotationGeometry: response.annotation_geometry,
        draftKind: kind,
        draftId: response.id,
        draftRevision: String(response.revision)
      }
    })
  } catch (error) {
    console.error('保存数据集标注模型失败:', error)
    const detail = error?.response?.data?.detail
    ElMessage.error(typeof detail === 'object' ? detail.message : (detail || '保存标注模型失败'))
  } finally {
    annotationModelSaving.value = false
  }
}

const annotateDataset = async (dataset) => {
  pendingAnnotationDataset.value = dataset
  try {
    const response = await getAnnotationDrafts(dataset.id || dataset.name)
    annotationDraftStatus.value = response?.drafts || { multimodal: { exists: false }, yolo: { exists: false } }
    const hasDraft = Object.values(annotationDraftStatus.value).some(item => item?.exists)
    if (hasDraft) {
      showAnnotationDraftDialog.value = true
      return
    }
  } catch (error) {
    ElMessage.warning(error?.response?.data?.detail || '读取临时草稿状态失败，将进入模型选择')
  }
  openAnnotationModelDialog(dataset, '')
}

/**
 * 删除单个数据集
 * @param {Object} dataset 
 */
const deleteDataset = async (dataset) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除数据集 "${dataset.name}" 吗？此操作不可恢复。`,
      '删除确认',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    // 调用删除API
    await requestDeleteDataset(dataset.id)
    
    // 从本地列表中移除
    const index = datasetList.value.findIndex(d => d.id === dataset.id)
    if (index > -1) {
      datasetList.value.splice(index, 1)
    }
    
  } catch (error) {
    if (error !== 'cancel') {
      console.error('删除数据集失败:', error)
      ElMessage.error('删除数据集失败')
    }
  }
}

// --- 辅助工具函数 ---

/**
 * 格式化文件大小
 * @param {number} bytes 
 */
const formatFileSize = (bytes) => {
  if (!bytes || bytes === 0) return '0 B'
  
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const formatDatasetSize = (bytes) => {
  const value = Number(bytes) || 0
  if (!value) return '0'
  const megabytes = value / (1024 * 1024)
  return parseFloat(megabytes.toFixed(megabytes >= 10 ? 2 : 1)).toString()
}

/**
 * 格式化日期显示
 * @param {string|Date} date 
 */
const formatDate = (date) => {
  if (!date) return '-'
  const parsedDate = new Date(date)
  if (Number.isNaN(parsedDate.getTime())) return '-'
  const year = parsedDate.getFullYear()
  const month = String(parsedDate.getMonth() + 1).padStart(2, '0')
  const day = String(parsedDate.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

const formatDateTime = (date) => {
  if (!date) return '-'
  const parsedDate = new Date(date)
  if (Number.isNaN(parsedDate.getTime())) return '-'
  const time = [parsedDate.getHours(), parsedDate.getMinutes(), parsedDate.getSeconds()]
    .map(value => String(value).padStart(2, '0'))
    .join(':')
  return `${formatDate(parsedDate)} ${time}`
}

// --- 生命周期钩子 ---
onMounted(() => {
  loadDatasets()
})

onBeforeUnmount(() => {
  currentUploadXhr?.abort()
  currentUploadXhr = null
  stopProcessingTimer()
  clearAugmentationPoll()
  clearDatasetCoverUrls()
  clearDetailPreviewUrls()
})
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

.uploader-select {
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

.upload-button {
  min-width: 112px;
  margin-left: 10px !important;
}

.merge-button {
  min-width: 112px;
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
  margin: 0;
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

.meta-row.subtle {
  font-size: 15px;
  color: #8a94a3;
}

.meta-gap {
  margin-left: 18px;
}

.dataset-card .dataset-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 20px;
  border-top: 1px solid #edf1f6;
}

.dataset-card .dataset-actions :deep(.el-button) {
  height: 36px;
  border-radius: 4px !important;
  font-size: 14px;
  letter-spacing: 0.12em;
}

.dataset-card .dataset-actions :deep(.dataset-action-primary) {
  flex: 1 1 0;
  min-width: 0;
  margin-left: 0;
  padding-right: 10px;
  padding-left: 10px;
}

.dataset-card .dataset-actions :deep(.dataset-action-delete) {
  flex: 0 0 72px;
  min-width: 72px;
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

.dataset-detail-dialog :deep(.el-dialog) {
  min-width: 760px;
  border-radius: 6px;
}

.dataset-detail-dialog :deep(.el-dialog__body) {
  padding-top: 10px;
}

.dataset-detail-dialog :deep(.el-table th.el-table__cell) {
  background: #f8f8f9;
  color: #515a6e;
}

.detail-thumbnail {
  width: 72px;
  height: 48px;
  border-radius: 3px;
  background: #f3f5f8;
}

.detail-thumbnail-placeholder {
  width: 72px;
  height: 48px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 3px;
  background: #f3f5f8;
  color: #b7bec9;
  font-size: 12px;
}

.detail-error {
  margin-top: 14px;
}

.detail-pagination {
  display: flex;
  justify-content: flex-end;
  padding-top: 20px;
}

.dataset-upload-dialog :deep(.el-dialog) {
  border-radius: 8px;
}

.dataset-upload-dialog :deep(.el-dialog__body) {
  max-height: 72vh;
  overflow: auto;
  padding-top: 10px;
}

.dataset-merge-dialog :deep(.el-dialog) {
  border-radius: 8px;
}

.dataset-merge-dialog :deep(.el-dialog__body) {
  max-height: 76vh;
  overflow: auto;
  padding-top: 10px;
}

.merge-form {
  margin-top: 18px;
}

.merge-form :deep(.el-form-item__label) {
  color: #303133;
  font-weight: 600;
}

.merge-candidate-list {
  width: 100%;
  min-height: 160px;
  max-height: 320px;
  margin-top: 10px;
  overflow: auto;
  border: 1px solid #dcdfe6;
  border-radius: 6px;
  background: #fff;
}

.merge-candidate-list :deep(.el-checkbox-group) {
  display: block;
}

.merge-candidate-item {
  width: 100%;
  min-height: 54px;
  margin: 0 !important;
  padding: 10px 14px;
  border-bottom: 1px solid #ebeef5;
}

.merge-candidate-item:last-child {
  border-bottom: 0;
}

.merge-candidate-item :deep(.el-checkbox__label) {
  min-width: 0;
  display: inline-flex;
  flex: 1;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.merge-candidate-name {
  overflow: hidden;
  color: #303133;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.merge-candidate-meta {
  flex: none;
  color: #909399;
  font-size: 12px;
}

.merge-selection-summary {
  width: 100%;
  margin-top: 8px;
  color: #606266;
  font-size: 13px;
}

.upload-description {
  margin: 0 0 20px;
  color: #606266;
  line-height: 1.6;
}

.upload-form :deep(.el-form-item__label) {
  color: #303133;
  font-weight: 600;
}

.file-picker-row {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 14px;
}

.selected-file-name {
  min-width: 0;
  overflow: hidden;
  color: #606266;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.upload-tip {
  width: 100%;
  margin-top: 8px;
  color: #909399;
  font-size: 12px;
}

.upload-tip code {
  color: #606266;
  font-family: inherit;
}

.upload-progress-panel {
  margin-top: 20px;
  padding: 16px;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  background: #fafafa;
}

.upload-progress-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 12px;
}

.upload-file-name {
  font-weight: 600;
  color: #303133;
  word-break: break-all;
}

.upload-progress-text {
  margin-top: 4px;
  color: #606266;
  font-size: 13px;
}

.upload-progress-footer {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin-top: 8px;
  color: #909399;
  font-size: 12px;
  flex-wrap: wrap;
}

.annotation-model-list {
  min-height: 120px;
  margin-top: 18px;
}

.annotation-model-group-title {
  margin: 12px 0 8px;
  color: #303133;
  font-size: 13px;
  font-weight: 600;
}

.annotation-draft-card-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 18px;
}

.annotation-draft-card {
  cursor: pointer;
  min-height: 126px;
}

.annotation-draft-card.is-disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.annotation-draft-card-title {
  color: #303133;
  font-weight: 600;
}

.annotation-draft-card-description {
  min-height: 42px;
  margin: 10px 0;
  color: #606266;
  font-size: 13px;
  line-height: 1.5;
}

.annotation-model-radio-group {
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
  width: 100%;
}

.annotation-model-option {
  width: 100%;
  height: auto;
  margin: 0 !important;
  padding: 14px 16px;
  align-items: flex-start;
}

.annotation-model-option :deep(.el-radio__label) {
  flex: 1;
  min-width: 0;
  white-space: normal;
}

.annotation-model-option-content {
  width: 100%;
}

.annotation-model-option-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: #303133;
  font-weight: 600;
}

.annotation-model-option-description {
  margin-top: 6px;
  color: #909399;
  font-size: 13px;
}

.augmentation-config {
  padding: 10px 0;
}

.augmentation-option-group {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.augmentation-help {
  width: 100%;
  margin-top: 6px;
  color: #909399;
  font-size: 12px;
  line-height: 1.5;
}

.augmentation-preview {
  margin-top: 20px;
  padding: 15px;
  background: #f5f7fa;
  border-radius: 8px;
}

.augmentation-preview h4 {
  margin: 0 0 15px;
  color: #303133;
  font-size: 16px;
}

.preview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
}

.preview-item {
  text-align: center;
}

.preview-item img {
  max-width: 100%;
  max-height: 120px;
  border: 2px solid #e4e7ed;
  border-radius: 4px;
  object-fit: contain;
}

.augmentation-progress {
  margin-top: 18px;
}

.augmentation-task-message {
  margin-top: 8px;
  color: #606266;
  font-size: 12px;
}


.augmentation-form {
  margin-top: 18px;
}

.augmentation-name-preview {
  margin-top: 6px;
  color: #909399;
  font-size: 12px;
}

.augmentation-switch-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px 18px;
  width: 100%;
}

.augmentation-switch-item,
.augmentation-option-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: #606266;
}

.augmentation-option-row {
  margin-top: 4px;
}

.augmentation-progress-panel {
  margin-top: 18px;
  padding: 14px;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  background: #fafafa;
}

.augmentation-progress-header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
  color: #606266;
  font-size: 13px;
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
  .uploader-select,
  .toolbar-button {
    width: 100%;
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
    text-align: left;
    font-size: 20px;
  }

  .dataset-card .dataset-actions {
    gap: 10px;
    flex-wrap: wrap;
  }

  .dataset-card .dataset-actions :deep(.el-button) {
    flex: 1;
    min-width: 100px;
    margin-left: 0;
  }

  .dataset-card .dataset-actions :deep(.dataset-action-primary),
  .dataset-card .dataset-actions :deep(.dataset-action-delete) {
    flex: 1 1 calc(50% - 5px);
    min-width: 100px;
  }
}

</style>

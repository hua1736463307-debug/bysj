/**
 * 网络请求工具类
 * 封装 axios 实例，配置请求拦截、响应拦截及后端 API 接口
 */
import axios from 'axios'
import { ElMessage } from 'element-plus'

// 普通请求超时时间：30秒
const DEFAULT_TIMEOUT = 30000
const LIST_TIMEOUT = 15000

// 大文件上传超时时间：5分钟
const UPLOAD_TIMEOUT = 300000
const VIDEO_UPLOAD_TIMEOUT = 600000
const VIDEO_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
const VIDEO_UPLOAD_CHUNK_THRESHOLD = 64 * 1024 * 1024
const VIDEO_UPLOAD_SESSION_KEY = 'deepvision:video-upload-sessions'

// 创建axios实例
const api = axios.create({
  baseURL: '/api',
  timeout: DEFAULT_TIMEOUT,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 登录状态工具
// 兼容不同页面/旧代码可能读取 token、access_token、auth_token 的情况。
export const getStoredToken = () => {
  return (
    localStorage.getItem('token') ||
    localStorage.getItem('access_token') ||
    localStorage.getItem('auth_token') ||
    ''
  )
}

export const getStoredUser = () => {
  try {
    const raw = localStorage.getItem('user')
    return raw ? JSON.parse(raw) : null
  } catch {
    localStorage.removeItem('user')
    return null
  }
}

export const setAuthHeader = (token = getStoredToken()) => {
  if (token) {
    api.defaults.headers.common.Authorization = `Bearer ${token}`
  } else {
    delete api.defaults.headers.common.Authorization
  }
}

export const saveAuth = (payload) => {
  const token = payload?.token || payload?.access_token
  const user = payload?.user

  if (token) {
    // token 为主键；access_token/auth_token 用于兼容项目中可能遗留的直接读取逻辑。
    localStorage.setItem('token', token)
    localStorage.setItem('access_token', token)
    localStorage.setItem('auth_token', token)
    setAuthHeader(token)
  }
  if (user) {
    localStorage.setItem('user', JSON.stringify(user))
  }
}

export const logoutAuth = () => {
  window.dispatchEvent(new CustomEvent('deepvision:before-logout'))
  localStorage.removeItem('token')
  localStorage.removeItem('access_token')
  localStorage.removeItem('auth_token')
  localStorage.removeItem('user')
  setAuthHeader('')
}

let redirectingAfterUnauthorized = false

export const authorizedFetch = async (url, options = {}) => {
  const headers = new Headers(options.headers || {})
  const token = getStoredToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)

  const response = await fetch(url, { ...options, headers })
  if (response.status === 401 && !redirectingAfterUnauthorized) {
    redirectingAfterUnauthorized = true
    let message = '登录已过期，请重新登录'
    try {
      const payload = await response.clone().json()
      message = payload?.detail || message
    } catch {
      // 非 JSON 401 响应使用统一提示。
    }
    logoutAuth()
    if (window.location.pathname !== '/login') {
      ElMessage.error(message)
      window.location.href = '/login'
    }
  }
  return response
}

// 页面刷新后恢复默认 Authorization，避免首次进入页面时后台接口不带 token。
setAuthHeader()

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    const token = getStoredToken()
    if (token) {
      config.headers = config.headers || {}
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    if (error.config?.silent) {
      return Promise.reject(error)
    }
    const status = error.response?.status
    const detail = error.response?.data?.detail
    const message = (
      typeof detail === 'string'
        ? detail
        : detail?.message || error.message || '请求失败'
    )

    if (status === 401) {
      logoutAuth()
      if (window.location.pathname !== '/login') {
        ElMessage.error(message || '登录已过期，请重新登录')
        window.location.href = '/login'
      }
    } else {
      ElMessage.error(message)
    }

    return Promise.reject(error)
  }
)

// API方法

// 登录
export const login = (username, password) => {
  return api.post('/auth/login', {
    username,
    password
  })
}

// 获取当前登录用户
export const getCurrentUser = () => {
  return api.get('/auth/me')
}

export const logoutCurrentSession = () => {
  return api.post('/auth/logout')
}

// 管理员：获取用户列表
export const getAdminUsers = (params = {}) => {
  return api.get('/admin/users', {
    params: {
      page: params.page || 1,
      page_size: params.page_size || 20,
      search: params.search || '',
      user_name: params.user_name || '',
      phone: params.phone || '',
      status: params.status || '',
      dept_id: params.dept_id || undefined,
      start_time: params.start_time || '',
      end_time: params.end_time || ''
    }
  })
}

// 管理员：新增用户
export const createAdminUser = (data) => {
  return api.post('/admin/users', data)
}

// 管理员：更新用户角色/状态
export const updateAdminUser = (userId, data) => {
  return api.put(`/admin/users/${userId}`, data)
}

// 管理员：删除用户
export const deleteAdminUser = (userId) => {
  return api.delete(`/admin/users/${userId}`)
}

// 管理员：重置用户密码
export const resetAdminUserPassword = (userId, password) => {
  return api.post(`/admin/users/${userId}/reset-password`, { password })
}

// 管理员：按需查看已加密保存的用户明码密码
export const revealAdminUserPassword = (userId) => {
  return api.post(`/admin/users/${userId}/reveal-password`)
}

// 平台管理：部门
export const getDepartments = (params = {}) => api.get('/system/departments', { params })
export const createDepartment = (data) => api.post('/system/departments', data)
export const updateDepartment = (id, data) => api.put(`/system/departments/${id}`, data)
export const deleteDepartment = (id) => api.delete(`/system/departments/${id}`)
export const saveDepartmentSort = (items) => api.put('/system/departments/sort/batch', { items })

// 平台管理：角色
export const getRoles = (params = {}) => api.get('/system/roles', { params })
export const getRoleOptions = () => api.get('/system/roles/options')
export const createRole = (data) => api.post('/system/roles', data)
export const updateRole = (id, data) => api.put(`/system/roles/${id}`, data)
export const deleteRole = (id) => api.delete(`/system/roles/${id}`)
export const updateRoleScope = (id, data) => api.put(`/system/roles/${id}/scope`, data)
export const getRoleUsers = (id) => api.get(`/system/roles/${id}/users`)
export const assignRoleUsers = (id, userIds) => api.put(`/system/roles/${id}/users`, { userIds })

// 平台管理：菜单
export const getMenus = (params = {}) => api.get('/system/menus', { params })
export const createMenu = (data) => api.post('/system/menus', data)
export const updateMenu = (id, data) => api.put(`/system/menus/${id}`, data)
export const deleteMenu = (id) => api.delete(`/system/menus/${id}`)
export const saveMenuSort = (items) => api.put('/system/menus/sort/batch', { items })

// 平台管理：在线用户
export const getOnlineSessions = (params = {}) => api.get('/system/online-sessions', { params })
export const forceLogoutSession = (sessionId) => api.delete(`/system/online-sessions/${encodeURIComponent(sessionId)}`)

// 平台管理：登录日志
export const getLoginLogs = (params = {}) => {
  return api.get('/system/login-logs', { params })
}
export const exportLoginLogs = (params = {}) => api.get('/system/login-logs/export', { params, responseType: 'blob' })
export const getSystemFaults = (params = {}) => api.get('/system/faults', { params })

// 平台管理：操作日志
export const getOperationLogs = (params = {}) => {
  return api.get('/system/operation-logs', { params })
}
export const exportOperationLogs = (params = {}) => api.get('/system/operation-logs/export', { params, responseType: 'blob' })

export const deleteLoginLogs = (ids) => api.delete(`/system/login-logs/${ids.join(',')}`)
export const cleanLoginLogs = () => api.delete('/system/login-logs')
export const unlockLoginUser = (username) => api.put(`/system/login-logs/unlock/${encodeURIComponent(username)}`)
export const deleteOperationLogs = (ids) => api.delete(`/system/operation-logs/${ids.join(',')}`)
export const cleanOperationLogs = () => api.delete('/system/operation-logs')

// 健康检查
export const checkApiHealth = async () => {
  try {
    const response = await api.get('/health/live', { silent: true, timeout: 10000 })
    return response.status === 'alive'
  } catch {
    return false
  }
}

export const getSystemHealth = () => api.get('/health', { silent: true, timeout: 10000 })

// 获取模型列表
export const getModels = () => {
  return api.get('/models')
}

export const getRecorders = (params) => api.get('/monitoring/recorders', { params })
export const createRecorder = (body) => api.post('/monitoring/recorders', body)
export const updateRecorder = (id, body) => api.put(`/monitoring/recorders/${encodeURIComponent(id)}`, body)
export const deleteRecorder = (id) => api.delete(`/monitoring/recorders/${encodeURIComponent(id)}`)
export const connectRecorder = (id) => api.post(`/monitoring/recorders/${encodeURIComponent(id)}/connect`, {}, { timeout: 540000 })
export const getRecorderChannelInventory = (id) => api.get(`/monitoring/recorders/${encodeURIComponent(id)}/channel-inventory`)

export const convertModelToOnnx = (modelId) => api.post(
  `/models/${encodeURIComponent(modelId)}/convert-onnx`,
  {},
  { timeout: 15000 }
)

// 获取任务列表
export const getTasks = () => {
  return api.get('/tasks')
}

export const getJobs = (params = {}) => api.get('/jobs', { params })
export const getJob = (taskId) => api.get(`/jobs/${encodeURIComponent(taskId)}`)
export const getJobEvents = (taskId, afterSequence = 0, limit = 200) => api.get(
  `/jobs/${encodeURIComponent(taskId)}/events`,
  { params: { after_sequence: afterSequence, limit } }
)
export const cancelJob = (taskId) => api.post(`/jobs/${encodeURIComponent(taskId)}/cancel`)

// 获取系统资源状态
export const getSystemResources = () => {
  return api.get('/system/resources', {
    timeout: 10000,
    params: {
      t: Date.now()
    }
  })
}

// 获取任务状态
export const getTaskStatus = (taskId) => {
  return getJob(taskId)
}

export const getTrainingTaskStatus = (taskId) => api.get(`/status/${encodeURIComponent(taskId)}`)

export const stopTraining = (taskId) => {
  return cancelJob(taskId)
}

export const startBatchAnnotation = (payload) => api.post('/annotation/ai-assist-batch/start', payload)
export const getBatchAnnotationStatus = (taskId) => api.get(`/annotation/ai-assist-batch/status/${encodeURIComponent(taskId)}`)
export const getBatchAnnotationResults = (taskId, page = 1, pageSize = 500) => api.get(
  `/annotation/ai-assist-batch/results/${encodeURIComponent(taskId)}`,
  { params: { page, page_size: pageSize } }
)
export const cancelBatchAnnotation = (taskId) => api.post(`/annotation/ai-assist-batch/cancel/${encodeURIComponent(taskId)}`)

// 获取数据集列表
export const getDatasets = (params = {}) => {
  return api.get('/datasets', {
    params: {
      page: params.page || 1,
      page_size: params.page_size || 20,
      t: Date.now()
    },
    timeout: LIST_TIMEOUT
  })
}

export const getDatasetAnnotationClasses = (datasetId) => api.get(
  `/datasets/${encodeURIComponent(datasetId)}/annotation-classes`
)

export const updateDatasetAnnotationClasses = (datasetId, classes, source = 'user', bindings = null, labelsZh = null, expectedClasses = null, retireClassIds = null, expectedInactiveClassIds = null) => {
  const payload = { classes, source }
  if (Array.isArray(bindings)) payload.annotation_class_bindings = bindings
  if (labelsZh && typeof labelsZh === 'object') payload.labels_zh = labelsZh
  if (Array.isArray(expectedClasses)) payload.expected_classes = expectedClasses
  if (Array.isArray(retireClassIds)) payload.retire_class_ids = retireClassIds
  if (Array.isArray(expectedInactiveClassIds)) payload.expected_inactive_class_ids = expectedInactiveClassIds
  return api.put(`/datasets/${encodeURIComponent(datasetId)}/annotation-classes`, payload)
}

// 获取训练数据集列表
export const saveDatasetAnnotationModel = (datasetId, payload) => {
  return api.post(`/datasets/${encodeURIComponent(datasetId)}/annotation-model`, payload)
}

export const getAnnotationDrafts = (datasetId) => api.get(
  `/datasets/${encodeURIComponent(datasetId)}/annotation-drafts`,
  { silent: true }
)

export const getAnnotationDraft = (datasetId, kind, draftId = '') => api.get(
  `/datasets/${encodeURIComponent(datasetId)}/annotation-drafts/${encodeURIComponent(kind)}`,
  { silent: true, params: draftId ? { draft_id: draftId } : {} }
)

export const putAnnotationDraft = (datasetId, kind, payload, options = {}) => api.put(
  `/datasets/${encodeURIComponent(datasetId)}/annotation-drafts/${encodeURIComponent(kind)}`,
  payload,
  { timeout: 30000, silent: true, ...options }
)

export const createAnnotationDraft = (datasetId, kind, payload) => api.post(
  `/datasets/${encodeURIComponent(datasetId)}/annotation-drafts/${encodeURIComponent(kind)}`,
  payload,
  { timeout: 60000, silent: true }
)

export const deleteAnnotationDraft = (datasetId, kind) => api.delete(
  `/datasets/${encodeURIComponent(datasetId)}/annotation-drafts/${encodeURIComponent(kind)}`,
  { silent: true }
)

export const getTrainingDatasets = (params = {}) => {
  return api.get('/training-datasets', {
    params: {
      page: params.page || 1,
      page_size: params.page_size || 20,
      name: params.name || undefined,
      creator: params.creator || undefined,
      t: Date.now()
    },
    timeout: LIST_TIMEOUT
  })
}

// 获取原始数据集图片列表
export const getDatasetImages = (datasetId, params = {}) => {
  return api.get(`/datasets/${encodeURIComponent(datasetId)}/images`, {
    params: {
      page: params.page || 1,
      page_size: params.page_size || 100,
      include_dimensions: Boolean(params.include_dimensions),
      dataset_role: params.dataset_role || params.datasetRole || undefined,
      draft_kind: params.draft_kind || params.draftKind || undefined,
      t: Date.now()
    },
    timeout: LIST_TIMEOUT
  })
}

// 上传数据集
export const uploadDataset = (file, onProgress) => {
  const formData = new FormData()
  formData.append('file', file)

  return api.post('/upload/dataset', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    },

    // 关键修改：
    // 原来数据集上传使用全局 30 秒超时，大文件容易上传失败。
    // 这里单独设置为 5 分钟。
    timeout: UPLOAD_TIMEOUT,

    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        )
        onProgress(progress)
      }
    }
  })
}

// 上传训练数据集
export const uploadTrainingDataset = (file, onProgress, ownerRoleId = null) => {
  const formData = new FormData()
  formData.append('file', file)
  if (ownerRoleId) formData.append('owner_role_id', String(ownerRoleId))

  return api.post('/upload/training-dataset', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    timeout: UPLOAD_TIMEOUT,
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        )
        onProgress(progress)
      }
    }
  })
}

// 将标注结果保存为训练数据集
export const exportAnnotationTrainingDataset = (payload) => {
  return api.post('/annotation/export-training-dataset', payload, {
    timeout: UPLOAD_TIMEOUT
  })
}

export const startAnnotationTrainingDatasetExport = (payload) => {
  return api.post('/annotation/export-training-dataset/start', payload, {
    timeout: UPLOAD_TIMEOUT
  })
}

export const preflightAnnotationTrainingDatasetExport = (payload) => {
  return api.post('/annotation/export-training-dataset/preflight', payload, {
    timeout: UPLOAD_TIMEOUT
  })
}

export const startDatasetAugmentation = (datasetId, payload) => {
  return api.post(`/datasets/${encodeURIComponent(datasetId)}/augmentation`, payload, {
    timeout: UPLOAD_TIMEOUT
  })
}

export const saveAnnotationVersion = (datasetId, payload) => {
  return api.post(`/datasets/${encodeURIComponent(datasetId)}/annotation-versions`, payload, {
    timeout: UPLOAD_TIMEOUT
  })
}

export const deriveDatasetClasses = (versionId, classNames, dryRun = false) => {
  return api.post(`/dataset-versions/${encodeURIComponent(versionId)}/derive-classes`, {
    class_names: classNames,
    dry_run: dryRun
  }, {
    timeout: UPLOAD_TIMEOUT
  })
}

// 开始训练
export const startTraining = (datasetPath, config) => {
  return api.post('/train', config, {
    params: { dataset_path: datasetPath }
  })
}

export const preflightTraining = (datasetPath, config) => {
  return api.post('/training/preflight', config, {
    params: { dataset_path: datasetPath }
  })
}

// 标准化推理配置参数
// 兼容页面中可能出现的 iouThreshold / iou_threshold、maxDetections / max_detections / max_det 等写法。
const appendInferenceParams = (params, config = {}) => {
  const confidence = config.confidence
  const iouThreshold = config.iou_threshold ?? config.iouThreshold
  const maxDetections = config.max_detections ?? config.maxDetections ?? config.max_det
  const saveResults = config.save_results ?? config.saveResults
  const trainingTaskId = config.training_task_id ?? config.trainingTaskId

  if (confidence !== undefined && confidence !== null) {
    params.append('confidence', confidence)
  }

  if (iouThreshold !== undefined && iouThreshold !== null) {
    params.append('iou_threshold', iouThreshold)
  }

  if (maxDetections !== undefined && maxDetections !== null) {
    params.append('max_detections', maxDetections)
    params.append('max_det', maxDetections)
  }

  if (saveResults !== undefined && saveResults !== null) {
    params.append('save_results', saveResults)
  }

  if (trainingTaskId) {
    params.append('training_task_id', trainingTaskId)
  }
}

// 单张图片推理
export const predictImage = (file, modelPath = 'yolov8n.pt', config = {}) => {
  const formData = new FormData()
  formData.append('file', file)

  // 构建查询参数
  const params = new URLSearchParams()
  params.append('model_path', modelPath)

  // 添加推理配置参数到查询字符串
  appendInferenceParams(params, config)

  return api.post(`/predict?${params.toString()}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

// 批量图片推理
export const predictBatch = (files, modelPath = 'yolov8n.pt', config = {}) => {
  const formData = new FormData()

  files.forEach(file => {
    formData.append('files', file)
  })

  // 构建查询参数
  const params = new URLSearchParams()
  params.append('model_path', modelPath)

  // 添加推理配置参数到查询字符串
  appendInferenceParams(params, config)

  return api.post(`/predict/batch?${params.toString()}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

// 导出模型
export const exportModel = (modelPath, config) => {
  return api.post('/export', config, {
    params: { model_path: modelPath }
  })
}

// 删除任务
export const deleteTask = (taskId) => {
  return api.delete(`/task/${encodeURIComponent(taskId)}`)
}

// 删除单个数据集
export const deleteDataset = (datasetId) => {
  return api.delete(`/datasets/${datasetId}`)
}

// 删除训练数据集
export const deleteTrainingDataset = (datasetId) => {
  return api.delete(`/training-datasets/${encodeURIComponent(datasetId)}`)
}

// 批量删除数据集
export const batchDeleteDatasets = (datasetIds) => {
  return api.post('/datasets/batch/delete', {
    dataset_ids: datasetIds
  })
}

// 获取推理配置
export const getInferenceConfigs = () => {
  return api.get('/inference/configs')
}

// 测试推理配置
export const testInferenceConfig = (file, modelPath, confidence, iouThreshold) => {
  const formData = new FormData()
  formData.append('file', file)

  return api.post('/inference/config/test', formData, {
    params: {
      model_path: modelPath,
      confidence: confidence,
      iou_threshold: iouThreshold
    }
  })
}

// 实时推理相关API
export const getRealtimeStats = (sessionId) => {
  return api.get(`/realtime/stats/${sessionId}`)
}

export const listRealtimeSessions = () => {
  return api.get('/realtime/sessions')
}

export const stopRealtimeSession = (sessionId) => {
  return api.delete(`/realtime/session/${encodeURIComponent(sessionId)}`)
}

// 视频源管理API
const getVideoUploadSessions = () => {
  try {
    return JSON.parse(localStorage.getItem(VIDEO_UPLOAD_SESSION_KEY) || '{}') || {}
  } catch {
    return {}
  }
}

const setVideoUploadSession = (fingerprint, uploadId) => {
  const sessions = getVideoUploadSessions()
  if (uploadId) sessions[fingerprint] = uploadId
  else delete sessions[fingerprint]
  localStorage.setItem(VIDEO_UPLOAD_SESSION_KEY, JSON.stringify(sessions))
}

const uploadVideoFileDirect = (file, onProgress) => {
  const formData = new FormData()
  formData.append('file', file)

  return api.post('/upload/video', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    timeout: VIDEO_UPLOAD_TIMEOUT,
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        )
        onProgress(progress)
      }
    }
  })
}

const uploadVideoFileChunked = async (file, onProgress) => {
  const fingerprint = `${file.name}:${file.size}:${file.lastModified || 0}`
  const sessions = getVideoUploadSessions()
  let uploadId = sessions[fingerprint] || ''
  let session = null

  if (uploadId) {
    try {
      session = await api.get(`/upload/video/chunks/${encodeURIComponent(uploadId)}`, { silent: true })
      if (Number(session.size) !== Number(file.size) || (session.original_filename || session.filename) !== file.name) {
        session = null
        uploadId = ''
      }
    } catch {
      session = null
      uploadId = ''
    }
  }

  const totalChunks = Math.ceil(file.size / VIDEO_UPLOAD_CHUNK_BYTES)
  if (!session) {
    session = await api.post('/upload/video/chunks/start', {
      filename: file.name,
      size: file.size,
      content_type: file.type || 'application/octet-stream',
      total_chunks: totalChunks
    })
    uploadId = session.upload_id
    setVideoUploadSession(fingerprint, uploadId)
  }

  const uploadedChunks = new Set((session.uploaded_chunks || []).map(Number))
  let uploadedBytes = 0
  for (const index of uploadedChunks) {
    const start = index * VIDEO_UPLOAD_CHUNK_BYTES
    uploadedBytes += Math.max(0, Math.min(VIDEO_UPLOAD_CHUNK_BYTES, file.size - start))
  }
  onProgress?.(Math.min(99, Math.round((uploadedBytes * 100) / file.size)))

  for (let index = 0; index < totalChunks; index += 1) {
    if (uploadedChunks.has(index)) continue
    const start = index * VIDEO_UPLOAD_CHUNK_BYTES
    const end = Math.min(file.size, start + VIDEO_UPLOAD_CHUNK_BYTES)
    const chunk = file.slice(start, end)
    const formData = new FormData()
    formData.append('chunk', chunk, `${file.name}.part${index}`)
    await api.post(
      `/upload/video/chunks/${encodeURIComponent(uploadId)}/parts/${index}`,
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
        onUploadProgress: (progressEvent) => {
          if (!onProgress || !progressEvent.total) return
          const current = (progressEvent.loaded / progressEvent.total) * chunk.size
          onProgress(Math.min(99, Math.round(((uploadedBytes + current) * 100) / file.size)))
        }
      }
    )
    uploadedBytes += chunk.size
    onProgress?.(Math.min(99, Math.round((uploadedBytes * 100) / file.size)))
  }

  const completed = await api.post(
    `/upload/video/chunks/${encodeURIComponent(uploadId)}/complete`,
    null,
    { timeout: VIDEO_UPLOAD_TIMEOUT }
  )
  setVideoUploadSession(fingerprint, '')
  onProgress?.(100)
  return completed
}

export const uploadVideoFile = (file, onProgress) => {
  if (Number(file?.size || 0) >= VIDEO_UPLOAD_CHUNK_THRESHOLD) {
    return uploadVideoFileChunked(file, onProgress)
  }
  return uploadVideoFileDirect(file, onProgress)
}

export const getVideoSources = () => {
  return api.get('/video/sources')
}

export const addVideoSource = (config) => {
  return api.post('/video/source/add', config)
}

export const deleteVideoSource = (sourceId) => {
  return api.delete(`/video/source/${sourceId}`)
}

export const testVideoSource = (config) => {
  return api.post('/video/source/test', config)
}

// 数据标注视频相关API
export const uploadAnnotationVideo = (file, onProgress) => {
  const formData = new FormData()
  formData.append('file', file)

  return api.post('/annotation/video/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    timeout: UPLOAD_TIMEOUT,
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        )
        onProgress(progress)
      }
    }
  })
}

export const extractVideoFrames = (videoId, videoPath, frameInterval = 1, maxFrames = 100) => {
  return api.post('/annotation/video/extract-frames', null, {
    params: {
      video_id: videoId,
      video_path: videoPath,
      frame_interval: frameInterval,
      max_frames: maxFrames
    },
    timeout: UPLOAD_TIMEOUT
  })
}

// AI辅助标注
export const aiAssistAnnotation = (
  imageData,
  modelPath = 'yolo11n.pt',
  confidence = 0.25,
  annotationClasses = [],
  annotationClassBindings = [],
  promptClasses = [],
  promptPageIndices = [],
  targetPageIndex = null,
  modelMetadata = {},
  modelId = '',
  sessionContext = {}
) => {
  return api.post('/annotation/ai-assist', {
    image_data: imageData,
    model_path: modelPath,
    model_id: modelId || modelMetadata?.id || '',
    confidence: confidence,
    annotation_classes: annotationClasses,
    annotation_class_bindings: annotationClassBindings,
    prompt_classes: promptClasses,
    prompt_page_indices: promptPageIndices,
    target_page_index: targetPageIndex,
    model_metadata: modelMetadata,
    dataset_id: sessionContext.dataset_id || '',
    annotation_draft_id: sessionContext.annotation_draft_id || ''
  }, {
    timeout: DEFAULT_TIMEOUT
  })
}

// 九宫格推理相关API
export const getGridSessions = () => {
  return api.get('/grid/sessions')
}

export const stopGridSession = (gridId) => {
  return api.delete(`/grid/session/${gridId}`)
}

// 九宫格WebSocket连接工具
export class GridWebSocket {
  constructor(gridId) {
    this.gridId = gridId
    this.websocket = null
    this.messageHandlers = {}
    this.isConnected = false
  }

  connect() {
    return new Promise((resolve, reject) => {
      try {
        // 根据环境确定WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const host = window.location.host
        const wsUrl = `${protocol}//${host}/api/ws/grid/${this.gridId}`

        this.websocket = new WebSocket(wsUrl)

        this.websocket.onopen = () => {
          this.isConnected = true
          console.log(`九宫格WebSocket连接已建立: ${this.gridId}`)
          resolve()
        }

        this.websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            this.handleMessage(data)
          } catch (error) {
            console.error('解析九宫格WebSocket消息失败:', error)
          }
        }

        this.websocket.onerror = (error) => {
          console.error('九宫格WebSocket错误:', error)
          reject(error)
        }

        this.websocket.onclose = () => {
          this.isConnected = false
          console.log('九宫格WebSocket连接已关闭')
          if (this.messageHandlers['close']) {
            this.messageHandlers['close']()
          }
        }
      } catch (error) {
        reject(error)
      }
    })
  }

  disconnect() {
    if (this.websocket) {
      this.websocket.close()
      this.websocket = null
      this.isConnected = false
    }
  }

  send(message) {
    if (this.websocket && this.isConnected) {
      this.websocket.send(JSON.stringify(message))
    } else {
      console.warn('九宫格WebSocket未连接，无法发送消息')
    }
  }

  on(eventType, handler) {
    this.messageHandlers[eventType] = handler
  }

  handleMessage(data) {
    const handler = this.messageHandlers[data.type]
    if (handler) {
      handler(data)
    }
  }
}

// WebSocket连接工具
export class RealtimeWebSocket {
  constructor(sessionId) {
    this.sessionId = sessionId
    this.websocket = null
    this.messageHandlers = {}
    this.isConnected = false
  }

  connect() {
    return new Promise((resolve, reject) => {
      try {
        // 根据环境确定WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const host = window.location.host
        const wsUrl = `${protocol}//${host}/api/ws/realtime/${this.sessionId}`

        this.websocket = new WebSocket(wsUrl)

        this.websocket.onopen = () => {
          this.isConnected = true
          console.log(`WebSocket连接已建立: ${this.sessionId}`)
          resolve()
        }

        this.websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            this.handleMessage(data)
          } catch (error) {
            console.error('解析WebSocket消息失败:', error)
          }
        }

        this.websocket.onerror = (error) => {
          console.error('WebSocket错误:', error)
          reject(error)
        }

        this.websocket.onclose = () => {
          this.isConnected = false
          console.log('WebSocket连接已关闭')
          if (this.messageHandlers['close']) {
            this.messageHandlers['close']()
          }
        }
      } catch (error) {
        reject(error)
      }
    })
  }

  disconnect() {
    if (this.websocket) {
      this.websocket.close()
      this.websocket = null
      this.isConnected = false
    }
  }

  send(message) {
    if (this.websocket && this.isConnected) {
      this.websocket.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket未连接，无法发送消息')
    }
  }

  on(eventType, handler) {
    this.messageHandlers[eventType] = handler
  }

  handleMessage(data) {
    const handler = this.messageHandlers[data.type]
    if (handler) {
      handler(data)
    }
  }
}


// --- 行为分析相关 API ---
// 注意：当前 axios 实例 baseURL 为 /api，因此这里调用的是后端 /api/behavior/... 路由。

export const getBackendOrigin = () => {
  // 静态媒体文件不要走 /api/static 代理路径，避免视频 Range 请求或前端代理导致 <video>/<img> 加载失败。
  // 本地开发时前端通常是 localhost:3000，后端是 localhost:8000。
  const explicit = import.meta.env.VITE_BACKEND_ORIGIN || import.meta.env.VITE_API_ORIGIN || ''
  if (explicit) {
    return explicit.replace(/\/$/, '')
  }

  const protocol = window.location.protocol || 'http:'
  const hostname = window.location.hostname || 'localhost'

  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return `${protocol}//${hostname}:8000`
  }

  // 如果你用 nginx 同源部署，也可让它走当前 origin。
  return window.location.origin
}

export const buildStaticUrl = (path) => {
  if (!path) return ''

  let value = String(path).trim().replace(/\\/g, '/')
  if (!value) return ''

  if (
    value.startsWith('http://') ||
    value.startsWith('https://') ||
    value.startsWith('data:') ||
    value.startsWith('blob:')
  ) {
    return value
  }

  // 后端内部路径转浏览器路径：
  // /app/static/xxx -> /static/xxx
  if (value.startsWith('/app/static/')) {
    value = value.replace('/app', '')
  }

  // 兼容 static/xxx -> /static/xxx
  if (value.startsWith('static/')) {
    value = `/${value}`
  }

  // 兼容 api/static/xxx -> /static/xxx
  // 避免 video/img 使用 /api/static 导致浏览器播放器加载失败
  if (value.startsWith('/api/static/')) {
    value = value.replace('/api/static/', '/static/')
  }

  if (value.startsWith('api/static/')) {
    value = value.replace('api/static/', '/static/')
  }

  // 图片上传目录也直接走后端静态挂载
  if (value.startsWith('image/')) {
    value = `/${value}`
  }

  if (value.startsWith('/api/image/')) {
    value = value.replace('/api/image/', '/image/')
  }

  if (value.startsWith('api/image/')) {
    value = value.replace('api/image/', '/image/')
  }

  // 静态资源强制走后端 8000 端口，避免前端代理 /api/static 影响视频 Range 请求
  if (value.startsWith('/static/') || value.startsWith('/image/')) {
    return encodeURI(`http://localhost:8000${value}`)
  }

  // 其它 API 路径仍然留给 axios 的 /api baseURL，不在这里处理
  return encodeURI(value.startsWith('/') ? value : `/${value}`)
}

export const startBehaviorTask = (payload) => {
  return api.post('/behavior/tasks/start', payload, {
    timeout: 10000
  })
}

export const getBehaviorTaskStatus = (taskId) => {
  return api.get(`/behavior/tasks/status/${taskId}`, {
    timeout: 10000,
    params: { t: Date.now() }
  })
}

export const getBehaviorTaskResult = (taskId) => {
  return api.get(`/behavior/tasks/result/${taskId}`, {
    timeout: 30000,
    params: { t: Date.now() }
  })
}

export const analyzeBehaviorVideo = (payload) => {
  return api.post('/behavior/analyze_video', payload, {
    timeout: UPLOAD_TIMEOUT
  })
}

export const deleteBehaviorTask = (taskId) => {
  return api.delete(`/behavior/tasks/${taskId}`)
}

export const getBehaviorEvents = (params = {}) => {
  return api.get('/behavior/events', { params })
}

export const getBehaviorEventRefreshConfig = () => {
  return api.get('/behavior/events/refresh-config')
}

export const getBehaviorEvent = (eventId) => {
  return api.get(`/behavior/events/${eventId}`)
}

export const deleteBehaviorEvent = (eventId) => {
  return api.delete(`/behavior/events/${encodeURIComponent(eventId)}`)
}

export const deleteBehaviorEvents = (eventIds) => {
  return api.delete('/behavior/events', { data: { event_ids: eventIds } })
}

export const updateBehaviorEventHandling = (eventId, payload) => {
  return api.patch(`/behavior/events/${encodeURIComponent(eventId)}/handling`, payload)
}

export const updateBehaviorEventFeedback = (eventId, payload) => {
  return api.put(`/behavior/events/${encodeURIComponent(eventId)}/feedback`, payload)
}

export const uploadMissedBehaviorEvent = (formData) => {
  return api.post('/behavior/feedback/missed', formData, {
    timeout: UPLOAD_TIMEOUT,
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

export const getBehaviorOptimizationSampleStats = () => {
  return api.get('/behavior/optimization-samples/stats', {
    timeout: 30000,
    params: { t: Date.now() }
  })
}

export const getDefaultBehaviorRuleProfile = () => {
  return api.get('/behavior/rule-profiles/default')
}

export const updateDefaultBehaviorRuleProfile = (config) => {
  return api.put('/behavior/rule-profiles/default', { config })
}

export const getSourceBehaviorRuleProfile = (sourceId) => {
  return api.get(`/behavior/rule-profiles/sources/${encodeURIComponent(sourceId)}`)
}

export const updateSourceBehaviorRuleProfile = (sourceId, config) => {
  return api.put(
    `/behavior/rule-profiles/sources/${encodeURIComponent(sourceId)}`,
    { config }
  )
}


// ==================== 行为样本复核 API ====================

export const getBehaviorReviewClips = (taskId) => {
  return api.get(`/behavior/review/tasks/${taskId}/clips`, {
    timeout: 30000,
    params: { t: Date.now() }
  })
}

export const reviewBehaviorClip = (payload) => {
  return api.post('/behavior/review/clips/review', payload, {
    timeout: 30000
  })
}

export const getBehaviorReviewStats = () => {
  return api.get('/behavior/review/stats', {
    timeout: 30000,
    params: { t: Date.now() }
  })
}

export const getBehaviorReviewDataset = () => {
  return api.get('/behavior/review/dataset', {
    timeout: 30000,
    params: { t: Date.now() }
  })
}


// ==================== 动作复核分类器 API ====================

export const getActionClassifierInfo = () => {
  return api.get('/behavior/action-classifier/info', {
    timeout: 30000,
    params: { t: Date.now() }
  })
}

export const getActionVerifierModels = () => {
  return api.get('/behavior/action-classifier/models', {
    timeout: 30000,
    params: { t: Date.now() }
  })
}

export const activateActionVerifierModel = (modelId) => {
  return api.post(`/behavior/action-classifier/models/${encodeURIComponent(modelId)}/activate`, null, {
    timeout: 30000
  })
}

export const getBehaviorModels = () => {
  return api.get('/behavior/models', { params: { t: Date.now() } })
}

export const startBehaviorModelTraining = (payload) => {
  return api.post('/behavior/models/train', payload, { timeout: 30000 })
}

export const getBehaviorModelTraining = (taskId) => {
  return api.get(`/behavior/models/training/${encodeURIComponent(taskId)}`, {
    params: { t: Date.now() }
  })
}

export const getBehaviorModelEvaluation = (modelId) => {
  return api.get(`/behavior/models/${encodeURIComponent(modelId)}/evaluation`)
}

export const getBehaviorModelShadowComparison = (modelId) => {
  return api.get(`/behavior/models/${encodeURIComponent(modelId)}/shadow-comparison`)
}

export const activateBehaviorModel = (modelId, reason = '') => {
  return api.post(`/behavior/models/${encodeURIComponent(modelId)}/activate`, { reason })
}

export const rollbackBehaviorModel = (modelId, reason) => {
  return api.post(`/behavior/models/${encodeURIComponent(modelId)}/rollback`, { reason })
}

// ActionTraining.vue 当前使用的命名
export const startActionClassifierTraining = (payload) => {
  return api.post('/behavior/action-classifier/train', payload, {
    timeout: 30000
  })
}

export const getActionClassifierTrainingStatus = (taskId) => {
  return api.get(`/behavior/action-classifier/status/${taskId}`, {
    timeout: 30000,
    params: { t: Date.now() }
  })
}

export const getActionClassifierTrainingResult = (taskId) => {
  return api.get(`/behavior/action-classifier/result/${taskId}`, {
    timeout: 30000,
    params: { t: Date.now() }
  })
}

export const predictActionClassifier = (payload) => {
  return api.post('/behavior/action-classifier/predict', payload, {
    timeout: 30000
  })
}

// 兼容另一套命名，避免页面使用旧名称时再次构建失败
export const trainActionClassifier = startActionClassifierTraining
export const getActionClassifierStatus = getActionClassifierTrainingStatus
export const getActionClassifierResult = getActionClassifierTrainingResult


export const listContinuousMonitors = () => api.get('/continuous-monitoring', { silent: true })
export const startContinuousMonitor = payload => api.post('/continuous-monitoring/start', payload)
export const stopContinuousMonitor = id => api.post(`/continuous-monitoring/${encodeURIComponent(id)}/stop`)
export const resumeContinuousMonitor = id => api.post(`/continuous-monitoring/${encodeURIComponent(id)}/resume`)

export default api

export const getMonitoringAlgorithms = params => api.get('/monitoring/algorithms', { params })
export const getAlgorithmCatalog = () => api.get('/monitoring/algorithm-catalog')
export const getPlatformAlgorithms = () => api.get('/system/algorithms')
export const createPlatformAlgorithm = body => api.post('/system/algorithms', body)
export const savePlatformAlgorithms = body => api.put('/system/algorithms', body)
export const getInferenceSwitchStatus = id => api.get(`/system/algorithms/inference-switch/${encodeURIComponent(id)}`, { silent: true, timeout: 5000 })
export const getInferenceServiceStatus = () => api.get('/system/inference-service/status', { silent: true, timeout: 5000 })
export const getPlatformAlgorithmModels = params => api.get('/system/algorithms/models', { params })
export const getPlatformAlgorithmModelClasses = id => api.get(`/system/algorithms/models/${encodeURIComponent(id)}/classes`, { timeout: 60000 })
export const deletePlatformAlgorithm = (code, revision) => api.delete(`/system/algorithms/${encodeURIComponent(code)}`, { data: { revision } })
export const getDeletedPlatformAlgorithms = () => api.get('/system/algorithms/deleted')
export const restorePlatformAlgorithm = (code, revision) => api.post(`/system/algorithms/${encodeURIComponent(code)}/restore`, { revision })
export const saveMonitoringAlgorithms = (recorder, channel, algorithms) => api.put(`/monitoring/algorithms/${encodeURIComponent(recorder)}/${encodeURIComponent(channel)}`, { algorithms })
export const getMonitoringTasks = () => api.get('/monitoring/tasks', { silent: true })
export const startMonitoringCamera = (recorder, channel) => api.post(`/monitoring/cameras/${encodeURIComponent(recorder)}/${encodeURIComponent(channel)}/start`)
export const stopMonitoringCamera = (recorder, channel) => api.post(`/monitoring/cameras/${encodeURIComponent(recorder)}/${encodeURIComponent(channel)}/stop`)
export const startMonitoringTask = body => api.post('/monitoring/tasks/start', body)
export const stopMonitoringTask = id => api.post(`/monitoring/tasks/${encodeURIComponent(id)}/stop`)
export const resumeMonitoringTask = id => api.post(`/monitoring/tasks/${encodeURIComponent(id)}/resume`)
export const getMonitoringPreview = (id, since = {}) => api.get(`/monitoring/tasks/${encodeURIComponent(id)}/preview`, { params: { since: JSON.stringify(since) }, silent: true, timeout: 10000 })
export const getMonitoringLiveFrames = cameras => api.post('/monitoring/live-frames', { cameras }, { silent: true, timeout: 5000 })
export const getMonitoringWebRTCSession = (recorder, channel) => api.post(`/monitoring/webrtc-session/${encodeURIComponent(recorder)}/${encodeURIComponent(channel)}`, {}, { silent: true, timeout: 6000 })
export const getAlarmRecording = id => api.get(`/behavior/events/${encodeURIComponent(id)}/recording`)

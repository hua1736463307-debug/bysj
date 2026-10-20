/**
 * 网络请求工具类
 * 封装 axios 实例，配置请求拦截、响应拦截及后端 API 接口
 */
import axios from 'axios'
import { ElMessage } from 'element-plus'

// 普通请求超时时间：30秒
const DEFAULT_TIMEOUT = 30000

// 大文件上传超时时间：5分钟
const UPLOAD_TIMEOUT = 300000

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
  localStorage.removeItem('token')
  localStorage.removeItem('access_token')
  localStorage.removeItem('auth_token')
  localStorage.removeItem('user')
  setAuthHeader('')
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
    const status = error.response?.status
    const message = error.response?.data?.detail || error.message || '请求失败'

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

// 管理员：获取用户列表
export const getAdminUsers = (params = {}) => {
  return api.get('/admin/users', {
    params: {
      page: params.page || 1,
      page_size: params.page_size || 20,
      search: params.search || ''
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

// 平台管理：登录日志
export const getLoginLogs = (params = {}) => {
  return api.get('/system/login-logs', { params })
}

// 平台管理：操作日志
export const getOperationLogs = (params = {}) => {
  return api.get('/system/operation-logs', { params })
}

// 健康检查
export const checkApiHealth = async () => {
  try {
    const response = await api.get('/health')
    return response.status === 'healthy'
  } catch {
    return false
  }
}

// 获取模型列表
export const getModels = () => {
  return api.get('/models')
}

// 获取任务列表
export const getTasks = () => {
  return api.get('/tasks')
}

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
  return api.get(`/status/${taskId}`)
}

// 获取数据集列表
export const getDatasets = (params = {}) => {
  return api.get('/datasets', {
    params: {
      page: params.page || 1,
      page_size: params.page_size || 20,
      t: Date.now()
    },
    timeout: 120000
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

// 开始训练
export const startTraining = (datasetPath, config) => {
  return api.post('/train', config, {
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
  return api.delete(`/task/${taskId}`)
}

// 删除单个数据集
export const deleteDataset = (datasetId) => {
  return api.delete(`/datasets/${datasetId}`)
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
  return api.delete(`/realtime/session/${sessionId}`)
}

// 视频源管理API
export const uploadVideoFile = (file, onProgress) => {
  const formData = new FormData()
  formData.append('file', file)

  return api.post('/upload/video', formData, {
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
export const aiAssistAnnotation = (imageData, modelPath = 'yolo11n.pt', confidence = 0.25) => {
  return api.post('/annotation/ai-assist', {
    image_data: imageData,
    model_path: modelPath,
    confidence: confidence
  }, {
    timeout: DEFAULT_TIMEOUT
  })
}

// 行为复核相关API
export const buildStaticUrl = (value) => {
  if (!value) return ''

  const raw = String(value).trim()
  if (!raw) return ''

  if (
    raw.startsWith('http://') ||
    raw.startsWith('https://') ||
    raw.startsWith('data:') ||
    raw.startsWith('blob:')
  ) {
    return raw
  }

  let path = raw.replace(/\\/g, '/')

  // 后端同时挂载了 /static 和 /api/static；前端统一使用 /api/static，
  // 避免 Vite/前端路由把 /static 当成前端自己的静态目录。
  if (path.startsWith('/api/static/')) {
    return path
  }

  if (path.startsWith('/app/static/')) {
    return path.replace('/app/static/', '/api/static/')
  }

  if (path.startsWith('static/')) {
    return `/api/${path}`
  }

  if (path.startsWith('/static/')) {
    return `/api${path}`
  }

  // 兼容后端返回 behavior_results/xxx.mp4 这类相对路径。
  if (path.startsWith('behavior_results/')) {
    return `/api/static/${path}`
  }

  return path.startsWith('/') ? path : `/api/static/${path}`
}

export const getBehaviorReviewTasks = () => {
  return api.get('/behavior/review/tasks', {
    params: { t: Date.now() },
    timeout: DEFAULT_TIMEOUT
  })
}

export const getBehaviorReviewClips = (taskId) => {
  return api.get(`/behavior/review/tasks/${encodeURIComponent(taskId)}/clips`, {
    params: { t: Date.now() },
    timeout: DEFAULT_TIMEOUT
  })
}

export const reviewBehaviorClip = (payload) => {
  return api.post('/behavior/review/clips/review', payload, {
    timeout: DEFAULT_TIMEOUT
  })
}

export const getBehaviorReviewStats = () => {
  return api.get('/behavior/review/stats', {
    params: { t: Date.now() },
    timeout: DEFAULT_TIMEOUT
  })
}

export const getBehaviorReviewDataset = () => {
  return api.get('/behavior/review/dataset', {
    params: { t: Date.now() },
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

export default api

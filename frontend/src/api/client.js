/**
 * 受控 API 客户端（PR-01 建立）。
 *
 * 约定：
 * - 同源 base URL（默认 /api/v1），不硬编码私网源站；
 * - 单处 token 注入、单处 401 处理；
 * - 统一解包 {code,message,data,request_id} envelope；
 * - 支持 AbortController 取消与超时。
 */
import axios from 'axios'

export const TOKEN_KEY = 'token'
export const USER_KEY = 'user'

const BASE_URL = import.meta.env.VITE_API_BASE || '/api/v1'

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || ''
}

export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token)
  else localStorage.removeItem(TOKEN_KEY)
}

export function getCurrentUser() {
  try {
    const raw = localStorage.getItem(USER_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    localStorage.removeItem(USER_KEY)
    return null
  }
}

export function saveAuth(payload) {
  const { token, user } = payload || {}
  setToken(token)
  if (user) localStorage.setItem(USER_KEY, JSON.stringify(user))
}

export function clearAuth() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
  localStorage.removeItem('access_token')
  localStorage.removeItem('auth_token')
}

export class ApiError extends Error {
  constructor(message, { code = 'request_failed', status = 0, data = null, requestId = '' } = {}) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
    this.data = data
    this.requestId = requestId
  }
}

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: { Accept: 'application/json' }
})

client.interceptors.request.use((config) => {
  const token = getToken()
  if (token && !config.headers?.Authorization) {
    config.headers = config.headers || {}
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

client.interceptors.response.use(
  (response) => {
    if (response.config?.responseType === 'blob') return response
    const body = response.data
    if (body && typeof body === 'object' && 'code' in body && 'data' in body) {
      if (body.code === 'ok') return body.data
      throw new ApiError(body.message || '请求失败', {
        code: body.code,
        status: response.status,
        data: body.data,
        requestId: body.request_id
      })
    }
    throw new ApiError('服务器响应不符合 v1 契约', { code: 'invalid_envelope', status: response.status })
  },
  (error) => {
    const response = error.response
    if (!response) {
      throw new ApiError(error.code === 'ECONNABORTED' ? '请求超时' : '网络异常，请检查连接', {
        code: error.code || 'network_error'
      })
    }
    if (response.status === 401) {
      clearAuth()
      if (window.location.pathname !== '/login') {
        window.location.assign('/login')
      }
    }
    const body = response.data || {}
    throw new ApiError(body.message || `请求失败 (${response.status})`, {
      code: body.code || 'http_error',
      status: response.status,
      data: body.data || null,
      requestId: body.request_id || response.headers?.['x-request-id'] || ''
    })
  }
)

export function createAbortController() {
  return new AbortController()
}

export const api = {
  get: (url, config) => client.get(url, config),
  post: (url, data, config) => client.post(url, data, config),
  put: (url, data, config) => client.put(url, data, config),
  patch: (url, data, config) => client.patch(url, data, config),
  delete: (url, config) => client.delete(url, config)
}

export default client

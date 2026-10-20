/**
 * 前端应用入口文件
 * 负责初始化 Vue 实例、挂载路由、注册插件及全局组件
 */
import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import axios from 'axios'
import ElementPlus from 'element-plus'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import './styles/ruoyi-theme.css'

import App from './App.vue'
import Login from './views/Login.vue'
import Annotation from './views/Annotation.vue'
import Dataset from './views/Dataset.vue'
import DatasetManager from './views/DatasetManager.vue'
import Training from './views/Training.vue'
import Inference from './views/Inference.vue'
import ModelHub from './views/ModelHub.vue'

const readCurrentUser = () => {
  try {
    const raw = localStorage.getItem('user')
    return raw ? JSON.parse(raw) : null
  } catch {
    localStorage.removeItem('user')
    return null
  }
}


const readAuthToken = () => {
  return (
    localStorage.getItem('token') ||
    localStorage.getItem('access_token') ||
    localStorage.getItem('auth_token') ||
    ''
  )
}

const clearAuthStorage = () => {
  localStorage.removeItem('token')
  localStorage.removeItem('access_token')
  localStorage.removeItem('auth_token')
  localStorage.removeItem('user')
}

// 兼容兜底：项目中如果有页面绕过 src/utils/api.js，直接使用 axios，也会自动带 token。
axios.interceptors.request.use((config) => {
  const token = readAuthToken()
  if (token) {
    config.headers = config.headers || {}
    if (!config.headers.Authorization) {
      config.headers.Authorization = `Bearer ${token}`
    }
  }
  return config
})

axios.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && window.location.pathname !== '/login') {
      clearAuthStorage()
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// 兼容兜底：项目中如果有页面直接使用 fetch 调 /api，也会自动带 token。
const nativeFetch = window.fetch.bind(window)
window.fetch = (input, init = {}) => {
  const token = readAuthToken()
  if (!token) {
    return nativeFetch(input, init)
  }

  const url = typeof input === 'string' ? input : input?.url
  const shouldAttachToken = typeof url === 'string' && (
    url.startsWith('/api') ||
    url.startsWith('http://localhost:8000') ||
    url.startsWith('http://127.0.0.1:8000')
  )

  if (!shouldAttachToken) {
    return nativeFetch(input, init)
  }

  const headers = new Headers(init.headers || (input?.headers || {}))
  if (!headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  return nativeFetch(input, {
    ...init,
    headers
  })
}

const routes = [
  { path: '/login', component: Login, meta: { guestOnly: true } },
  { path: '/', redirect: '/raw-dataset' },
  { path: '/raw-dataset', component: Dataset, meta: { requiresAuth: true } },
  { path: '/dataset', redirect: '/raw-dataset' },
  { path: '/image', redirect: '/raw-dataset' },
  { path: '/dataset-annotation', component: Annotation, meta: { requiresAuth: true } },
  { path: '/annotation', redirect: '/dataset-annotation' },
  { path: '/train-dataset', component: DatasetManager, meta: { requiresAuth: true } },
  { path: '/dataset-manager', redirect: '/train-dataset' },
  { path: '/model-training', component: Training, meta: { requiresAuth: true } },
  { path: '/training', redirect: '/model-training' },
  { path: '/model-hub', component: ModelHub, meta: { requiresAuth: true } },
  { path: '/models', redirect: '/model-hub' },
  { path: '/inference-service', component: Inference, meta: { requiresAuth: true } },
  { path: '/inference', redirect: '/inference-service' },
  { path: '/:pathMatch(.*)*', redirect: '/' }
]
const router = createRouter({ history: createWebHistory(), routes })
router.beforeEach(to => {
  const token = readAuthToken()
  if (to.meta.requiresAuth && !token) return '/login'
  if (to.meta.guestOnly && token) return '/'
})
const app = createApp(App)
app.use(router)
app.use(ElementPlus, { locale: zhCn })
for (const [name, icon] of Object.entries(ElementPlusIconsVue)) app.component(name, icon)
app.mount('#app')

/**
 * 前端应用入口文件
 * 负责初始化 Vue 实例、挂载路由、注册插件及全局组件
 */
import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import { getToken } from './api/client'
import './styles/medical.css'
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
  const token = getToken()
  if (to.meta.requiresAuth && !token) return '/login'
  if (to.meta.guestOnly && token) return '/'
})
const app = createApp(App)
app.use(router)
app.use(ElementPlus, { locale: zhCn })
for (const [name, icon] of Object.entries(ElementPlusIconsVue)) app.component(name, icon)
app.mount('#app')

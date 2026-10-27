<template>
  <router-view v-if="route.path === '/login'" />
  <el-container v-else style="min-height:100vh">
    <el-header style="display:flex;align-items:center;gap:20px;border-bottom:1px solid #ddd">
      <strong>胸片五标签辅助分析</strong>
      <router-link v-for="item in navigation" :key="item.path" :to="item.path">{{ item.label }}</router-link>
      <span style="flex:1"></span>
      <el-dropdown trigger="click">
        <span style="cursor:pointer">历史研究模块</span>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item v-for="item in legacyNavigation" :key="item.path">
              <router-link :to="item.path">{{ item.label }}</router-link>
            </el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
      <el-button text @click="logout">退出登录</el-button>
    </el-header>
    <el-main><router-view /></el-main>
  </el-container>
</template>
<script setup>
import {useRoute,useRouter} from 'vue-router'
import { logout as logoutSession } from './api/auth'
const route=useRoute(),router=useRouter()
const navigation=[
  {path:'/cxr/dashboard',label:'概览'},
  {path:'/cxr/data',label:'胸片数据'},
  {path:'/cxr/training',label:'训练与校准'},
  {path:'/cxr/analysis',label:'病例分析'},
  {path:'/cxr/review',label:'人工复核'},
  {path:'/cxr/reports',label:'报告'}
]
// 旧 BCCD 检测链路保留为历史入口，不承载胸片分类输出。
const legacyNavigation=[
  {path:'/raw-dataset',label:'原始数据（检测）'},
  {path:'/dataset-annotation',label:'数据标注（检测）'},
  {path:'/train-dataset',label:'训练数据（检测）'},
  {path:'/model-training',label:'模型训练（检测）'},
  {path:'/model-hub',label:'模型管理'},
  {path:'/inference-service',label:'推理验证（检测）'}
]
async function logout(){try{await logoutSession()}finally{router.replace('/login')}}
</script>

<template>
  <router-view v-if="route.path === '/login'" />
  <el-container v-else style="min-height:100vh">
    <el-header style="display:flex;align-items:center;gap:24px;border-bottom:1px solid #ddd">
      <strong>图像训推实验平台</strong>
      <router-link v-for="item in navigation" :key="item.path" :to="item.path">{{ item.label }}</router-link>
      <el-button text @click="logout">退出登录</el-button>
    </el-header>
    <el-main><router-view /></el-main>
  </el-container>
</template>
<script setup>
import {useRoute,useRouter} from 'vue-router'
import { logout as logoutSession } from './api/auth'
const route=useRoute(),router=useRouter()
const navigation=[{path:'/raw-dataset',label:'原始数据'},{path:'/dataset-annotation',label:'数据标注'},{path:'/train-dataset',label:'训练数据'},{path:'/model-training',label:'模型训练'},{path:'/model-hub',label:'模型管理'},{path:'/inference-service',label:'推理验证'}]
async function logout(){try{await logoutSession()}finally{router.replace('/login')}}
</script>

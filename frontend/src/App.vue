<template>
  <router-view v-if="route.path === '/login'" />
  <div v-else class="admin-layout" :class="{ collapsed }">
    <aside class="admin-sidebar">
      <router-link class="admin-brand" to="/"><el-icon :size="25"><FirstAidKit /></el-icon><strong v-if="!collapsed">医疗辅助诊断平台</strong></router-link>
      <el-menu :default-active="route.path" :collapse="collapsed" router background-color="#304156" text-color="#bfcbd9" active-text-color="#409eff">
        <el-menu-item index="/cxr/dashboard"><el-icon><DataBoard /></el-icon><template #title>工作台</template></el-menu-item>
        <el-sub-menu index="cxr"><template #title><el-icon><FirstAidKit /></el-icon><span>胸片辅助分析</span></template>
          <el-menu-item v-for="item in navigation.slice(1)" :key="item.path" :index="item.path">{{ item.label }}</el-menu-item>
        </el-sub-menu>
        <el-sub-menu index="legacy"><template #title><el-icon><Collection /></el-icon><span>血细胞研究</span></template>
          <el-menu-item v-for="item in legacyNavigation" :key="item.path" :index="item.path">{{ item.label }}</el-menu-item>
        </el-sub-menu>
      </el-menu>
      <div v-if="!collapsed" class="sidebar-note">医学影像 · 研究辅助</div>
    </aside>
    <div class="admin-body">
      <header class="admin-navbar">
        <el-button text :aria-label="collapsed ? '展开菜单' : '收起菜单'" @click="collapsed = !collapsed"><el-icon :size="20"><Expand v-if="collapsed"/><Fold v-else/></el-icon></el-button>
        <el-breadcrumb separator="/"><el-breadcrumb-item :to="{path:'/'}">首页</el-breadcrumb-item><el-breadcrumb-item>{{ current.label }}</el-breadcrumb-item></el-breadcrumb>
        <div class="navbar-spacer"/><span class="research-note">仅供研究与辅助参考</span>
        <el-dropdown @command="logout"><span class="admin-user"><el-avatar :size="30"><UserFilled /></el-avatar><span>个人中心</span><el-icon><ArrowDown /></el-icon></span><template #dropdown><el-dropdown-menu><el-dropdown-item command="logout">退出登录</el-dropdown-item></el-dropdown-menu></template></el-dropdown>
      </header>
      <nav class="admin-tabs" aria-label="已访问页面"><router-link v-for="item in tabs" :key="item.path" :to="item.path" :class="{active:route.path===item.path}"><span v-if="route.path===item.path" class="tab-dot"/>{{item.label}}<button v-if="item.path !== '/cxr/dashboard'" aria-label="关闭标签" @click.prevent="closeTab(item.path)">×</button></router-link></nav>
      <main class="admin-content"><router-view /></main>
      <footer class="admin-footer">医疗辅助诊断平台 · 研究结果需经人工复核</footer>
    </div>
  </div>
</template>
<script setup>
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { logout as logoutSession } from './api/auth'
const route=useRoute(),router=useRouter(),collapsed=ref(false)
const navigation=[{path:'/cxr/dashboard',label:'工作台'},{path:'/cxr/data',label:'胸片数据'},{path:'/cxr/training',label:'训练与校准'},{path:'/cxr/analysis',label:'病例分析'},{path:'/cxr/review',label:'人工复核'},{path:'/cxr/reports',label:'分析报告'}]
const legacyNavigation=[{path:'/raw-dataset',label:'原始数据'},{path:'/dataset-annotation',label:'数据标注'},{path:'/train-dataset',label:'训练数据'},{path:'/model-training',label:'模型训练'},{path:'/model-hub',label:'模型管理'},{path:'/inference-service',label:'推理验证'}]
const current=computed(()=>[...navigation,...legacyNavigation].find(item=>item.path===route.path)||{path:route.path,label:'工作台'})
const tabs=ref([navigation[0]])
watch(()=>route.path,()=>{if(route.path!=='/login'&&!tabs.value.some(item=>item.path===route.path))tabs.value.push(current.value)},{immediate:true})
function closeTab(path){const index=tabs.value.findIndex(item=>item.path===path);tabs.value=tabs.value.filter(item=>item.path!==path);if(route.path===path)router.push(tabs.value[Math.max(0,index-1)].path)}
async function logout(){try{await logoutSession()}finally{router.replace('/login')}}
</script>

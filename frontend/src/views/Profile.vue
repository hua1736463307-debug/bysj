<template>
  <el-card style="max-width:650px">
    <template #header><h2>个人中心</h2></template>
    <el-descriptions :column="1"><el-descriptions-item label="账号">{{ user?.username }}</el-descriptions-item><el-descriptions-item label="角色">{{ user?.role }}</el-descriptions-item></el-descriptions>
    <el-divider>修改密码</el-divider>
    <el-form label-width="110px" @submit.prevent="changePassword">
      <el-form-item label="当前密码"><el-input v-model="currentPassword" type="password" show-password autocomplete="current-password" /></el-form-item>
      <el-form-item label="新密码"><el-input v-model="newPassword" type="password" show-password autocomplete="new-password" /></el-form-item>
      <el-form-item label="确认新密码"><el-input v-model="confirmation" type="password" show-password autocomplete="new-password" /></el-form-item>
      <el-form-item><el-button native-type="submit" type="primary" :loading="busy">保存并重新登录</el-button></el-form-item>
    </el-form>
    <el-alert v-if="error" :title="error" type="error" :closable="false" />
  </el-card>
</template>
<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { api, clearAuth, getCurrentUser } from '../api/client'
const router = useRouter(), user = getCurrentUser()
const currentPassword = ref(''), newPassword = ref(''), confirmation = ref(''), busy = ref(false), error = ref('')
async function changePassword() {
  if (busy.value) return
  error.value = ''
  if (newPassword.value !== confirmation.value) { error.value = '两次新密码不一致'; return }
  busy.value = true
  try {
    await api.post('/auth/password', { current_password: currentPassword.value, new_password: newPassword.value })
    clearAuth()
    await router.replace('/login')
  } catch (e) { error.value = e.message }
  finally { busy.value = false }
}
</script>

<template>
  <el-card>
    <template #header><h2>在线会话</h2></template>
    <p>列出尚未过期且未注销的登录会话，有效会话可能来自暂时离开的用户。</p>
    <el-button :disabled="busy" @click="act(refresh)">刷新</el-button>
    <el-alert v-if="error" :title="error" type="error" :closable="false" />
    <el-table :data="sessions" v-loading="busy" stripe>
      <el-table-column prop="username" label="用户名" width="140" />
      <el-table-column prop="client_ip" label="登录 IP" width="150" />
      <el-table-column prop="user_agent" label="客户端" min-width="200" show-overflow-tooltip />
      <el-table-column prop="issued_at" label="登录时间" min-width="180" />
      <el-table-column prop="expires_at" label="到期时间" min-width="180" />
      <el-table-column label="操作" width="120"><template #default="{ row }"><el-button :disabled="busy" @click="revoke(row)">强制注销</el-button></template></el-table-column>
    </el-table>
    <el-pagination v-model:current-page="page" :page-size="20" :total="total" :disabled="busy" layout="total, prev, pager, next" @current-change="act(refresh)" />
  </el-card>
</template>
<script setup>
import { ref, onMounted } from 'vue'
import { ElMessageBox } from 'element-plus'
import { listSessions, revokeSession } from '../api/users'
import { useTaskAction } from '../composables/useTaskAction'
const { error, busy, act } = useTaskAction()
const sessions = ref([]), page = ref(1), total = ref(0)
async function refresh() {
  const result = await listSessions({ page: page.value, page_size: 20 })
  sessions.value = result.items
  total.value = result.total
}
async function revoke(session) {
  try { await ElMessageBox.confirm('注销 ' + session.username + ' 的此登录会话？', '强制注销') }
  catch { return }
  await act(async () => { await revokeSession(session.session_id); await refresh() })
}
onMounted(() => act(refresh))
</script>

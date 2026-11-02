<template>
  <section class="users-page">
    <el-card>
      <template #header><h2>账户管理</h2></template>
      <el-form inline @submit.prevent="searchUsers">
        <el-form-item label="用户名称"><el-input v-model="search" clearable placeholder="用户名或昵称" /></el-form-item>
        <el-form-item><el-button native-type="submit" :disabled="busy">搜索</el-button></el-form-item>
        <el-form-item><el-button type="primary" :disabled="busy" @click="openEditor()">新增用户</el-button></el-form-item>
      </el-form>
      <el-alert v-if="error" :title="error" type="error" :closable="false" role="alert" />
      <el-table :data="users" v-loading="busy" stripe border>
        <el-table-column prop="id" label="编号" width="80" />
        <el-table-column prop="username" label="用户名" min-width="130" />
        <el-table-column prop="display_name" label="昵称" min-width="130" />
        <el-table-column label="角色" width="110"><template #default="{ row }">{{ roles[row.role] }}</template></el-table-column>
        <el-table-column label="状态" width="100"><template #default="{ row }"><el-tag :type="row.status === 'active' ? 'success' : 'info'">{{ row.status === 'active' ? '正常' : '停用' }}</el-tag></template></el-table-column>
        <el-table-column prop="last_login_at" label="最近登录" min-width="180" />
        <el-table-column label="操作" min-width="370" fixed="right"><template #default="{ row }">
          <template v-if="row.role !== 'admin'">
            <el-button size="small" :disabled="busy" @click="openEditor(row)">编辑 / 启停</el-button>
            <el-button size="small" :disabled="busy" @click="openPassword(row)">重置密码</el-button>
            <el-button size="small" :disabled="busy" @click="removeUser(row)">删除</el-button>
            <el-button size="small" :disabled="busy" @click="openProfile(row)">组织与角色</el-button>
          </template><span v-else>管理员账号受保护</span>
        </template></el-table-column>
      </el-table>
      <el-pagination v-model:current-page="page" :page-size="20" :total="total" :disabled="busy" layout="total, prev, pager, next" @current-change="loadUsers" />
    </el-card>
    <el-dialog v-model="editing" :title="selectedId ? '编辑用户' : '新增用户'" width="480px" :close-on-click-modal="false" :before-close="closeEditor">
      <el-form label-width="90px" @submit.prevent="save">
        <el-form-item label="用户名"><el-input v-model="form.username" :disabled="!!selectedId" maxlength="64" /></el-form-item>
        <el-form-item label="昵称"><el-input v-model="form.nickname" maxlength="64" /></el-form-item>
        <el-form-item v-if="!selectedId" label="密码"><el-input v-model="form.password" type="password" show-password maxlength="72" /></el-form-item>
        <el-form-item label="角色"><el-select v-model="form.role"><el-option label="研究员" value="researcher" /><el-option label="复核员" value="reviewer" /></el-select></el-form-item>
        <el-form-item v-if="selectedId" label="状态"><el-select v-model="form.status"><el-option label="正常" value="active" /><el-option label="停用" value="disabled" /></el-select></el-form-item>
        <el-alert v-if="dialogError" :title="dialogError" type="error" :closable="false" />
      </el-form>
      <template #footer><el-button :disabled="saving" @click="editing = false">取消</el-button><el-button type="primary" :loading="saving" @click="save">保存</el-button></template>
    </el-dialog>
    <el-dialog v-model="resetting" title="重置密码" width="480px" :close-on-click-modal="false" :before-close="closePassword">
      <p>重置 {{ passwordUser?.username }} 的密码后，其已有登录会话将失效。</p>
      <el-input v-model="newPassword" aria-label="新密码" type="password" show-password maxlength="72" placeholder="至少 8 位" />
      <el-alert v-if="dialogError" :title="dialogError" type="error" :closable="false" />
      <template #footer><el-button :disabled="saving" @click="resetting = false">取消</el-button><el-button type="primary" :loading="saving" @click="savePassword">确认重置</el-button></template>
    </el-dialog>
    <el-dialog v-model="profileVisible" title="组织与角色" width="480px">
      <el-form label-width="90px">
        <el-form-item label="部门"><el-select v-model="profile.department_id" clearable><el-option v-for="item in departments" :key="item.id" :label="item.name" :value="item.id" /></el-select></el-form-item>
        <el-form-item label="自定义角色"><el-select v-model="profile.role_id" clearable><el-option v-for="item in customRoles" :key="item.id" :label="item.name" :value="item.id" /></el-select></el-form-item>
      </el-form>
      <el-alert v-if="dialogError" :title="dialogError" type="error" :closable="false" />
      <template #footer><el-button type="primary" :loading="saving" @click="saveUserProfile">保存</el-button></template>
    </el-dialog>
  </section>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { listUsers, createUser, updateUser, resetPassword, deleteUser } from '../api/users'
import { listDepartments, listRoles, getProfile, saveProfile } from '../api/system'

const roles = { admin: '管理员', researcher: '研究员', reviewer: '复核员' }
const users = ref([]), total = ref(0), page = ref(1), search = ref('')
const busy = ref(false), saving = ref(false), error = ref(''), dialogError = ref('')
const editing = ref(false), selectedId = ref(null), resetting = ref(false)
const passwordUser = ref(null), newPassword = ref('')
const form = reactive({ username: '', nickname: '', password: '', role: 'researcher', status: 'active' })
const profileVisible = ref(false), profileUser = ref(null), departments = ref([]), customRoles = ref([])
const profile = reactive({ department_id: null, role_id: null })
async function openProfile(user) {
  busy.value = true
  dialogError.value = ''
  try {
    const [departmentResult, roleResult, current] = await Promise.all([listDepartments(), listRoles(), getProfile(user.id)])
    departments.value = departmentResult.items.filter(item => item.status === 'active')
    customRoles.value = roleResult.items.filter(item => item.status === 'active')
    Object.assign(profile, current)
    profileUser.value = user
    profileVisible.value = true
  } catch (e) { error.value = e.message }
  finally { busy.value = false }
}
async function saveUserProfile() {
  if (saving.value) return
  saving.value = true
  try {
    await saveProfile(profileUser.value.id, { department_id: profile.department_id || null, role_id: profile.role_id || null })
    profileVisible.value = false
    await loadUsers()
  } catch (e) { dialogError.value = e.message }
  finally { saving.value = false }
}

async function loadUsers() {
  busy.value = true
  error.value = ''
  try {
    const result = await listUsers({ page: page.value, page_size: 20, search: search.value })
    users.value = result.items
    total.value = result.total
  } catch (e) { error.value = e.message }
  finally { busy.value = false }
}
function searchUsers() { page.value = 1; loadUsers() }
function openEditor(user) {
  selectedId.value = user?.id || null
  Object.assign(form, { username: user?.username || '', nickname: user?.display_name || '', password: '', role: user?.role || 'researcher', status: user?.status || 'active' })
  dialogError.value = ''
  editing.value = true
}
function closeEditor(done) { if (!saving.value) done() }
function closePassword(done) { if (!saving.value) done() }
async function save() {
  if (saving.value) return
  saving.value = true
  dialogError.value = ''
  try {
    if (selectedId.value) {
      await updateUser(selectedId.value, { nickname: form.nickname, role: form.role, status: form.status })
    } else {
      await createUser({ username: form.username, nickname: form.nickname, password: form.password, role: form.role })
    }
    form.password = ''
    editing.value = false
    ElMessage.success('用户已保存')
    await loadUsers()
  } catch (e) { dialogError.value = e.message }
  finally { saving.value = false }
}
function openPassword(user) {
  passwordUser.value = user
  newPassword.value = ''
  dialogError.value = ''
  resetting.value = true
}
async function savePassword() {
  if (saving.value) return
  saving.value = true
  dialogError.value = ''
  try {
    await resetPassword(passwordUser.value.id, newPassword.value)
    newPassword.value = ''
    resetting.value = false
    ElMessage.success('密码已重置')
  } catch (e) { dialogError.value = e.message }
  finally { saving.value = false }
}
onMounted(loadUsers)
async function removeUser(user) {
  try {
    await ElMessageBox.confirm(`删除账号 ${user.username}？历史研究记录会保留，已有会话将失效。`, '删除账号')
  } catch { return }
  busy.value = true
  try {
    await deleteUser(user.id)
    if (users.value.length === 1 && page.value > 1) page.value--
    await loadUsers()
  } catch (e) { error.value = e.message }
  finally { busy.value = false }
}
</script>

<style scoped>
.users-page h2 { margin: 0; font-size: 18px; }
.el-pagination { margin-top: 20px; }
.el-alert { margin-bottom: 16px; }
</style>

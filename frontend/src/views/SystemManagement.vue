<template>
  <el-card>
    <template #header><h2>{{ title }}</h2></template>
    <el-alert v-if="error" :title="error" type="error" :closable="false" />
    <el-form inline @submit.prevent="searchLogs">
      <el-form-item v-if="isLog" label="用户名"><el-input v-model="search" clearable /></el-form-item>
      <el-form-item><el-button :disabled="busy" native-type="submit">刷新 / 搜索</el-button></el-form-item>
      <el-form-item v-if="!isLog && panel !== 'menus'"><el-button type="primary" @click="edit()">新增</el-button></el-form-item>
    </el-form>
    <el-table :data="rows" v-loading="busy" :row-key="panel === 'menus' ? 'path' : 'id'" stripe border>
      <template v-if="panel === 'departments'">
        <el-table-column prop="name" label="部门名称" />
        <el-table-column label="上级部门"><template #default="{ row }">{{ departmentName(row.parent_id) }}</template></el-table-column>
        <el-table-column prop="sort_order" label="排序" width="90" />
      </template>
      <template v-else-if="panel === 'roles'">
        <el-table-column prop="name" label="角色名称" />
        <el-table-column label="业务身份"><template #default="{ row }">{{ row.base_role === 'reviewer' ? '复核员' : '研究员' }}</template></el-table-column>
        <el-table-column label="允许模块" min-width="250"><template #default="{ row }">{{ row.modules.map(key => modules[key]).join('、') }}</template></el-table-column>
      </template>
      <template v-else-if="panel === 'menus'">
        <el-table-column prop="label" label="菜单名称" />
        <el-table-column prop="path" label="页面路径" min-width="230" />
        <el-table-column prop="sort_order" label="排序" width="90" />
        <el-table-column label="显示"><template #default="{ row }">{{ row.visible ? '显示' : '隐藏' }}</template></el-table-column>
      </template>
      <template v-else>
        <el-table-column prop="created_at" label="时间" min-width="190" />
        <el-table-column prop="username" label="用户名" width="140" />
        <el-table-column prop="action" label="操作" min-width="250" show-overflow-tooltip />
        <el-table-column prop="status" label="响应状态" width="100" />
        <el-table-column prop="client_ip" label="来源 IP" width="140" />
        <el-table-column prop="request_id" label="请求 ID" min-width="170" show-overflow-tooltip />
      </template>
      <el-table-column v-if="!isLog && panel !== 'menus'" label="状态" width="100"><template #default="{ row }"><el-tag :type="row.status === 'active' ? 'success' : 'info'">{{ row.status === 'active' ? '正常' : '停用' }}</el-tag></template></el-table-column>
      <el-table-column v-if="!isLog" label="操作" width="180"><template #default="{ row }"><el-button size="small" :disabled="busy" @click="edit(row)">编辑</el-button><el-button v-if="panel !== 'menus'" size="small" :disabled="busy" @click="remove(row)">删除</el-button></template></el-table-column>
    </el-table>
    <el-pagination v-if="isLog" v-model:current-page="page" :page-size="20" :total="total" :disabled="busy" layout="total, prev, pager, next" @current-change="act(load)" />
    <el-dialog v-model="visible" :title="editingId ? '编辑' : '新增'" width="min(620px, 95vw)" :close-on-click-modal="false">
      <el-form label-width="110px">
        <el-form-item label="名称"><el-input v-model="form.name" maxlength="64" /></el-form-item>
        <template v-if="panel === 'menus'">
          <el-form-item label="页面路径"><el-input :model-value="form.path" disabled /></el-form-item>
          <el-form-item label="排序"><el-input-number v-model="form.sort_order" :min="0" :max="100000" /></el-form-item>
          <el-form-item label="显示"><el-switch v-model="form.visible" /></el-form-item>
        </template>
        <template v-if="panel === 'departments'">
          <el-form-item label="上级部门"><el-select v-model="form.parent_id" clearable><el-option v-for="department in rows.filter(item => item.id !== editingId)" :key="department.id" :label="department.name" :value="department.id" /></el-select></el-form-item>
          <el-form-item label="排序"><el-input-number v-model="form.sort_order" :min="0" :max="100000" /></el-form-item>
        </template>
        <template v-else-if="panel === 'roles'">
          <el-form-item label="业务身份"><el-select v-model="form.base_role"><el-option label="研究员" value="researcher" /><el-option label="复核员" value="reviewer" /></el-select></el-form-item>
          <el-form-item label="模块权限"><el-checkbox-group v-model="form.modules"><el-checkbox v-for="(label, key) in modules" :key="key" :label="key">{{ label }}</el-checkbox></el-checkbox-group></el-form-item>
          <p>训练、分析与标注所依赖的数据和模型模块将自动一并授权。角色变更后，关联用户需重新登录。</p>
        </template>
        <el-form-item v-if="panel !== 'menus'" label="状态"><el-select v-model="form.status"><el-option label="正常" value="active" /><el-option label="停用" value="disabled" /></el-select></el-form-item>
      </el-form>
      <el-alert v-if="error" :title="error" type="error" :closable="false" />
      <template #footer><el-button :disabled="busy" @click="visible = false">取消</el-button><el-button type="primary" :loading="busy" @click="act(save)">保存</el-button></template>
    </el-dialog>
  </el-card>
</template>
<script setup>
import { computed, reactive, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessageBox } from 'element-plus'
import { listDepartments, saveDepartment, deleteDepartment, listRoles, saveRole, deleteRole, listLogs, listMenus, saveMenu } from '../api/system'
import { useTaskAction } from '../composables/useTaskAction'
const route = useRoute()
const { busy, error, act } = useTaskAction()
const panel = computed(() => route.meta.systemPanel)
const titles = { departments: '部门管理', roles: '角色与模块授权', menus: '菜单管理', login: '登录日志', operation: '操作日志', fault: '系统故障日志' }
const title = computed(() => titles[panel.value])
const isLog = computed(() => ['login', 'operation', 'fault'].includes(panel.value))
const rows = ref([]), modules = ref({}), page = ref(1), total = ref(0), search = ref('')
const visible = ref(false), editingId = ref(null)
const form = reactive({ name: '', parent_id: null, sort_order: 0, base_role: 'researcher', modules: [], status: 'active', path: '', visible: true })
function departmentName(id) { return rows.value.find(row => row.id === id)?.name || '顶级部门' }
async function load() {
  const requestedPanel = panel.value
  const result = isLog.value ? await listLogs({ kind: panel.value, search: search.value, page: page.value, page_size: 20 })
    : panel.value === 'departments' ? await listDepartments() : panel.value === 'menus' ? await listMenus() : await listRoles()
  if (panel.value !== requestedPanel) return
  rows.value = result.items
  modules.value = result.modules || {}
  total.value = result.total || 0
}
function searchLogs() { page.value = 1; act(load) }
function edit(row) {
  editingId.value = row?.id || row?.path || null
  Object.assign(form, { name: row?.name || '', parent_id: row?.parent_id || null, sort_order: row?.sort_order || 0,
    base_role: row?.base_role || 'researcher', modules: [...(row?.modules || [])], status: row?.status || 'active' })
  if (panel.value === 'menus') Object.assign(form, { name: row.label, path: row.path, visible: row.visible })
  error.value = ''
  visible.value = true
}
async function save() {
  if (panel.value === 'menus') {
    await saveMenu({ path: form.path, label: form.name, sort_order: form.sort_order, visible: form.visible })
    window.dispatchEvent(new Event('navigation-updated'))
  } else if (panel.value === 'departments') {
    await saveDepartment(editingId.value, { name: form.name, parent_id: form.parent_id || null, sort_order: form.sort_order, status: form.status })
  } else {
    await saveRole(editingId.value, { name: form.name, base_role: form.base_role, modules: form.modules, status: form.status })
  }
  visible.value = false
  await load()
}
async function remove(row) {
  try { await ElMessageBox.confirm('删除「' + row.name + '」？有关联用户或子部门时会拒绝删除。', '确认删除') }
  catch { return }
  await act(async () => {
    if (panel.value === 'departments') await deleteDepartment(row.id)
    else await deleteRole(row.id)
    await load()
  })
}
watch(panel, () => { rows.value = []; visible.value = false; page.value = 1; search.value = ''; act(load) }, { immediate: true })
</script>

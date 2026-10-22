<template>
  <main class="medical-page">
    <h1>原始数据</h1><p>导入图像与 YOLO 标签，保留来源信息；导入完成后可标注和冻结版本。</p>
    <fieldset><legend>导入数据集</legend>
      <div class="toolbar"><label>名称 <input aria-label="数据集名称" v-model="name" /></label><label>来源 URL <input v-model="sourceUrl" /></label><label>授权说明 <input v-model="license" /></label></div>
      <div class="toolbar"><label>类别映射 <input aria-label="类别映射" v-model="classMap" size="48" /></label><input aria-label="导入文件" type="file" multiple accept=".zip,.jpg,.jpeg,.png,.bmp,.txt,.yaml,.json" @change="files = Array.from($event.target.files)" /><el-button type="primary" :disabled="busy || !files.length" @click="act(upload)">开始导入</el-button></div>
      <p class="muted">可选多张图片、同名 .txt 标签，或一个 ZIP；类别映射格式为 JSON。</p>
    </fieldset>
    <p v-if="error" role="alert" class="error">{{ error }}</p>
    <JobProgress :job-id="jobId" @terminal="refresh" />
    <div class="toolbar"><el-button @click="act(refresh)">刷新列表</el-button><span>共 {{ total }} 个数据集</span></div>
    <table><thead><tr><th>名称 / ID</th><th>来源</th><th>最新版本</th><th>图像数</th><th>操作</th></tr></thead><tbody>
      <tr v-for="d in datasets" :key="d.dataset_id"><td>{{ d.name }}<div class="muted">{{ d.dataset_id }}</div></td><td>{{ d.source?.url || '未填写' }}<div>{{ d.source?.license || '授权信息未填写' }}</div></td><td>{{ d.latest_version?.state || '无版本' }}</td><td>{{ d.latest_version?.asset_count ?? 0 }}</td><td><router-link :to="{ path:'/dataset-annotation', query:{ dataset:d.dataset_id, version:d.latest_version?.version_id } }">标注</router-link> · <router-link to="/train-dataset">版本管理</router-link> <el-button type="danger" text @click="act(() => remove(d))">删除</el-button></td></tr>
    </tbody></table>
    <el-pagination v-model:current-page="page" :page-size="10" :total="total" layout="prev, pager, next" @current-change="act(refresh)" />
  </main>
</template>
<script setup>
import { ref, onMounted } from 'vue'
import { ElMessageBox } from 'element-plus'
import { listDatasets, importDataset, deleteDataset } from '../api/datasets'
import { useTaskAction } from '../composables/useTaskAction'
import JobProgress from '../components/medical/JobProgress.vue'
const { error, busy, act } = useTaskAction()
const datasets=ref([]), total=ref(0), page=ref(1), name=ref(''), sourceUrl=ref(''), license=ref(''), files=ref([]), jobId=ref('')
const classMap=ref('{"0":"RBC","1":"WBC","2":"Platelets"}')
async function refresh(){ const data=await listDatasets({page:page.value,page_size:10}); datasets.value=data.items;total.value=data.total }
async function upload(){ const classes=JSON.parse(classMap.value); if(!classes || Array.isArray(classes) || typeof classes!=='object') throw new Error('类别映射必须是 JSON 对象'); const form=new FormData(); for(const file of files.value) form.append('files',file);form.append('name',name.value || '新数据集');form.append('source_url',sourceUrl.value);form.append('license',license.value);form.append('class_map',JSON.stringify(classes)); jobId.value=(await importDataset(form)).job_id }
async function remove(d){ await ElMessageBox.confirm(`删除数据集「${d.name}」？被引用的数据集会拒绝删除。`,'删除数据集');await deleteDataset(d.dataset_id);await refresh() }
onMounted(()=>act(refresh))
</script>

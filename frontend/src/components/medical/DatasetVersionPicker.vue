<template>
  <div class="toolbar">
    <label>数据集 <select aria-label="数据集" v-model="datasetId" @change="loadVersions"><option value="">请选择</option><option v-for="d in datasets" :key="d.dataset_id" :value="d.dataset_id">{{ d.name }}</option></select></label>
    <label>版本 <select aria-label="版本" v-model="versionId" @change="choose"><option value="">请选择</option><option v-for="v in versions" :key="v.version_id" :value="v.version_id">{{ v.state }} · {{ v.asset_count }} 图 · {{ v.version_id.slice(0,8) }}</option></select></label>
    <el-button @click="refresh">刷新数据</el-button><span class="error" v-if="error">{{ error }}</span>
  </div>
</template>
<script setup>
import { ref, onMounted } from 'vue'
import { listDatasets, listVersions } from '../../api/datasets'
const props = defineProps({ frozenOnly: Boolean, taskType: { type: String, default: 'detect' } })
const emit = defineEmits(['change'])
const datasets = ref([]), versions = ref([]), datasetId = ref(''), versionId = ref(''), error = ref('')
let generation = 0
function choose() { emit('change', { datasetId: datasetId.value, version: versions.value.find(v => v.version_id === versionId.value) || null }) }
async function loadVersions() {
  const ticket = ++generation; versionId.value = ''; versions.value = []; choose()
  if (!datasetId.value) return
  try { const data = await listVersions(datasetId.value); if(ticket !== generation) return; versions.value = data.items.filter(v => !props.frozenOnly || v.state === 'frozen'); versionId.value = versions.value[0]?.version_id || ''; choose() }
  catch(e) { error.value = e.message }
}
async function refresh() {
  try {
    const all = []; let page = 1, data
    do { data = await listDatasets({ task: props.taskType, page: page++, page_size: 100 }); all.push(...data.items) } while (all.length < data.total)
    datasets.value = all; error.value = ''
    if (!all.some(d => d.dataset_id === datasetId.value)) datasetId.value = all[0]?.dataset_id || ''
    await loadVersions()
  } catch(e) { error.value = e.message }
}
async function select(dataset, version) {
  datasetId.value = dataset
  await loadVersions()
  if (!versions.value.some(item => item.version_id === version)) throw new Error('训练来源版本已不可用')
  versionId.value = version
  choose()
}
defineExpose({ refresh, select }); onMounted(refresh)
</script>

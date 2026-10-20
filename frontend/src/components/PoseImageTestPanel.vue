<template>
  <section class="pose-image-tests">
    <el-alert type="info" :closable="false" title="上传图片后，按当前突然倒地流程逐人测试；测试不会发送告警。红框：倒地，绿框：未倒地，橙框：证据不足。" />
    <el-alert type="warning" :closable="false" title="该分类器判断单帧倒地姿态，不包含地面、床或沙发支撑判断。分数不是经过校准的准确率；请结合实际图片核对结果。" />
    <p>模型：{{ taskId }}；本次测试阈值：{{ state.threshold ?? '—' }}；训练推荐值：{{ state.recommended_threshold ?? '—' }}；当前告警：{{ state.active ? `分类器 ${state.active.model_id}` : '原几何规则' }}</p>
    <div class="test-actions">
      <input ref="fileInput" aria-label="选择本地测试图片" type="file" accept="image/jpeg,image/png,image/webp" multiple :disabled="operating" @change="choose" />
      <el-button type="primary" :loading="busy" :disabled="operating || showArchived || !files.length || !ready" @click="run">测试所选图片（{{ files.length }}）</el-button>
      <el-button :disabled="operating || (!files.length && !selected)" @click="clearSelection">清空已选图片</el-button>
      <el-button :disabled="operating" @click="reload">刷新结果</el-button>
      <el-button type="danger" plain :loading="clearing" :disabled="operating || !state.active_count" @click="clearHistory">清空测试记录</el-button>
      <el-button :disabled="operating" @click="switchHistory">{{ showArchived ? '返回测试记录' : `已清空记录（${state.archived_count || 0}）` }}</el-button>
    </div>
    <p>每次最多 10 张，每张不超过 5 MiB、1200 万像素；请先停止持续监控，等待 GPU 训练完成。</p>
    <el-alert v-if="error" type="error" :closable="false" :title="error" />
    <p>{{ showArchived ? '已清空记录' : '测试记录' }}共 {{ state.total || 0 }} 条，显示最近 20 条。清空仅整理本人当前模型的测试历史，可恢复；不改变模型或当前告警配置。</p>
    <el-table :data="state.tests || []" @row-click="view" highlight-current-row>
      <el-table-column label="测试时间" min-width="160"><template #default="{ row }">{{ row.created_at }}</template></el-table-column>
      <el-table-column label="人员 / 倒地 / 拒判"><template #default="{ row }">{{ row.result.person_count }} / {{ row.result.fall_count }} / {{ row.result.person_count-row.result.valid_people }}</template></el-table-column>
      <el-table-column label="操作"><template #default="{ row }"><el-button link type="primary" @click.stop="view(row)">查看结果</el-button><el-button v-if="showArchived" link type="primary" :loading="restoring===row.id" :disabled="operating" @click.stop="restore(row)">恢复记录</el-button></template></el-table-column>
    </el-table>
    <div v-if="selected" class="test-result">
      <p>图片测试 {{ selected.id }}；该次阈值：{{ selected.result.spec.threshold }}；耗时 {{ selected.result.elapsed_seconds }} 秒；{{ selected.result.person_count ? '逐人结果如下' : '未得到可关联的人员姿态，不能据此认为画面中一定无人倒地' }}。</p>
      <img v-if="preview" :src="preview" alt="本地图片姿态分类标注结果" class="test-preview" />
      <el-table :data="selected.result.people">
        <el-table-column prop="person" label="人员" width="80" />
        <el-table-column label="分类结果"><template #default="{ row }">{{ row.valid ? (row.fallen ? '倒地' : '未倒地') : '证据不足（拒判）' }}</template></el-table-column>
        <el-table-column label="倒地分数"><template #default="{ row }">{{ row.probability == null ? '—' : row.probability.toFixed(4) }}</template></el-table-column>
        <el-table-column prop="keypoints_available" label="有效关键点" />
        <el-table-column label="判断依据"><template #default="{ row }">{{ reasons[row.reason] || row.reason }}</template></el-table-column>
      </el-table>
      <el-checkbox v-if="admin && !showArchived" v-model="reviewed">我已核对本地测试结果，确认用于突然倒地告警，并了解支撑场景尚未验证</el-checkbox>
    </div>
    <div v-if="admin" class="test-actions">
      <el-button type="danger" :loading="binding" :disabled="operating || !canActivate || !reviewed" @click="bind(true)">启用到突然倒地告警</el-button>
      <el-button :loading="binding" :disabled="operating || state.active?.model_id !== taskId" @click="bind(false)">停用分类器，恢复原规则</el-button>
    </div>
    <p v-else>告警分类器的启用和停用由管理员操作。</p>
  </section>
</template>

<script setup>
import {computed,onMounted,onUnmounted,ref} from 'vue'
import {ElMessage,ElMessageBox} from 'element-plus'
import api from '../api/api.js'
import {createPoseRequestId} from '../utils/poseTraining.mjs'
const props=defineProps({taskId:{type:String,required:true}})
const state=ref({tests:[]}),files=ref([]),busy=ref(false),binding=ref(false),ready=ref(false),admin=ref(false),selected=ref(null),preview=ref(''),reviewed=ref(false),error=ref('')
const fileInput=ref(null),clearing=ref(false),restoring=ref(''),showArchived=ref(false)
const operating=computed(()=>busy.value || binding.value || clearing.value || !!restoring.value)
let stopped=false,viewVersion=0
const reasons={CLASSIFIER_FALL:'倒地分数达到锁定阈值',CLASSIFIER_NON_FALL:'倒地分数低于锁定阈值',INSUFFICIENT_KEYPOINTS:'关键点不足或肩髋不可见',INVALID_KEYPOINTS:'关键点或人员框无效',DEGENERATE_TORSO:'躯干关键点异常',CLASSIFIER_UNAVAILABLE:'分类器加载或校验失败，不告警'}
const canActivate=computed(()=>!showArchived.value && !selected.value?.result.history_archived && selected.value?.result.valid_people>0 && selected.value.result.spec.pipeline_sha256===state.value.pipeline_sha256 && selected.value.result.spec.model_id===props.taskId && selected.value.result.spec.threshold===state.value.threshold)
const message=e=>typeof e.response?.data?.detail==='string' ? e.response.data.detail : e.message || '操作失败，请重试'
async function reload() {
  try { const value=await api.get(`/pose-training/jobs/${props.taskId}/image-tests`,{params:{archived:showArchived.value}});if(stopped)return;state.value=value;ready.value=true }
  catch(e){if(!stopped){error.value=message(e);ready.value=false}}
}
function resetPreview(){viewVersion++;selected.value=null;reviewed.value=false;if(preview.value){URL.revokeObjectURL(preview.value);preview.value=''}}
function clearSelection(){files.value=[];if(fileInput.value)fileInput.value.value='';resetPreview();error.value=''}
async function switchHistory(){if(operating.value)return;showArchived.value=!showArchived.value;resetPreview();await reload()}
async function clearHistory(){
  if(operating.value || !state.value.active_count)return
  try{
    await ElMessageBox.confirm('清空当前模型下你本人的全部测试记录？记录可在“已清空记录”中恢复，不影响训练模型或当前告警配置。','清空测试记录',{confirmButtonText:'清空记录',cancelButtonText:'取消',type:'warning'})
    if(stopped)return
    clearing.value=true;error.value='';let count=0,before=null
    do{
      const result=await api.delete(`/pose-training/jobs/${props.taskId}/image-tests`,{params:before?{before}:{}})
      count+=result.archived_count;before=result.before
      if(stopped)return
      if(!result.remaining_count)break
      if(!result.archived_count)throw new Error('剩余记录正在更新，请刷新后重试')
    }while(!stopped)
    clearSelection();showArchived.value=false;await reload();ElMessage.success(`已清空 ${count} 条测试记录，可在已清空记录中恢复`)
  }catch(e){if(!stopped && e!=='cancel' && e!=='close')error.value=message(e)}finally{clearing.value=false}
}
async function restore(row){
  if(operating.value)return
  restoring.value=row.id;error.value=''
  try{await api.post(`/pose-training/image-tests/${row.id}/restore`);if(stopped)return;if(selected.value?.id===row.id)resetPreview();await reload();ElMessage.success('测试记录已恢复')}
  catch(e){if(!stopped)error.value=message(e)}finally{restoring.value=''}
}
function choose(event) {
  const chosen=Array.from(event.target.files || [])
  if(chosen.length>10 || chosen.some(f=>f.size>5*1024*1024)){ElMessage.error('最多选择 10 张图片，每张最大 5 MiB');files.value=[];event.target.value='';return}
  try { files.value=chosen.map(file=>({file,requestId:createPoseRequestId()}));error.value='' }
  catch(e){error.value=message(e);files.value=[]}
}
async function run() {
  if(busy.value)return
  busy.value=true;error.value=''
  try {
    await reload();if(!ready.value)throw new Error(error.value || '请刷新测试配置后重试')
    for(const item of files.value){
      if(stopped)break
      const signature=`${props.taskId}:${state.value.pipeline_sha256}:${state.value.threshold}`
      if(item.signature && item.signature!==signature)item.requestId=createPoseRequestId()
      item.signature=signature
      const data=new FormData();data.append('file',item.file);data.append('request_id',item.requestId);data.append('threshold',String(state.value.threshold))
      const row=await api.post(`/pose-training/jobs/${props.taskId}/image-tests`,data,{timeout:90000,headers:{'Content-Type':'multipart/form-data'}})
      if(stopped)break
      await reload();await view(row)
    }
  }catch(e){if(!stopped)error.value=message(e)}finally{busy.value=false}
}
async function view(row) {
  const version=++viewVersion;selected.value=row;reviewed.value=false
  if(preview.value){URL.revokeObjectURL(preview.value);preview.value=''}
  try { const blob=await api.get(`/pose-training/image-tests/${row.id}/files/preview`,{responseType:'blob'});if(!stopped && version===viewVersion)preview.value=URL.createObjectURL(blob) }
  catch(e){if(!stopped && version===viewVersion)error.value=message(e)}
}
async function bind(enable) {
  if(binding.value)return
  try {
    await ElMessageBox.confirm(enable ? '启用后，突然倒地告警将使用本分类器判断；其他告警保持原配置。请确认已核对测试图片。' : '停用后恢复突然倒地原有几何规则。','更新倒地告警配置')
    binding.value=true
    await api.post(`/pose-training/jobs/${props.taskId}/alarm-binding`,{revision:state.value.revision,test_id:enable ? selected.value.id : null,enable,reviewed:enable && reviewed.value})
    await reload();ElMessage.success(enable ? '倒地分类器已启用' : '已恢复原几何规则')
  }catch(e){if(e!=='cancel' && e!=='close')error.value=message(e)}finally{binding.value=false}
}
onMounted(async()=>{await reload();try{const response=await api.get('/auth/me');if(!stopped)admin.value=response.user?.role==='admin'}catch{}})
onUnmounted(()=>{stopped=true;viewVersion++;if(preview.value)URL.revokeObjectURL(preview.value)})
</script>

<style scoped>
.test-actions { display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:16px 0; }
.pose-image-tests :deep(.el-button.is-link) { background:transparent !important;box-shadow:none !important; }
.test-preview { max-width:100%;max-height:520px;object-fit:contain;display:block;margin:12px auto; }
.test-result { margin:20px 0; }
</style>

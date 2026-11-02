<template>
  <el-dialog :model-value="visible" :title="titles[operation]" width="min(650px, 95vw)" :close-on-click-modal="false" :before-close="close">
    <el-alert title="操作将生成派生版本。来源版本须先冻结，合并会检查跨划分重复；增强仅作用于训练集。" type="info" :closable="false" />
    <p>来源版本 {{ versionIds.map(id => id.slice(0, 8)).join('、') }}</p>
    <el-form label-width="110px" style="margin-top:20px">
      <el-form-item v-if="operation === 'merge'" label="新数据集名称"><el-input v-model="name" /></el-form-item>
      <template v-if="operation === 'augment'">
        <el-form-item label="旋转"><el-checkbox-group v-model="rotations"><el-checkbox :label="90">90°</el-checkbox><el-checkbox :label="180">180°</el-checkbox><el-checkbox :label="270">270°</el-checkbox></el-checkbox-group></el-form-item>
        <el-form-item label="镜像 / 灰度"><el-checkbox v-model="flip">水平镜像</el-checkbox><el-checkbox v-model="grayscale">灰度</el-checkbox></el-form-item>
        <el-form-item label="亮度"><el-checkbox-group v-model="brightness"><el-checkbox :label="0.8">0.8 倍</el-checkbox><el-checkbox :label="1.2">1.2 倍</el-checkbox></el-checkbox-group></el-form-item>
        <el-form-item label="对比度"><el-checkbox-group v-model="contrast"><el-checkbox :label="0.8">0.8 倍</el-checkbox><el-checkbox :label="1.2">1.2 倍</el-checkbox></el-checkbox-group></el-form-item>
      </template>
      <template v-if="operation === 'preannotate'">
        <el-form-item label="检测模型"><el-select v-model="modelId" filterable><el-option v-for="model in models" :key="model.model_id" :label="model.name" :value="model.model_id" /></el-select></el-form-item>
        <el-form-item label="置信度"><el-input-number v-model="confidence" :min="0" :max="1" :step=".05" /></el-form-item>
        <el-form-item label="写入方式"><el-radio-group v-model="annotationMode"><el-radio label="append">追加</el-radio><el-radio label="replace">替换</el-radio></el-radio-group></el-form-item>
        <el-form-item label="类别映射"><el-input v-model="mapping" type="textarea" :rows="3" placeholder='留空按类别名称匹配，或填写 {"模型类别ID": 目标类别ID}' /></el-form-item>
        <p>本次处理 {{ assetIds.length ? assetIds.length + ' 张所选图片' : '整个版本' }}。AI 结果写入可编辑草稿版本，需人工检查。</p>
      </template>
    </el-form>
    <template v-if="operation === 'augment'">
      <el-button :disabled="busy" @click="preview">生成增强预览</el-button>
      <div class="augmentation-previews"><figure v-for="item in previews" :key="item.label"><img :src="item.url" :alt="item.label" /><figcaption>{{ item.label }}</figcaption></figure></div>
    </template>
    <el-alert v-if="error" :title="error" type="error" :closable="false" />
    <template #footer><el-button :disabled="busy" @click="close()">取消</el-button><el-button type="primary" :loading="busy" @click="submit">创建后台任务</el-button></template>
  </el-dialog>
</template>
<script setup>
import { ref, watch, onBeforeUnmount } from 'vue'
import { createDerivative, getVersion, listAssets, augmentationPreview } from '../../api/datasets'
import { listModels } from '../../api/models'
const props = defineProps({ visible: Boolean, operation: String, versionIds: { type: Array, default: () => [] }, assetIds: { type: Array, default: () => [] } })
const emit = defineEmits(['update:visible', 'submitted'])
const titles = { merge: '合并数据集', augment: '数据增强', preannotate: 'AI 辅助预标注' }
const busy = ref(false), error = ref(''), models = ref([]), modelId = ref('')
const name = ref('合并数据集'), rotations = ref([]), flip = ref(false), grayscale = ref(false)
const brightness = ref([]), contrast = ref([]), confidence = ref(.5), annotationMode = ref('append'), mapping = ref('')
const requestKey = ref('')
const requestBody = ref('')
const previews = ref([])
let previewGeneration = 0
function clearPreviews() { ++previewGeneration; for (const item of previews.value) URL.revokeObjectURL(item.url); previews.value = [] }
onBeforeUnmount(clearPreviews)
watch([rotations, brightness, contrast, flip, grayscale], clearPreviews, { deep: true })
async function preview() {
  clearPreviews()
  const ticket = previewGeneration
  const operations = [...rotations.value.map(value => ['rotate', value]), ...brightness.value.map(value => ['brightness', value]), ...contrast.value.map(value => ['contrast', value])]
  if (flip.value) operations.push(['flip', 1])
  if (grayscale.value) operations.push(['grayscale', 1])
  if (!operations.length) { error.value = '请先选择增强方式'; return }
  busy.value = true
  error.value = ''
  try {
    const version = await getVersion(props.versionIds[0])
    const assets = await listAssets(version.dataset_id, { version_id: version.version_id, page: 1, page_size: 1 })
    if (!assets.items.length) throw new Error('来源版本没有图片')
    const results = await Promise.all(operations.map(([operation, value]) => augmentationPreview(assets.items[0].asset_id, operation, value)))
    if (ticket === previewGeneration) previews.value = results.map((result, index) => ({
      label: operations[index].join(' '), url: URL.createObjectURL(result.data),
    }))
  } catch (e) { error.value = e.message }
  finally { busy.value = false }
}
function close(done) { if (!busy.value) { emit('update:visible', false); if (typeof done === 'function') done() } }
watch(() => props.visible, async visible => {
  if (!visible) { clearPreviews(); return }
  error.value = ''
  requestKey.value = crypto.randomUUID()
  requestBody.value = ''
  if (props.operation === 'preannotate') {
    busy.value = true
    try { models.value = (await listModels({ task: 'detect', page_size: 200 })).items }
    catch (e) { error.value = e.message }
    finally { busy.value = false }
  }
})
async function submit() {
  if (busy.value) return
  busy.value = true
  error.value = ''
  try {
    const payload = {
      operation: props.operation, version_ids: props.versionIds, name: name.value,
      rotations: rotations.value, flip: flip.value, grayscale: grayscale.value,
      brightness: brightness.value, contrast: contrast.value,
      model_id: modelId.value || null, confidence: confidence.value,
      class_mapping: mapping.value.trim() ? JSON.parse(mapping.value) : {},
      annotation_mode: annotationMode.value, asset_ids: props.assetIds,
    }
    const encoded = JSON.stringify(payload)
    if (requestBody.value && requestBody.value !== encoded) requestKey.value = crypto.randomUUID()
    requestBody.value = encoded
    const job = await createDerivative(payload, requestKey.value)
    emit('submitted', job.job_id)
    emit('update:visible', false)
  } catch (e) { error.value = e.message }
  finally { busy.value = false }
}
</script>
<style scoped>
.augmentation-previews { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.augmentation-previews figure { margin: 12px 0; text-align: center; }
.augmentation-previews img { max-width: 100%; height: 120px; object-fit: contain; }
</style>

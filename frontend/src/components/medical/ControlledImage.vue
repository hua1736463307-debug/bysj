<template><img v-if="url" :src="url" :alt="alt" @load="$emit('load', $event)" /><span v-else-if="error" class="error">{{ error }}</span><span v-else>正在读取图像…</span></template>
<script setup>
import { ref, watch, onBeforeUnmount } from 'vue'
import { api } from '../../api/client'
const props = defineProps({ assetId: [Number, String], artifactId: String, alt: { type: String, default: '受控图像' } })
defineEmits(['load'])
const url = ref(''), error = ref(''); let ticket = 0
watch(() => [props.assetId, props.artifactId], async () => {
  const current = ++ticket; if(url.value) URL.revokeObjectURL(url.value); url.value = ''; error.value = ''
  if(!props.assetId && !props.artifactId) return
  try { const response = await api.get(props.artifactId ? `/artifacts/${props.artifactId}` : `/assets/${props.assetId}/image`, { responseType: 'blob' }); if(current === ticket) url.value = URL.createObjectURL(response.data) }
  catch(e) { if(current === ticket) error.value = e.message }
}, { immediate: true })
onBeforeUnmount(() => { ++ticket; if(url.value) URL.revokeObjectURL(url.value) })
</script>

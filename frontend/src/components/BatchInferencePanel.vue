<template>
  <Inference
    embedded
    :training-task-id="trainingTaskId"
    :locked-model="lockedModel"
    @busy-change="emit('busy-change', $event)"
  />
</template>

<script setup>
import Inference from '../views/Inference.vue'

defineProps({
  trainingTaskId: { type: String, default: '' },
  lockedModel: { type: Object, default: null }
})

const emit = defineEmits(['busy-change'])
</script>

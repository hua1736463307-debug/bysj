<template>
  <section v-if="job" class="job-card" aria-live="polite">
    <strong>任务 {{ job.status }} · {{ job.stage || job.type }}</strong>
    <el-progress :percentage="Math.round((job.progress || 0) * 100)" />
    <small>{{ job.job_id }}</small>
    <p v-if="job.error" class="error">{{ job.error.message || job.error.detail || job.error.code }}</p>
    <el-button v-if="!terminal.has(job.status)" @click="cancel">取消任务</el-button>
    <el-button v-if="job.status === 'interrupted'" @click="resume">恢复任务</el-button>
    <p v-if="error" class="error">{{ error }}</p>
  </section>
</template>
<script setup>
import { ref, watch, onBeforeUnmount } from 'vue'
import { getJob, cancelJob, resumeJob } from '../../api/training'
const props = defineProps({ jobId: String })
const emit = defineEmits(['terminal', 'update'])
const job = ref(null), error = ref(''), terminal = new Set(['completed', 'failed', 'cancelled', 'interrupted'])
let timer, generation = 0
async function poll(ticket = generation) {
  try {
    const data = await getJob(props.jobId)
    if (ticket !== generation) return
    job.value = data; error.value = ''; emit('update', data)
    if (terminal.has(data.status)) emit('terminal', data)
    else timer = setTimeout(() => poll(ticket), 1500)
  } catch (e) {
    if (ticket !== generation) return
    error.value = e.message; timer = setTimeout(() => poll(ticket), 5000)
  }
}
async function cancel() { try { await cancelJob(props.jobId); clearTimeout(timer); await poll() } catch(e) { error.value = e.message } }
async function resume() { try { await resumeJob(props.jobId); clearTimeout(timer); await poll() } catch(e) { error.value = e.message } }
watch(() => props.jobId, id => { clearTimeout(timer); ++generation; job.value = null; if (id) poll() }, { immediate: true })
onBeforeUnmount(() => { ++generation; clearTimeout(timer) })
</script>

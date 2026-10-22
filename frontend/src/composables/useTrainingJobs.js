import { onBeforeUnmount, ref } from 'vue'
import { getJob, listJobs } from '../api/training'

const TERMINAL = new Set(['completed', 'failed', 'cancelled'])
const MAX_INTERVAL = 15000

/**
 * 训练任务轮询：指数退避、离开页面停止、终态停止。
 */
export function useTrainingJobs({ intervalMs = 2000, onJobUpdate } = {}) {
  const jobs = ref([])
  const loading = ref(false)
  const error = ref(null)
  let timer = null
  let stopped = false

  async function refresh(params = {}) {
    loading.value = true
    try {
      const data = await listJobs({ type: 'training', ...params })
      jobs.value = data.items || []
      if (onJobUpdate) jobs.value.forEach(onJobUpdate)
      schedule()
      return jobs.value
    } catch (err) {
      error.value = err
      schedule(true)
      return jobs.value
    } finally {
      loading.value = false
    }
  }

  function schedule(backoff = false) {
    stopTimer()
    if (stopped) return
    if (jobs.value.every((job) => TERMINAL.has(job.status))) return
    const delay = backoff ? Math.min(MAX_INTERVAL, intervalMs * 2) : intervalMs
    timer = setTimeout(() => {
      refresh().catch(() => {})
    }, delay)
  }

  async function pollJob(jobId) {
    const job = await getJob(jobId)
    const index = jobs.value.findIndex((item) => item.job_id === jobId)
    if (index >= 0) jobs.value[index] = job
    else jobs.value = [job, ...jobs.value]
    if (!TERMINAL.has(job.status)) {
      timer = setTimeout(() => pollJob(jobId).catch(() => {}), intervalMs)
    }
    return job
  }

  function stopTimer() {
    if (timer) clearTimeout(timer)
    timer = null
  }

  function stop() {
    stopped = true
    stopTimer()
  }

  onBeforeUnmount(stop)

  return { jobs, loading, error, refresh, pollJob, stop }
}

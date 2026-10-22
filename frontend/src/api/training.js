import { api } from './client'

export function preflightTraining(payload) {
  return api.post('/training/preflight', payload)
}

export function startTraining(payload, idempotencyKey) {
  const config = idempotencyKey ? { headers: { 'Idempotency-Key': idempotencyKey } } : undefined
  return api.post('/training/jobs', payload, config)
}

export function listJobs(params = {}) {
  return api.get('/jobs', { params })
}

export function getJob(jobId, { signal } = {}) {
  return api.get(`/jobs/${jobId}`, { signal })
}

export function getJobEvents(jobId, afterSequence = 0) {
  return api.get(`/jobs/${jobId}/events`, { params: { after_sequence: afterSequence } })
}

export function cancelJob(jobId) {
  return api.post(`/jobs/${jobId}/cancel`)
}

export function resumeJob(jobId) {
  return api.post(`/jobs/${jobId}/resume`)
}

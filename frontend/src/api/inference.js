import { api } from './client'

export function createInference({ modelId, files = [], assetIds = [], params = {} }) {
  const form = new FormData()
  form.append('model_id', modelId)
  if (assetIds.length) form.append('asset_ids', assetIds.join(','))
  form.append('params', JSON.stringify(params))
  for (const file of files) form.append('files', file)
  return api.post('/inference/images', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000
  })
}

export function listRuns(params = {}) {
  return api.get('/inference/runs', { params })
}

export function getRun(runId, { signal, page = 1, page_size = 50 } = {}) {
  return api.get(`/inference/runs/${runId}`, { signal, params: { page, page_size } })
}

export function exportRun(runId, format = 'json') {
  return api.get(`/inference/runs/${runId}/export`, { params: { format }, responseType: 'blob' })
}

export function getArtifact(artifactId) {
  return api.get(`/artifacts/${artifactId}`, { responseType: 'blob' })
}

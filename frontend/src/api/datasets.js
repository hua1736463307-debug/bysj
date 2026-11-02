import { api } from './client'

export function listDatasets(params = {}) {
  return api.get('/datasets', { params })
}

export function getDataset(datasetId) {
  return api.get(`/datasets/${datasetId}`)
}

export function listVersions(datasetId) {
  return api.get(`/datasets/${datasetId}/versions`)
}

export function listAssets(datasetId, params = {}) {
  return api.get(`/datasets/${datasetId}/assets`, { params })
}

export function createVersion(datasetId, payload) {
  return api.post(`/datasets/${datasetId}/versions`, payload)
}

export function createTrainTiles(datasetId, payload, key) {
  return api.post(`/datasets/${datasetId}/train-tile-versions`, payload, {
    headers: { 'Idempotency-Key': key }
  })
}

export function deleteDataset(datasetId) {
  return api.delete(`/datasets/${datasetId}`)
}

export function deleteAsset(assetId) {
  return api.delete(`/assets/${assetId}`)
}

export function importDataset(formData, { signal } = {}) {
  return api.post('/datasets/import', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    signal,
    timeout: 300000
  })
}

export function createAnnotationVersion(datasetId, baseVersionId) {
  return api.post(`/datasets/${datasetId}/annotation-versions`, { base_version_id: baseVersionId }, { timeout: 300000 })
}

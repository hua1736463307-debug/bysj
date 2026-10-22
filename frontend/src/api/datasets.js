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

import { api } from './client'

export function listModels(params = {}) {
  return api.get('/models', { params })
}
export const uploadModel = form => api.post('/models/upload', form, { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 300000 })

export function getModel(modelId) {
  return api.get(`/models/${modelId}`)
}

export function registerModel(payload) {
  return api.post('/models/register', payload)
}

export function deleteModel(modelId) {
  return api.delete(`/models/${modelId}`)
}

export function downloadModel(modelId) {
  return api.get(`/models/${modelId}/download`, { responseType: 'blob' })
}

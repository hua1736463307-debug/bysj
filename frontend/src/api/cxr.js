/**
 * CXR 五标签辅助分析 API 客户端。
 *
 * 只通过统一 `client.js` 发请求：token 注入与 401 处理保持单点，
 * 不新增 axios 实例，也不把 token 写进报告或导出文件。
 */
import { api } from './client'

export function getSchema() {
  return api.get('/cxr/schema')
}

export function getDashboard() {
  return api.get('/cxr/dashboard')
}

export function listCxrDatasets(params = {}) {
  return api.get('/cxr/datasets', { params })
}

export function getCxrDataset(datasetId) {
  return api.get(`/cxr/datasets/${datasetId}`)
}

export function importCxrDataset(payload, idempotencyKey) {
  return api.post('/cxr/datasets/import', payload, {
    headers: { 'Idempotency-Key': idempotencyKey },
    timeout: 300000
  })
}

export function freezeCxrVersion(datasetId, payload, idempotencyKey) {
  return api.post(`/cxr/datasets/${datasetId}/versions`, payload, {
    headers: { 'Idempotency-Key': idempotencyKey }
  })
}

export function getVersionSummary(datasetId, versionId) {
  return api.get(`/cxr/datasets/${datasetId}/versions/${versionId}/summary`)
}

export function preflightTraining(payload) {
  return api.post('/cxr/training/preflight', payload)
}

export function createTrainingJob(payload, idempotencyKey) {
  return api.post('/cxr/training/jobs', payload, {
    headers: { 'Idempotency-Key': idempotencyKey }
  })
}

export function listCxrModels(params = {}) {
  return api.get('/cxr/models', { params })
}

export function createEvaluation(payload, idempotencyKey) {
  return api.post('/cxr/evaluations', payload, {
    headers: { 'Idempotency-Key': idempotencyKey }
  })
}

export function listEvaluations(params = {}) {
  return api.get('/cxr/evaluations', { params })
}

export function createAnalysisBatch(payload, idempotencyKey) {
  return api.post('/cxr/analysis-batches', payload, {
    headers: { 'Idempotency-Key': idempotencyKey }
  })
}

export function uploadAnalysisBatch({ modelId, files, idempotencyKey }) {
  const form = new FormData()
  form.append('model_id', modelId)
  for (const file of files) form.append('files', file)
  return api.post('/cxr/analysis-batches/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data', 'Idempotency-Key': idempotencyKey },
    timeout: 300000
  })
}

export function listBatches(params = {}) {
  return api.get('/cxr/analysis-batches', { params })
}

export function getBatch(batchId, params = {}) {
  return api.get(`/cxr/analysis-batches/${batchId}`, { params })
}

export function getAnalysis(analysisId) {
  return api.get(`/cxr/analyses/${analysisId}`)
}

export function createExplanation(analysisId, labelKey, idempotencyKey) {
  return api.post(
    `/cxr/analyses/${analysisId}/explanations`,
    { label_key: labelKey, method: 'gradcam' },
    { headers: idempotencyKey ? { 'Idempotency-Key': idempotencyKey } : {} }
  )
}

export function createReview(payload) {
  return api.post('/cxr/reviews', payload)
}

export function listReviews(params = {}) {
  return api.get('/cxr/reviews', { params })
}

export function getReview(reviewId) {
  return api.get(`/cxr/reviews/${reviewId}`)
}

export function updateReview(reviewId, payload) {
  return api.patch(`/cxr/reviews/${reviewId}`, payload)
}

export function submitReview(reviewId) {
  return api.post(`/cxr/reviews/${reviewId}/submit`)
}

export function decideReview(reviewId, approve, reason) {
  return api.post(`/cxr/reviews/${reviewId}/decision`, { approve, reason })
}

export function createReport(payload) {
  return api.post('/cxr/reports', payload)
}

export function listReports(params = {}) {
  return api.get('/cxr/reports', { params })
}

export function getReport(reportId) {
  return api.get(`/cxr/reports/${reportId}`)
}

export function exportReport(reportId, format = 'html') {
  return api.get(`/cxr/reports/${reportId}/export`, { params: { format }, responseType: 'blob' })
}

export const LABEL_FLAG_TEXT = {
  above_threshold: '高于阈值',
  below_threshold: '低于阈值',
  borderline: '接近阈值',
  not_available: '无可用评分'
}

export const OPINION_TEXT = {
  supported: '支持',
  not_supported: '不支持',
  indeterminate: '无法判断'
}

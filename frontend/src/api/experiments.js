import { api } from './client'

export function listExperiments(params = {}) {
  return api.get('/experiments', { params })
}

export function getExperiment(experimentId) {
  return api.get(`/experiments/${experimentId}`)
}

export function listEvaluations(params = {}) {
  return api.get('/evaluations', { params })
}

export function createEvaluation(payload) {
  return api.post('/evaluations', payload)
}

export function getEvaluation(evaluationId) {
  return api.get(`/evaluations/${evaluationId}`)
}

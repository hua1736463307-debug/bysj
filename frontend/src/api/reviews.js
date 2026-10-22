import { api } from './client'

export function listReviews(params = {}) {
  return api.get('/reviews', { params })
}

export function getReview(reviewId) {
  return api.get(`/reviews/${reviewId}`)
}

export function createReview(payload) {
  return api.post('/reviews', payload)
}

export function updateReview(reviewId, payload) {
  return api.patch(`/reviews/${reviewId}`, payload)
}

export function submitReview(reviewId) {
  return api.post(`/reviews/${reviewId}/submit`)
}

export function decideReview(reviewId, approve, reason) {
  return api.post(`/reviews/${reviewId}/decision`, { approve, reason })
}

export function exportVersion(parentVersionId, reviewIds, idempotencyKey) {
  return api.post('/reviews/export-version', {
    parent_version_id: parentVersionId,
    review_ids: reviewIds,
    idempotency_key: idempotencyKey
  })
}

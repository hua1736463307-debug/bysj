import { api } from './client'

export function getAnnotations(imageId, { signal } = {}) {
  return api.get(`/annotations/${imageId}`, { signal })
}

export function saveAnnotations(imageId, { expectedRevision, boxes, deletedIds = [], page = 1, geometry = 'rect' }) {
  return api.put(`/annotations/${imageId}`, {
    expected_revision: expectedRevision,
    boxes,
    deleted_ids: deletedIds,
    page,
    geometry
  })
}

export function exportAnnotations(datasetId, versionId, format = 'yolo-txt') {
  return api.post('/annotations/export', {
    dataset_id: datasetId,
    version_id: versionId,
    format
  })
}

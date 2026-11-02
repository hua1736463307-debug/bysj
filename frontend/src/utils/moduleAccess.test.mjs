import test from 'node:test'
import assert from 'node:assert/strict'
import { canAccessPath, firstAuthorizedPath } from './moduleAccess.mjs'

test('custom roles land on an accessible module and cannot open system pages', () => {
  const user = { role: 'researcher', modules: ['datasets', 'annotations'] }
  assert.equal(firstAuthorizedPath(user), '/raw-dataset')
  assert.equal(canAccessPath(user, '/cxr/data'), false)
  assert.equal(canAccessPath(user, '/system/user'), false)
  assert.equal(canAccessPath(user, '/dataset-annotation'), true)
  assert.equal(firstAuthorizedPath({ role: 'reviewer', modules: [] }), '/access-denied')
})

import test from 'node:test'
import assert from 'node:assert/strict'
import { moveBox, resizeBox, hitBox } from './boxGeometry.mjs'

test('move and resize stay within normalized image bounds', () => {
  const box = { cx: .5, cy: .5, w: .4, h: .4, class_id: 2 }
  assert.deepEqual(moveBox(box, 2, -2), { ...box, cx: .8, cy: .2 })
  const enlarged = resizeBox(box, 2, 2)
  assert.equal(enlarged.w, .7)
  assert.equal(enlarged.cx, .65)
  assert.equal(enlarged.class_id, 2)
  assert(resizeBox(box, 0, 0).w > 0)
  assert.equal(box.cx, .5)
})

test('selection prefers the topmost overlapping box', () => {
  const box = { cx: .5, cy: .5, w: .4, h: .4 }
  assert.equal(hitBox([box, box], { x: .5, y: .5 }), 1)
  assert.equal(hitBox([box], { x: 0, y: 0 }), -1)
})

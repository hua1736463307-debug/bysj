import assert from 'node:assert/strict'
import test from 'node:test'

import { createBoundedHistory } from './annotationHistory.mjs'

test('undo and redo keep per-image stacks', () => {
  const history = createBoundedHistory(10)
  history.push('img-1', { boxes: [1] })
  history.push('img-1', { boxes: [1, 2] })
  history.push('img-2', { boxes: ['a'] })

  assert.equal(history.canUndo('img-1'), true)
  const previous = history.undo('img-1', { boxes: [1, 2, 3] })
  assert.deepEqual(previous, { boxes: [1, 2] })
  assert.equal(history.canRedo('img-1'), true)

  const next = history.redo('img-1', { boxes: [1, 2] })
  assert.deepEqual(next, { boxes: [1, 2, 3] })

  // 另一张图的历史不受影响
  assert.equal(history.size('img-2').past, 1)
  assert.equal(history.canUndo('img-2'), true)
})

test('history is bounded and push clears redo stack', () => {
  const history = createBoundedHistory(3)
  for (let index = 0; index < 5; index += 1) {
    history.push('img', { step: index })
  }
  assert.equal(history.size('img').past, 3)

  history.undo('img', { step: 99 })
  assert.equal(history.canRedo('img'), true)
  history.push('img', { step: 100 })
  assert.equal(history.canRedo('img'), false)
})

test('undo on empty stack returns null', () => {
  const history = createBoundedHistory(3)
  assert.equal(history.undo('missing', {}), null)
  assert.equal(history.redo('missing', {}), null)
})

/**
 * 有界的按图像撤销/重做栈（纯函数，便于 Node 测试）。
 * 每张图独立保存历史，避免切页时互相污染。
 */
export function createBoundedHistory(limit = 50) {
  const stacks = new Map()

  function ensure(imageId) {
    if (!stacks.has(imageId)) stacks.set(imageId, { past: [], future: [] })
    return stacks.get(imageId)
  }

  function push(imageId, state) {
    const stack = ensure(imageId)
    stack.past.push(state)
    if (stack.past.length > limit) stack.past.shift()
    stack.future = []
  }

  function undo(imageId, current) {
    const stack = ensure(imageId)
    if (!stack.past.length) return null
    const previous = stack.past.pop()
    stack.future.push(current)
    return previous
  }

  function redo(imageId, current) {
    const stack = ensure(imageId)
    if (!stack.future.length) return null
    const next = stack.future.pop()
    stack.past.push(current)
    return next
  }

  function reset(imageId) {
    stacks.set(imageId, { past: [], future: [] })
  }

  function canUndo(imageId) {
    return ensure(imageId).past.length > 0
  }

  function canRedo(imageId) {
    return ensure(imageId).future.length > 0
  }

  function size(imageId) {
    const stack = ensure(imageId)
    return { past: stack.past.length, future: stack.future.length }
  }

  return { push, undo, redo, reset, canUndo, canRedo, size }
}

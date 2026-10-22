import { computed, ref } from 'vue'
import { createBoundedHistory } from '../utils/annotationHistory.mjs'

/**
 * 按 image_id 保存有界撤销/重做历史。
 * 只保存标注快照，不保存画布缩放/平移（视图变化不产生历史）。
 */
export function useAnnotationHistory(limit = 50) {
  const history = createBoundedHistory(limit)
  const version = ref(0)

  function refresh() {
    version.value += 1
  }

  return {
    push(imageId, state) {
      history.push(imageId, state)
      refresh()
    },
    undo(imageId, current) {
      const value = history.undo(imageId, current)
      refresh()
      return value
    },
    redo(imageId, current) {
      const value = history.redo(imageId, current)
      refresh()
      return value
    },
    reset(imageId) {
      history.reset(imageId)
      refresh()
    },
    canUndo: (imageId) =>
      computed(() => {
        version.value
        return history.canUndo(imageId)
      }),
    canRedo: (imageId) =>
      computed(() => {
        version.value
        return history.canRedo(imageId)
      }),
    size: (imageId) => history.size(imageId)
  }
}

import { onBeforeUnmount, ref } from 'vue'
import { getAnnotations, saveAnnotations } from '../api/annotations'
import { ApiError } from '../api/client'

/**
 * 标注草稿会话：
 * - 会话键 = owner + dataset + source_version + geometry（由后端强制）；
 * - 保存必须携带 expected_revision，冲突时保留本地状态并返回冲突信息；
 * - 切换图片/数据集时中止旧请求，迟到响应不得覆盖新会话；
 * - 卸载前 flush，避免丢失最后一笔修改。
 */
export function useAnnotationDraft({ imageId, datasetId, versionId, throttleMs = 800 } = {}) {
  const revision = ref(0)
  const annotations = ref([])
  const saving = ref(false)
  const dirty = ref(false)
  const lastError = ref(null)
  const conflict = ref(null)

  let controller = null
  let timer = null
  let pendingBoxes = null
  let pendingDeleted = []
  let currentImage = imageId

  async function load(nextImageId) {
    if (controller) controller.abort()
    controller = new AbortController()
    currentImage = nextImageId
    conflict.value = null
    const data = await getAnnotations(nextImageId, { signal: controller.signal })
    revision.value = data.revision
    annotations.value = data.annotations || []
    dirty.value = false
    return data
  }

  async function flush() {
    if (!dirty.value || pendingBoxes === null) return null
    const boxes = pendingBoxes
    const deleted = pendingDeleted
    pendingBoxes = null
    pendingDeleted = []
    saving.value = true
    try {
      const data = await saveAnnotations(currentImage, {
        expectedRevision: revision.value,
        boxes,
        deletedIds: deleted
      })
      revision.value = data.revision
      annotations.value = data.annotations
      dirty.value = false
      lastError.value = null
      return data
    } catch (error) {
      if (error instanceof ApiError && error.code === 'revision_conflict') {
        conflict.value = {
          serverRevision: error.data?.server_revision ?? null,
          localBoxes: boxes
        }
      } else {
        lastError.value = error
        // 失败保持本地未保存状态
        pendingBoxes = boxes
        pendingDeleted = deleted
        dirty.value = true
      }
      throw error
    } finally {
      saving.value = false
    }
  }

  function scheduleSave(boxes, deletedIds = []) {
    pendingBoxes = boxes
    pendingDeleted = deletedIds
    dirty.value = true
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => {
      flush().catch(() => {})
    }, throttleMs)
  }

  function setImage(nextImageId) {
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
    pendingBoxes = null
    pendingDeleted = []
    dirty.value = false
    return load(nextImageId)
  }

  onBeforeUnmount(() => {
    if (timer) clearTimeout(timer)
    if (controller) controller.abort()
    flush().catch(() => {})
  })

  return {
    revision,
    annotations,
    saving,
    dirty,
    lastError,
    conflict,
    load,
    flush,
    scheduleSave,
    setImage,
    datasetId,
    versionId
  }
}

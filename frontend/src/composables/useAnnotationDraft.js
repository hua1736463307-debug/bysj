import { onBeforeUnmount, ref } from 'vue'
import { getAnnotations, saveAnnotations } from '../api/annotations'
import { getCurrentUser } from '../api/client'

// Serialize saves; never let a delayed response clear newer local edits.
export function useAnnotationDraft({ throttleMs = 800 } = {}) {
  const revision = ref(0), annotations = ref([]), saving = ref(false), dirty = ref(false)
  const lastError = ref(null), conflict = ref(null)
  let currentImage, serial = 0, loadSerial = 0, pending = null, inFlight = null, timer, controller
  const clone = value => JSON.parse(JSON.stringify(value))
  const key = id => `bysj:draft:${getCurrentUser()?.id}:${id}`
  function persist() {
    if (pending && currentImage) {
      try { localStorage.setItem(key(currentImage), JSON.stringify({ ...pending, revision: revision.value })) }
      catch { lastError.value = new Error('本地草稿存储失败，请保持页面打开并手动保存') }
    }
  }
  async function load(id, { discard = false } = {}) {
    const ticket = ++loadSerial
    controller?.abort(); controller = new AbortController()
    const data = await getAnnotations(id, { signal: controller.signal })
    if (ticket !== loadSerial) return null
    currentImage = id; revision.value = data.revision; annotations.value = clone(data.annotations || [])
    pending = null; dirty.value = false; conflict.value = null; lastError.value = null
    if (discard) localStorage.removeItem(key(id))
    let local
    try { local = JSON.parse(localStorage.getItem(key(id)) || 'null') } catch { local = null }
    if (local && data.editable) {
      pending = { boxes: local.boxes, deleted: local.deleted || [], serial: ++serial }
      annotations.value = clone(local.boxes); dirty.value = true
      if (local.revision !== data.revision) conflict.value = { serverRevision: data.revision, localBoxes: clone(local.boxes) }
    }
    return data
  }
  function flush() {
    clearTimeout(timer)
    if (inFlight) return inFlight
    if (conflict.value) return Promise.reject(new Error('标注冲突：请先处理服务器版本与本地草稿'))
    inFlight = (async () => {
      saving.value = true
      try {
        while (pending) {
          const snapshot = clone(pending), id = currentImage
          const data = await saveAnnotations(id, { expectedRevision: revision.value, boxes: snapshot.boxes, deletedIds: snapshot.deleted })
          revision.value = data.revision
          if (pending?.serial === snapshot.serial) {
            pending = null; annotations.value = data.annotations; dirty.value = false
            localStorage.removeItem(key(id))
          } else { persist() }
          lastError.value = null
        }
      } catch (error) {
        lastError.value = error
        if (error.code === 'revision_conflict') conflict.value = { serverRevision: error.data?.server_revision, localBoxes: clone(pending?.boxes || annotations.value) }
        persist(); throw error
      } finally { saving.value = false }
    })().finally(() => { inFlight = null })
    return inFlight
  }
  function scheduleSave(boxes, deletedIds = []) {
    pending = { boxes: clone(boxes), deleted: clone(deletedIds), serial: ++serial }
    annotations.value = clone(boxes); dirty.value = true; persist()
    clearTimeout(timer); timer = setTimeout(() => flush().catch(() => {}), throttleMs)
  }
  async function setImage(id) { await flush(); return load(id) }
  function beforeUnload(event) { if (dirty.value) { persist(); event.preventDefault(); event.returnValue = '' } }
  window.addEventListener('beforeunload', beforeUnload)
  onBeforeUnmount(() => {
    clearTimeout(timer); controller?.abort(); ++loadSerial; persist()
    window.removeEventListener('beforeunload', beforeUnload)
  })
  return { revision, annotations, saving, dirty, lastError, conflict, load, flush, scheduleSave, setImage }
}

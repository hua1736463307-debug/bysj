import test from 'node:test'
import assert from 'node:assert/strict'
import vm from 'node:vm'
import { readFileSync } from 'node:fs'

const source = readFileSync(new URL('../src/views/Annotation.vue', import.meta.url), 'utf8')
const extract = (name, next) => source.slice(source.indexOf(`const ${name} =`), source.indexOf(`const ${next} =`))
const ref = value => ({ value })
function context(count = 3, annotationBytes = 1) {
  const ctx = {
    TextEncoder, Date, annotationDraftSavePromise: null, annotationDraftClearAllVersion: 0,
    annotationDraftDebounceTimer: null, annotationViewActive: true, annotationDraftTimer: null,
    window: { clearTimeout() {}, setTimeout() { return 1 }, clearInterval() {} },
    cacheCurrentPageAnnotations() {}, getImageAnnotationCacheKey: image => image?.relative_path,
    cloneAnnotations: value => structuredClone(value || []),
    annotationDraftDirtyPaths: new Set(), annotationDraftDirtyVersions: new Map(), annotationCache: new Map(),
    currentDatasetRole: ref('original'), currentImage: ref({ relative_path: '0.jpg' }), currentDatasetId: ref('source'), annotationDraftKind: ref('yolo'),
    annotationDraftId: ref('draft'), annotationDraftBatchTaskId: ref(''),
    annotationDraftSourceVersionId: ref('version'), annotationDraftRevision: ref(1),
    annotationSessionModel: ref('model.pt'), aiAssistConfig: { model: 'model.pt' }, selectedAnnotationModel: ref('model'),
    annotationClassScope: ref('dataset'), classLabelsZh: ref({}), inactiveClassIds: ref([]), annotationGeometry: ref('rect'), classList: ref(['phone']), annotationClassBindings: ref([]),
    imagePage: ref(1), selectedImagePath: ref(''), annotationDraftClearAll: ref(false),
    canSaveAnnotationDraftNow: () => true, annotationDraftConflict: ref(false),
    annotationDraftMetadataDirty: ref(false), annotationDraftChangeVersion: ref(count),
    annotationDraftSaving: ref(false), annotationDraftTemporaryDatasetId: ref(''),
    annotationDraftLastSavedAt: ref(''), annotationDraftWarning: ref(''), draftEndpoint: () => '/draft',
    sent: [], revision: 1
  }
  for (let i = 0; i < count; i++) {
    const path = `${i}.jpg`
    ctx.annotationDraftDirtyPaths.add(path); ctx.annotationDraftDirtyVersions.set(path, i + 1)
    ctx.annotationCache.set(path, { relative_path: path, annotations: [{ class: 0, note: 'x'.repeat(annotationBytes) }] })
  }
  ctx.annotationDraftPending = { get value() {
    return ctx.annotationDraftMetadataDirty.value || ctx.annotationDraftDirtyPaths.size > 0 || Boolean(ctx.annotationDraftBatchTaskId.value)
  } }
  ctx.putAnnotationDraft = async (dataset, kind, payload) => {
    assert.equal(payload.revision, ctx.revision)
    ctx.sent.push(structuredClone(payload)); ctx.revision++
    return { id: 'draft', revision: ctx.revision, batch_sync: payload.annotation_task_id ? { has_more: false } : null }
  }
  ctx.authorizedFetch = async (url, options) => {
    const result = await ctx.putAnnotationDraft('source', 'yolo', JSON.parse(options.body))
    return { ok: true, json: async () => result }
  }
  vm.createContext(ctx)
  vm.runInContext(extract('buildAnnotationDraftPayload', 'startAnnotationDraftTimer'), ctx)
  return ctx
}
const flush = (ctx, options = {}) => vm.runInContext(`flushAnnotationDraft(${JSON.stringify(options)})`, ctx)

test('normal save drains 501 images in bounded requests', async () => {
  const ctx = context(501)
  assert.equal(await flush(ctx), true)
  assert.deepEqual(ctx.sent.map(p => p.assets.length), [200, 200, 101])
  assert.equal(ctx.annotationDraftDirtyPaths.size, 0)
})

test('exit saves all pending images that fit', async () => {
  const ctx = context(3)
  assert.equal(await flush(ctx, { keepalive: true }), true)
  assert.equal(ctx.sent[0].assets.length, 3)
})

test('exit size cap acknowledges only submitted images', async () => {
  const ctx = context(3, 20000)
  assert.equal(await flush(ctx, { keepalive: true }), false)
  assert.equal(ctx.sent[0].assets.length, 2)
  assert.equal(ctx.annotationDraftDirtyPaths.size, 1)
  assert.ok(new TextEncoder().encode(JSON.stringify(ctx.sent[0])).length <= 48 * 1024)
  assert.equal(await flush(ctx), true)
  assert.equal(ctx.sent[1].assets[0].relative_path, '2.jpg')
})

test('oversize single exit image remains pending', async () => {
  const ctx = context(1, 60000)
  assert.equal(await flush(ctx, { keepalive: true }), false)
  assert.equal(ctx.sent.length, 0); assert.equal(ctx.annotationDraftDirtyPaths.size, 1)
  assert.equal(await flush(ctx), true)
})

test('failed network save retains pending annotations and retries', async () => {
  const ctx = context(); const save = ctx.putAnnotationDraft
  ctx.putAnnotationDraft = async () => { throw new Error('network') }
  assert.equal(await flush(ctx), false); assert.equal(ctx.annotationDraftDirtyPaths.size, 3)
  ctx.putAnnotationDraft = save
  assert.equal(await flush(ctx), true); assert.equal(ctx.annotationDraftWarning.value, '')
})

test('simultaneous flushes never submit the same revision concurrently', async () => {
  const ctx = context(); const save = ctx.putAnnotationDraft; let active = 0; let maxActive = 0
  ctx.putAnnotationDraft = async (...args) => {
    active++; maxActive = Math.max(maxActive, active)
    await new Promise(resolve => setImmediate(resolve))
    const result = await save(...args); active--; return result
  }
  await Promise.all([flush(ctx), flush(ctx), flush(ctx, { keepalive: true })])
  assert.equal(maxActive, 1); assert.equal(ctx.sent.length, 1)
})

test('edits during an in-flight save are saved in the next revision', async () => {
  const ctx = context(1); const save = ctx.putAnnotationDraft; let first = true
  ctx.putAnnotationDraft = async (...args) => {
    if (first) {
      first = false; ctx.annotationDraftChangeVersion.value++
      ctx.annotationDraftDirtyVersions.set('0.jpg', 2)
      ctx.annotationCache.get('0.jpg').annotations = [{ class: 0, x: 0.9 }]
    }
    return save(...args)
  }
  assert.equal(await flush(ctx), true)
  assert.equal(ctx.sent.length, 2); assert.equal(ctx.sent[1].assets[0].annotations[0].x, 0.9)
})

test('batch checkpoints are drained without loading or rendering images', async () => {
  const ctx = context(0); ctx.annotationDraftBatchTaskId.value = 'task'
  const save = ctx.putAnnotationDraft
  ctx.putAnnotationDraft = async (...args) => {
    const response = await save(...args)
    response.batch_sync.has_more = ctx.sent.length < 3
    return response
  }
  assert.equal(await flush(ctx), true)
  assert.equal(ctx.sent.length, 3); assert.ok(ctx.sent.every(p => p.assets.length === 0))
  assert.equal(ctx.annotationDraftBatchTaskId.value, '')
})

test('older backend cannot silently claim a complete batch save', async () => {
  const ctx = context(0); ctx.annotationDraftBatchTaskId.value = 'task'
  ctx.putAnnotationDraft = async () => ({ id: 'draft', revision: 2 })
  assert.equal(await flush(ctx), false); assert.equal(ctx.annotationDraftBatchTaskId.value, 'task')
})


test('training annotations can change images and leave without creating an original draft', async () => {
  const ctx = context(0); ctx.currentDatasetRole.value = 'training'
  assert.equal(await flush(ctx), true); assert.equal(ctx.sent.length, 0)
})

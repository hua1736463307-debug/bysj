import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const annotation = readFileSync(new URL('../src/views/Annotation.vue', import.meta.url), 'utf8').replace(/\r\n/g, '\n')
const dataset = readFileSync(new URL('../src/views/Dataset.vue', import.meta.url), 'utf8')
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor

test('model selection creates a draft and navigates with its immutable identity', async () => {
  const body = dataset.match(/const confirmAnnotationModel = async \(\) => \{([\s\S]*?)\n\}\n/)?.[1]
  assert.ok(body)
  const model = { path: '/models/fpi.pt', name: 'best.pt', type: 'trained', geometry: 'rect' }
  const navigations = []; const calls = []
  const environment = {
    pendingAnnotationDataset: { value: { id: 'old-source' } },
    filteredAnnotationModels: { value: [model] }, annotationModels: { value: [model] },
    selectedAnnotationModel: { value: model.path }, annotationSelectionKind: { value: 'yolo' },
    pendingDraftReplacement: { value: null },
    modelSelectionValue: m => m.path, modelIdentity: m => m.path, modelSupportsTextPrompts: () => false,
    annotationModelSaving: { value: false }, showAnnotationModelDialog: { value: true },
    createAnnotationDraft: async (...args) => { calls.push(args); return { id: 'new-draft', revision: 1,
      model_path: model.path, model_name: 'FPI', annotation_geometry: 'rect' } },
    router: { push: value => navigations.push(value) }, datasetTitle: () => 'source',
    ElMessage: { warning: message => assert.fail(message), error: message => assert.fail(message) }, console
  }
  await new AsyncFunction(...Object.keys(environment), body)(...Object.values(environment))
  assert.equal(calls.length, 1)
  assert.equal(calls[0][1], 'yolo')
  assert.equal(navigations[0].query.draftId, 'new-draft')
  assert.equal(navigations[0].query.draftRevision, '1')
  assert.equal(environment.annotationModelSaving.value, false)
})

test('restoring isolated metadata clears unrelated inactive IDs and Chinese labels', () => {
  const body = annotation.match(/const applyDraftClassContext = \(response\) => \{([\s\S]*?)\n\}\n/)?.[1]
  const calls = []; const scope = { value: 'dataset' }
  const apply = new Function('response', 'annotationClassScope', 'applyAnnotationClasses', body)
  apply({ class_scope: 'model_session', class_names: ['phone'], class_bindings: [],
    class_context: { inactive_class_ids: [], labels_zh: {} } }, scope, (...args) => calls.push(args))
  assert.equal(scope.value, 'model_session')
  assert.deepEqual(calls[0], [['phone'], [], {}, []])
})

test('page responses restore the same isolated context as metadata responses', () => {
  const body = annotation.match(/const applyDraftClassContext = \(response\) => \{([\s\S]*?)\n\}\n/)?.[1]
  const calls = []
  new Function('response', 'annotationClassScope', 'applyAnnotationClasses', body)({
    draft_class_scope: 'model_session', draft_class_names: ['phone'], draft_class_bindings: [],
    draft_class_context: { inactive_class_ids: [], labels_zh: { phone: '手机' } }
  }, { value: '' }, (...args) => calls.push(args))
  assert.deepEqual(calls[0], [['phone'], [], { phone: '手机' }, []])
})

test('session class persistence saves the draft and never calls dataset class update', async () => {
  const body = annotation.match(/const persistDatasetAnnotationClasses = async \(source = 'user', expectedClasses = null\) => \{([\s\S]*?)\n\}\n/)?.[1]
  let saved = 0; let marked = 0
  const env = { currentDatasetId: { value: 'source' }, currentDatasetRole: { value: 'original' },
    annotationClassScope: { value: 'model_session' }, markDraftMetadataDirty: () => marked++,
    flushAnnotationDraft: async () => { saved++; return true },
    updateDatasetAnnotationClasses: () => assert.fail('global dataset mutation') }
  const result = await new AsyncFunction('source','expectedClasses',...Object.keys(env),body)('user',null,...Object.values(env))
  assert.equal(result, true); assert.equal(saved, 1); assert.equal(marked, 1)
})

test('page requests include the current draft ID as well as its kind', () => {
  const body = annotation.match(/const buildImageListQuery = \(page\) => \{([\s\S]*?)\n\}\n/)?.[1]
  const env = { imagePageSize: { value: 100 }, currentDatasetRole: { value: 'original' },
    annotationDraftKind: { value: 'yolo' }, annotationDraftId: { value: 'new-draft' }, currentDatasetId: { value: 'source' } }
  const url = new Function('page',...Object.keys(env),body)(1,...Object.values(env))
  const query = new URL(url, 'http://test').searchParams
  assert.equal(query.get('draft_id'), 'new-draft'); assert.equal(query.get('draft_kind'), 'yolo')
})

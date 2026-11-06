import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import test from 'node:test'
import { computed, reactive, ref, watch } from 'vue'
import { deferred } from './draft-harness.mjs'

function setup(file, names, overrides = {}) {
  const source = readFileSync(new URL('../src/' + file, import.meta.url), 'utf8')
    .split('<script setup>')[1].split('</script>')[0].replace(/^import .*$/gm, '')
  const context = {
    computed, reactive, ref, watch, onMounted() {}, onBeforeUnmount() {},
    defineExpose() {}, useRoute: () => ({ query: {} }), getCurrentUser: () => ({ role: 'admin' }),
    useTaskAction: () => ({ error: ref(''), busy: ref(false), act: action => action() }),
    ...overrides,
  }
  vm.createContext(context)
  vm.runInContext(source + '\nthis.component = {' + names.join(',') + '}', context)
  return context.component
}

test('dataset assets ignore late responses from a previous selection', async () => {
  const requests = { A: deferred(), B: deferred() }
  const component = setup('views/DatasetManager.vue', ['change', 'assets', 'version', 'total'], {
    listAssets: dataset => requests[dataset].promise,
  })
  const first = component.change({ datasetId: 'A', version: { version_id: 'vA' } })
  const second = component.change({ datasetId: 'B', version: { version_id: 'vB' } })
  requests.B.resolve({ items: [{ asset_id: 2 }], total: 1 })
  await second
  requests.A.resolve({ items: [{ asset_id: 1 }], total: 99 })
  await first
  assert.equal(component.version.value.version_id, 'vB')
  assert.equal(component.assets.value[0].asset_id, 2)
  assert.equal(component.total.value, 1)
})

test('clearing the version invalidates a pending asset response', async () => {
  const request = deferred()
  const component = setup('views/DatasetManager.vue', ['change', 'assets', 'total'], {
    listAssets: () => request.promise,
  })
  const loading = component.change({ datasetId: 'A', version: { version_id: 'vA' } })
  await component.change({ datasetId: 'B', version: null })
  request.resolve({ items: [{ asset_id: 1 }], total: 10 })
  await loading
  assert.equal(component.assets.value.length, 0)
  assert.equal(component.total.value, 0)
})

test('a delete confirmation cannot apply after switching the selected version', async () => {
  const confirmation = deferred(), deleted = []
  const component = setup('views/DatasetManager.vue', ['change', 'remove', 'assets'], {
    listAssets: async dataset => ({ items: [{ asset_id: dataset, relative_path: dataset + '.png' }], total: 1 }),
    ElMessageBox: { confirm: () => confirmation.promise },
    deleteAsset: async id => deleted.push(id),
  })
  await component.change({ datasetId: 'A', version: { version_id: 'vA' } })
  const removing = component.remove(component.assets.value[0])
  await component.change({ datasetId: 'B', version: { version_id: 'vB' } })
  confirmation.resolve()
  await assert.rejects(removing, /重新选择要删除的图片/)
  assert.equal(deleted.length, 0)
  await component.remove(component.assets.value[0])
  assert.deepEqual(deleted, ['B'])
})

test('version-management navigation keeps the clicked dataset and version', () => {
  let navigation
  const component = setup('views/Dataset.vue', ['manageVersions'], {
    useRouter: () => ({ push: value => { navigation = value } }),
  })
  component.manageVersions({ dataset_id: 'B', latest_version: { version_id: 'vB-old' } })
  assert.equal(navigation.path, '/train-dataset')
  assert.equal(navigation.query.dataset, 'B')
  assert.equal(navigation.query.version, 'vB-old')
})

function picker(props = {}) {
  const changes = []
  const component = setup('components/medical/DatasetVersionPicker.vue', ['refresh', 'select', 'error'], {
    defineProps: () => ({ taskType: 'detect', ...props }),
    defineEmits: () => (_, value) => changes.push(value),
    listDatasets: async () => ({ items: [{ dataset_id: 'A' }, { dataset_id: 'B' }], total: 2 }),
    listVersions: async dataset => ({ items: [
      { version_id: dataset + '-new', state: 'draft' },
      { version_id: dataset + '-old', state: 'frozen' },
    ] }),
  })
  return { component, changes }
}

test('picker opens a linked older version instead of the first dataset or latest version', async () => {
  const { component, changes } = picker({ initialDatasetId: 'B', initialVersionId: 'B-old' })
  await component.refresh()
  assert.equal(changes.at(-1).datasetId, 'B')
  assert.equal(changes.at(-1).version.version_id, 'B-old')
})

test('picker preserves default frozen selection and programmatic retraining selection', async () => {
  const { component, changes } = picker({ frozenOnly: true })
  await component.refresh()
  assert.equal(changes.at(-1).version.version_id, 'A-old')
  await component.select('B', 'B-old')
  assert.equal(changes.at(-1).version.version_id, 'B-old')
})

test('unavailable deep-linked versions do not fall back to an unrelated version', async () => {
  const { component, changes } = picker({ initialDatasetId: 'B', initialVersionId: 'missing' })
  await component.refresh()
  assert.match(component.error.value, /已不可用/)
  assert.equal(changes.at(-1).version, null)
})

test('formal ISIC mode resets smoke overrides and sends the selected registered arm and seed', () => {
  const component = setup('views/IsicWorkspace.vue', ['purpose', 'trainingParams', 'request', 'report', 'checkedRequest'])
  Object.assign(component.trainingParams, { epochs: 2, batch: 3, image_size: 128, lr: .01,
    seed: 100, class_weight: 'none', patience: 2, amp: false, weight_decay: .02, num_workers: 4 })
  component.report.value = { ok: true }
  component.checkedRequest.value = 'smoke-preflight'
  component.purpose.value = 'formal'
  assert.equal(component.report.value, null)
  assert.equal(component.checkedRequest.value, '')
  assert.deepEqual(JSON.parse(JSON.stringify(component.request.value.params)), {
    epochs: 30, batch: 16, lr: .0003, seed: 42, device: '0', class_weight: 'balanced',
    patience: 8, image_size: 224, weight_decay: .0001, amp: true, num_workers: 0,
  })
  component.trainingParams.seed = 44
  component.trainingParams.class_weight = 'none'
  assert.equal(component.request.value.params.seed, 44)
  assert.equal(component.request.value.params.class_weight, 'none')
})

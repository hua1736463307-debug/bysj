<template>
  <main class="medical-page">
    <h1>人工复核</h1>
    <p class="muted">
      逐标签保留模型值与独立审核意见（支持/不支持/无法判断）。提交与审批分离；五项未填全或存在
      "无法判断"时不能批准，系统不会自动通过。审核者记录为实际操作账号，不代表临床执业资格。
    </p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>

    <section class="toolbar">
      <label>状态
        <select v-model="state" @change="act(refresh)">
          <option value="">全部</option>
          <option value="draft">草稿</option>
          <option value="submitted">待审批</option>
          <option value="approved">已批准</option>
          <option value="rejected">已驳回</option>
        </select>
      </label>
      <el-button :disabled="busy" @click="act(refresh)">刷新</el-button>
    </section>

    <table>
      <thead><tr><th>复核</th><th>分析</th><th>状态</th><th>revision</th><th>审核者</th><th>操作</th></tr></thead>
      <tbody>
        <tr v-for="item in items" :key="item.review_id">
          <td class="muted">{{ item.review_id.slice(0, 8) }}</td>
          <td class="muted">{{ item.analysis_id.slice(0, 8) }}</td>
          <td>{{ item.state }}</td>
          <td>{{ item.revision }}</td>
          <td>{{ item.reviewer || '—' }}</td>
          <td><el-button :disabled="busy" @click="act(() => open(item.review_id))">打开</el-button></td>
        </tr>
      </tbody>
    </table>

    <section v-if="current">
      <h2>复核 {{ current.review_id.slice(0, 8) }} · {{ current.state }} · revision {{ current.revision }}</h2>
      <table>
        <thead><tr><th>观察项</th><th>模型评分</th><th>阈值</th><th>提示</th><th>人工意见</th><th>理由</th></tr></thead>
        <tbody>
          <tr v-for="row in matrix" :key="row.label_key">
            <td>{{ row.display_name }}</td>
            <td>{{ fmt(row.calibrated_score ?? row.raw_score) }}</td>
            <td>{{ fmt(row.threshold) }}</td>
            <td class="muted">{{ flagText(row.flag) }}</td>
            <td>
              <select v-model="draft[row.label_key].opinion" :disabled="current.state !== 'draft'">
                <option value="">未填写</option>
                <option value="supported">支持</option>
                <option value="not_supported">不支持</option>
                <option value="indeterminate">无法判断</option>
              </select>
            </td>
            <td>
              <input v-model="draft[row.label_key].reason" :disabled="current.state !== 'draft'" placeholder="判读依据（可选）" />
            </td>
          </tr>
        </tbody>
      </table>
      <p class="muted">原模型分数以快照保存，人工编辑不会覆盖它。</p>
      <div class="toolbar">
        <label>复核说明 <input v-model="reason" :disabled="current.state !== 'draft'" /></label>
        <el-button :disabled="busy || current.state !== 'draft'" @click="act(save)">保存草稿</el-button>
        <el-button :disabled="busy || current.state !== 'draft'" @click="act(submit)">提交复核</el-button>
        <el-button type="primary" :disabled="busy || current.state !== 'submitted'" @click="act(() => decide(true))">批准</el-button>
        <el-button :disabled="busy || current.state !== 'submitted'" @click="act(() => decide(false))">驳回</el-button>
        <el-button :disabled="busy || !['submitted', 'approved'].includes(current.state)" @click="act(makeReport)">
          生成报告
        </el-button>
      </div>
      <p v-if="current.decided_by" class="muted">审批：{{ current.decided_by }} · {{ current.decided_at }}</p>
    </section>
  </main>
</template>
<script setup>
import { onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { LABEL_FLAG_TEXT, createReport, decideReview, getReview, listReviews, submitReview, updateReview } from '../../api/cxr'
import { useTaskAction } from '../../composables/useTaskAction'

const { error, busy, act } = useTaskAction()
const router = useRouter()
const items = ref([]), state = ref(''), current = ref(null), matrix = ref([]), draft = reactive({}), reason = ref('')

function fmt(value) { return Number.isFinite(value) ? Number(value).toFixed(4) : '—' }
function flagText(flag) { return LABEL_FLAG_TEXT[flag] || flag }
async function refresh() { items.value = (await listReviews({ state: state.value || undefined, page_size: 50 })).items }
async function open(reviewId) {
  const review = await getReview(reviewId)
  current.value = review
  matrix.value = review.matrix
  reason.value = review.reason || ''
  for (const key of Object.keys(draft)) delete draft[key]
  for (const row of review.matrix) {
    draft[row.label_key] = { opinion: row.opinion || '', reason: row.opinion_reason || '' }
  }
}
function payloadItems() {
  return Object.entries(draft)
    .filter(([, value]) => value.opinion)
    .map(([labelKey, value]) => ({ label_key: labelKey, opinion: value.opinion, reason: value.reason || null }))
}
async function save() {
  const review = await updateReview(current.value.review_id, {
    expected_revision: current.value.revision,
    reason: reason.value || null,
    items: payloadItems()
  })
  await open(review.review_id)
}
async function submit() {
  const review = await submitReview(current.value.review_id)
  await open(review.review_id)
  await refresh()
}
async function makeReport() {
  await createReport({ analysis_id: current.value.analysis_id, review_id: current.value.review_id })
  router.push('/cxr/reports')
}
async function decide(approve) {
  const review = await decideReview(current.value.review_id, approve, reason.value || null)
  await open(review.review_id)
  await refresh()
}
onMounted(() => act(refresh))
</script>

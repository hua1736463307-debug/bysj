<template>
  <main class="medical-page">
    <h1>辅助分析报告</h1>
    <p class="muted">
      报告引用输入、五项评分、阈值/校准、模型版本与真实审核意见。无批准复核只出"待复核研究草稿"；
      最终报告锁定复核 revision 生成不可变快照，后续变化出新版本并引用上一版。
    </p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>

    <section class="toolbar">
      <el-button :disabled="busy" @click="act(refresh)">刷新报告</el-button>
      <span class="muted">草稿不可作为诊断依据；导出只读快照，不重新计算。</span>
    </section>

    <table>
      <thead><tr><th>报告</th><th>版本</th><th>状态</th><th>复核 revision</th><th>签发</th><th>操作</th></tr></thead>
      <tbody>
        <tr v-for="item in items" :key="item.report_id">
          <td class="muted">{{ item.report_id.slice(0, 8) }}<div class="muted">{{ item.analysis_id.slice(0, 8) }}</div></td>
          <td>v{{ item.version }}</td>
          <td :class="item.state === 'final' ? '' : 'warn'">{{ item.state === 'final' ? '最终' : '待复核草稿' }}</td>
          <td>{{ item.review_revision }}</td>
          <td>{{ item.issued_by || '—' }}<div class="muted">{{ item.issued_at || '' }}</div></td>
          <td>
            <el-button size="small" :disabled="busy" @click="act(() => open(item.report_id))">查看</el-button>
            <el-button size="small" :disabled="busy" @click="act(() => download(item.report_id, 'html'))">导出 HTML</el-button>
            <el-button size="small" :disabled="busy" @click="act(() => download(item.report_id, 'json'))">导出 JSON</el-button>
          </td>
        </tr>
      </tbody>
    </table>

    <section v-if="snapshot">
      <h2>{{ snapshot.title }} · {{ snapshot.case?.case_id }} · v{{ snapshot.version }}</h2>
      <p :class="snapshot.state === 'final' ? 'muted' : 'warn'">
        {{ snapshot.state === 'final' ? '最终报告（已引用批准复核）' : '待复核研究草稿（未经人工批准）' }}
      </p>
      <table>
        <tbody>
          <tr><td>任务 / 契约</td><td>{{ snapshot.task }} / {{ snapshot.label_schema }}</td></tr>
          <tr><td>模型</td><td class="hash">{{ snapshot.model?.model_id }} · {{ snapshot.model?.sha256?.slice(0, 16) }}…</td></tr>
          <tr><td>校准 / 阈值来源</td><td>{{ snapshot.model?.calibration_status }} / {{ snapshot.model?.threshold_source }}</td></tr>
          <tr><td>输入影像</td><td class="hash">{{ snapshot.input?.sha256?.slice(0, 16) }}… · {{ snapshot.input?.width }}×{{ snapshot.input?.height }} · {{ snapshot.input?.kind }}</td></tr>
          <tr><td>复核</td><td>{{ snapshot.review?.state }} · revision {{ snapshot.review?.revision }} · {{ snapshot.review?.reviewer || '未指定' }}</td></tr>
        </tbody>
      </table>
      <table>
        <thead><tr><th>观察项</th><th>原始</th><th>校准</th><th>阈值</th><th>提示</th><th>人工意见</th></tr></thead>
        <tbody>
          <tr v-for="row in snapshot.labels || []" :key="row.label_key">
            <td>{{ row.display_name }}</td>
            <td>{{ fmt(row.raw_score) }}</td>
            <td>{{ fmt(row.calibrated_score) }}</td>
            <td>{{ fmt(row.threshold) }}</td>
            <td class="muted">{{ flagText(row.flag) }}</td>
            <td>{{ opinionText(row.opinion) }}</td>
          </tr>
        </tbody>
      </table>
      <h3>限制与免责</h3>
      <ul class="muted"><li v-for="(line, index) in snapshot.limitations || []" :key="index">{{ line }}</li></ul>
    </section>
  </main>
</template>
<script setup>
import { onMounted, ref } from 'vue'
import { LABEL_FLAG_TEXT, OPINION_TEXT, exportReport, getReport, listReports } from '../../api/cxr'
import { useTaskAction } from '../../composables/useTaskAction'
import { saveBlob } from '../../utils/download'

const { error, busy, act } = useTaskAction()
const items = ref([]), snapshot = ref(null)

function fmt(value) { return Number.isFinite(value) ? Number(value).toFixed(4) : '—' }
function flagText(flag) { return LABEL_FLAG_TEXT[flag] || flag }
function opinionText(opinion) { return OPINION_TEXT[opinion] || '未填写' }
async function refresh() { items.value = (await listReports({ page_size: 50 })).items }
async function open(reportId) { snapshot.value = (await getReport(reportId)).snapshot }
async function download(reportId, format) {
  saveBlob(await exportReport(reportId, format), `${reportId.slice(0, 8)}.${format}`)
}
onMounted(() => act(refresh))
</script>

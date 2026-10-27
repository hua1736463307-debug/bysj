<template>
  <main class="medical-page">
    <h1>胸片五标签辅助分析</h1>
    <p class="muted">
      研究原型：输出五项模型评分与人工复核状态，不是临床诊断结论。未超过阈值不代表排除该征象或其他疾病。
    </p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <section class="toolbar">
      <span>标签契约：{{ schema?.label_schema || 'cxr-five-v1' }}</span>
      <span>任务：{{ schema?.task || 'cxr_multilabel' }}</span>
      <el-button :disabled="busy" @click="act(load)">刷新统计</el-button>
    </section>
    <el-row :gutter="12">
      <el-col :span="6"><el-card shadow="never"><div class="muted">数据集 / 版本</div><h2>{{ stats.datasets ?? '—' }} / {{ stats.versions ?? '—' }}</h2></el-card></el-col>
      <el-col :span="6"><el-card shadow="never"><div class="muted">分析（完成 / 失败）</div><h2>{{ stats.analyses?.completed ?? '—' }} / {{ stats.analyses?.failed ?? '—' }}</h2></el-card></el-col>
      <el-col :span="6"><el-card shadow="never"><div class="muted">复核（草稿 / 待审 / 已批准）</div><h2>{{ stats.reviews?.draft ?? '—' }} / {{ stats.reviews?.submitted ?? '—' }} / {{ stats.reviews?.approved ?? '—' }}</h2></el-card></el-col>
      <el-col :span="6"><el-card shadow="never"><div class="muted">报告（草稿 / 最终）</div><h2>{{ stats.reports?.draft ?? '—' }} / {{ stats.reports?.final ?? '—' }}</h2></el-card></el-col>
    </el-row>
    <p class="muted">{{ stats.note }}</p>
    <h2>五项观察</h2>
    <table>
      <thead><tr><th>顺序</th><th>稳定 key</th><th>展示名</th><th>原始列</th></tr></thead>
      <tbody>
        <tr v-for="label in schema?.labels || []" :key="label.key">
          <td>{{ label.index }}</td><td class="hash">{{ label.key }}</td>
          <td>{{ label.display_name }}</td><td>{{ label.source_column }}</td>
        </tr>
      </tbody>
    </table>
    <p class="muted">
      四态：{{ (schema?.raw_states || []).join(' / ') }}；监督只使用明确 positive/negative，
      uncertain 与空值保留原状态但不参与训练。
    </p>
  </main>
</template>
<script setup>
import { ref, onMounted } from 'vue'
import { getDashboard, getSchema } from '../../api/cxr'
import { useTaskAction } from '../../composables/useTaskAction'
const { error, busy, act } = useTaskAction()
const stats = ref({}), schema = ref(null)
async function load() { stats.value = await getDashboard(); schema.value = await getSchema() }
onMounted(() => act(load))
</script>

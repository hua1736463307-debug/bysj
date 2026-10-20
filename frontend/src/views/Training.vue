<template>
  <div class="training-page">
    <el-tabs v-model="trainingChain">
      <el-tab-pane label="YOLO 模型训练" name="yolo" />
      <el-tab-pane label="单帧姿态分类" name="pose" />
    </el-tabs>
    <PoseTrainingPanel v-if="trainingChain === 'pose'" />
    <div v-show="trainingChain === 'yolo'">
    <div class="filter-panel">
      <el-form class="filter-form" inline @submit.prevent>
        <el-form-item label="训练项目名称">
          <el-input
            v-model.trim="filters.projectName"
            class="filter-input"
            clearable
            placeholder="请输入训练项目名称"
            @keyup.enter="handleSearch"
          />
        </el-form-item>
        <el-form-item label="训练人">
          <el-input
            v-model.trim="filters.trainer"
            class="filter-input"
            clearable
            placeholder="请输入训练人"
            @keyup.enter="handleSearch"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">
            <el-icon><Search /></el-icon>
            搜索
          </el-button>
          <el-button @click="resetFilters">
            <el-icon><Refresh /></el-icon>
            重置
          </el-button>
        </el-form-item>
      </el-form>
    </div>

    <div class="table-actions">
      <el-button type="primary" class="new-training-button" @click="openTrainingDialog">
        <span class="new-training-icon"><el-icon><Plus /></el-icon></span>
        <span>新增训练</span>
      </el-button>
    </div>

    <div class="table-panel">
      <el-table :data="pagedTrainingRows" class="training-table" row-key="id">
        <el-table-column label="训练项目名称" min-width="260" align="center">
          <template #default="{ row }">
            {{ row.projectName }}
          </template>
        </el-table-column>
        <el-table-column label="数据集名称" min-width="260" align="center">
          <template #default="{ row }">
            {{ row.datasetName || '-' }}
          </template>
        </el-table-column>
        <el-table-column label="训练轮次" width="140" align="center">
          <template #default="{ row }">
            {{ row.epochs }}
          </template>
        </el-table-column>
        <el-table-column label="批次大小" width="140" align="center">
          <template #default="{ row }">
            {{ row.batchSize }}
          </template>
        </el-table-column>
        <el-table-column label="学习率" width="140" align="center">
          <template #default="{ row }">
            {{ formatLearningRate(row.learningRate) }}
          </template>
        </el-table-column>
        <el-table-column label="训练人" width="140" align="center">
          <template #default="{ row }">
            {{ row.trainer }}
          </template>
        </el-table-column>
        <el-table-column label="模型状态" width="180" align="center">
          <template #default="{ row }">
            <el-button
              v-if="row.status === 'completed' && row.testModel?.available"
              class="model-test-button"
              size="small"
              type="primary"
              @click="openModelTest(row)"
            >模型测试</el-button>
            <el-tooltip v-else-if="row.status === 'completed'" :content="row.testModel?.reason || '训练模型不可用'">
              <el-tag type="warning" size="small">模型不可用</el-tag>
            </el-tooltip>
            <el-tag v-else :type="getTaskTagType(row.status)" size="small">
              {{ getTaskStatusText(row.status || 'unknown') }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="240" align="center">
          <template #default="{ row }">
            <div class="training-row-actions">
              <el-button link type="primary" aria-label="查看训练详情" @click="viewTaskDetail(row)">
                <el-icon><View /></el-icon>
                详情
              </el-button>
              <el-button link type="primary" aria-label="重新训练" @click="retrainTask(row)">
                <el-icon><Refresh /></el-icon>
                重新训练
              </el-button>
              <el-button link type="danger" aria-label="删除训练记录" @click="deleteTrainingRecord(row)">
                <el-icon><Delete /></el-icon>
                删除
              </el-button>
            </div>
          </template>
        </el-table-column>
      </el-table>

      <el-empty v-if="filteredTrainingRows.length === 0" description="暂无训练项目" />

      <div class="pagination-wrap">
        <span class="total-text">共 {{ filteredTrainingRows.length }} 条</span>
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :total="filteredTrainingRows.length"
          :page-sizes="[10, 20, 50, 100]"
          layout="sizes, prev, pager, next, jumper"
        />
      </div>
    </div>

    <el-dialog
      v-model="showTrainingDialog"
      title="新增训练"
      width="1240px"
      class="training-config-dialog"
      destroy-on-close
    >
      <div class="dialog-layout">
        <div class="config-column">
          <el-card class="config-card">
            <template #header>
              <div class="card-header">
                <span>训练配置</span>
              </div>
            </template>

            <el-form :model="trainingConfig" label-width="120px" class="training-form">
              <el-row :gutter="20">
                <el-col :xs="24" :sm="12">
                  <el-form-item label="数据集路径">
                    <el-input
                      v-model="trainingConfig.datasetPath"
                      placeholder="选择数据集路径"
                      readonly
                    >
                      <template #append>
                        <el-button @click="selectDataset">选择</el-button>
                      </template>
                    </el-input>
                  </el-form-item>
                </el-col>

                <el-col :xs="24" :sm="12">
                  <el-form-item label="基础/续训模型">
                    <el-select
                      v-model="trainingConfig.modelName"
                      placeholder="选择预训练模型、上传模型或训练得到的 best.pt"
                      filterable
                      :disabled="trainingConfig.modelLocked"
                      style="width: 100%;"
                    >
                      <el-option
                        v-for="model in availableModels"
                        :key="model.path || model.name"
                        :label="getModelOptionLabel(model)"
                        :value="model.path || model.name"
                      >
                        <div class="model-option">
                          <span class="model-option-name">{{ model.displayName || model.name }}</span>
                          <span class="model-option-type">{{ getModelTypeText(model.type) }}</span>
                        </div>
                      </el-option>
                    </el-select>
                    <div v-if="trainingConfig.modelLocked" class="model-binding-hint model-binding-fixed-hint">
                      已按该训练数据集导出时的标注模型绑定：{{ trainingConfig.boundModelName }}；不可更改
                    </div>
                    <div v-else-if="trainingConfig.boundModelName" class="model-binding-hint">
                      最近一次标注模型：{{ trainingConfig.boundModelName }}；当前训练可重新选择
                    </div>
                  </el-form-item>
                </el-col>
              </el-row>

              <el-alert
                :title="isSelectedYoloe
                  ? 'YOLOE 分割专用 PE 微调：AdamW / lr0=0.001 / 150 epochs'
                  : '标准 YOLO 训练流程'"
                :type="isSelectedYoloe ? 'success' : 'info'"
                :closable="false"
                show-icon
                style="margin-bottom: 16px;"
              />

              <el-row :gutter="20">
                <el-col :xs="24" :sm="12">
                  <el-form-item label="训练轮数">
                    <el-input-number
                      v-model="trainingConfig.epochs"
                      :min="1"
                      :max="1000"
                      controls-position="right"
                    />
                  </el-form-item>
                </el-col>

                <el-col :xs="24" :sm="12">
                  <el-form-item label="批次大小">
                    <el-input-number
                      v-model="trainingConfig.batchSize"
                      :min="1"
                      :max="64"
                      controls-position="right"
                    />
                  </el-form-item>
                </el-col>
              </el-row>

              <el-row :gutter="20">
                <el-col :xs="24" :sm="12">
                  <el-form-item label="图片尺寸">
                    <el-select v-model="trainingConfig.imgSize" :disabled="Boolean(trainingConfig.datasetOptimalImageSize)">
                      <el-option label="320x320" :value="320" />
                      <el-option label="416x416" :value="416" />
                      <el-option label="512x512" :value="512" />
                      <el-option label="640x640" :value="640" />
                      <el-option label="1280x1280（小目标推荐）" :value="1280" />
                    </el-select>
                    <div v-if="trainingConfig.datasetOptimalImageSize" class="model-binding-hint">
                      已按该训练数据集裁剪结果自动选择并锁定：{{ trainingConfig.datasetOptimalImageSize }}x{{ trainingConfig.datasetOptimalImageSize }}
                    </div>
                  </el-form-item>
                </el-col>

                <el-col :xs="24" :sm="12">
                  <el-form-item label="学习率">
                    <el-input-number
                      v-model="trainingConfig.learningRate"
                      :min="0.0001"
                      :max="0.1"
                      :step="0.001"
                      controls-position="right"
                      style="width: 100%;"
                    />
                  </el-form-item>
                </el-col>
              </el-row>

              <el-row :gutter="20">
                <el-col :xs="24" :sm="12">
                  <el-form-item label="设备">
                    <el-select v-model="trainingConfig.device">
                      <el-option label="自动选择" value="auto" />
                      <el-option label="CPU" value="cpu" />
                      <el-option label="GPU 0" value="0" />
                      <el-option label="GPU 1" value="1" />
                    </el-select>
                  </el-form-item>
                </el-col>

                <el-col :xs="24" :sm="12">
                  <el-form-item label="项目名称">
                    <el-input v-model="trainingConfig.projectName" placeholder="输入项目名称" />
                  </el-form-item>
                </el-col>
              </el-row>

              <el-form-item label="训练类别">
                <el-select
                  v-model="trainingConfig.classNames"
                  multiple
                  placeholder="当前固定使用数据集完整类别"
                  style="width: 100%;"
                  filterable
                  disabled
                >
                  <el-option
                    v-for="item in annotationClasses"
                    :key="item"
                    :label="item"
                    :value="item"
                  />
                </el-select>
                <el-alert
                  title="为保护源标签，类别子集训练将在派生数据集版本功能完成后开放。"
                  type="warning"
                  :closable="false"
                  show-icon
                  style="margin-top: 8px;"
                />
              </el-form-item>

              <el-card v-if="trainingPreflight" class="preflight-card" shadow="never">
                <template #header>
                  <div class="card-header">
                    <span>训练预检 · {{ trainingPreflight.training_profile }}</span>
                    <el-tag :type="trainingPreflight.errors?.length ? 'danger' : 'success'">
                      {{ trainingPreflight.errors?.length ? '阻断' : '通过' }}
                    </el-tag>
                  </div>
                </template>
                <div class="preflight-summary">
                  <span>图片 {{ trainingPreflight.asset_count || 0 }}</span>
                  <span>实例 {{ trainingPreflight.annotation_count || 0 }}</span>
                  <span v-if="trainingPreflight.optimal_image_size">训练尺寸 {{ trainingPreflight.optimal_image_size }}x{{ trainingPreflight.optimal_image_size }}</span>
                  <span>大类 {{ trainingPreflight.major_class_ids?.length || 0 }}</span>
                  <span>稀疏类 {{ trainingPreflight.sparse_class_ids?.length || 0 }}</span>
                  <span>零样本类 {{ trainingPreflight.zero_class_ids?.length || 0 }}</span>
                </div>
                <el-alert
                  v-for="item in trainingPreflight.errors || []"
                  :key="`error-${item.code}`"
                  :title="item.message"
                  type="error"
                  :closable="false"
                  show-icon
                />
                <el-alert
                  v-for="item in trainingPreflight.warnings || []"
                  :key="`warning-${item.code}`"
                  :title="item.message"
                  type="warning"
                  :closable="false"
                  show-icon
                />
                <el-alert
                  v-if="gpuRuntimeUnavailable"
                  title="当前无剩余运算力，可提交训练排队，GPU 恢复后自动继续。"
                  type="error"
                  :closable="false"
                  show-icon
                />
              </el-card>

              <el-form-item>
                <el-button @click="runTrainingPreflight" :loading="isPreflighting">
                  执行预检
                </el-button>
                <el-button
                  type="primary"
                  @click="startTraining"
                  :loading="isTraining"
                  :disabled="!canStartTraining"
                >
                  <el-icon><VideoPlay /></el-icon>
                  {{ isTraining ? '训练中...' : '开始训练' }}
                </el-button>
                <el-button @click="resetConfig">
                  <el-icon><RefreshRight /></el-icon>
                  重置配置
                </el-button>
              </el-form-item>
            </el-form>
          </el-card>

        </div>

        <div class="side-panels">
          <el-card
            v-if="Object.keys(selectedTaskMajorMaskMetrics).length"
            class="status-card major-metrics-card"
          >
            <template #header>
              <div class="card-header">
                <span>大类验收指标（Mask）</span>
                <el-tag type="success">实例数 &gt; 20</el-tag>
              </div>
            </template>
            <div class="major-metric-summary">
              <div>
                <small>mAP50-95(M)</small>
                <strong>{{ Number(selectedTaskMajorMaskMetrics.map50_95 || 0).toFixed(4) }}</strong>
              </div>
              <div>
                <small>mAP50(M)</small>
                <strong>{{ Number(selectedTaskMajorMaskMetrics.map50 || 0).toFixed(4) }}</strong>
              </div>
              <div>
                <small>Recall(M)</small>
                <strong>{{ Number(selectedTaskMajorMaskMetrics.recall || 0).toFixed(4) }}</strong>
              </div>
              <div>
                <small>大类 mAP50-95(B)</small>
                <strong>{{ Number(selectedTaskGroupedMetrics?.major_classes?.box?.map50_95 || 0).toFixed(4) }}</strong>
              </div>
              <div>
                <small>全部类 mAP50-95(M)</small>
                <strong>{{ Number(selectedTaskGroupedMetrics?.all_classes?.mask?.map50_95 || 0).toFixed(4) }}</strong>
              </div>
            </div>
            <div v-for="item in selectedTaskMajorMaskRows" :key="item.class_id" class="major-class-row">
              <span>{{ item.name }}</span>
              <span>Recall {{ Number(item.recall || 0).toFixed(4) }}</span>
            </div>
            <div class="epoch-meta">
              <span>最佳 epoch：{{ selectedTask.bestEpoch || '-' }}</span>
              <span>当前学习率：{{ formatLearningRate(selectedTask.currentLearningRate) }}</span>
            </div>
            <div class="epoch-metrics">
              <div v-for="(value, key) in selectedTaskLatestMetrics" :key="key">
                <small>{{ key }}</small>
                <span>{{ Number(value).toFixed(5) }}</span>
              </div>
            </div>
          </el-card>

          <el-card class="status-card">
            <template #header>
              <div class="card-header">
                <span>训练状态</span>
                <el-button text :disabled="!currentDialogTask" @click="refreshCurrentTask">
                  <el-icon><Refresh /></el-icon>
                </el-button>
              </div>
            </template>

            <div class="training-tasks">
              <div v-if="!currentDialogTask" class="empty-tasks">
                <el-empty description="启动训练后显示当前任务状态" />
              </div>

              <div v-else class="task-item">
                <div class="task-header">
                  <span class="task-name">{{ currentDialogTask.name }}</span>
                  <el-tag :type="getTaskTagType(currentDialogTask.status)" size="small">
                    {{ getTaskStatusText(currentDialogTask.status) }}
                  </el-tag>
                </div>

                <el-progress
                  :percentage="Math.round(currentDialogTask.progress)"
                  :status="getProgressStatus(currentDialogTask.status)"
                  :stroke-width="8"
                />

                <div class="task-info">
                  <small>{{ currentDialogTask.message }}</small>
                </div>

                <div class="training-log status-training-log">
                  <h4>训练日志</h4>
                  <el-scrollbar height="180px">
                    <div class="log-content">
                      <div v-if="currentDialogTask.logs.length === 0" class="log-empty">
                        暂无训练日志
                      </div>
                      <div
                        v-for="(log, index) in currentDialogTask.logs"
                        :key="index"
                        class="log-line"
                      >
                        {{ log }}
                      </div>
                    </div>
                  </el-scrollbar>
                </div>

                <div class="task-actions">
                  <el-button size="small" @click="viewTaskDetail(currentDialogTask)">
                    <el-icon><View /></el-icon>
                    详情
                  </el-button>
                  <el-button
                    v-if="currentDialogTask.status === 'running'"
                    size="small"
                    type="danger"
                    @click="stopTask(currentDialogTask)"
                  >
                    <el-icon><VideoPause /></el-icon>
                    停止
                  </el-button>
                </div>
              </div>
            </div>
          </el-card>

          <el-card class="resource-card">
            <template #header>
              <div class="card-header">
                <span>系统资源</span>
                <el-button size="small" text @click="refreshSystemResources">
                  <el-icon><Refresh /></el-icon>
                </el-button>
              </div>
            </template>

            <el-alert
              v-if="systemStatus.error"
              :title="systemStatus.error"
              type="warning"
              show-icon
              :closable="false"
              class="resource-alert"
            />

            <div class="resource-item">
              <div class="resource-label">
                <span>CPU使用率</span>
                <span>{{ systemStatus.cpu }}%</span>
              </div>
              <el-progress :percentage="systemStatus.cpu" :stroke-width="8" />
            </div>

            <div class="resource-item">
              <div class="resource-label">
                <span>内存使用率</span>
                <span>{{ systemStatus.memory }}%</span>
              </div>
              <el-progress :percentage="systemStatus.memory" :stroke-width="8" />
              <div v-if="systemStatus.memoryText" class="resource-extra">
                内存：{{ systemStatus.memoryText }}
              </div>
            </div>

            <div class="resource-item">
              <div class="resource-label">
                <span>GPU使用率</span>
                <span>{{ systemStatus.gpu }}%</span>
              </div>
              <el-progress
                :percentage="systemStatus.gpu"
                :stroke-width="8"
                :status="systemStatus.gpu > 0 ? undefined : 'exception'"
              />
              <div class="resource-extra">
                <div>{{ systemStatus.gpuName }}</div>
                <div v-if="systemStatus.gpuMemoryText">显存：{{ systemStatus.gpuMemoryText }}</div>
                <div v-if="systemStatus.gpuTemperature !== null">温度：{{ systemStatus.gpuTemperature }}℃</div>
              </div>
            </div>
          </el-card>
        </div>
      </div>
    </el-dialog>

    <el-dialog v-model="showDatasetDialog" width="900px" title="选择数据集">
      <el-table
        :data="datasetList"
        v-loading="isLoadingDatasets"
        highlight-current-row
        :row-class-name="getRowClassName"
        @row-click="selectDatasetRow"
      >
        <el-table-column width="50">
          <template #default="{ row }">
            <el-radio
              v-model="selectedDatasetId"
              :label="row.id"
              @change="selectDatasetRow(row)"
            >
              <span></span>
            </el-radio>
          </template>
        </el-table-column>
        <el-table-column prop="name" label="数据集名称" min-width="220" />
        <el-table-column prop="image_count" label="图片数" width="100" align="center">
          <template #default="{ row }">
            {{ row.image_count || row.imageCount || 0 }}
          </template>
        </el-table-column>
        <el-table-column prop="size" label="大小" width="120" align="center">
          <template #default="{ row }">
            {{ formatFileSize(row.size || 0) }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="110" align="center">
          <template #default="{ row }">
            <el-tag :type="row.status === 'ready' ? 'success' : 'info'" size="small">
              {{ row.status === 'ready' ? '就绪' : '默认' }}
            </el-tag>
          </template>
        </el-table-column>
      </el-table>

      <div v-if="selectedDatasetInfo" class="selected-info">
        已选择：<strong>{{ selectedDatasetInfo.name }}</strong>
        <span>{{ selectedDatasetInfo.image_count || selectedDatasetInfo.imageCount || 0 }} 张图片</span>
      </div>

      <template #footer>
        <el-button @click="showDatasetDialog = false">取消</el-button>
        <el-button @click="loadDatasetList(true)" :loading="isLoadingDatasets">
          刷新列表
        </el-button>
        <el-button type="primary" @click="confirmDatasetSelection" :disabled="!selectedDatasetId">
          确定选择
        </el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="showTaskDialog"
      :title="`训练详情 - ${selectedTask?.name}`"
      width="1240px"
      class="training-config-dialog"
      destroy-on-close
      @closed="clearTaskDetailTimer"
    >
      <div v-if="selectedTask" class="dialog-layout detail-dialog-layout">
        <div class="config-column">
          <el-card class="config-card">
            <template #header>
              <div class="card-header">
                <span>训练配置</span>
                <el-button text @click="refreshTaskDetail">
                  <el-icon><Refresh /></el-icon>
                </el-button>
              </div>
            </template>

            <el-form label-width="120px" class="training-form detail-training-form">
              <el-row :gutter="20">
                <el-col :xs="24" :sm="12">
                  <el-form-item label="数据集名称">
                    <span class="detail-value">{{ selectedTaskDatasetName }}</span>
                  </el-form-item>
                </el-col>
                <el-col :xs="24" :sm="12">
                  <el-form-item label="基础/续训模型">
                    <span class="detail-value">{{ selectedTaskConfig.model_name || '-' }}</span>
                  </el-form-item>
                </el-col>
                <el-col :xs="24" :sm="12">
                  <el-form-item label="训练模式">
                    <span class="detail-value">{{ selectedTaskConfig.training_profile || 'standard_yolo' }}</span>
                  </el-form-item>
                </el-col>
              </el-row>

              <el-row :gutter="20">
                <el-col :xs="24" :sm="12">
                  <el-form-item label="训练轮数">
                    <span class="detail-value">{{ selectedTaskConfig.epochs ?? '-' }}</span>
                  </el-form-item>
                </el-col>
                <el-col :xs="24" :sm="12">
                  <el-form-item label="批次大小">
                    <span class="detail-value">{{ selectedTaskConfig.batch_size ?? '-' }}</span>
                  </el-form-item>
                </el-col>
              </el-row>

              <el-row :gutter="20">
                <el-col :xs="24" :sm="12">
                  <el-form-item label="图片尺寸">
                    <span class="detail-value">{{ selectedTaskConfig.img_size ? `${selectedTaskConfig.img_size} x ${selectedTaskConfig.img_size}` : '-' }}</span>
                  </el-form-item>
                </el-col>
                <el-col :xs="24" :sm="12">
                  <el-form-item label="学习率">
                    <span class="detail-value">{{ formatLearningRate(selectedTaskConfig.learning_rate) }}</span>
                  </el-form-item>
                </el-col>
              </el-row>

              <el-row :gutter="20">
                <el-col :xs="24" :sm="12">
                  <el-form-item label="设备">
                    <span class="detail-value">{{ formatTrainingDevice(selectedTaskConfig.device) }}</span>
                  </el-form-item>
                </el-col>
                <el-col :xs="24" :sm="12">
                  <el-form-item label="项目名称">
                    <span class="detail-value">{{ selectedTaskConfig.project_name || selectedTask.name || '-' }}</span>
                  </el-form-item>
                </el-col>
              </el-row>

              <el-row :gutter="20">
                <el-col :xs="24" :sm="12">
                  <el-form-item label="任务 ID">
                    <span class="detail-value detail-mono">{{ selectedTask.id }}</span>
                  </el-form-item>
                </el-col>
                <el-col :xs="24" :sm="12">
                  <el-form-item label="开始时间">
                    <span class="detail-value">{{ formatDate(selectedTask.startTime) }}</span>
                  </el-form-item>
                </el-col>
              </el-row>

              <el-form-item label="训练类别">
                <span class="detail-value">{{ formatTaskClasses(selectedTaskConfig.class_names) }}</span>
              </el-form-item>
            </el-form>
          </el-card>
        </div>

        <div class="side-panels">
          <el-card class="status-card">
            <template #header>
              <div class="card-header">
                <span>训练状态</span>
                <el-button text @click="refreshTaskDetail">
                  <el-icon><Refresh /></el-icon>
                </el-button>
              </div>
            </template>

            <div class="training-tasks">
              <div class="task-item">
                <div class="task-header">
                  <span class="task-name">{{ selectedTask.name }}</span>
                  <el-tag :type="getTaskTagType(selectedTask.status)" size="small">
                    {{ getTaskStatusText(selectedTask.status) }}
                  </el-tag>
                </div>

                <el-progress
                  :percentage="Math.round(selectedTask.progress || 0)"
                  :status="getProgressStatus(selectedTask.status)"
                  :stroke-width="8"
                />

                <div class="task-info">
                  <small>{{ selectedTask.message || '等待训练日志更新' }}</small>
                </div>

                <div class="training-log status-training-log">
                  <h4>训练日志</h4>
                  <el-scrollbar height="180px">
                    <div class="log-content">
                      <div v-if="selectedTask.logs.length === 0" class="log-empty">
                        暂无训练日志
                      </div>
                      <div
                        v-for="(log, index) in selectedTask.logs"
                        :key="index"
                        class="log-line"
                      >
                        {{ log }}
                      </div>
                    </div>
                  </el-scrollbar>
                </div>

                <div class="task-actions">
                  <el-button
                    v-if="['pending', 'running', 'interrupted', 'cancelling'].includes(selectedTask.status)"
                    size="small"
                    type="danger"
                    :loading="selectedTask.status === 'cancelling'"
                    :disabled="selectedTask.status === 'cancelling'"
                    @click="stopTask(selectedTask)"
                  >
                    <el-icon><VideoPause /></el-icon>
                    {{ selectedTask.status === 'cancelling' ? '正在取消' : '取消任务' }}
                  </el-button>
                </div>
              </div>
            </div>
          </el-card>

          <el-card class="resource-card">
            <template #header>
              <div class="card-header">
                <span>系统资源</span>
                <el-button size="small" text @click="refreshSystemResources">
                  <el-icon><Refresh /></el-icon>
                </el-button>
              </div>
            </template>

            <el-alert
              v-if="systemStatus.error"
              :title="systemStatus.error"
              type="warning"
              show-icon
              :closable="false"
              class="resource-alert"
            />

            <div class="resource-item">
              <div class="resource-label">
                <span>CPU 使用率</span>
                <span>{{ systemStatus.cpu }}%</span>
              </div>
              <el-progress :percentage="systemStatus.cpu" :stroke-width="8" />
            </div>

            <div class="resource-item">
              <div class="resource-label">
                <span>内存使用率</span>
                <span>{{ systemStatus.memory }}%</span>
              </div>
              <el-progress :percentage="systemStatus.memory" :stroke-width="8" />
              <div v-if="systemStatus.memoryText" class="resource-extra">
                内存：{{ systemStatus.memoryText }}
              </div>
            </div>

            <div class="resource-item">
              <div class="resource-label">
                <span>GPU 使用率</span>
                <span>{{ systemStatus.gpu }}%</span>
              </div>
              <el-progress
                :percentage="systemStatus.gpu"
                :stroke-width="8"
                :status="systemStatus.gpu > 0 ? undefined : 'exception'"
              />
              <div class="resource-extra">
                <div>{{ systemStatus.gpuName }}</div>
                <div v-if="systemStatus.gpuMemoryText">显存：{{ systemStatus.gpuMemoryText }}</div>
                <div v-if="systemStatus.gpuTemperature !== null">温度：{{ systemStatus.gpuTemperature }}℃</div>
              </div>
            </div>
          </el-card>
        </div>
      </div>

      <template #footer>
        <el-button @click="showTaskDialog = false">关闭</el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="showModelTestDialog"
      :title="`模型测试 - ${modelTestTask?.projectName || ''}`"
      width="92vw"
      class="model-test-dialog"
      destroy-on-close
      append-to-body
      :close-on-click-modal="!modelTestBusy"
      :close-on-press-escape="!modelTestBusy"
      :before-close="closeModelTest"
    >
      <BatchInferencePanel
        v-if="modelTestTask"
        :key="`${modelTestTask.id}-${modelTestVersion}`"
        embedded
        :training-task-id="modelTestTask.id"
        :locked-model="modelTestTask.testModel"
        @busy-change="modelTestBusy = $event"
      />
    </el-dialog>

    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import PoseTrainingPanel from '../components/PoseTrainingPanel.vue'
const trainingChain = ref('yolo')
import { ElMessage, ElMessageBox } from 'element-plus'
import { useRoute } from 'vue-router'
import {
  getModels,
  getJobEvents,
  getSystemResources,
  getTaskStatus as apiGetTaskStatus,
  getTrainingTaskStatus,
  getTasks as apiGetTasks,
  getTrainingDatasets,
  deleteTask,
  preflightTraining,
  startTraining as apiStartTraining,
  stopTraining as apiStopTraining
} from '../utils/api'
import BatchInferencePanel from '../components/BatchInferencePanel.vue'
import { modelSupportsTextPrompts } from '../utils/modelCapabilities.mjs'

const route = useRoute()

const isTraining = ref(false)
const loadingTasks = ref(false)
const showTrainingDialog = ref(false)
const showDatasetDialog = ref(false)
const showTaskDialog = ref(false)
const showModelTestDialog = ref(false)
const modelTestTask = ref(null)
const modelTestVersion = ref(0)
const modelTestBusy = ref(false)
const isLoadingDatasets = ref(false)
const currentPage = ref(1)
const pageSize = ref(10)
const selectedTask = ref(null)
const selectedDatasetId = ref('')
const selectedDatasetInfo = ref(null)
const annotationClasses = ref([])
const availableModels = ref([])
const trainingPreflight = ref(null)
const isPreflighting = ref(false)
const datasetList = ref([])
const activeTasks = ref([])
const currentDialogTaskId = ref('')
const taskTimers = new Map()
const detailTaskTimer = ref(null)
const maxDetailLogLines = 1000

const filters = reactive({
  projectName: '',
  trainer: ''
})

const trainingConfig = reactive({
  datasetPath: '',
  datasetName: '',
  modelName: 'yolov8n.pt',
  boundModelName: '',
  modelLocked: false,
  annotationGeometry: '',
  trainingProfile: 'auto',
  epochs: 100,
  batchSize: 16,
  imgSize: 640,
  datasetOptimalImageSize: null,
  learningRate: 0.01,
  device: 'auto',
  projectName: '',
  majorClassThreshold: 20,
  classNames: []
})

const systemStatus = reactive({
  cpu: 0,
  memory: 0,
  memoryText: '',
  gpu: 0,
  gpuName: '未检测到GPU',
  gpuMemoryText: '',
  gpuTemperature: null,
  gpuStatus: 'unavailable',
  gpuReason: 'cuda_unavailable',
  gpuError: '',
  error: ''
})

const resourceTimer = ref(null)

const trainingRecords = ref([])

const syncTrainingRecord = (taskId, payload = {}) => {
  const record = trainingRecords.value.find(item => String(item.id) === String(taskId))
  if (!record) return
  record.status = payload.status || record.status
  if (payload.test_model) record.testModel = payload.test_model
}

const normalizeDatasetPath = (value) => {
  let path = String(value || '').trim().replace(/\\/g, '/').replace(/\/+$/, '')
  if (/\.(ya?ml)$/i.test(path)) {
    path = path.slice(0, path.lastIndexOf('/'))
  }
  return path.replace(/\/+$/, '')
}

const datasetPathBasename = (value) => {
  const path = normalizeDatasetPath(value)
  const basename = path.split('/').filter(Boolean).pop() || ''
  return basename.replace(/_[0-9a-f]{10,64}$/i, '')
}

const SUPPORTED_IMAGE_SIZES = [320, 416, 512, 640, 1280]

const normalizeSupportedImageSize = (value) => {
  const parsed = Number(value)
  return SUPPORTED_IMAGE_SIZES.includes(parsed) ? parsed : null
}

const getDatasetOptimalImageSize = (dataset) => {
  const summary = dataset?.image_size && typeof dataset.image_size === 'object'
    ? dataset.image_size
    : (dataset?.imageSize && typeof dataset.imageSize === 'object' ? dataset.imageSize : {})
  const splitConfig = dataset?.split_config && typeof dataset.split_config === 'object'
    ? dataset.split_config
    : {}
  const explicitCandidates = [
    dataset?.optimal_image_size,
    dataset?.image_size_optimal,
    summary.optimal_size,
    summary.recommended_size,
    splitConfig.optimal_image_size
  ]
  for (const candidate of explicitCandidates) {
    const normalized = normalizeSupportedImageSize(candidate)
    if (normalized) return normalized
  }

  const counts = summary.target_size_counts
    || splitConfig.image_size_target_counts
    || splitConfig.target_size_counts
    || {}
  const ranked = Object.entries(counts)
    .map(([size, count]) => [normalizeSupportedImageSize(size), Number(count)])
    .filter(([size, count]) => size && Number.isFinite(count) && count > 0)
    .sort((left, right) => right[1] - left[1] || right[0] - left[0])
  return ranked[0]?.[0] || null
}

const applyDatasetImageSize = (dataset) => {
  const optimal = getDatasetOptimalImageSize(dataset)
  trainingConfig.datasetOptimalImageSize = optimal
  if (optimal) trainingConfig.imgSize = optimal
  return optimal
}

const getDatasetAnnotationBinding = (dataset) => {
  const nested = dataset?.annotation_model
  if (nested && typeof nested === 'object' && (nested.path || nested.model_path)) {
    return {
      path: String(nested.path || nested.model_path).trim(),
      name: String(nested.name || nested.model_name || nested.path || nested.model_path).trim(),
      geometry: String(nested.geometry || dataset.annotation_geometry || '').trim().toLowerCase()
    }
  }
  const path = String(dataset?.annotation_model_path || '').trim()
  if (!path) return null
  return {
    path,
    name: String(dataset.annotation_model_name || path).trim(),
    geometry: String(dataset.annotation_geometry || '').trim().toLowerCase()
  }
}

const isFixedTrainingDataset = (dataset, binding = getDatasetAnnotationBinding(dataset)) => {
  if (!binding?.path) return false
  const role = String(
    dataset?.dataset_role
    || dataset?.datasetRole
    || dataset?.role
    || 'training'
  ).trim().toLowerCase()
  if (role !== 'training') return false
  const mode = String(
    dataset?.model_binding_mode
    || dataset?.annotation_model_selection_mode
    || ''
  ).trim().toLowerCase()
  return mode !== 'reselectable'
}

const applyDatasetAnnotationBinding = (dataset) => {
  applyDatasetImageSize(dataset)
  const binding = getDatasetAnnotationBinding(dataset)
  const fixed = isFixedTrainingDataset(dataset, binding)
  trainingConfig.modelLocked = fixed
  if (!binding?.path) {
    const previousBoundModel = trainingConfig.boundModelName
    trainingConfig.boundModelName = ''
    trainingConfig.annotationGeometry = ''
    if (previousBoundModel && trainingConfig.modelName === previousBoundModel) {
      trainingConfig.modelName = availableModels.value[0]?.path || availableModels.value[0]?.name || 'yolov8n.pt'
    }
    return null
  }

  const key = binding.path
  if (!availableModels.value.some(model => (model.path || model.name) === key)) {
    availableModels.value.push({
      path: key,
      name: key,
      displayName: binding.name || key,
      type: 'bound',
      status: 'active',
      supportsTextPrompts: modelSupportsTextPrompts(binding, key)
    })
  }
  trainingConfig.modelName = key
  trainingConfig.boundModelName = binding.name || key
  trainingConfig.annotationGeometry = binding.geometry || ''
  return binding
}

const findTrainingDatasetForJob = (input = {}) => {
  const config = input.config || {}
  const versionId = String(input.dataset_version_id || config._dataset_version_id || '')
  const datasetPath = normalizeDatasetPath(input.dataset_path)
  const datasetId = datasetPath.split('/').filter(Boolean).pop() || ''

  return datasetList.value.find((dataset) => {
    if (versionId && String(dataset.version_id || dataset.versionId || '') === versionId) return true

    const candidatePath = normalizeDatasetPath(dataset.path || dataset.root_path || dataset.storage_path)
    if (datasetPath && candidatePath && candidatePath === datasetPath) return true

    return Boolean(datasetId && String(dataset.id || '') === datasetId)
  }) || null
}

const getTrainingDatasetDisplayName = (input = {}) => {
  const dataset = findTrainingDatasetForJob(input)
  return String(dataset?.name || dataset?.id || datasetPathBasename(input.dataset_path) || '-')
}

const canStartTraining = computed(() => {
  return trainingConfig.datasetPath && trainingConfig.projectName && !isTraining.value
})

const gpuRuntimeUnavailable = computed(() => (
  trainingConfig.device !== 'cpu' && systemStatus.gpuStatus !== 'ready'
))

const currentDialogTask = computed(() => {
  return activeTasks.value.find(item => String(item.id) === String(currentDialogTaskId.value)) || null
})

const selectedTaskInput = computed(() => selectedTask.value?.input || {})
const selectedTaskConfig = computed(() => selectedTaskInput.value.config || {})
const selectedTaskDatasetName = computed(() => getTrainingDatasetDisplayName(selectedTaskInput.value))
const selectedTaskGroupedMetrics = computed(() => (
  selectedTask.value?.groupedMetrics
  || selectedTask.value?.result?.grouped_metrics
  || selectedTask.value?.results?.grouped_metrics
  || {}
))
const selectedTaskLatestMetrics = computed(() => selectedTask.value?.latestMetrics || {})
const selectedTaskMajorMaskMetrics = computed(() => (
  selectedTaskGroupedMetrics.value?.major_classes?.mask || {}
))
const selectedTaskMajorMaskRows = computed(() => {
  const majorIds = new Set(selectedTaskGroupedMetrics.value?.major_class_ids || [])
  return (selectedTaskGroupedMetrics.value?.per_class?.mask || [])
    .filter(item => majorIds.has(item.class_id))
})

const selectedModel = computed(() => (
  availableModels.value.find(model => (model.path || model.name) === trainingConfig.modelName) || null
))
const isSelectedYoloe = computed(() => {
  const model = selectedModel.value
  return modelSupportsTextPrompts(model || {}, trainingConfig.modelName)
})

watch(() => [trainingConfig.modelName, isSelectedYoloe.value], () => {
  trainingConfig.trainingProfile = isSelectedYoloe.value ? 'yoloe_pe_segment' : 'standard_yolo'
  trainingConfig.epochs = isSelectedYoloe.value ? 150 : 100
  trainingConfig.learningRate = isSelectedYoloe.value ? 0.001 : 0.01
  trainingConfig.batchSize = 16
  trainingConfig.imgSize = trainingConfig.datasetOptimalImageSize || 640
  trainingPreflight.value = null
})

const filteredTrainingRows = computed(() => {
  const projectName = filters.projectName.trim().toLowerCase()
  const trainer = filters.trainer.trim().toLowerCase()

  return trainingRecords.value.filter((row) => {
    const matchProject = !projectName || row.projectName.toLowerCase().includes(projectName)
    const matchTrainer = !trainer || row.trainer.toLowerCase().includes(trainer)
    return matchProject && matchTrainer
  })
})

const pagedTrainingRows = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  return filteredTrainingRows.value.slice(start, start + pageSize.value)
})

const handleSearch = () => {
  currentPage.value = 1
}

const resetFilters = () => {
  filters.projectName = ''
  filters.trainer = ''
  currentPage.value = 1
}

const openTrainingDialog = () => {
  currentDialogTaskId.value = ''
  showTrainingDialog.value = true
  if (datasetList.value.length === 0) {
    loadDatasetList(true)
  }
  refreshSystemResources()
}

const retrainTask = (row) => {
  resetConfig()
  trainingConfig.projectName = `${row.projectName}_retrain`
  trainingConfig.datasetName = row.datasetName
  trainingConfig.epochs = Number(row.epochs) || 100
  trainingConfig.batchSize = Number(row.batchSize) || 16
  trainingConfig.learningRate = Number(row.learningRate) || 0.01

  const dataset = datasetList.value.find(item => String(item.id || '') === String(row.datasetId || ''))
    || datasetList.value.find(item => normalizeDatasetPath(item.path) === normalizeDatasetPath(row.datasetPath))
    || datasetList.value.find(item => item.name === row.datasetName)
  if (dataset) {
    selectedDatasetId.value = dataset.id
    selectedDatasetInfo.value = dataset
    trainingConfig.datasetPath = dataset.path || ''
    trainingConfig.datasetName = dataset.name || dataset.id || row.datasetName
    applyDatasetAnnotationBinding(dataset)
  } else {
    trainingConfig.datasetPath = row.datasetPath || ''
    applyDatasetAnnotationBinding(null)
  }

  openTrainingDialog()
}

const deleteTrainingRecord = async (row) => {
  try {
    await ElMessageBox.confirm(`确定要删除训练项目 "${row.projectName}" 吗？`, '删除确认', {
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      type: 'warning'
    })
  } catch {
    return
  }

  try {
    await deleteTask(row.id)
    clearTaskTimer(row.id)
    await refreshTasks()
    ElMessage.success('训练记录已删除')
  } catch (error) {
    console.error('删除训练记录失败:', error)
  }
}

const selectDataset = () => {
  showDatasetDialog.value = true
  if (datasetList.value.length === 0) {
    loadDatasetList(true)
  }
}

const selectDatasetRow = (row) => {
  selectedDatasetId.value = row.id
  selectedDatasetInfo.value = row
}

const getRowClassName = ({ row }) => {
  return selectedDatasetId.value === row.id ? 'selected-row' : ''
}

const confirmDatasetSelection = () => {
  const selectedDataset = datasetList.value.find(dataset => dataset.id === selectedDatasetId.value)
  if (!selectedDataset) return

  trainingConfig.datasetPath = selectedDataset.path || ''
  trainingConfig.datasetName = selectedDataset.name || selectedDataset.id || ''
  applyDatasetAnnotationBinding(selectedDataset)

  if (!trainingConfig.projectName) {
    trainingConfig.projectName = `${trainingConfig.datasetName}_training`
  }

  showDatasetDialog.value = false
  ElMessage.success(`已选择数据集: ${trainingConfig.datasetName}`)
}

const buildTrainingRequest = () => ({
  model_name: trainingConfig.modelName,
  training_profile: trainingConfig.trainingProfile,
  epochs: trainingConfig.epochs,
  batch_size: trainingConfig.batchSize,
  img_size: trainingConfig.imgSize,
  device: trainingConfig.device,
  learning_rate: trainingConfig.learningRate,
  optimizer: isSelectedYoloe.value ? 'AdamW' : null,
  patience: isSelectedYoloe.value ? 30 : null,
  major_class_threshold: trainingConfig.majorClassThreshold,
  prepare_latest_annotations: true,
  class_names: [],
  project_name: trainingConfig.projectName
})

const runTrainingPreflight = async () => {
  if (!trainingConfig.datasetPath || !trainingConfig.modelName) return null
  isPreflighting.value = true
  try {
    trainingPreflight.value = await preflightTraining(
      trainingConfig.datasetPath,
      buildTrainingRequest()
    )
    const optimal = normalizeSupportedImageSize(trainingPreflight.value?.optimal_image_size)
    if (optimal) {
      trainingConfig.datasetOptimalImageSize = optimal
      trainingConfig.imgSize = optimal
    }
    return trainingPreflight.value
  } finally {
    isPreflighting.value = false
  }
}

const startTraining = async () => {
  if (!canStartTraining.value) {
    ElMessage.warning('请完善训练配置')
    return
  }

  isTraining.value = true

  try {
    const preflight = await runTrainingPreflight()
    if (!preflight || (preflight.errors || []).length) {
      ElMessage.error('训练预检未通过，请先修复阻断问题')
      return
    }

    const config = buildTrainingRequest()
    const result = await apiStartTraining(trainingConfig.datasetPath, config)
    const taskId = result.task_id
    if (!taskId) throw new Error('后端未返回训练任务 ID')

    const newTask = {
      id: taskId,
      name: trainingConfig.projectName,
      status: result.status || 'pending',
      progress: 0,
      message: '训练准备中...',
      startTime: new Date(),
      logs: [result.message || '训练任务已进入队列'],
      eventCursor: 0
    }

    activeTasks.value.unshift(newTask)
    currentDialogTaskId.value = taskId
    trainingRecords.value.unshift({
      id: taskId,
      projectName: trainingConfig.projectName,
      datasetName: trainingConfig.datasetName || selectedDatasetInfo.value?.name || trainingConfig.datasetPath,
      datasetId: selectedDatasetInfo.value?.id || selectedDatasetId.value,
      datasetPath: trainingConfig.datasetPath,
      epochs: trainingConfig.epochs,
      batchSize: trainingConfig.batchSize,
      learningRate: trainingConfig.learningRate,
      trainer: getCurrentTrainer(),
      status: result.status || 'pending',
      testModel: null
    })

    monitorTask(taskId)
    ElMessage.success(result.message || '训练任务已进入队列')
  } catch (error) {
    console.error('启动训练失败:', error)
    const detail = error?.response?.data?.detail
    const message = typeof detail === 'object' ? detail.message : detail
    ElMessage.error(message || error?.message || '启动训练失败')
  } finally {
    isTraining.value = false
  }
}

const monitorTask = (taskId) => {
  const task = activeTasks.value.find(item => item.id === taskId)
  if (!task) return

  clearTaskTimer(taskId)
  const poll = async () => {
    if (document.hidden) {
      taskTimers.set(taskId, window.setTimeout(poll, 5000))
      return
    }
    try {
      const status = await getTrainingTaskStatus(taskId)
      task.status = status.status || task.status
      task.progress = Number(status.progress ?? task.progress ?? 0)
      task.message = status.message || task.message || ''
      syncTrainingRecord(taskId, status)
      const eventData = await getJobEvents(taskId, task.eventCursor || 0, 200)
      const events = Array.isArray(eventData?.events) ? eventData.events : []
      events.forEach(event => {
        task.logs.push(`[${formatDate(event.created_at)}] ${event.message}`)
      })
      if (events.length > 0) {
        task.eventCursor = Number(eventData.next_sequence || events[events.length - 1].sequence || task.eventCursor || 0)
      }
      if (['completed', 'failed', 'cancelled'].includes(task.status)) {
        clearTaskTimer(taskId)
        if (task.status === 'completed') {
          window.dispatchEvent(new CustomEvent('modelTrainingCompleted', {
            detail: { taskId: task.id, projectName: task.name }
          }))
        }
        return
      }
    } catch (error) {
      task.message = error?.response?.data?.detail || error?.message || '任务状态查询失败，稍后重试'
    }
    taskTimers.set(taskId, window.setTimeout(poll, 2000))
  }
  poll()
}

const clearTaskTimer = (taskId) => {
  const timer = taskTimers.get(taskId)
  if (timer) {
    clearTimeout(timer)
    taskTimers.delete(taskId)
  }
}

const clearTaskDetailTimer = () => {
  if (detailTaskTimer.value) {
    clearTimeout(detailTaskTimer.value)
    detailTaskTimer.value = null
  }
}

const formatTaskLog = (event) => {
  const timestamp = formatDate(event?.created_at)
  const message = String(event?.message || '').trim()
  return timestamp === '-' ? message : `[${timestamp}] ${message}`
}

const applyLogState = (task, events, nextSequence) => {
  const eventCursor = Number(
    nextSequence ?? events[events.length - 1]?.sequence ?? task.eventCursor ?? 0
  )
  const detail = {
    ...task,
    logs: [
      ...(Array.isArray(task.logs) ? task.logs : []),
      ...events.map(formatTaskLog).filter(Boolean)
    ].slice(-maxDetailLogLines),
    eventCursor: Number.isFinite(eventCursor) ? eventCursor : 0
  }

  events.forEach((event) => {
    const payload = event?.payload || {}
    if (payload.progress !== undefined && !Number.isNaN(Number(payload.progress))) {
      detail.progress = Number(payload.progress)
    }
    if (event?.message) {
      detail.message = event.message
    }
    if (payload.grouped_metrics) {
      detail.groupedMetrics = payload.grouped_metrics
    }
    if (payload.metrics) {
      detail.latestMetrics = payload.metrics
    }
    if (payload.learning_rate !== undefined) {
      detail.currentLearningRate = payload.learning_rate
    }
    if (payload.best_epoch !== undefined) {
      detail.bestEpoch = payload.best_epoch
    }
  })
  return detail
}

const scheduleTaskDetailRefresh = () => {
  clearTaskDetailTimer()
  if (
    !showTaskDialog.value
    || !selectedTask.value
    || ['completed', 'failed', 'cancelled'].includes(selectedTask.value.status)
  ) return
  detailTaskTimer.value = window.setTimeout(refreshTaskDetail, 2000)
}

const refreshTaskDetail = async () => {
  if (!showTaskDialog.value) return
  const taskId = selectedTask.value?.id
  if (!taskId) return

  try {
    const previous = selectedTask.value || {}
    const eventCursor = Number(previous.eventCursor || 0)
    const [taskData, eventData] = await Promise.all([
      apiGetTaskStatus(taskId),
      getJobEvents(taskId, eventCursor, eventCursor > 0 ? 200 : maxDetailLogLines)
    ])
    const events = Array.isArray(eventData?.events) ? eventData.events : []
    const task = {
      ...previous,
      ...taskData,
      name: taskData?.input?.config?.project_name || previous.name || taskId,
      startTime: taskData?.created_at || previous.startTime || null,
      input: taskData?.input || previous.input || {}
    }
    selectedTask.value = applyLogState(task, events, eventData?.next_sequence)
  } catch (error) {
    if (selectedTask.value) {
      selectedTask.value = {
        ...selectedTask.value,
        message: error?.response?.data?.detail || error?.message || '训练详情加载失败，稍后重试'
      }
    }
  } finally {
    scheduleTaskDetailRefresh()
  }
}

const refreshCurrentTask = () => {
  if (currentDialogTask.value) {
    monitorTask(currentDialogTask.value.id)
  }
}

const resetConfig = () => {
  Object.assign(trainingConfig, {
    datasetPath: '',
    datasetName: '',
    modelName: 'yolov8n.pt',
    boundModelName: '',
    modelLocked: false,
    annotationGeometry: '',
    trainingProfile: 'standard_yolo',
    epochs: 100,
    batchSize: 16,
    imgSize: 640,
    datasetOptimalImageSize: null,
    learningRate: 0.01,
    device: 'auto',
    projectName: '',
    majorClassThreshold: 20,
    classNames: []
  })
  trainingPreflight.value = null
  selectedDatasetId.value = ''
  selectedDatasetInfo.value = null
}

const refreshTasks = async () => {
  loadingTasks.value = true
  try {
    const data = await apiGetTasks()
    const rows = Array.isArray(data) ? data : (data?.tasks || [])
    activeTasks.value = rows.map(item => ({
      id: item.task_id || item.id,
      name: item.project_name || item.name || item.task_id || item.id,
      status: item.status || 'pending',
      progress: Number(item.progress || 0),
      message: item.message || '',
      startTime: item.created_at || item.start_time || null,
      logs: [],
      eventCursor: 0,
      input: item.input || {}
    })).filter(item => item.id)
    trainingRecords.value = rows.map(item => {
      const input = item.input || {}
      const config = input.config || {}
      const dataset = findTrainingDatasetForJob(input)
      return {
        id: item.task_id || item.id,
        projectName: config.project_name || item.task_id || item.id,
        datasetName: getTrainingDatasetDisplayName(input),
        datasetId: dataset?.id || '',
        datasetPath: input.dataset_path || dataset?.path || '',
        epochs: config.epochs || '-',
        batchSize: config.batch_size || '-',
        learningRate: config.learning_rate || '-',
        trainer: item.owner_username || '-',
        status: item.status || 'pending',
        testModel: item.test_model || null
      }
    }).filter(item => item.id)
    activeTasks.value
      .filter(item => !['completed', 'failed', 'cancelled'].includes(item.status))
      .forEach(item => monitorTask(item.id))
  } finally {
    loadingTasks.value = false
  }
}

const viewTaskDetail = (task) => {
  clearTaskDetailTimer()
  const storedTask = activeTasks.value.find(item => String(item.id) === String(task.id)) || {}
  selectedTask.value = {
    ...task,
    ...storedTask,
    name: storedTask.name || task.projectName || task.name || task.id,
    startTime: storedTask.startTime || task.startTime || null,
    logs: (Array.isArray(storedTask.logs)
      ? [...storedTask.logs]
      : (Array.isArray(task.logs) ? [...task.logs] : [])
    ).slice(-maxDetailLogLines),
    eventCursor: Number(storedTask.eventCursor ?? task.eventCursor ?? 0) || 0
  }
  showTaskDialog.value = true
  refreshTaskDetail()
  refreshSystemResources()
}

const stopTask = async (task) => {
  if (task.status === 'cancelling') return
  try {
    await ElMessageBox.confirm('确定要取消这个训练任务吗？等待中的任务将退出队列，运行中的任务将停止训练。', '取消任务确认', {
      confirmButtonText: '取消任务',
      cancelButtonText: '返回',
      type: 'warning'
    })
  } catch {
    return
  }

  if (!['pending', 'running', 'interrupted'].includes(task.status)) return
  const previousStatus = task.status
  task.status = 'cancelling'
  task.message = '正在请求取消...'
  try {
    const result = await apiStopTraining(task.id)
    task.status = result.status === 'interrupted' ? 'cancelling' : (result.status || 'cancelling')
    task.message = result.message || '正在取消训练任务'
    monitorTask(task.id)
    ElMessage.success(task.message)
  } catch (error) {
    task.status = previousStatus
    task.message = error?.response?.data?.detail || error?.message || '取消训练失败'
  }
}

const getTaskTagType = (status) => {
  const typeMap = {
    pending: 'info',
    running: 'warning',
    cancelling: 'warning',
    interrupted: 'warning',
    completed: 'success',
    failed: 'danger',
    cancelled: 'info'
  }
  return typeMap[status] || 'info'
}

const getTaskStatusText = (status) => {
  const textMap = {
    pending: '等待中',
    running: '运行中',
    cancelling: '取消中',
    interrupted: '已中断',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消'
  }
  return textMap[status] || '未知状态'
}

const openModelTest = async (row) => {
  try {
    const status = await getTrainingTaskStatus(row.id)
    syncTrainingRecord(row.id, status)
    if (status.status !== 'completed' || !status.test_model?.available) {
      ElMessage.warning(status.test_model?.reason || '训练模型当前不可用')
      return
    }
    modelTestTask.value = { ...row, status: status.status, testModel: status.test_model }
    modelTestVersion.value += 1
    modelTestBusy.value = false
    showModelTestDialog.value = true
  } catch (error) {
    ElMessage.error(error?.response?.data?.detail || error?.message || '训练模型状态复核失败')
  }
}

const closeModelTest = (done) => {
  if (modelTestBusy.value) {
    ElMessage.warning('推理处理中，请等待完成后再关闭')
    return
  }
  modelTestTask.value = null
  done()
}

const getProgressStatus = (taskStatus) => {
  if (taskStatus === 'failed') return 'exception'
  if (taskStatus === 'completed') return 'success'
  return undefined
}

const formatFileSize = (bytes) => {
  if (!bytes || bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`
}

const formatDate = (date) => {
  if (!date) return '-'
  const parsed = new Date(date)
  if (Number.isNaN(parsed.getTime())) return '-'
  return parsed.toLocaleString('zh-CN', { hour12: false })
}

const formatLearningRate = (value) => {
  const number = Number(value)
  if (Number.isNaN(number)) return value || '-'
  return number.toFixed(4)
}

const getCurrentTrainer = () => {
  try {
    const raw = localStorage.getItem('user')
    const user = raw ? JSON.parse(raw) : null
    return user?.username || (user?.role === 'admin' ? '超级管理员' : '') || '管理员'
  } catch {
    return '管理员'
  }
}

const getModelTypeText = (type) => {
  const typeMap = {
    pretrained: '预训练模型',
    uploaded: '上传模型',
    trained: '训练产物'
  }
  return typeMap[type] || type || '未知类型'
}

const getModelOptionLabel = (model) => {
  if (!model) return ''

  const parts = [model.displayName || model.name || model.path, getModelTypeText(model.type)]

  if (model.type === 'trained') {
    if (model.status === 'candidate') parts.push('候选未激活')
    if (model.isLatest) parts.push('最新')
    if (model.dataset && model.dataset !== '未知数据集') parts.push(`数据集: ${model.dataset}`)
    if (model.epochs) parts.push(String(model.epochs).replace(/^\s+/, ''))
  }

  if (model.size) parts.push(model.size)
  return parts.filter(Boolean).join(' | ')
}

const clampPercent = (value) => {
  const number = Number(value)
  if (Number.isNaN(number)) return 0
  return Math.max(0, Math.min(100, Math.round(number)))
}

const formatGB = (value) => {
  const number = Number(value)
  if (Number.isNaN(number)) return '0 GB'
  return `${number.toFixed(2)} GB`
}

const formatMB = (value) => {
  const number = Number(value)
  if (Number.isNaN(number)) return '0 MB'
  return `${Math.round(number)} MB`
}

const refreshSystemResources = async () => {
  try {
    const data = await getSystemResources()
    const gpus = Array.isArray(data.gpus) ? data.gpus : []
    let selectedGpu = null

    if (trainingConfig.device !== 'cpu' && trainingConfig.device !== 'auto') {
      selectedGpu = gpus.find(gpu => String(gpu.index) === String(trainingConfig.device))
    }

    if (!selectedGpu && gpus.length > 0) {
      selectedGpu = gpus[0]
    }

    systemStatus.cpu = clampPercent(data.cpu)
    systemStatus.memory = clampPercent(data.memory)
    systemStatus.memoryText = data.memory_used !== undefined && data.memory_total !== undefined
      ? `${formatGB(data.memory_used)} / ${formatGB(data.memory_total)}`
      : ''

    if (selectedGpu) {
      systemStatus.gpu = clampPercent(selectedGpu.utilization)
      systemStatus.gpuName = selectedGpu.name || `GPU ${selectedGpu.index}`
      systemStatus.gpuMemoryText = selectedGpu.memory_used !== undefined && selectedGpu.memory_total !== undefined
        ? `${formatMB(selectedGpu.memory_used)} / ${formatMB(selectedGpu.memory_total)}`
        : ''
      systemStatus.gpuTemperature = selectedGpu.temperature ?? null
    } else {
      systemStatus.gpu = clampPercent(data.gpu || 0)
      systemStatus.gpuName = data.gpu_name || '未检测到GPU'
      systemStatus.gpuMemoryText = ''
      systemStatus.gpuTemperature = null
    }

    systemStatus.gpuStatus = data.gpu_status || 'unavailable'
    systemStatus.gpuReason = data.gpu_reason || 'cuda_unavailable'
    systemStatus.gpuError = data.gpu_error || ''

    systemStatus.error = ''
  } catch (error) {
    console.error('刷新系统资源失败:', error)
    systemStatus.error = '系统资源读取失败，请检查后端 /system/resources 接口'
    systemStatus.gpuStatus = 'unknown'
    systemStatus.gpuReason = 'resource_request_failed'
    systemStatus.gpuError = error?.message || ''
  }
}

const pollSystemResources = async () => {
  if (!document.hidden) {
    await refreshSystemResources()
  }
  resourceTimer.value = window.setTimeout(pollSystemResources, document.hidden ? 10000 : 3000)
}

const loadDatasetList = async (forceRefresh = false) => {
  isLoadingDatasets.value = true

  try {
    const response = await getTrainingDatasets({
      page: 1,
      page_size: 100,
      t: forceRefresh ? Date.now() : undefined
    })
    datasetList.value = response.datasets || []
    return datasetList.value
  } catch (error) {
    console.error('加载数据集列表失败:', error)
    ElMessage.error('加载数据集列表失败')
    return []
  } finally {
    isLoadingDatasets.value = false
  }
}

const handleDatasetFromRoute = () => {
  const datasetId = String(route.query.datasetId || '').trim()
  const datasetPath = String(route.query.datasetPath || '').trim()
  if (!datasetId && !datasetPath) return

  const dataset = datasetList.value.find(item => String(item.id) === datasetId)
  if (dataset) {
    selectedDatasetId.value = dataset.id
    selectedDatasetInfo.value = dataset
    trainingConfig.datasetPath = dataset.path || datasetPath
    trainingConfig.datasetName = dataset.name || dataset.id
    trainingConfig.projectName = `${trainingConfig.datasetName}_training`
    applyDatasetAnnotationBinding(dataset)
  } else {
    trainingConfig.datasetPath = datasetPath
    trainingConfig.datasetName = datasetId
    trainingConfig.projectName = datasetId ? `${datasetId}_training` : ''
    applyDatasetAnnotationBinding(null)
  }

  showTrainingDialog.value = true
}

onMounted(() => {
  loadDatasetList(true).then(() => {
    refreshTasks()
    handleDatasetFromRoute()
  })

  getModels().then(response => {
    if (response && response.models) {
      availableModels.value = response.models
        .filter(model => (
          ['pretrained', 'uploaded', 'trained'].includes(model.type)
          && String(model.format || 'pt').toLowerCase() === 'pt'
        ))
        .map(model => ({
          ...model,
          name: model.name,
          path: model.path || model.name,
          size: formatFileSize(model.size || 0),
          type: model.type,
          status: model.status || 'active',
          displayName: model.display_name || model.displayName || model.name,
          dataset: model.dataset || '',
          epochs: model.epochs || '',
          baseModel: model.base_model || '',
          supportsTextPrompts: modelSupportsTextPrompts(model),
          isLatest: Boolean(model.is_latest)
        }))
      if (selectedDatasetInfo.value) {
        applyDatasetAnnotationBinding(selectedDatasetInfo.value)
      }
    }
  }).catch(error => {
    console.error('加载模型列表失败:', error)
    ElMessage.error('加载模型列表失败')
  })

  const savedClasses = localStorage.getItem('annotationClasses')
  if (savedClasses) {
    try {
      const parsedClasses = JSON.parse(savedClasses)
      if (Array.isArray(parsedClasses) && parsedClasses.length > 0) {
        annotationClasses.value = parsedClasses
      }
    } catch (error) {
      console.error('Failed to parse annotation classes', error)
    }
  }

  pollSystemResources()
})

onUnmounted(() => {
  clearTaskDetailTimer()

  if (resourceTimer.value) {
    clearTimeout(resourceTimer.value)
    resourceTimer.value = null
  }

  for (const timer of taskTimers.values()) {
    clearTimeout(timer)
  }
  taskTimers.clear()
})
</script>

<style scoped>
.model-test-button.el-button--primary {
  color: #fff !important;
  background: #1765ad !important;
  border-color: #1765ad !important;
  font-weight: 600;
}
.model-test-button.el-button--primary:hover,
.model-test-button.el-button--primary:focus-visible {
  color: #fff !important;
  background: #12518c !important;
  border-color: #12518c !important;
}
.training-page {
  min-height: 100%;
  padding: 24px 24px 32px;
  background: #fff;
  color: #303133;
}

.filter-panel {
  margin-bottom: 20px;
}

.filter-form {
  display: flex;
  align-items: center;
  gap: 22px;
  flex-wrap: wrap;
}

.filter-form :deep(.el-form-item) {
  margin: 0;
}

.filter-form :deep(.el-form-item__label) {
  color: #303133;
  font-size: 16px;
  font-weight: 700;
}

.filter-input {
  width: 290px;
}

.table-actions {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 14px;
}

.new-training-button {
  min-width: 148px;
  height: 44px;
  padding: 0 18px 0 7px;
  border: 0;
  border-radius: 6px;
  background: #1677ff;
  box-shadow: 0 6px 14px rgba(22, 119, 255, 0.24);
  font-size: 15px;
  font-weight: 600;
  transition: background-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
}

.new-training-button:hover,
.new-training-button:focus-visible {
  background: #0958d9;
  box-shadow: 0 9px 18px rgba(22, 119, 255, 0.3);
  transform: translateY(-1px);
}

.new-training-icon {
  width: 30px;
  height: 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-right: 8px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.2);
}

.new-training-icon :deep(.el-icon) {
  font-size: 18px;
  font-weight: 700;
}

.training-table {
  width: 100%;
  font-size: 15px;
}

.training-table :deep(.el-table__header th) {
  height: 54px;
  background: #f5f7fa;
  color: #303133;
  font-weight: 700;
}

.training-table :deep(.el-table__row td) {
  height: 68px;
}

.training-row-actions {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-height: 28px;
  white-space: nowrap;
}

.training-row-actions :deep(.el-button.is-link) {
  height: auto;
  padding: 2px;
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none;
  font-size: 12px;
}

.training-row-actions :deep(.el-button + .el-button) {
  margin-left: 0;
}

.training-row-actions :deep(.el-button--primary.is-link) {
  color: #1890ff !important;
}

.training-row-actions :deep(.el-button--danger.is-link) {
  color: #e34848 !important;
}

.model-test-dialog :deep(.el-dialog) {
  max-width: 1440px;
  margin-top: 4vh;
}

.model-test-dialog :deep(.el-dialog__body) {
  max-height: 82vh;
  overflow: auto;
  padding: 0 16px 16px;
}

.pagination-wrap {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 16px;
  padding: 24px 0 0;
}

.total-text {
  color: #606266;
  font-size: 14px;
}

.training-config-dialog :deep(.el-dialog) {
  width: min(1280px, calc(100vw - 32px)) !important;
  max-width: 1280px;
  border-radius: 4px;
}

:global(.training-config-dialog .el-dialog__body) {
  max-height: calc(100vh - 180px);
  overflow: auto;
  background: #f4f6f9;
  padding: 18px;
}

.dialog-layout {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  align-items: stretch;
}

.config-column {
  grid-column: 1 / -1;
  min-width: 0;
}

.config-card,
.status-card,
.resource-card {
  border: 1px solid #ebeef5;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-weight: 700;
}

.training-form {
  padding: 10px 0 0;
}

.detail-training-form :deep(.el-form-item) {
  margin-bottom: 18px;
}

.detail-value {
  display: block;
  min-height: 32px;
  width: 100%;
  overflow-wrap: anywhere;
  color: #303133;
  line-height: 1.6;
}

.detail-mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
}

.side-panels {
  display: contents;
}

.training-tasks {
  max-height: 360px;
  overflow-y: auto;
}

.task-item {
  border: 1px solid #e4e7ed;
  border-radius: 6px;
  padding: 14px;
  margin-bottom: 10px;
  background: white;
}

.task-header,
.resource-label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.task-name {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 700;
}

.task-info {
  margin: 10px 0;
  color: #606266;
}

.task-actions {
  text-align: right;
}

.empty-tasks {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 250px;
}

.empty-tasks :deep(.el-empty) {
  padding: 16px 0;
}

.resource-alert {
  margin-bottom: 12px;
}

.resource-item {
  margin-bottom: 16px;
}

.resource-label {
  color: #606266;
  font-size: 14px;
}

.resource-extra {
  margin-top: 8px;
  color: #909399;
  font-size: 12px;
  line-height: 1.6;
}

.selected-info {
  margin-top: 14px;
  padding: 12px;
  border: 1px solid #bfdbfe;
  border-radius: 6px;
  background: #f0f9ff;
  color: #606266;
}

.selected-info strong {
  color: #303133;
  margin-right: 8px;
}

:deep(.selected-row) {
  background-color: #ecf5ff !important;
}

.model-option {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
}

.model-option-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-option-type {
  flex-shrink: 0;
  color: #909399;
  font-size: 12px;
}

.model-binding-hint {
  margin-top: 6px;
  color: #909399;
  font-size: 12px;
  line-height: 1.4;
}

.model-binding-fixed-hint {
  color: #e6a23c;
}

.training-log {
  margin-top: 20px;
}

.training-log h4 {
  margin: 0 0 10px;
}

.status-training-log {
  margin-top: 14px;
}

.preflight-card {
  margin-bottom: 16px;
}

.preflight-summary,
.major-metric-summary {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.preflight-card :deep(.el-alert) {
  margin-top: 8px;
}

.major-metric-summary > div {
  display: grid;
  gap: 4px;
}

.major-metric-summary strong {
  font-size: 20px;
  color: #409eff;
}

.major-class-row {
  display: flex;
  justify-content: space-between;
  padding: 6px 0;
  border-top: 1px solid #ebeef5;
}

.epoch-meta,
.epoch-metrics {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 10px;
}

.epoch-metrics > div {
  display: grid;
  min-width: 130px;
  gap: 2px;
}

.log-content {
  padding: 10px;
  border-radius: 4px;
  background: #f5f7fa;
  font-family: monospace;
  font-size: 12px;
}

.log-line {
  margin-bottom: 5px;
  color: #606266;
}

.log-empty {
  color: #909399;
  text-align: center;
}

@media (max-width: 1180px) {
  .dialog-layout {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .config-column {
    grid-column: 1 / -1;
  }
}

@media (max-width: 900px) {
  .training-page {
    padding: 16px;
  }

  .filter-form {
    display: block;
  }

  .filter-form :deep(.el-form-item) {
    display: block;
    margin-bottom: 12px;
  }

  .filter-input {
    width: 100%;
  }

  .table-actions {
    justify-content: stretch;
  }

  .new-training-button {
    width: 100%;
  }

  .pagination-wrap {
    justify-content: flex-start;
    flex-wrap: wrap;
  }

  .side-panels {
    display: contents;
  }

  .dialog-layout {
    grid-template-columns: 1fr;
  }

  .config-column {
    grid-column: auto;
  }
}
</style>

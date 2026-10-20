<!-- 
  数据标注视图
  核心功能组件，支持图片上传、矩形框标注、类别管理及标注结果保存
-->
<template>
  <div class="annotation-page">
    <!-- 固定显示的一键批量AI标注按钮：用于避免工具栏过窄、换行、遮挡导致按钮看不见 -->
    <!--
    <el-button
      class="floating-ai-batch-button"
      type="success"
      size="large"
      @click="openAIAssistDialog('batch')"
      :disabled="!selectedImagePath || isBatchAIAssisting"
      :loading="isBatchAIAssisting"
    >
      <el-icon><MagicStick /></el-icon>
      一键AI标注全部
    </el-button>
    -->
    <!-- 工具栏 -->
    <div class="toolbar">
      <div class="toolbar-left">
        <!--<el-upload
          ref="uploadRef"
          :show-file-list="false"
          :before-upload="handleImageUpload"
          accept="image/*"
          multiple
        >
          <el-button type="primary">
            <el-icon><Upload /></el-icon>
            上传图片
          </el-button>
        </el-upload>-->
        
        <div class="dataset-context-group">
          <div class="dataset-name-display" :title="currentDatasetName || '未指定数据集'">
            数据集：{{ currentDatasetName || '未指定数据集' }}
          </div>
          <el-button
            @click="loadImagesFromSelectedPath"
            :loading="loadingPathImages"
            :disabled="!currentDatasetId"
          >
            <el-icon><RefreshRight /></el-icon>
            刷新图片
          </el-button>
        </div>

        
        <!--<el-upload
          ref="videoUploadRef"
          :show-file-list="false"
          :before-upload="handleVideoUpload"
          accept="video/*"
        >
          <el-button type="success">
            <el-icon><VideoCamera /></el-icon>
            上传视频
          </el-button>
        </el-upload>-->
        
        <el-divider direction="vertical" />
        
        <el-button @click="undo" :disabled="!canUndo" title="撤销 (Ctrl+Z)">
          <el-icon><RefreshLeft /></el-icon>
          撤销
        </el-button>
        
        <el-button @click="redo" :disabled="!canRedo" title="重做：删除整个数据集的全部标注 (Ctrl+Y)">
          <el-icon><RefreshRight /></el-icon>
          重做
        </el-button>
        
        <el-divider direction="vertical" />
        
        <el-button @click="openAIAssistDialog('single')" :disabled="!currentImage || isBatchAIAssisting" type="primary">
          <el-icon><MagicStick /></el-icon>
          AI辅助标注
        </el-button>

        <el-button 
          @click="openAIAssistDialog('batch')" 
          :disabled="!currentDatasetId || isBatchAIAssisting"
          :loading="isBatchAIAssisting"
          type="success"
        >
          <el-icon><MagicStick /></el-icon>
          一键AI标注全部
        </el-button>
        
        <el-divider direction="vertical" />
        
        <el-button v-if="false" @click="saveAnnotations" :disabled="!currentImage">
          <el-icon><Download /></el-icon>
          保存标注
        </el-button>
        
        <el-button @click="showDataAugmentationDialog" :disabled="!currentImage" type="warning">
          <el-icon><MagicStick /></el-icon>
          数据增强
        </el-button>
        <el-button @click="openExportDatasetDialog" :disabled="!currentDatasetId || imageList.length === 0" :loading="isExportingDataset">
          <el-icon><Files /></el-icon>
          导出数据集
        </el-button>
      </div>
      
      <div class="toolbar-center">
        <el-select v-model="currentTool" placeholder="选择工具" style="width: 140px">
          <el-option label="选择" value="select" />
          <el-option v-if="!isPolygonAnnotationMode" label="矩形框" value="rect" />
          <el-option v-if="!isRectAnnotationMode" label="多边形" value="polygon" />
          <el-option label="删除" value="delete" />
        </el-select>
        <el-tag v-if="selectedAnnotationModel" size="small" :type="isPolygonAnnotationMode ? 'success' : 'primary'">
          {{ selectedAnnotationModel }} · {{ isPolygonAnnotationMode ? '多边形' : '矩形框' }}
        </el-tag>
        
        <el-select v-model="currentClass" placeholder="选择类别" style="width: 150px; margin-left: 10px">
          <el-option 
            v-for="{ name: cls, index } in activeClassOptions"
            :key="index"
            :label="`${index}: ${getClassDisplayName(cls)}`"
            :value="index" 
          />
        </el-select>
        
        <el-button @click="showClassDialog = true" type="text">
          <el-icon><Setting /></el-icon>
          编辑类别
        </el-button>
      </div>
      
      <div class="toolbar-right">
        <span v-if="annotationDraftSaving" class="annotation-draft-saving">正在保存临时标注…</span>
        <span v-else-if="annotationDraftPending" class="annotation-draft-saving">有标注待保存</span>
        <span v-else-if="annotationDraftLastSavedAt && !annotationDraftWarning" class="annotation-draft-saved">已临时保存</span>
        <div v-if="isBatchAIAssisting" class="ai-batch-progress">
          <span>后台AI标注 {{ aiBatchProgress.current }} / {{ aiBatchProgress.total }}{{ aiBatchStatusMessage ? ` · ${aiBatchStatusMessage}` : '' }}</span>
          <el-progress 
            :percentage="aiBatchPercentage" 
            :stroke-width="8" 
            :show-text="false" 
            style="width: 140px"
          />
          <el-button size="small" type="danger" :disabled="aiBatchCancelRequested" @click="cancelAIAssistAll">
            {{ aiBatchCancelRequested ? '正在取消…' : '取消标注' }}
          </el-button>
        </div>
        <div v-else-if="isExportingDataset" class="ai-batch-progress">
          <span>{{ exportDatasetProgress.message || '后台导出训练数据集' }}</span>
          <el-progress
            :percentage="exportDatasetProgress.percentage"
            :stroke-width="8"
            :show-text="false"
            style="width: 140px"
          />
        </div>
        <span v-else-if="currentImage">
          {{ (imagePage - 1) * imagePageSize + currentImageIndex + 1 }} / {{ imageTotal || imageList.length }}
        </span>
      </div>
    </div>
    <el-alert
      v-if="annotationDraftWarning"
      class="annotation-draft-warning"
      :title="annotationDraftWarning"
      type="warning"
      :closable="false"
      show-icon
    />
    
    <!-- 主要内容区域 -->
    <div class="main-content">
      <!-- 左侧图片列表 -->
      <div class="sidebar">
        <div class="sidebar-header">
          <h3>图片列表 ({{ imageList.length }} / {{ imageTotal || imageList.length }})</h3>
          <div class="image-source">来源：{{ currentDatasetPath || currentDatasetName || '未选择数据集' }}</div>
        </div>
        <div class="image-list" v-loading="loadingPathImages">
          <el-empty
            v-if="!loadingPathImages && imageList.length === 0"
            :description="currentDatasetId ? '当前数据集暂无图片' : '请从数据集页面进入标注'"
          />
          <div 
            v-for="(image, index) in imageList" 
            :key="image.relative_path || image.id || image.filename || index"
            class="image-item"
            :class="{ active: currentImageIndex === index }"
            @click="selectImage(index)"
          >
            <img :src="getImageDisplayUrl(image)" :alt="image.name" />
            <div class="image-info">
              <div class="image-name" :title="image.name">{{ image.name }}</div>
              <div class="annotation-count">
                标注: {{ image.annotations?.length || 0 }}
              </div>
            </div>
            <el-button 
              class="delete-image-btn"
              type="danger" 
              size="small"
              circle
              @click.stop="deleteImage(index)"
            >
              <el-icon><Close /></el-icon>
            </el-button>
          </div>

          <div v-if="imageTotal > imagePageSize" class="image-list-pagination">
            <el-pagination
              small
              background
              layout="prev, pager, next"
              :current-page="imagePage"
              :page-size="imagePageSize"
              :total="imageTotal"
              :pager-count="5"
              @current-change="handleImagePageChange"
            />
          </div>
        </div>
      </div>
      
      <!-- 中间画布区域 -->
      <div class="canvas-area">
        <div v-if="currentImage" class="canvas-controls" aria-label="画布缩放控制">
          <el-tooltip content="放大" placement="bottom">
            <el-button circle :disabled="!canZoomIn" aria-label="放大图片" @click="zoomIn">
              <el-icon><ZoomIn /></el-icon>
            </el-button>
          </el-tooltip>
          <el-tooltip content="缩小" placement="bottom">
            <el-button circle :disabled="!canZoomOut" aria-label="缩小图片" @click="zoomOut">
              <el-icon><ZoomOut /></el-icon>
            </el-button>
          </el-tooltip>
          <span class="zoom-level" aria-live="polite">{{ zoomLabel }}</span>
          <el-tooltip content="重置为实际尺寸 (100%)" placement="bottom">
            <el-button circle :disabled="!currentImage" aria-label="重置为实际尺寸" @click="resetZoom">
              <el-icon><Refresh /></el-icon>
            </el-button>
          </el-tooltip>
        </div>
        <div class="canvas-container" v-if="currentImage">
          <!-- 多边形绘制提示 -->
          <div v-if="currentTool === 'polygon' && currentImage" class="polygon-tip">
            <el-alert 
              v-if="!isDrawingPolygon"
              title="点击画布开始绘制多边形"
              type="info"
              :closable="false"
              show-icon
            />
            <el-alert 
              v-else
              :title="`已标记 ${polygonPoints.length} 个点，点击起始点完成绘制（至少需要3个点）`"
              type="warning"
              :closable="false"
              show-icon
            >
              <template #default>
                <el-button size="small" type="danger" @click="cancelPolygon">取消 (Esc / 右键)</el-button>
              </template>
            </el-alert>
          </div>
          
          <canvas 
            ref="canvasRef"
            :width="canvasWidth"
            :height="canvasHeight"
            @mousedown="handleMouseDown"
            @mousemove="handleMouseMove"
            @mouseup="handleMouseUp"
            @contextmenu="handleRightClick"
            @wheel="handleWheel"
          ></canvas>
        </div>
        <div v-else class="empty-canvas">
          <el-empty :description="currentDatasetId ? '请选择左侧图片开始标注' : '请从数据集页面进入标注'" />
        </div>
      </div>
      
      <!-- 右侧属性面板 -->
      <div class="properties-panel">
        <div class="panel-header">
          <h3>标注列表</h3>
        </div>
        <div v-if="selectedAnnotationData" class="selected-annotation-editor" @click.stop>
          <div class="selected-annotation-title">
            <span>已选标注 #{{ selectedAnnotation + 1 }}</span>
            <el-tag size="small" :type="selectedAnnotationData.type === 'polygon' ? 'success' : 'info'">
              {{ selectedAnnotationData.type === 'polygon' ? '多边形' : '矩形' }}
            </el-tag>
          </div>
          <el-select
            :model-value="selectedAnnotationClass"
            class="selected-annotation-class"
            size="small"
            aria-label="修改选中标注类别"
            @update:model-value="handleSelectedAnnotationClassChange"
          >
            <el-option
              v-for="{ name: cls, index } in activeClassOptions"
              :key="index"
              :label="`${index}: ${getClassDisplayName(cls)}`"
              :value="index"
            />
          </el-select>
          <div v-if="selectedAnnotationModelOrigin" class="annotation-model-origin">
            AI原始：{{ selectedAnnotationModelOrigin }}
          </div>
        </div>
        <div class="annotations-list">
          <div 
            v-for="(annotation, index) in currentAnnotations" 
            :key="index"
            class="annotation-item"
            :class="{ selected: selectedAnnotation === index }"
            @click="selectAnnotation(index)"
          >
            <div class="annotation-header">
              <span class="class-label">
                {{ getClassDisplayName(classList[annotation.class]) }}
                <el-tag v-if="annotation.type === 'polygon'" size="small" type="success" style="margin-left: 5px;">多边形</el-tag>
                <el-tag v-else size="small" type="info" style="margin-left: 5px;">矩形</el-tag>
              </span>
              <el-button 
                type="danger" 
                size="small" 
                text
                @click.stop="deleteAnnotation(index)"
              >
                <el-icon><Delete /></el-icon>
              </el-button>
            </div>
            <div class="annotation-coords">
              <small v-if="annotation.type === 'polygon'">
                顶点数: {{ annotation.points?.length || 0 }}
              </small>
              <small v-else>
                x: {{ annotation.x.toFixed(3) }}, 
                y: {{ annotation.y.toFixed(3) }}, 
                w: {{ annotation.width.toFixed(3) }}, 
                h: {{ annotation.height.toFixed(3) }}
              </small>
            </div>
          </div>
        </div>
        
        <!-- 快捷键说明 -->
        <div class="shortcuts-help">
          <h4>快捷键</h4>
          <div class="shortcut-item">
            <kbd>A</kbd> <span>上一张</span>
          </div>
          <div class="shortcut-item">
            <kbd>D</kbd> <span>下一张</span>
          </div>
          <div class="shortcut-item">
            <kbd>Del</kbd> <span>删除选中</span>
          </div>
          <div class="shortcut-item">
            <kbd>1-9</kbd> <span>选择类别</span>
          </div>
          <div class="shortcut-item">
            <kbd>Esc</kbd> <span>取消绘制</span>
          </div>
          <div class="shortcut-item">
            <kbd>右键</kbd> <span>取消绘制</span>
          </div>
        </div>
        
        <!-- 多边形工具说明 -->
        <div v-if="currentTool === 'polygon'" class="tool-help">
          <h4>多边形工具使用说明</h4>
          <ul>
            <li>点击画布添加顶点</li>
            <li>点击起始点完成绘制</li>
            <li>右键或按Esc取消绘制</li>
            <li>至少需要3个顶点</li>
            <li>可拖动整个多边形移动</li>
          </ul>
        </div>
      </div>
    </div>
    
    <!-- 类别编辑对话框 -->
    <el-dialog v-model="showClassDialog" title="编辑类别" width="620px">
      <div class="class-editor">
        <el-form :model="classForm" label-width="80px">
          <el-form-item label="类别列表">
            <div class="class-list">
              <div 
                v-for="(cls, index) in classForm.classes" 
                :key="index"
                class="class-item"
              >
                <el-input v-model="classForm.classes[index]" placeholder="英文类别（训练使用）" />
                <el-input v-model="classForm.labelsZh[index]" placeholder="中文名称（前端显示）" />
                <el-button 
                  type="danger" 
                  text
                  @click="removeClass(index)"
                  :disabled="classForm.classes.length <= 1"
                >
                  <el-icon><Delete /></el-icon>
                </el-button>
              </div>
            </div>
            <el-button class="add-class-button" @click="addClass" type="primary" text>
              <el-icon><Plus /></el-icon>
              添加类别
            </el-button>
          </el-form-item>
        </el-form>
      </div>
      <template #footer>
        <el-button @click="showClassDialog = false">取消</el-button>
        <el-button type="primary" @click="saveClasses">保存</el-button>
      </template>
    </el-dialog>

    <!-- 数据增强对话框 -->
    <el-dialog
      v-model="showAugmentationDialog"
      title="数据增强"
      width="600px"
      center
    >
      <div class="augmentation-config">
        <el-form :model="augmentationConfig" label-width="120px">
          <el-form-item label="旋转角度">
            <el-checkbox-group v-model="augmentationConfig.rotations" class="augmentation-option-group">
              <el-checkbox v-for="angle in ROTATION_OPTIONS" :key="angle" :label="angle">
                {{ angle }}°
              </el-checkbox>
            </el-checkbox-group>
            <div class="augmentation-help">可多选；每个角度固定生成 1 张</div>
          </el-form-item>

          <el-form-item label="镜像">
            <el-radio-group v-model="augmentationConfig.mirror" class="augmentation-option-group">
              <el-radio v-for="option in MIRROR_OPTIONS" :key="option.value" :label="option.value">
                {{ option.label }}
              </el-radio>
            </el-radio-group>
            <div class="augmentation-help">三选一；不镜像不额外生成图片</div>
          </el-form-item>

          <el-form-item label="亮度">
            <el-checkbox-group v-model="augmentationConfig.brightnessLevels" class="augmentation-option-group">
              <el-checkbox v-for="option in BRIGHTNESS_OPTIONS" :key="option.value" :label="option.value">
                {{ option.label }}
              </el-checkbox>
            </el-checkbox-group>
            <div class="augmentation-help">亮度倍率；每个档位固定生成 1 张</div>
          </el-form-item>

          <el-form-item label="灰度强度">
            <el-checkbox-group v-model="augmentationConfig.grayscaleLevels" class="augmentation-option-group">
              <el-checkbox v-for="option in GRAYSCALE_OPTIONS" :key="option.value" :label="option.value">
                {{ option.label }}
              </el-checkbox>
            </el-checkbox-group>
            <div class="augmentation-help">灰度化强度；30% 为轻度，100% 为完全灰度</div>
          </el-form-item>

          <el-form-item label="对比度">
            <el-checkbox-group v-model="augmentationConfig.contrastLevels" class="augmentation-option-group">
              <el-checkbox v-for="option in CONTRAST_OPTIONS" :key="option.value" :label="option.value">
                {{ option.label }}
              </el-checkbox>
            </el-checkbox-group>
            <div class="augmentation-help">对比度倍率：60% 低、100% 标准、140% 高</div>
          </el-form-item>

          <el-alert
            :title="augmentationOperationCount > 0
              ? `本次将生成 ${augmentationOperationCount} 张增强图片，原始图片始终保留`
              : '请选择至少一种增强操作'"
            :type="augmentationOperationCount > 0 ? 'info' : 'warning'"
            :closable="false"
            show-icon
          />
        </el-form>

        <div class="augmentation-preview" v-if="augmentationPreview.length > 0">
          <h4>预览效果</h4>
          <div class="preview-grid">
            <div v-for="(preview, index) in augmentationPreview" :key="index" class="preview-item">
              <img :src="preview" :alt="`预览${index + 1}`" />
            </div>
          </div>
        </div>
      </div>

      <template #footer>
        <el-button @click="generatePreview" :disabled="augmentationOperationCount === 0" :loading="isGeneratingPreview">
          生成预览
        </el-button>
        <el-button @click="showAugmentationDialog = false">取消</el-button>
        <el-button type="primary" @click="applyDataAugmentation" :disabled="augmentationOperationCount === 0" :loading="isApplyingAugmentation">
          应用增强
        </el-button>
      </template>
    </el-dialog>

    <!-- 训练数据集导出对话框 -->
    <el-dialog
      v-model="showExportDatasetDialog"
      title="导出训练数据集"
      width="520px"
      :close-on-click-modal="!isCheckingExport"
      :close-on-press-escape="!isCheckingExport"
    >
      <el-form :model="exportDatasetConfig" label-width="140px">
        <el-form-item label="训练数据集名称" required>
          <el-input
            v-model="exportDatasetConfig.name"
            maxlength="100"
            show-word-limit
            placeholder="请输入训练数据集名称"
          />
        </el-form-item>
        <el-alert
          title="未标注图片将在确认后从训练数据集中剔除"
          type="info"
          :closable="false"
          show-icon
        />
        <el-alert
          v-if="exportDatasetPreflight"
          :title="exportPreflightTitle"
          :description="exportPreflightDescription"
          :type="exportDatasetPreflight.can_export === false ? 'error' : 'info'"
          :closable="false"
          show-icon
        />
        <div v-if="exportDatasetPreflight?.quality" class="export-quality-summary">
          <div class="export-quality-metrics">
            <span>原图 {{ exportDatasetPreflight.quality.source_image_count || 0 }}</span>
            <span>有标注原图 {{ exportDatasetPreflight.quality.annotated_source_image_count || 0 }}</span>
            <span>切片 {{ exportDatasetPreflight.quality.tile_count || 0 }}</span>
            <span>来源组 {{ exportDatasetPreflight.quality.annotated_source_group_count || 0 }}</span>
            <span>train/val/test 来源组 {{ exportDatasetPreflight.quality.split_source_group_counts?.train || 0 }}/{{ exportDatasetPreflight.quality.split_source_group_counts?.val || 0 }}/{{ exportDatasetPreflight.quality.split_source_group_counts?.test || 0 }}</span>
            <span>目标尺寸 {{ exportDatasetPreflight.quality.target_size || '-' }} px</span>
            <span>手机框最小边 P05 {{ exportDatasetPreflight.quality.mobilephone_output_short_side_pixels?.p05 ?? '-' }} px</span>
          </div>
          <div v-if="exportDatasetPreflight.quality.mobilephone_sample_previews?.length" class="export-quality-samples">
            <span>抽样预览</span>
            <el-link
              v-for="sample in exportDatasetPreflight.quality.mobilephone_sample_previews.slice(0, 8)"
              :key="`${sample.asset_id}-${sample.relative_path}`"
              href="#"
              type="primary"
              @click.prevent="openExportSamplePreview(sample)"
            >{{ sample.relative_path }}</el-link>
          </div>
        </div>
      </el-form>
      <template #footer>
        <el-button @click="showExportDatasetDialog = false" :disabled="isCheckingExport">
          取消
        </el-button>
        <el-button
          type="primary"
          @click="exportDataset"
          :loading="isCheckingExport || isExportingDataset"
        >
          导出
        </el-button>
      </template>
    </el-dialog>

    <!-- 视频帧提取对话框 -->
    <el-dialog
      v-model="showVideoFrameDialog"
      title="视频帧提取"
      width="600px"
      center
    >
      <div class="video-frame-config">
        <el-form :model="videoFrameConfig" label-width="120px">
          <el-form-item label="视频信息">
            <div class="video-info-display">
              <p><strong>文件名:</strong> {{ currentVideoInfo?.filename }}</p>
              <p><strong>时长:</strong> {{ formatDuration(currentVideoInfo?.duration) }}</p>
              <p><strong>帧率:</strong> {{ currentVideoInfo?.fps }} fps</p>
              <p><strong>总帧数:</strong> {{ currentVideoInfo?.frame_count }}</p>
              <p><strong>分辨率:</strong> {{ currentVideoInfo?.width }}x{{ currentVideoInfo?.height }}</p>
            </div>
          </el-form-item>
          
          <el-form-item label="提取间隔">
            <el-radio-group v-model="videoFrameConfig.intervalType">
              <el-radio label="fps">按帧率提取</el-radio>
              <el-radio label="count">按数量提取</el-radio>
            </el-radio-group>
          </el-form-item>
          
          <el-form-item v-if="videoFrameConfig.intervalType === 'fps'" label="每秒提取帧数">
            <el-input-number 
              v-model="videoFrameConfig.frameInterval" 
              :min="0.1" 
              :max="30" 
              :step="0.5"
              controls-position="right"
            />
            <span style="margin-left: 10px; color: #909399;">
              每秒提取 {{ videoFrameConfig.frameInterval }} 帧
            </span>
          </el-form-item>
          
          <el-form-item v-else label="总共提取帧数">
            <el-input-number 
              v-model="videoFrameConfig.maxFrames" 
              :min="1" 
              :max="500" 
              :step="10"
              controls-position="right"
            />
            <span style="margin-left: 10px; color: #909399;">
              从视频中均匀提取 {{ videoFrameConfig.maxFrames }} 帧
            </span>
          </el-form-item>
          
          <el-form-item>
            <el-alert
              type="info"
              :closable="false"
              show-icon
            >
              <template #default>
                <div>
                  预计提取约 <strong>{{ estimatedFrameCount }}</strong> 帧
                </div>
              </template>
            </el-alert>
          </el-form-item>
        </el-form>
      </div>
      
      <template #footer>
        <el-button @click="showVideoFrameDialog = false">取消</el-button>
        <el-button type="primary" @click="extractFrames" :loading="isExtractingFrames">
          开始提取
        </el-button>
      </template>
    </el-dialog>

    <!-- AI辅助标注对话框 -->
    <el-dialog
      v-model="showAIAssistDialog"
      :title="aiAssistRunMode === 'batch' ? '一键AI标注全部' : 'AI辅助预标注'"
      width="500px"
      center
    >
      <div class="ai-assist-config">
        <el-form :model="aiAssistConfig" label-width="120px">
          <el-form-item label="检测模型">
            <el-select
              v-model="aiAssistConfig.model"
              placeholder="选择模型"
              :disabled="Boolean(annotationSessionModel)"
              @change="handleAIAssistModelChange"
            >
              <el-option 
                v-for="model in availableModels" 
                :key="model.path" 
                :label="`${model.display_name || model.name} ${model.type ? `[${model.type}]` : ''}`" 
                :value="model.path" 
              />
            </el-select>
            <span v-if="annotationSessionModel" class="ai-model-binding-hint">
              本次标注使用进入页面时选择的模型，AI 一键标注不可切换
            </span>
          </el-form-item>

          <el-form-item v-if="isYOLOEModelSelected" label="YOLOE提示词">
            <el-input
              v-model="aiAssistConfig.promptText"
              type="textarea"
              :rows="3"
              placeholder="留空使用当前类别列表；也可每行输入一个检测类别"
            />
            <span style="font-size: 12px; color: #909399; display: block; margin-top: 5px;">
              当前传入：{{ aiAssistPromptSummary }}
            </span>
          </el-form-item>
          
          <el-form-item label="置信度阈值">
            <el-slider 
              v-model="aiAssistConfig.confidence" 
              :min="0.01" 
              :max="1" 
              :step="0.01"
              show-input
            />
            <span style="font-size: 12px; color: #909399;">
              当前阈值：{{ Number(aiAssistConfig.confidence).toFixed(2) }}。阈值越高，检测越严格，标注框越少。
            </span>
          </el-form-item>
          
          <el-form-item label="类别映射">
            <el-radio-group v-model="aiAssistConfig.classMode">
              <el-radio label="auto">按当前页面类别自动映射</el-radio>
              <el-radio label="manual">手动指定目标类别</el-radio>
            </el-radio-group>
          </el-form-item>
          
          <el-form-item label="目标类别" v-if="aiAssistConfig.classMode === 'manual'">
            <el-select 
              v-model="aiAssistConfig.targetClass" 
              placeholder="选择或输入类别"
              filterable
              allow-create
              default-first-option
            >
              <el-option 
                v-for="{ name: cls, index } in activeClassOptions"
                :key="index" 
                :label="getClassDisplayName(cls)"
                :value="index" 
              />
            </el-select>
            <span style="font-size: 12px; color: #909399; display: block; margin-top: 5px;">
              所有检测到的目标将统一标记为此类别
            </span>
          </el-form-item>
          
          <el-form-item label="处理方式">
            <el-radio-group v-model="aiAssistConfig.mode">
              <el-radio label="replace">替换现有标注</el-radio>
              <el-radio label="append">追加到现有标注</el-radio>
            </el-radio-group>
          </el-form-item>
          
          <el-alert
            type="info"
            :closable="false"
            show-icon
            style="margin-bottom: 15px;"
          >
            <template #default>
              <div v-if="aiAssistRunMode === 'batch'">
                <strong>稳定批量模式：</strong>后端会直接扫描 ./data/image 并创建后台 AI 标注任务。
                左侧列表继续分页显示，不需要一次性加载全部图片；当前页面已加载 {{ imageList.length }} / {{ imageTotal || imageList.length }} 张。
              </div>
              <div v-else-if="aiAssistConfig.classMode === 'auto'">
                <strong>自动模式：</strong>检测到的模型类别将自动添加到类别列表
              </div>
              <div v-else>
                <strong>手动模式：</strong>所有检测到的目标将统一标记为指定的类别
              </div>
            </template>
          </el-alert>
        </el-form>
      </div>
      
      <template #footer>
        <el-button
          :disabled="isBatchAIAssisting && aiBatchCancelRequested"
          @click="isBatchAIAssisting ? cancelAIAssistAll() : (showAIAssistDialog = false)"
        >
          {{ isBatchAIAssisting ? (aiBatchCancelRequested ? '正在取消…' : '取消标注') : '取消' }}
        </el-button>
        <el-button
          v-if="aiAssistRunMode === 'single'"
          type="primary"
          @click="runAIAssist"
          :loading="isAIAssisting"
        >
          <el-icon><MagicStick /></el-icon>
          标注当前图片
        </el-button>
        <el-button
          v-else
          type="success"
          @click="runAIAssistAll"
          :loading="isBatchAIAssisting"
        >
          <el-icon><MagicStick /></el-icon>
          按当前阈值标注全部图片
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
/**
 * 数据标注视图逻辑
 * 处理图片/视频上传、Canvas 绘制、标注逻辑、AI 辅助及数据增强
 */
import { ref, reactive, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useRoute, onBeforeRouteLeave } from 'vue-router'
import { Upload, Download, Files, Setting, Delete, Plus, MagicStick, VideoCamera, Close, Refresh, RefreshLeft, RefreshRight, ZoomIn, ZoomOut } from '@element-plus/icons-vue'
import { saveAs } from 'file-saver'
import {
  authorizedFetch,
  preflightAnnotationTrainingDatasetExport,
  startAnnotationTrainingDatasetExport,
  uploadAnnotationVideo,
  extractVideoFrames,
  aiAssistAnnotation,
  getDatasets,
  getDatasetAnnotationClasses,
  updateDatasetAnnotationClasses,
  saveDatasetAnnotationModel,
  deleteAnnotationDraft,
  getAnnotationDraft,
  putAnnotationDraft,
  getJob,
  startDatasetAugmentation,
  startBatchAnnotation,
  getBatchAnnotationStatus,
  getBatchAnnotationResults,
  cancelBatchAnnotation
} from '../utils/api'
import { getModels, getTrainingDatasets } from '../utils/api'
import { relabelAnnotation, validateAnnotationClassNames } from '../utils/annotationLabeling.mjs'
import { modelSupportsTextPrompts, modelAnnotationGeometry } from '../utils/modelCapabilities.mjs'

import {
  ROTATION_OPTIONS,
  MIRROR_OPTIONS,
  BRIGHTNESS_OPTIONS,
  GRAYSCALE_OPTIONS,
  CONTRAST_OPTIONS,
  buildAugmentationOperations,
  getAugmentationOperationLabel,
  getAugmentedImageDimensions,
  transformAnnotations
} from '../utils/imageAugmentation.mjs'

const route = useRoute()
const DEFAULT_NEW_ANNOTATION_CLASSES = ['object']

// --- UI 状态控制 ---
const canvasRef = ref(null)
const uploadRef = ref(null)
const videoUploadRef = ref(null)
const showClassDialog = ref(false)
const showAugmentationDialog = ref(false)
const showVideoFrameDialog = ref(false)
const showAIAssistDialog = ref(false)
const aiAssistRunMode = ref('single') // single: 当前图片；batch: 全部图片

// --- AI 辅助标注相关配置 ---
const isAIAssisting = ref(false)
const isBatchAIAssisting = ref(false)
const aiBatchCancelRequested = ref(false)
const isCancellingBatch = ref(false)
const aiBatchProgress = reactive({
  current: 0,
  total: 0,
  success: 0,
  failed: 0,
  added: 0
})
const aiBatchTaskId = ref('')
const aiBatchStatusMessage = ref('')
const aiBatchResultCache = reactive(new Map()) // filename -> { annotations, taskId, raw }
const thumbnailObjectUrls = new Set()

const aiBatchPercentage = computed(() => {
  if (!aiBatchProgress.total) return 0
  return Math.round((aiBatchProgress.current / aiBatchProgress.total) * 100)
})
const availableModels = ref([])
const selectedAnnotationModel = ref('')
const annotationSessionModel = ref('')
const annotationGeometry = ref('rect')
const aiAssistConfig = reactive({
  model: 'yolo11n.pt',
  confidence: 0.25,
  mode: 'replace',    // 处理方式: 'replace' 替换或 'append' 追加
  classMode: 'auto',  // 类别模式: 'auto' 自动识别或 'manual' 手动指定
  targetClass: 0,     // 手动模式下的目标类别索引
  promptText: ''      // YOLOE 文本提示词；留空时使用当前类别
})

const selectedAIAssistModel = computed(() => {
  return availableModels.value.find(model => (
    model.path === aiAssistConfig.model ||
    model.name === aiAssistConfig.model ||
    model.display_name === aiAssistConfig.model
  )) || null
})

const isPolygonAnnotationMode = computed(() => annotationGeometry.value === 'polygon')
const isRectAnnotationMode = computed(() => !isPolygonAnnotationMode.value)

const isYOLOEModelSelected = computed(() => {
  const model = selectedAIAssistModel.value || {}
  return modelSupportsTextPrompts(model, aiAssistConfig.model)
})

const splitPromptText = (value) => {
  return String(value || '')
    .split(/[\n,，;；]+/)
    .map(item => item.trim())
    .filter(Boolean)
}

const getManualTargetClassName = () => {
  const target = aiAssistConfig.targetClass
  if (typeof target === 'string') return target.trim()
  return classList.value[target] || ''
}

const getAIAssistPromptClasses = () => {
  if (!isYOLOEModelSelected.value) return []
  const customPrompts = splitPromptText(aiAssistConfig.promptText)
  if (customPrompts.length > 0) return customPrompts

  if (aiAssistConfig.classMode === 'manual') {
    const targetName = getManualTargetClassName()
    return targetName ? [targetName] : []
  }

  return activeClassOptions.value
    .map(item => String(item.name || '').trim())
    .filter(Boolean)
}

const getAIAssistPromptPageIndices = () => {
  const prompts = getAIAssistPromptClasses()
  if (!prompts.length) return []
  if (aiAssistConfig.classMode === 'manual' && Number.isInteger(Number(aiAssistConfig.targetClass))) {
    return prompts.map(() => Number(aiAssistConfig.targetClass))
  }
  return prompts.map(prompt => classList.value.findIndex(
    item => String(item || '').trim().toLocaleLowerCase() === String(prompt || '').trim().toLocaleLowerCase()
  ))
}

const getAIAssistModelMetadata = () => {
  const model = selectedAIAssistModel.value || {}
  return {
    id: model.id || '',
    sha256: model.sha256 || '',
    source: model.source || model.type || '',
    model_family: model.model_family || '',
    supports_text_prompts: model.supports_text_prompts === true,
    base_model: model.base_model || '',
    lineage_root_model: model.lineage_root_model || ''
  }
}

const getAIAssistModelId = () => String(selectedAIAssistModel.value?.id || '').trim()

const aiAssistPromptSummary = computed(() => {
  const prompts = getAIAssistPromptClasses()
  if (prompts.length === 0) return '未设置'
  const visiblePrompts = prompts.slice(0, 8).join('、')
  return prompts.length > 8 ? `${visiblePrompts} 等 ${prompts.length} 个` : visiblePrompts
})

// --- 视频处理相关 ---
const currentVideoInfo = ref(null)
const isExtractingFrames = ref(false)
const videoFrameConfig = reactive({
  intervalType: 'fps', // 提取方式: 'fps' 每秒帧数 或 'count' 总帧数
  frameInterval: 1,    // 帧间隔
  maxFrames: 100       // 最大提取数量
})

// --- 数据增强配置 ---
const augmentationConfig = reactive({
  rotations: [],
  mirror: 'none',
  brightnessLevels: [],
  grayscaleLevels: [],
  contrastLevels: []
})
const augmentationPreview = ref([])
const isGeneratingPreview = ref(false)
const isApplyingAugmentation = ref(false)
const augmentationOperationCount = computed(() => buildAugmentationOperations(augmentationConfig).length)
const showExportDatasetDialog = ref(false)
const exportDatasetConfig = reactive({
  name: '',
  requiredGeometry: 'bbox'
})
const exportDatasetPreflight = ref(null)
const exportPreflightTitle = computed(() => {
  const preflight = exportDatasetPreflight.value
  if (!preflight) return ''
  if (preflight.can_export === false) return '预检阻止导出'
  return `预检：原图 ${preflight.quality?.source_image_count || 0} 张，切片 ${preflight.quality?.tile_count || 0} 张`
})
const exportPreflightDescription = computed(() => {
  const preflight = exportDatasetPreflight.value
  if (!preflight) return ''
  if (preflight.can_export === false) {
    return (preflight.errors || []).map(item => item.message || item.code).join('；')
  }
  return (preflight.quality?.warnings || []).map(item => item.message || item.code).join('；') || '预检通过，可继续导出'
})


const isCheckingExport = ref(false)
const isExportingDataset = ref(false)
const exportDatasetProgress = reactive({
  percentage: 0,
  message: ''
})

// --- 画布与视图控制 ---
const canvasWidth = ref(800)
const canvasHeight = ref(600)
const scale = ref(1)   // 缩放比例
const offsetX = ref(0) // 画布偏移 X
const offsetY = ref(0) // 画布偏移 Y
const minZoom = 0.1
const maxZoom = 4
const zoomStep = 0.2
const zoomLabel = computed(() => `${Math.round(scale.value * 100)}%`)
const canZoomIn = computed(() => Boolean(currentImageObj) && scale.value < maxZoom)
const canZoomOut = computed(() => Boolean(currentImageObj) && scale.value > minZoom)

// --- 标注工具与类别 ---
const currentTool = ref('rect') // 当前工具: rect(矩形), select(选择), polygon(多边形), delete(删除)
const currentClass = ref(0)     // 当前选中的类别索引
const classList = ref([...DEFAULT_NEW_ANNOTATION_CLASSES]) // 当前数据集的类别列表
const inactiveClassIds = ref([]) // 保留训练编号，只在标注界面隐藏停用类别
const activeClassOptions = computed(() => classList.value
  .map((name, index) => ({ name, index }))
  .filter(item => !inactiveClassIds.value.includes(item.index)))
const classLabelsZh = ref({}) // 英文类别到中文展示名；不参与训练
const getClassDisplayName = name => classLabelsZh.value[name] || name || ''
const annotationClassBindings = ref([]) // 页面类别到模型原生类别的稳定映射

// --- 数据集图片读取配置 ---
const currentDatasetId = ref('')
const currentDatasetRole = ref('original')
const currentDatasetName = ref('')
const currentDatasetPath = ref('')
const selectedImagePath = ref('')
const datasetOptions = ref([])
const loadingDatasets = ref(false)
const loadingPathImages = ref(false)
const imagePage = ref(1)
const imagePageSize = ref(100)
const imageTotal = ref(0)

// --- 临时标注草稿（仅原始数据集，按用户和标注类型隔离） ---
const annotationDraftKind = ref(String(route.query.draftKind || '').trim().toLowerCase())
const annotationDraftId = ref(String(route.query.draftId || '').trim())
const annotationClassScope = ref('dataset')
const annotationSessionDatasetId = ref(String(route.query.datasetId || '').trim())
const annotationDraftRevision = ref(Number(route.query.draftRevision) || 0)
const annotationDraftSourceVersionId = ref('')
const annotationDraftTemporaryDatasetId = ref('')
const annotationDraftDirtyPaths = reactive(new Set())
const annotationDraftDirtyVersions = new Map()
const annotationDraftChangeVersion = ref(0)
const annotationDraftMetadataDirty = ref(false)
const annotationDraftClearAll = ref(false)
const annotationDraftSaving = ref(false)
const annotationDraftWarning = ref('')
const annotationDraftConflict = ref(false)
const annotationDraftLastSavedAt = ref('')
const annotationDraftBatchTaskId = ref('')
const annotationDraftResumeBatch = ref(null)
const annotationDraftPending = computed(() => annotationDraftMetadataDirty.value || annotationDraftDirtyPaths.size > 0 || Boolean(annotationDraftBatchTaskId.value))
let annotationDraftClearAllVersion = 0
let annotationDraftSavePromise = null
let annotationDraftDebounceTimer = null
let annotationViewActive = true
let annotationDraftTimer = null
let annotationDraftVisibilityHandler = null
let annotationDraftPagehideHandler = null
let annotationDraftLogoutHandler = null

// --- 图片管理 ---
const imageList = ref([]) // 当前路径下的图片数据
const currentImageIndex = ref(-1)
const currentImage = computed(() => 
  currentImageIndex.value >= 0 ? imageList.value[currentImageIndex.value] : null
)
const annotationCache = new Map()
const datasetClearAllActive = ref(false)
const datasetHasAnnotations = ref(false)
const datasetClearAllRestoredKeys = reactive(new Set())
const datasetClearAllBaseline = new Map()

const cloneAnnotations = (annotations) => JSON.parse(JSON.stringify(annotations || []))

const getImageAnnotationCacheKey = (image) => String(
  image?.relative_path || image?.filename || image?.name || ''
).replace(/\\/g, '/')

const getCachedAnnotationEntry = (image) => {
  const keys = [
    image?.relative_path,
    image?.filename,
    image?.name
  ].filter(Boolean).map(key => String(key).replace(/\\/g, '/'))

  return keys.map(key => annotationCache.get(key)).find(Boolean) || null
}

const cacheImageAnnotations = (image) => {
  const key = getImageAnnotationCacheKey(image)
  if (!key) return

  if (Array.isArray(image.annotations) && image.annotations.length > 0) {
    datasetHasAnnotations.value = true
  }

  annotationCache.set(key, {
    relative_path: image.relative_path || '',
    width: Number(image.width) || 0,
    height: Number(image.height) || 0,
    annotations: cloneAnnotations(image.annotations),
    aiBatchTaskId: image._aiBatchTaskApplied || ''
  })
}

const cacheCurrentPageAnnotations = () => {
  imageList.value.forEach(cacheImageAnnotations)
}

// --- 标注交互状态 ---
const isDrawing = ref(false)
const startPoint = ref({ x: 0, y: 0 })
const selectedAnnotation = ref(-1)
const currentAnnotations = computed(() => 
  currentImage.value?.annotations || []
)
const selectedAnnotationData = computed(() => (
  selectedAnnotation.value >= 0 ? currentAnnotations.value[selectedAnnotation.value] || null : null
))
const selectedAnnotationClass = computed(() => {
  const classIndex = Number(selectedAnnotationData.value?.class)
  return Number.isInteger(classIndex) && classIndex >= 0 && classIndex < classList.value.length ? classIndex : 0
})
const selectedAnnotationModelOrigin = computed(() => {
  const annotation = selectedAnnotationData.value
  const modelName = String(annotation?.model_class_name || '').trim()
  if (modelName) return modelName
  const modelClassId = Number(annotation?.model_class_id)
  return Number.isInteger(modelClassId) && modelClassId >= 0 ? `类别 ${modelClassId}` : ''
})

// --- 标注框拖拽与缩放状态 ---
const isDragging = ref(false)
const isResizing = ref(false)
const isMovingPolygonPoint = ref(false)
const dragStartPos = ref({ x: 0, y: 0 })
const resizeHandle = ref('') // 缩放手柄: 'tl', 'tr', 'bl', 'br', 'l', 'r', 't', 'b'
const resizeStartAnnotation = ref(null)
const movingPolygonPointIndex = ref(-1)

// --- 多边形标注状态 ---
const polygonPoints = ref([])      // 当前正在绘制的点集
const isDrawingPolygon = ref(false)
const hoveredPointIndex = ref(-1)  // 悬停点索引

// --- 历史记录 (撤销/重做) ---
const historyByImage = reactive(new Map())
const historyRevision = ref(0)
const maxHistorySize = 50

const getImageHistoryKey = (image = currentImage.value) => getImageAnnotationCacheKey(image)

const ensureImageHistory = (image = currentImage.value) => {
  const key = getImageHistoryKey(image)
  if (!key) return null

  let entry = historyByImage.get(key)
  if (!entry) {
    const baseline = datasetClearAllActive.value && datasetClearAllBaseline.has(key)
      ? cloneAnnotations(datasetClearAllBaseline.get(key))
      : cloneAnnotations(image?.annotations)
    const currentState = cloneAnnotations(image?.annotations)
    const states = JSON.stringify(baseline) === JSON.stringify(currentState)
      ? [currentState]
      : [baseline, currentState]
    entry = { states, index: states.length - 1 }
    historyByImage.set(key, entry)
    historyRevision.value += 1
  }
  return entry
}

/**
 * 保存当前标注状态到历史记录栈
 */
const saveHistory = (image = currentImage.value) => {
  if (!image) return

  const key = getImageHistoryKey(image)
  const imageHistory = ensureImageHistory(image)
  if (!key || !imageHistory) return
  
  // 深拷贝当前标注状态
  const state = cloneAnnotations(image.annotations)
  const currentState = imageHistory.states[imageHistory.index]
  if (currentState && JSON.stringify(currentState) === JSON.stringify(state)) {
    return
  }

  // 移除当前索引之后的所有历史记录
  if (imageHistory.index < imageHistory.states.length - 1) {
    imageHistory.states = imageHistory.states.slice(0, imageHistory.index + 1)
  }
  imageHistory.states.push(state)
  
  // 限制历史记录大小
  if (imageHistory.states.length > maxHistorySize) {
    imageHistory.states.shift()
  }
  imageHistory.index = imageHistory.states.length - 1
  datasetClearAllRestoredKeys.delete(key)
  historyRevision.value += 1
}

// 撤销
const undo = () => {
  const image = currentImage.value
  const imageHistory = ensureImageHistory(image)
  if (!image || !imageHistory || !canUndo.value) return
  
  imageHistory.index--
  image.annotations = cloneAnnotations(imageHistory.states[imageHistory.index])
  const key = getImageHistoryKey(image)
  if (datasetClearAllActive.value && key && image.annotations.length > 0) {
    datasetClearAllRestoredKeys.add(key)
  }
  cacheImageAnnotations(image)
  annotationDraftClearAll.value = false
  markImageDirty(image)
  historyRevision.value += 1
  drawCanvas()
  ElMessage.success('已撤销')
}

// 重做
const redo = async () => {
  if (!canRedo.value) return

  try {
    await ElMessageBox.confirm(
      `重做将删除数据集“${currentDatasetName.value || currentDatasetId.value}”中的全部标注，确定继续吗？`,
      '重做：删除全部标注',
      {
        confirmButtonText: '删除全部',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    applyDatasetClearAll()
    ElMessage.success('已重做：已删除整个数据集的全部标注')
  } catch (error) {
    if (error !== 'cancel' && error !== 'close') {
      console.error('重做删除全部标注失败:', error)
      ElMessage.error(error?.message || '重做失败')
    }
  }
}

// 判断是否可以撤销/重做
const canUndo = computed(() => {
  historyRevision.value
  const imageHistory = historyByImage.get(getImageHistoryKey(currentImage.value))
  return Boolean(imageHistory && imageHistory.index > 0)
})
const canRedo = computed(() => {
  historyRevision.value
  if (datasetClearAllActive.value && datasetClearAllRestoredKeys.size > 0) return true
  if (datasetClearAllActive.value || !currentDatasetId.value || !imageTotal.value) return false
  return datasetHasAnnotations.value || imageList.value.some(image => Array.isArray(image?.annotations) && image.annotations.length > 0)
})

const applyDatasetClearAll = () => {
  if (!currentDatasetId.value) return

  datasetClearAllActive.value = true
  annotationDraftClearAll.value = true
  annotationDraftClearAllVersion += 1
  markDraftMetadataDirty()
  datasetHasAnnotations.value = false
  annotationCache.forEach((entry, key) => {
    if (!datasetClearAllBaseline.has(key)) {
      datasetClearAllBaseline.set(key, cloneAnnotations(entry.annotations))
    }
    entry.annotations = []
    entry.aiBatchTaskId = ''
  })
  imageList.value.forEach(image => {
    const key = getImageHistoryKey(image)
    if (key && !datasetClearAllBaseline.has(key)) {
      datasetClearAllBaseline.set(key, cloneAnnotations(image.annotations))
    }

    image.annotations = []
    image._aiBatchTaskApplied = ''
    ensureImageHistory(image)
    saveHistory(image)
    cacheImageAnnotations(image)
  })

  datasetClearAllRestoredKeys.clear()
  historyRevision.value += 1
  selectedAnnotation.value = -1
  drawCanvas()
}

// 类别编辑表单
const classForm = reactive({
  classes: [...classList.value],
  labelsZh: classList.value.map(name => classLabelsZh.value[name] || ''),
  ids: classList.value.map((_, index) => index)
})

// 画布上下文
let ctx = null
let currentImageObj = null

const getApiBaseUrl = () => {
  const envBaseUrl = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL
  if (envBaseUrl) return String(envBaseUrl).replace(/\/$/, '')
  return '/api'
}

const pendingRequestControllers = new Set()

const authenticatedFetch = async (url, options = {}) => {
  const controller = new AbortController()
  const timeoutMs = Number(options.timeoutMs || 30000)
  const requestOptions = { ...options }
  delete requestOptions.timeoutMs
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs)
  pendingRequestControllers.add(controller)
  try {
    return await authorizedFetch(url, { ...requestOptions, signal: controller.signal })
  } finally {
    window.clearTimeout(timeout)
    pendingRequestControllers.delete(controller)
  }
}

const buildApiUrl = (path) => {
  const baseUrl = getApiBaseUrl()
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${baseUrl}${normalizedPath}`
}

const draftEndpoint = () => {
  if (!currentDatasetId.value || !annotationDraftKind.value) return ''
  return buildApiUrl(`/datasets/${encodeURIComponent(currentDatasetId.value)}/annotation-drafts/${encodeURIComponent(annotationDraftKind.value)}`)
}

const markImageDirty = (image = currentImage.value) => {
  if (!image || currentDatasetRole.value !== 'original' || !annotationDraftKind.value) return
  const key = getImageAnnotationCacheKey(image)
  if (!key) return
  cacheImageAnnotations(image)
  annotationDraftChangeVersion.value += 1
  annotationDraftDirtyPaths.add(key)
  annotationDraftDirtyVersions.set(key, annotationDraftChangeVersion.value)
  scheduleAnnotationDraftSave()
}

const markDraftMetadataDirty = () => {
  if (currentDatasetRole.value !== 'original' || !annotationDraftKind.value) return
  annotationDraftMetadataDirty.value = true
  annotationDraftChangeVersion.value += 1
  scheduleAnnotationDraftSave()
}

const canSaveAnnotationDraftNow = () => Boolean(
  currentDatasetRole.value === 'original' &&
  currentDatasetId.value &&
  annotationDraftKind.value &&
  !isDrawing.value &&
  !isDrawingPolygon.value &&
  !isDragging.value &&
  !isResizing.value &&
  !isMovingPolygonPoint.value
)

const buildAnnotationDraftPayload = ({ keepalive = false } = {}) => {
  cacheCurrentPageAnnotations()
  const dirtyKeys = [...annotationDraftDirtyPaths]
  const selectedKey = getImageAnnotationCacheKey(currentImage.value)
  const orderedKeys = dirtyKeys.includes(selectedKey)
    ? [selectedKey, ...dirtyKeys.filter(key => key !== selectedKey)] : dirtyKeys
  const payload = {
    draft_id: annotationDraftId.value || undefined,
    annotation_task_id: keepalive ? undefined : annotationDraftBatchTaskId.value || undefined,
    source_version_id: annotationDraftSourceVersionId.value || undefined,
    model_path: annotationSessionModel.value || aiAssistConfig.model || selectedAnnotationModel.value,
    model_name: selectedAnnotationModel.value || annotationSessionModel.value || aiAssistConfig.model,
    annotation_geometry: annotationGeometry.value,
    class_names: [...classList.value],
    class_bindings: cloneAnnotations(annotationClassBindings.value),
    ...(annotationClassScope.value === 'model_session' ? {
      class_context: { labels_zh: { ...classLabelsZh.value }, inactive_class_ids: [...inactiveClassIds.value] }
    } : {}),
    last_page: imagePage.value,
    last_image_path: currentImage.value?.relative_path || selectedImagePath.value || '',
    revision: annotationDraftRevision.value,
    clear_all_annotations: annotationDraftClearAll.value,
    assets: []
  }
  const maxBytes = keepalive ? 48 * 1024 : 8 * 1024 * 1024
  const byteSize = value => new TextEncoder().encode(JSON.stringify(value)).byteLength
  if (byteSize(payload) > maxBytes) throw new Error('草稿类别信息超过本次保存大小上限，请保持页面打开后重试')
  for (const key of orderedKeys) {
    const item = annotationCache.get(key)
    if (!item) throw new Error('待保存标注缓存缺失，请保持页面打开')
    const asset = { relative_path: item.relative_path, width: Number(item.width) || 0,
      height: Number(item.height) || 0, annotations: cloneAnnotations(item.annotations) }
    payload.assets.push(asset)
    if (payload.assets.length > 200 || byteSize(payload) > maxBytes) {
      payload.assets.pop()
      if (!payload.assets.length) throw new Error('当前图片标注超过本次保存大小上限，请保持页面打开后重试')
      break
    }
  }
  return payload
}

const parseDraftKeepaliveResponse = async (response) => {
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const error = new Error(data?.detail || data?.message || `HTTP ${response.status}`)
    error.status = response.status
    throw error
  }
  return data
}

const stopAnnotationDraftTimer = () => {
  if (annotationDraftTimer) {
    window.clearInterval(annotationDraftTimer)
    annotationDraftTimer = null
  }
}

const scheduleAnnotationDraftSave = () => {
  if (annotationDraftDebounceTimer) window.clearTimeout(annotationDraftDebounceTimer)
  annotationDraftDebounceTimer = window.setTimeout(() => {
    annotationDraftDebounceTimer = null
    if (annotationViewActive) void flushAnnotationDraft()
  }, 750)
}

const saveAnnotationDraft = async ({ keepalive = false } = {}) => {
  if (!canSaveAnnotationDraftNow() || annotationDraftConflict.value) return false
  if (!annotationDraftPending.value) return true
  annotationDraftSaving.value = true
  try {
    do {
      const sentVersion = annotationDraftChangeVersion.value
      const sentClearAllVersion = annotationDraftClearAllVersion
      const payload = buildAnnotationDraftPayload({ keepalive })
      const sentKeys = payload.assets.map(item => item.relative_path)
      if (!payload.model_path) return false
      const response = keepalive
        ? await parseDraftKeepaliveResponse(await authorizedFetch(draftEndpoint(), {
          method: 'PUT', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload), keepalive: true
        }))
        : await putAnnotationDraft(currentDatasetId.value, annotationDraftKind.value, payload)
      annotationDraftId.value = response?.id || annotationDraftId.value
      annotationDraftRevision.value = Number(response?.revision) || annotationDraftRevision.value
      annotationDraftSourceVersionId.value = response?.source_version_id || annotationDraftSourceVersionId.value
      annotationDraftTemporaryDatasetId.value = response?.temporary_dataset_id || annotationDraftTemporaryDatasetId.value
      annotationDraftLastSavedAt.value = response?.updated_at || new Date().toISOString()
      // Acknowledge only assets actually included in this request, and retain newer edits.
      sentKeys.forEach(key => {
        if (annotationDraftDirtyVersions.get(key) <= sentVersion) {
          annotationDraftDirtyPaths.delete(key)
          annotationDraftDirtyVersions.delete(key)
        }
      })
      if (annotationDraftChangeVersion.value === sentVersion) {
        annotationDraftMetadataDirty.value = false
      }
      if (payload.clear_all_annotations && sentClearAllVersion === annotationDraftClearAllVersion) annotationDraftClearAll.value = false
      if (payload.annotation_task_id && response?.batch_sync?.has_more === false) {
        annotationDraftBatchTaskId.value = ''
      } else if (payload.annotation_task_id && !response?.batch_sync) {
        throw new Error('服务端尚不支持完整草稿同步，请保持页面打开')
      }
      annotationDraftWarning.value = ''
      if (keepalive) break
    } while (annotationDraftPending.value && canSaveAnnotationDraftNow())
    return !annotationDraftPending.value
  } catch (error) {
    if (Number(error?.status || error?.response?.status) === 409) {
      annotationDraftConflict.value = true
      annotationDraftWarning.value = error?.response?.data?.detail || '临时标注版本已变化，自动保存已暂停；请保持页面打开并核对。'
      stopAnnotationDraftTimer()
    } else {
      annotationDraftWarning.value = error?.message || '临时标注保存失败，系统将继续重试；请保持页面打开。'
    }
    return false
  } finally {
    annotationDraftSaving.value = false
  }
}

const flushAnnotationDraft = async (options = {}) => {
  if (currentDatasetRole.value !== 'original' || !currentDatasetId.value || !annotationDraftKind.value) return true
  if (annotationDraftSavePromise) {
    if (options.keepalive) return false
    if (!await annotationDraftSavePromise) return false
    return flushAnnotationDraft(options)
  }
  annotationDraftSavePromise = saveAnnotationDraft(options)
  try { return await annotationDraftSavePromise }
  finally { annotationDraftSavePromise = null }
}

const startAnnotationDraftTimer = () => {
  stopAnnotationDraftTimer()
  if (currentDatasetRole.value !== 'original' || !annotationDraftKind.value) return
  annotationDraftTimer = window.setInterval(() => {
    void flushAnnotationDraft()
  }, 5000)
}

const resetAnnotationDraftState = () => {
  annotationClassScope.value = 'dataset'
  annotationDraftId.value = ''
  annotationDraftRevision.value = 0
  annotationDraftSourceVersionId.value = ''
  annotationDraftTemporaryDatasetId.value = ''
  annotationDraftDirtyPaths.clear()
  annotationDraftDirtyVersions.clear()
  annotationDraftMetadataDirty.value = false
  annotationDraftClearAll.value = false
  annotationDraftWarning.value = ''
  annotationDraftConflict.value = false
  annotationDraftLastSavedAt.value = ''
  annotationDraftBatchTaskId.value = ''
  annotationDraftResumeBatch.value = null
}

const getDatasetOptionValue = (dataset) => {
  return String(dataset?.id || dataset?.name || dataset?.path || '')
}

const getDatasetOptionLabel = (dataset) => {
  const path = dataset?.path || dataset?.dataset_path || ''
  return String(path || dataset?.name || dataset?.id || dataset?.original_filename || '未命名数据集')
}

const clampNormalizedCoordinate = (value) => Math.min(1, Math.max(0, value))

const normalizePolygonPoints = (rawPoints) => {
  if (!Array.isArray(rawPoints)) return []

  const pairs = rawPoints.length > 0 && rawPoints.every(point => !Array.isArray(point) && (typeof point !== 'object' || point === null))
    ? (rawPoints.length >= 6 && rawPoints.length % 2 === 0
        ? Array.from({ length: rawPoints.length / 2 }, (_, index) => [rawPoints[index * 2], rawPoints[index * 2 + 1]])
        : [])
    : rawPoints.map(point => {
        if (Array.isArray(point)) return [point[0], point[1]]
        return [point?.x, point?.y]
      })

  return pairs
    .map(([x, y]) => ({ x: Number(x), y: Number(y) }))
    .filter(point => Number.isFinite(point.x) && Number.isFinite(point.y))
    .map(point => ({
      x: clampNormalizedCoordinate(point.x),
      y: clampNormalizedCoordinate(point.y)
    }))
}

const normalizeAnnotation = (annotation, { bboxCoordinateMode = 'topleft' } = {}) => {
  if (!annotation || typeof annotation !== 'object') return null

  const classId = Number(annotation.class ?? annotation.class_id ?? annotation.category_id ?? 0)
  const safeClass = Number.isFinite(classId) && classId >= 0 ? Math.trunc(classId) : 0

  const rawPoints = annotation.points ?? annotation.polygon ?? annotation.segmentation
  const points = normalizePolygonPoints(rawPoints)
  if ((annotation.type === 'polygon' || rawPoints) && points.length >= 3) {
    return points.length >= 3 ? { ...annotation, class: safeClass, type: 'polygon', points } : null
  }

  let x = Number(annotation.x)
  let y = Number(annotation.y)
  let width = Number(annotation.width)
  let height = Number(annotation.height)
  const bbox = Array.isArray(annotation.bbox) ? annotation.bbox.map(Number) : null
  let bboxAlreadyNormalized = false

  if ((!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(width) || !Number.isFinite(height)) && bbox?.length >= 4) {
    const [cx, cy, bw, bh] = bbox
    if ([cx, cy, bw, bh].every(Number.isFinite)) {
      x = cx
      y = cy
      width = bw
      height = bh
      bboxAlreadyNormalized = true
    }
  }

  if (![x, y, width, height].every(Number.isFinite)) return null
  if (bboxCoordinateMode === 'center' && !bboxAlreadyNormalized) {
    x -= width / 2
    y -= height / 2
  }
  width = Math.max(0, width)
  height = Math.max(0, height)
  x = clampNormalizedCoordinate(x)
  y = clampNormalizedCoordinate(y)
  width = Math.min(width, 1 - x)
  height = Math.min(height, 1 - y)
  return { ...annotation, class: safeClass, type: 'bbox', x, y, width, height }
}

const normalizeAnnotations = (annotations, options = {}) => (Array.isArray(annotations) ? annotations : [])
  .map(annotation => normalizeAnnotation(annotation, options))
  .filter(Boolean)

const findDatasetOption = (datasetId) => {
  const id = String(datasetId || '')
  return datasetOptions.value.find(dataset => getDatasetOptionValue(dataset) === id)
}

const normalizeAnnotationClassNames = (values) => {
  const source = Array.isArray(values) ? values : []
  const names = []
  const seen = new Set()
  source.forEach(value => {
    const name = String(value ?? '').trim()
    if (name && !seen.has(name)) {
      names.push(name)
      seen.add(name)
    }
  })
  return names.length > 0 ? names : [...DEFAULT_NEW_ANNOTATION_CLASSES]
}

const normalizeAnnotationClassBindings = (values, names = classList.value) => {
  if (!Array.isArray(values)) return []
  const bindings = []
  const seen = new Set()
  values.forEach(value => {
    const pageIndex = Number(value?.page_index ?? value?.pageIndex)
    if (!Number.isInteger(pageIndex) || pageIndex < 0 || pageIndex >= names.length || seen.has(pageIndex)) return
    const modelClassId = Number(value?.model_class_id ?? value?.modelClassId ?? value?.class_id)
    const modelClassName = String(value?.model_class_name ?? value?.modelClassName ?? value?.class_name ?? '').trim()
    if (!Number.isInteger(modelClassId) && !modelClassName) return
    bindings.push({
      page_index: pageIndex,
      name: names[pageIndex],
      model_class_id: Number.isInteger(modelClassId) && modelClassId >= 0 ? modelClassId : null,
      model_class_name: modelClassName
    })
    seen.add(pageIndex)
  })
  return bindings.sort((a, b) => a.page_index - b.page_index)
}

const applyAnnotationClasses = (values, bindings = [], labelsZh = classLabelsZh.value, inactiveIds = inactiveClassIds.value) => {
  classList.value = normalizeAnnotationClassNames(values)
  inactiveClassIds.value = [...new Set((Array.isArray(inactiveIds) ? inactiveIds : [])
    .map(Number).filter(index => Number.isInteger(index) && index >= 0 && index < classList.value.length))].sort((a, b) => a - b)
  classLabelsZh.value = Object.fromEntries(classList.value
    .filter(name => typeof labelsZh?.[name] === 'string' && labelsZh[name].trim())
    .map(name => [name, labelsZh[name].trim()]))
  annotationClassBindings.value = normalizeAnnotationClassBindings(bindings, classList.value)
  classForm.ids = activeClassOptions.value.map(item => item.index)
  classForm.classes = activeClassOptions.value.map(item => item.name)
  classForm.labelsZh = activeClassOptions.value.map(item => classLabelsZh.value[item.name] || '')
  if (inactiveClassIds.value.includes(currentClass.value)) currentClass.value = activeClassOptions.value[0]?.index ?? 0
}

const applyDraftClassContext = (response) => {
  const scope = response.class_scope || response.draft_class_scope || 'dataset'
  annotationClassScope.value = scope
  const classes = response.class_names || response.draft_class_names
  const bindings = response.class_bindings || response.draft_class_bindings || []
  if (!Array.isArray(classes)) return
  if (scope === 'model_session') {
    const context = response.class_context || response.draft_class_context || {}
    applyAnnotationClasses(classes, bindings, context.labels_zh || {}, context.inactive_class_ids || [])
  } else applyAnnotationClasses(classes, bindings)
}

const applyDatasetClasses = (dataset) => {
  if (currentDatasetRole.value !== 'training' && dataset?.dataset_role !== 'training') return
  const classMap = dataset?.class_map || dataset?.classMap || dataset?.classes
  let names = []
  if (Array.isArray(classMap)) {
    names = classMap.map(item => typeof item === 'string' ? item : item?.name).filter(Boolean)
  } else if (classMap && typeof classMap === 'object') {
    names = Object.entries(classMap)
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([, value]) => typeof value === 'string' ? value : value?.name)
      .filter(Boolean)
  }
  if (names.length > 0) {
    applyAnnotationClasses(names, names.map((name, index) => ({
      page_index: index,
      name,
      model_class_id: index,
      model_class_name: name
    })), {}, [])
  }
}

const applyDatasetOption = (dataset) => {
  if (currentDatasetId.value !== getDatasetOptionValue(dataset)) inactiveClassIds.value = []
  currentDatasetId.value = getDatasetOptionValue(dataset)
  currentDatasetRole.value = dataset?.dataset_role || currentDatasetRole.value || 'original'
  applyDatasetClasses(dataset)
  currentDatasetName.value = dataset?.name || dataset?.original_filename || currentDatasetId.value
  currentDatasetPath.value = dataset?.path || dataset?.dataset_path || ''
  selectedImagePath.value = currentDatasetPath.value || currentDatasetName.value
}

const loadDatasetAnnotationClasses = async (datasetId = currentDatasetId.value) => {
  const targetId = String(datasetId || '').trim()
  if (!targetId || currentDatasetRole.value === 'training') {
    if (currentDatasetRole.value === 'training') applyDatasetClasses(findDatasetOption(targetId))
    else applyAnnotationClasses(DEFAULT_NEW_ANNOTATION_CLASSES, [], {}, [])
    return [...classList.value]
  }

  try {
    const response = await getDatasetAnnotationClasses(targetId)
    const classes = normalizeAnnotationClassNames(response?.classes || response?.annotation_classes)
    const bindings = normalizeAnnotationClassBindings(
      response?.bindings || response?.annotation_class_bindings,
      classes
    )
    if (targetId === currentDatasetId.value) applyAnnotationClasses(classes, bindings, response?.labels_zh || {}, response?.inactive_class_ids || [])
    return classes
  } catch (error) {
    console.warn('读取数据集类别失败，使用新数据集默认类别:', error)
    if (targetId === currentDatasetId.value) applyAnnotationClasses(DEFAULT_NEW_ANNOTATION_CLASSES, [], {}, [])
    return [...DEFAULT_NEW_ANNOTATION_CLASSES]
  }
}

const persistDatasetAnnotationClasses = async (source = 'user', expectedClasses = null) => {
  if (!currentDatasetId.value || currentDatasetRole.value !== 'original') return true
  if (annotationClassScope.value === 'model_session') {
    markDraftMetadataDirty()
    return await flushAnnotationDraft({ force: true })
  }
  try {
    await updateDatasetAnnotationClasses(
      currentDatasetId.value,
      classList.value,
      source,
      annotationClassBindings.value,
      classLabelsZh.value,
      expectedClasses
    )
    markDraftMetadataDirty()
    return true
  } catch (error) {
    console.warn('保存数据集类别失败:', error)
    ElMessage.error(error?.response?.data?.detail || '保存数据集类别失败')
    return false
  }
}

const clearDatasetSelection = () => {
  inactiveClassIds.value = []
  aiBatchTaskId.value = ''
  aiBatchResultCache.clear()
  annotationCache.clear()
  historyByImage.clear()
  datasetClearAllBaseline.clear()
  datasetClearAllRestoredKeys.clear()
  datasetClearAllActive.value = false
  historyRevision.value += 1
  currentDatasetId.value = ''
  currentDatasetName.value = ''
  currentDatasetPath.value = ''
  selectedImagePath.value = ''
  imageList.value = []
  imagePage.value = 1
  imageTotal.value = 0
  datasetHasAnnotations.value = false
  resetAnnotationDraftState()
  resetCanvasState()
}

const handleDatasetSelectionChange = async (datasetId) => {
  if (annotationClassScope.value === 'model_session' && String(datasetId || '') !== annotationSessionDatasetId.value) {
    currentDatasetId.value = annotationSessionDatasetId.value
    ElMessage.warning('本次标注已绑定原始数据集；更换数据集请从原始数据集页面新建标注。')
    return
  }
  if (!await flushAnnotationDraft({ force: true })) return
  const manualPath = String(datasetId || '').trim()
  if (!manualPath) {
    clearDatasetSelection()
    return
  }

  if (manualPath !== currentDatasetId.value) {
    aiBatchTaskId.value = ''
    aiBatchResultCache.clear()
    annotationCache.clear()
    historyByImage.clear()
    datasetClearAllBaseline.clear()
    datasetClearAllRestoredKeys.clear()
    datasetClearAllActive.value = false
    datasetHasAnnotations.value = false
    historyRevision.value += 1
    resetAnnotationDraftState()
  }

  const selectedDataset = findDatasetOption(datasetId)
  if (!selectedDataset) {
    currentDatasetId.value = manualPath
    currentDatasetName.value = manualPath
    currentDatasetPath.value = manualPath
    selectedImagePath.value = manualPath
  } else {
    applyDatasetOption(selectedDataset)
  }

  if (currentDatasetRole.value === 'original') {
    await loadDatasetAnnotationClasses(currentDatasetId.value)
  }
  await loadImagesFromSelectedPath()
}

const loadDatasetOptions = async () => {
  if (!currentDatasetId.value) {
    datasetOptions.value = []
    return []
  }

  loadingDatasets.value = true
  try {
    const data = currentDatasetRole.value === 'training'
      ? await getTrainingDatasets({ page: 1, page_size: 200 })
      : await getDatasets({ page: 1, page_size: 200 })
    datasetOptions.value = Array.isArray(data.datasets) ? data.datasets : []

    const selectedDataset = findDatasetOption(currentDatasetId.value)
    if (selectedDataset) {
      applyDatasetOption(selectedDataset)
    }

    return datasetOptions.value
  } catch (error) {
    console.error('加载数据集路径失败:', error)
    ElMessage.error('加载数据集路径失败')
    return []
  } finally {
    loadingDatasets.value = false
  }
}

const getBackendAssetBaseUrl = () => {
  const envBaseUrl = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL

  if (envBaseUrl && /^https?:\/\//i.test(String(envBaseUrl))) {
    return String(envBaseUrl)
      .replace(/\/api\/?$/, '')
      .replace(/\/$/, '')
  }

  if (typeof window !== 'undefined') {
    return `${window.location.protocol}//${window.location.hostname}:8000`
  }

  return 'http://localhost:8000'
}

const extractFileNameFromPath = (value, fallback = 'image.jpg') => {
  const text = String(value || fallback).split('?')[0].split('#')[0]
  const normalized = text.replace(/\\/g, '/')
  const name = normalized.split('/').filter(Boolean).pop() || fallback

  try {
    return decodeURIComponent(name)
  } catch (error) {
    return name
  }
}

const isLocalImageDiskPath = (value) => {
  const text = String(value || '').replace(/\\/g, '/').trim()
  return (
    text.startsWith('./data/image/') ||
    text.startsWith('data/image/') ||
    text.startsWith('/app/data/image/') ||
    text.includes('/data/image/')
  )
}

const buildImageAssetUrl = (rawUrl, filename) => {
  const safeFilename = extractFileNameFromPath(filename || rawUrl || 'image.jpg')
  const backendBaseUrl = getBackendAssetBaseUrl()

  if (!rawUrl) {
    return `${backendBaseUrl}/image/${encodeURIComponent(safeFilename)}`
  }

  const text = String(rawUrl).trim()

  if (text.startsWith('data:') || text.startsWith('blob:')) {
    return text
  }

  if (/^https?:\/\//i.test(text)) {
    return text
  }

  if (text.startsWith('/api/')) {
    return `${backendBaseUrl}${text}`
  }

  if (isLocalImageDiskPath(text)) {
    const name = extractFileNameFromPath(text, safeFilename)
    return `${backendBaseUrl}/image/${encodeURIComponent(name)}`
  }

  if (text.startsWith('/api/dataset-files/')) {
    return `${backendBaseUrl}${text.replace('/api/dataset-files/', '/dataset-files/')}`
  }

  if (text.startsWith('/dataset-files/')) {
    return `${backendBaseUrl}${text}`
  }

  if (text.startsWith('/api/image/')) {
    const name = extractFileNameFromPath(text.replace(/^\/api\/image\//, ''), safeFilename)
    return `${backendBaseUrl}/image/${encodeURIComponent(name)}`
  }

  if (text.startsWith('/image/')) {
    const name = extractFileNameFromPath(text.replace(/^\/image\//, ''), safeFilename)
    return `${backendBaseUrl}/image/${encodeURIComponent(name)}`
  }

  const name = extractFileNameFromPath(text, safeFilename)
  return `${backendBaseUrl}/image/${encodeURIComponent(name)}`
}

const buildImageProxyUrl = (filename) => {
  const safeFilename = extractFileNameFromPath(filename || 'image.jpg')
  return buildApiUrl(`/image/${encodeURIComponent(safeFilename)}`)
}

const buildImageDataApiUrl = (filename) => {
  const safeFilename = extractFileNameFromPath(filename || 'image.jpg')
  return buildApiUrl(`/images/${encodeURIComponent(safeFilename)}/data`)
}

const buildDatasetImageDataApiUrl = (datasetId, relativePath) => {
  const params = new URLSearchParams({
    image_path: String(relativePath || '')
  })
  return buildApiUrl(`/datasets/${encodeURIComponent(datasetId)}/images/data?${params.toString()}`)
}

const parseImageDataResponse = async (response) => {
  const text = await response.text()
  let data = null

  if (text) {
    try {
      data = JSON.parse(text)
    } catch (error) {
      data = { message: text }
    }
  }

  if (!response.ok) {
    const detail = data?.detail || data?.message || `HTTP ${response.status}`
    throw new Error(detail)
  }

  const dataUrl = data?.data
  if (typeof dataUrl !== 'string' || !dataUrl.startsWith('data:image/')) {
    throw new Error('后端未返回有效的图片 base64 数据')
  }

  return dataUrl
}

const fetchImageDataUrlFromApi = async (filename) => {
  const response = await authenticatedFetch(buildImageDataApiUrl(filename), { method: 'GET', cache: 'no-store' })
  return await parseImageDataResponse(response)
}

const fetchDatasetImageDataUrlFromApi = async (datasetId, relativePath) => {
  const response = await authenticatedFetch(buildDatasetImageDataApiUrl(datasetId, relativePath), { method: 'GET', cache: 'no-store' })
  return await parseImageDataResponse(response)
}

const getImageDisplayUrl = (image) => {
  if (!image) return ''
  if (image.thumbnailUrl) return image.data || image.previewUrl || ''
  return image.data || image.previewUrl || image.url || image.remoteUrl || ''
}

const blobToDataUrl = (blob) => new Promise((resolve, reject) => {
  const reader = new FileReader()
  reader.onload = () => resolve(reader.result)
  reader.onerror = () => reject(new Error('图片数据读取失败'))
  reader.readAsDataURL(blob)
})

const dataUrlToBlob = (dataUrl) => {
  const parts = String(dataUrl || '').split(',')
  if (parts.length < 2) {
    throw new Error('无效的图片 data URL')
  }

  const header = parts[0]
  const base64Data = parts.slice(1).join(',')
  const mimeMatch = header.match(/data:([^;]+);base64/)
  const mimeType = mimeMatch ? mimeMatch[1] : 'image/jpeg'
  const binary = atob(base64Data)
  const bytes = new Uint8Array(binary.length)

  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i)
  }

  return new Blob([bytes], { type: mimeType })
}

const fetchImageAsDataUrl = async (url) => {
  const response = await authenticatedFetch(url, { method: 'GET', cache: 'no-store' })
  if (!response.ok) {
    throw new Error(`图片读取失败: HTTP ${response.status}`)
  }

  const blob = await response.blob()
  if (!blob.type || !blob.type.startsWith('image/')) {
    throw new Error('读取到的内容不是图片')
  }

  return await blobToDataUrl(blob)
}

const ensureImageDataUrl = async (image) => {
  if (!image) {
    throw new Error('未选择图片')
  }

  if (typeof image.data === 'string' && image.data.startsWith('data:image/')) {
    return image.data
  }

  if (typeof image.url === 'string' && image.url.startsWith('data:image/')) {
    image.data = image.url
    return image.data
  }

  const filename = image.filename || image.name || extractFileNameFromPath(image.url, 'image.jpg')

  if (image.dataset_id && image.relative_path) {
    const dataUrl = await fetchImageAsDataUrl(image.fetchUrl || image.url)
    image.data = dataUrl
    image.url = dataUrl
    image.previewUrl = dataUrl
    return image.data
  }

  // 优先通过后端 API 获取 base64，避免浏览器跨域 fetch 静态图片导致 Failed to fetch。
  try {
    const dataUrl = await fetchImageDataUrlFromApi(filename)
    image.data = dataUrl
    image.url = dataUrl
    image.previewUrl = dataUrl
    return image.data
  } catch (apiError) {
    console.warn(`通过 /images/${filename}/data 读取图片失败，尝试静态资源兜底`, apiError)
  }

  // 兜底：只使用同源 /api/image 地址，避免再次触发跨域 CORS 问题。
  const candidates = [
    image.fetchUrl,
    buildImageProxyUrl(filename)
  ].filter(Boolean)

  const uniqueCandidates = [...new Set(candidates)]
  let lastError = null

  for (const candidate of uniqueCandidates) {
    try {
      const dataUrl = await fetchImageAsDataUrl(candidate)
      image.data = dataUrl
      image.url = dataUrl
      image.previewUrl = dataUrl
      return image.data
    } catch (error) {
      lastError = error
      console.warn(`图片数据读取失败，尝试下一个地址: ${candidate}`, error)
    }
  }

  throw lastError || new Error('无法读取图片数据，请确认后端 /images/{filename}/data 接口可用')
}


const parseApiResponse = async (response) => {
  const text = await response.text()
  let data = null

  if (text) {
    try {
      data = JSON.parse(text)
    } catch (error) {
      data = { message: text }
    }
  }

  if (!response.ok) {
    const detail = data?.detail || data?.message || `HTTP ${response.status}`
    throw new Error(detail)
  }

  return data || {}
}

const normalizeRemoteImageRecord = (record) => {
  const name = record.name || record.filename || record.original_name || extractFileNameFromPath(record.url, 'image.jpg')
  const filename = record.filename || extractFileNameFromPath(name, 'image.jpg')
  const rawUrl = record.url || record.image_url || record.preview_url || record.saved_path || record.path || `/image/${encodeURIComponent(filename)}`
  const url = buildImageAssetUrl(rawUrl, filename)
  const datasetId = record.dataset_id || currentDatasetId.value
  const bboxCoordinateMode = currentDatasetRole.value === 'training' ? 'center' : 'topleft'
  const relativePath = record.relative_path || record.saved_path || ''
  const fetchUrl = datasetId && relativePath
    ? buildImageAssetUrl(record.fetch_url || rawUrl, filename)
    : buildImageProxyUrl(filename)
  const thumbnailUrl = record.thumbnail_url
    ? buildImageAssetUrl(record.thumbnail_url, filename)
    : ''
  const cacheKeys = [
    relativePath,
    filename,
    name
  ].filter(Boolean)
  const cachedEntry = getCachedAnnotationEntry({ relative_path: relativePath, filename, name })
  const batchCached = cacheKeys.map(key => aiBatchResultCache.get(key)).find(Boolean)
  const imageKey = getImageAnnotationCacheKey({ relative_path: relativePath, filename, name })
  const recordAnnotations = cachedEntry
    ? normalizeAnnotations(cachedEntry.annotations, { bboxCoordinateMode: 'topleft' })
    : normalizeAnnotations(record.annotations, { bboxCoordinateMode })

  if (datasetClearAllActive.value) {
    if (imageKey && !datasetClearAllBaseline.has(imageKey)) {
      datasetClearAllBaseline.set(imageKey, cloneAnnotations(cachedEntry?.annotations ?? recordAnnotations))
    }
  }

  let annotations = datasetClearAllActive.value
    ? normalizeAnnotations(cachedEntry?.annotations || [], { bboxCoordinateMode: 'topleft' })
    : recordAnnotations
  let appliedBatchTaskId = cachedEntry?.aiBatchTaskId || ''

  if (!datasetClearAllActive.value && batchCached && batchCached.taskId !== appliedBatchTaskId && Array.isArray(batchCached.annotations)) {
    const generatedAnnotations = cloneAnnotations(batchCached.annotations)
    if (aiAssistConfig.mode === 'replace') {
      annotations = normalizeAnnotations(generatedAnnotations, { bboxCoordinateMode: 'topleft' })
    } else {
      annotations = [
        ...JSON.parse(JSON.stringify(annotations || [])),
        ...normalizeAnnotations(generatedAnnotations, { bboxCoordinateMode: 'topleft' })
      ]
    }
    appliedBatchTaskId = batchCached.taskId || aiBatchTaskId.value || ''
  }

  const normalized = {
    ...record,
    id: record.id || filename,
    filename,
    name,
    dataset_id: datasetId,
    dataset_name: record.dataset_name || currentDatasetName.value,
    relative_path: relativePath,
    rawUrl,
    url,
    remoteUrl: url,
    previewUrl: '',
    thumbnailUrl,
    fetchUrl,
    width: Number(record.width) || 0,
    height: Number(record.height) || 0,
    size: Number(record.size) || 0,
    sourcePath: currentDatasetName.value || selectedImagePath.value,
    annotations: normalizeAnnotations(annotations, { bboxCoordinateMode: 'topleft' }),
    _aiBatchTaskApplied: appliedBatchTaskId
  }

  cacheImageAnnotations(normalized)
  return normalized
}

const revokeThumbnailObjectUrls = () => {
  thumbnailObjectUrls.forEach(url => URL.revokeObjectURL(url))
  thumbnailObjectUrls.clear()
}

const hydrateThumbnails = async (images) => {
  const queue = images.filter(image => image.thumbnailUrl && !image.previewUrl)
  const workers = Array.from({ length: Math.min(6, queue.length) }, async () => {
    while (queue.length > 0) {
      const image = queue.shift()
      let lastError = null

      for (let attempt = 0; attempt < 3 && !image.previewUrl; attempt += 1) {
        try {
          const response = await authenticatedFetch(image.thumbnailUrl, { method: 'GET', timeoutMs: 30000 })
          if (!response.ok) throw new Error(`缩略图请求失败: HTTP ${response.status}`)
          const objectUrl = URL.createObjectURL(await response.blob())
          thumbnailObjectUrls.add(objectUrl)
          image.previewUrl = objectUrl
        } catch (error) {
          lastError = error
          if (attempt < 2) {
            await new Promise(resolve => setTimeout(resolve, 300 * (attempt + 1)))
          }
        }
      }

      if (!image.previewUrl && lastError?.name !== 'AbortError') {
        console.warn('缩略图加载失败:', image.thumbnailUrl, lastError)
      }
    }
  })
  await Promise.allSettled(workers)
}

const hydrateImageListThumbnails = (startIndex, imageCount) => {
  const thumbnailTargets = Array.from(
    { length: imageCount },
    (_, offset) => imageList.value[startIndex + offset]
  ).filter(Boolean)

  if (thumbnailTargets.length > 0) {
    void hydrateThumbnails(thumbnailTargets)
  }
}

const resetCanvasState = () => {
  currentImageIndex.value = -1
  selectedAnnotation.value = -1
  currentImageObj = null

  if (canvasRef.value) {
    ctx = canvasRef.value.getContext('2d')
    ctx.clearRect(0, 0, canvasWidth.value, canvasHeight.value)
  }
}

const buildImageListQuery = (page) => {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(imagePageSize.value),
    include_dimensions: 'false',
    dataset_role: currentDatasetRole.value || 'original'
  })
  if (currentDatasetRole.value === 'original' && annotationDraftKind.value && annotationDraftId.value) {
    params.set('draft_kind', annotationDraftKind.value)
    params.set('draft_id', annotationDraftId.value)
  }

  return `/datasets/${encodeURIComponent(currentDatasetId.value)}/images?${params.toString()}`
}

const fetchImagesPageFromSelectedPath = async ({ page = imagePage.value, silent = false } = {}) => {
  if (!currentDatasetId.value) {
    ElMessage.warning('请从数据集页面进入标注')
    return
  }

  const targetPage = Math.max(1, Number(page) || 1)
  cacheCurrentPageAnnotations()
  loadingPathImages.value = true

  try {
    const response = await authenticatedFetch(buildApiUrl(buildImageListQuery(targetPage)), { method: 'GET' })
    const data = await parseApiResponse(response)
    currentDatasetName.value = data.dataset_name || currentDatasetName.value
    currentDatasetPath.value = data.dataset_path || currentDatasetPath.value
    selectedImagePath.value = currentDatasetPath.value || currentDatasetName.value
    const images = Array.isArray(data.images) ? data.images : []
    const normalizedImages = images.map(normalizeRemoteImageRecord)

    if (data.draft_id) annotationDraftId.value = String(data.draft_id)
    if (data.draft_revision != null) annotationDraftRevision.value = Math.max(annotationDraftRevision.value, Number(data.draft_revision) || 0)
    if (data.draft_source_version_id) annotationDraftSourceVersionId.value = String(data.draft_source_version_id)
    annotationDraftTemporaryDatasetId.value = String(data.draft_temporary_dataset_id || '')
    if (data.draft_class_names) applyDraftClassContext(data)

    if (!datasetClearAllActive.value) {
      datasetHasAnnotations.value = data.has_annotations === true || normalizedImages.some(
        image => Array.isArray(image?.annotations) && image.annotations.length > 0
      )
    }

    imageTotal.value = Number(data.total) || normalizedImages.length
    imagePage.value = Number(data.page) || targetPage
    revokeThumbnailObjectUrls()
    imageList.value = normalizedImages
    resetCanvasState()

    if (imageList.value.length > 0) {
      const firstImageLoading = selectImage(0)
      await nextTick()
      hydrateImageListThumbnails(0, normalizedImages.length)
      await firstImageLoading
      if (!silent) ElMessage.success(`已读取第 ${imagePage.value} 页，共 ${imageTotal.value} 张图片`)
    } else if (!silent) {
      ElMessage.warning(`${currentDatasetName.value || currentDatasetId.value} 下暂无图片`)
    }
  } catch (error) {
    console.error('读取图片路径失败:', error)
    ElMessage.error(error?.message || `读取 ${selectedImagePath.value} 失败`)
  } finally {
    loadingPathImages.value = false
  }
}

const loadImagesFromSelectedPath = async () => {
  if (!await flushAnnotationDraft({ force: true })) return
  imageTotal.value = 0
  await fetchImagesPageFromSelectedPath({ page: 1 })
}

const restoreAnnotationDraftLocation = async () => {
  if (!annotationDraftId.value || !annotationDraftKind.value) return
  const targetPage = Math.max(1, Number(route.query.draftPage) || 1)
  if (targetPage > 1) await fetchImagesPageFromSelectedPath({ page: targetPage, silent: true })
  const targetPath = String(route.query.draftImagePath || '').replace(/\\/g, '/').trim()
  if (!targetPath) return
  const targetIndex = imageList.value.findIndex(image => String(image.relative_path || '').replace(/\\/g, '/') === targetPath)
  if (targetIndex >= 0) await selectImage(targetIndex)
}

const handleImagePageChange = async (page) => {
  if (loadingPathImages.value || page === imagePage.value) return
  if (!await flushAnnotationDraft({ force: true })) return
  await fetchImagesPageFromSelectedPath({ page, silent: true })
}

const handleImagePathChange = () => {
  loadImagesFromSelectedPath()
}

// 方法
const handleImageUpload = (file) => {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const img = new Image()
      img.onload = () => {
        const imageData = {
          name: file.name,
          url: e.target.result,
          width: img.width,
          height: img.height,
          annotations: []
        }
        imageList.value.push(imageData)
        
        if (currentImageIndex.value === -1) {
          selectImage(0)
        }
        
        ElMessage.success(`已上传图片: ${file.name}`)
        resolve(false) // 阻止默认上传
      }
      img.src = e.target.result
    }
    reader.readAsDataURL(file)
  })
}

// 视频上传处理
const handleVideoUpload = async (file) => {
  try {
    ElMessage.info('正在上传视频，请稍候...')
    
    const response = await uploadAnnotationVideo(file, (progress) => {
      console.log(`上传进度: ${progress}%`)
    })
    
    currentVideoInfo.value = response
    showVideoFrameDialog.value = true
    
    ElMessage.success('视频上传成功，请配置帧提取参数')
    return false // 阻止默认上传
    
  } catch (error) {
    console.error('视频上传失败:', error)
    ElMessage.error('视频上传失败')
    return false
  }
}

// 提取视频帧
const extractFrames = async () => {
  if (!currentVideoInfo.value) return
  
  isExtractingFrames.value = true
  
  try {
    ElMessage.info('正在提取视频帧，请稍候...')
    
    // 计算实际的帧间隔参数
    let frameInterval = videoFrameConfig.frameInterval
    let maxFrames = videoFrameConfig.maxFrames
    
    if (videoFrameConfig.intervalType === 'count') {
      // 如果按数量提取，计算帧间隔
      const totalFrames = currentVideoInfo.value.frame_count
      frameInterval = totalFrames / maxFrames / currentVideoInfo.value.fps
    }
    
    const response = await extractVideoFrames(
      currentVideoInfo.value.video_id,
      currentVideoInfo.value.video_path,
      frameInterval,
      maxFrames
    )
    
    // 将提取的帧添加到图片列表
    response.frames.forEach((frame, index) => {
      const imageData = {
        name: `${currentVideoInfo.value.filename}_frame_${frame.frame_index}.jpg`,
        url: frame.data,
        width: frame.width,
        height: frame.height,
        annotations: [],
        frameIndex: frame.frame_index,
        timestamp: frame.timestamp
      }
      imageList.value.push(imageData)
    })
    
    // 自动选择第一帧
    if (currentImageIndex.value === -1 && imageList.value.length > 0) {
      selectImage(0)
    }
    
    showVideoFrameDialog.value = false
    currentVideoInfo.value = null
    
    ElMessage.success(`成功提取 ${response.extracted_frames} 帧，已添加到图片列表`)
    
  } catch (error) {
    console.error('提取视频帧失败:', error)
    ElMessage.error('提取视频帧失败')
  } finally {
    isExtractingFrames.value = false
  }
}

// 格式化时长
const formatDuration = (seconds) => {
  if (!seconds) return '0:00'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

// 显示AI辅助标注对话框
const openAIAssistDialog = async (mode = 'single') => {
  if (mode === 'single' && !currentImage.value) {
    ElMessage.warning('请先选择一张图片')
    return
  }

  if (mode === 'batch' && !currentDatasetId.value) {
    ElMessage.warning('请先从原始数据集页面选择数据集')
    return
  }

  aiAssistRunMode.value = mode
  showAIAssistDialog.value = true

  if (availableModels.value.length === 0) {
    await loadAvailableModels()
  }
  if (annotationSessionModel.value) {
    aiAssistConfig.model = annotationSessionModel.value
  }
}

// AI辅助标注配置校验
const validateAIAssistConfig = () => {
  if (annotationSessionModel.value) {
    aiAssistConfig.model = annotationSessionModel.value
  }
  const confidence = Number(aiAssistConfig.confidence)
  if (!Number.isFinite(confidence) || confidence <= 0 || confidence > 1) {
    ElMessage.warning('置信度阈值必须在 0.01 到 1 之间')
    return false
  }
  aiAssistConfig.confidence = clampNumber(confidence, 0.01, 1)

  if (!aiAssistConfig.model) {
    ElMessage.warning('请选择检测模型')
    return false
  }

  if (isYOLOEModelSelected.value && getAIAssistPromptClasses().length === 0) {
    ElMessage.warning('请设置YOLOE提示词或先编辑类别列表')
    return false
  }

  if (aiAssistConfig.classMode === 'manual' &&
      (aiAssistConfig.targetClass === null || aiAssistConfig.targetClass === undefined || aiAssistConfig.targetClass === '')) {
    ElMessage.warning('请选择或输入目标类别')
    return false
  }
  if (aiAssistConfig.classMode === 'manual' && inactiveClassIds.value.includes(Number(aiAssistConfig.targetClass))) {
    ElMessage.warning('目标类别已停用，请选择其他类别')
    return false
  }
  return true
}

// 将数值限制在指定范围内
const clampNumber = (value, min = 0, max = 1) => {
  const numberValue = Number(value)
  if (!Number.isFinite(numberValue)) return min
  return Math.max(min, Math.min(max, numberValue))
}

// 确保类别存在，并返回类别索引
const ensureClassIndex = (className) => {
  const safeClassName = String(className || 'unknown').trim() || 'unknown'
  let classIndex = classList.value.indexOf(safeClassName)
  let isNew = false

  if (classIndex === -1) {
    classList.value.push(safeClassName)
    classForm.ids = classList.value.map((_, index) => index)
      .filter(index => !inactiveClassIds.value.includes(index))
    classForm.classes = classForm.ids.map(index => classList.value[index])
    classForm.labelsZh = classForm.ids.map(index => classLabelsZh.value[classList.value[index]] || '')
    persistDatasetAnnotationClasses('user')
    classIndex = classList.value.length - 1
    isNew = true
  }

  return { classIndex, isNew }
}

// 解析AI检测结果应该写入哪个类别
const resolveAIAssistClassIndex = (ann) => {
  const backendPageIndex = Number(ann?.page_class_id)
  if (Number.isInteger(backendPageIndex) && backendPageIndex >= 0 && backendPageIndex < classList.value.length) {
    return { classIndex: backendPageIndex, isNew: false }
  }

  if (aiAssistConfig.classMode === 'manual') {
    let targetClassIndex = aiAssistConfig.targetClass

    // el-select allow-create 时，用户可能输入字符串类别名
    if (typeof targetClassIndex === 'string') {
      return ensureClassIndex(targetClassIndex)
    }

    return { classIndex: targetClassIndex, isNew: false }
  }

  // 自动模式：使用后端返回的类别名，没有类别名时使用 class_xxx 兜底
  const className = ann.class_name || `class_${ann.class ?? 'unknown'}`
  return ensureClassIndex(className)
}

const normalizeAnnotationPoints = (rawPoints) => {
  if (!rawPoints) return []

  let points = Array.isArray(rawPoints) ? rawPoints : []
  if (points.length > 0 && points.every(point => typeof point !== 'object')) {
    if (points.length < 6 || points.length % 2 !== 0) return []
    points = Array.from({ length: points.length / 2 }, (_, index) => [
      points[index * 2],
      points[index * 2 + 1]
    ])
  }

  const normalized = points
    .map(point => {
      const x = Number(Array.isArray(point) ? point[0] : point?.x)
      const y = Number(Array.isArray(point) ? point[1] : point?.y)
      if (!Number.isFinite(x) || !Number.isFinite(y)) return null
      return {
        x: clampNumber(x, 0, 1),
        y: clampNumber(y, 0, 1)
      }
    })
    .filter(Boolean)

  return normalized.length >= 3 ? normalized : []
}

const getPointBounds = (points) => {
  const xs = points.map(point => point.x)
  const ys = points.map(point => point.y)
  const minX = Math.min(...xs)
  const minY = Math.min(...ys)
  const maxX = Math.max(...xs)
  const maxY = Math.max(...ys)
  return {
    x: clampNumber(minX, 0, 0.999),
    y: clampNumber(minY, 0, 0.999),
    width: clampNumber(maxX - minX, 0.001, 1 - minX),
    height: clampNumber(maxY - minY, 0.001, 1 - minY)
  }
}

// 将AI返回的检测结果标准化为当前前端使用的相对坐标格式
const createAIAssistAnnotation = (ann, classIndex) => {
  const points = normalizeAnnotationPoints(ann.points || ann.polygon || ann.segmentation)
  let x = Number(ann.x)
  let y = Number(ann.y)
  let width = Number(ann.width)
  let height = Number(ann.height)

  if ((!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(width) || !Number.isFinite(height)) && Array.isArray(ann.bbox) && ann.bbox.length >= 4) {
    const [cx, cy, bw, bh] = ann.bbox.map(Number)
    if ([cx, cy, bw, bh].every(Number.isFinite)) {
      x = cx - bw / 2
      y = cy - bh / 2
      width = bw
      height = bh
    }
  }

  if (isPolygonAnnotationMode.value) {
    if (points.length >= 3) {
      const bounds = getPointBounds(points)
      return {
        type: 'polygon',
        class: classIndex,
        points,
        ...bounds
      }
    }

    if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) {
      return null
    }
    x = clampNumber(x, 0, 0.999)
    y = clampNumber(y, 0, 0.999)
    width = clampNumber(width, 0.001, 1 - x)
    height = clampNumber(height, 0.001, 1 - y)
    return {
      type: 'polygon',
      class: classIndex,
      points: [
        { x, y },
        { x: x + width, y },
        { x: x + width, y: y + height },
        { x, y: y + height }
      ],
      x,
      y,
      width,
      height
    }
  }

  if (points.length >= 3) {
    const bounds = getPointBounds(points)
    return {
      type: 'rect',
      class: classIndex,
      ...bounds
    }
  }

  if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) {
    return null
  }

  x = clampNumber(x, 0, 0.999)
  y = clampNumber(y, 0, 0.999)
  width = clampNumber(width, 0.001, 1 - x)
  height = clampNumber(height, 0.001, 1 - y)

  return {
    type: 'rect',
    class: classIndex,
    x,
    y,
    width,
    height
  }
}

// 把AI检测结果写入指定图片
const applyAIAssistResultToImage = (image, annotations) => {
  const result = {
    added: 0,
    newClasses: 0
  }

  if (!Array.isArray(annotations) || annotations.length === 0) {
    return result
  }

  if (aiAssistConfig.mode === 'replace') {
    image.annotations = []
  }

  annotations.forEach(ann => {
    const { classIndex, isNew } = resolveAIAssistClassIndex(ann)
    if (inactiveClassIds.value.includes(classIndex)) return
    const annotation = createAIAssistAnnotation(ann, classIndex)

    if (annotation) {
      image.annotations.push(annotation)
      result.added += 1
      if (isNew) result.newClasses += 1
    }
  })

  return result
}

// 执行当前图片AI辅助标注
const runAIAssist = async () => {
  if (!currentImage.value) return
  if (!validateAIAssistConfig()) return

  isAIAssisting.value = true

  try {
    ElMessage.info('正在使用AI检测当前图片，请稍候...')

    const imageData = await ensureImageDataUrl(currentImage.value)
    if (!await persistDatasetAnnotationClasses('user')) return
    const response = await aiAssistAnnotation(
      imageData,
      aiAssistConfig.model,
      aiAssistConfig.confidence,
      classList.value,
      annotationClassBindings.value,
      getAIAssistPromptClasses(),
      getAIAssistPromptPageIndices(),
      aiAssistConfig.classMode === 'manual' && Number.isInteger(Number(aiAssistConfig.targetClass))
        ? Number(aiAssistConfig.targetClass)
        : null,
      getAIAssistModelMetadata(),
      getAIAssistModelId(),
      { dataset_id: currentDatasetId.value, annotation_draft_id: annotationDraftId.value }
    )

    const annotations = response.annotations || []
    if (Array.isArray(response.unmapped_page_classes) && response.unmapped_page_classes.length > 0) {
      ElMessage.warning('部分页面类别没有对应的模型原生类别，已自动过滤')
    }
    if (annotations.length === 0) {
      ElMessage.warning('未检测到任何目标；请检查页面类别与模型类别映射或降低置信度阈值')
      return
    }

    const result = applyAIAssistResultToImage(currentImage.value, annotations)

    saveHistory()
    markImageDirty(currentImage.value)
    if (result.newClasses > 0) markDraftMetadataDirty()
    drawCanvas()
    showAIAssistDialog.value = false

    if (result.newClasses > 0) {
      ElMessage.info(`已自动添加 ${result.newClasses} 个新类别到类别列表`)
    }

    if (aiAssistConfig.classMode === 'manual') {
      const targetName = getClassDisplayName(classList.value[currentImage.value.annotations[currentImage.value.annotations.length - 1]?.class]) || '目标类别'
      ElMessage.success(`AI检测完成，已将 ${result.added} 个目标标记为「${targetName}」`)
    } else {
      ElMessage.success(`AI检测完成，已添加 ${result.added} 个标注框`)
    }
  } catch (error) {
    console.error('AI辅助标注失败:', error)
    ElMessage.error(error.response?.data?.detail || 'AI辅助标注失败，请检查模型是否存在')
  } finally {
    isAIAssisting.value = false
  }
}

// 将后端批量标注返回的原始检测结果转换成当前前端标注格式
const convertBackendBatchAnnotations = (annotations = []) => {
  const converted = []

  if (!Array.isArray(annotations)) return converted

  annotations.forEach(ann => {
    const { classIndex } = resolveAIAssistClassIndex(ann)
    if (inactiveClassIds.value.includes(classIndex)) return
    const annotation = createAIAssistAnnotation(ann, classIndex)
    if (annotation) converted.push(annotation)
  })

  return converted
}

const applyGeneratedAnnotationsToImage = (image, generatedAnnotations, taskId = aiBatchTaskId.value) => {
  if (!image || !Array.isArray(generatedAnnotations)) return

  // 避免同一个后端任务结果被重复追加。
  if (image._aiBatchTaskApplied === taskId) return

  const clonedAnnotations = JSON.parse(JSON.stringify(generatedAnnotations))
  if (aiAssistConfig.mode === 'replace') {
    image.annotations = clonedAnnotations
  } else {
    image.annotations = [
      ...JSON.parse(JSON.stringify(image.annotations || [])),
      ...clonedAnnotations
    ]
  }

  image._aiBatchTaskApplied = taskId
  markImageDirty(image)
}

const applyBackendBatchCacheToLoadedImages = () => {
  const currentImageBefore = currentImage.value
    ? cloneAnnotations(currentImage.value.annotations)
    : null

  imageList.value.forEach(image => {
    const keys = [image.asset_key, image.path, image.saved_path, image.relative_path, image.filename, image.name].filter(Boolean)
    const cached = keys.map(key => aiBatchResultCache.get(key)).find(Boolean)
    if (!datasetClearAllActive.value && cached) {
      applyGeneratedAnnotationsToImage(image, cached.annotations || [], cached.taskId)
    }
    cacheImageAnnotations(image)
  })

  if (currentImage.value) {
    selectedAnnotation.value = -1
    const currentImageChanged = JSON.stringify(currentImageBefore) !== JSON.stringify(currentImage.value.annotations)
    if (currentImageChanged) {
      saveHistory()
    }
    drawCanvas()
  }
}

const startBackendBatchAnnotationTask = async () => {
  if (!await persistDatasetAnnotationClasses('user')) throw new Error('无法保存当前标注会话')
  markDraftMetadataDirty()
  if (!await flushAnnotationDraft({ force: true }) || !annotationDraftId.value) {
    throw new Error(annotationDraftWarning.value || '无法保存标注草稿，未启动 AI 任务')
  }
  return await startBatchAnnotation({
    model_path: aiAssistConfig.model,
    model_id: getAIAssistModelId(),
    confidence: aiAssistConfig.confidence,
    dataset_id: currentDatasetId.value,
    annotation_draft_id: annotationDraftId.value || undefined,
    annotation_mode: aiAssistConfig.mode,
    annotation_classes: classList.value,
    annotation_class_bindings: annotationClassBindings.value,
    prompt_classes: getAIAssistPromptClasses(),
    prompt_page_indices: getAIAssistPromptPageIndices(),
    target_page_index: aiAssistConfig.classMode === 'manual' && Number.isInteger(Number(aiAssistConfig.targetClass))
      ? Number(aiAssistConfig.targetClass)
      : null,
    model_metadata: getAIAssistModelMetadata()
  })
}

const fetchBackendBatchAnnotationStatus = async (taskId) => {
  return await getBatchAnnotationStatus(taskId)
}

const fetchBackendBatchAnnotationResultsPage = async (taskId, page = 1, pageSize = 500) => {
  return await getBatchAnnotationResults(taskId, page, pageSize)
}

const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms))

const annotationTaskStorageKey = () => `deepvision:annotation-task:${currentDatasetId.value || 'default'}`

const updateBatchProgress = (data = {}) => {
  aiBatchProgress.total = Number(data.total) || aiBatchProgress.total || 0
  aiBatchProgress.current = Number(data.processed) || 0
  aiBatchProgress.success = Number(data.success) || 0
  aiBatchProgress.failed = Number(data.failed) || 0
  aiBatchProgress.added = Number(data.added) || 0
  aiBatchStatusMessage.value = data.current_filename || data.message || ''
  if (data.status === 'cancelling') aiBatchCancelRequested.value = true
  if (aiBatchCancelRequested.value) aiBatchStatusMessage.value = '正在取消，等待当前图片处理结束'
}

const cancelAIAssistAll = async () => {
  if (!isBatchAIAssisting.value || isCancellingBatch.value) return
  aiBatchCancelRequested.value = true
  aiBatchStatusMessage.value = '正在取消，等待当前图片处理结束'
  // Task creation may still be in flight. Submit cancellation once its ID arrives.
  if (!aiBatchTaskId.value) return
  isCancellingBatch.value = true
  try {
    await cancelBatchAnnotation(aiBatchTaskId.value)
  } catch (error) {
    aiBatchCancelRequested.value = false
    aiBatchStatusMessage.value = '取消请求失败，可重试'
    ElMessage.error(error?.response?.data?.detail || error?.message || '取消请求失败，请重试')
  } finally {
    isCancellingBatch.value = false
  }
}

const syncBackendBatchResultsToCache = async (taskId) => {
  let page = 1
  const pageSize = 500
  let hasMore = true

  while (hasMore) {
    const data = await fetchBackendBatchAnnotationResultsPage(taskId, page, pageSize)
    const items = Array.isArray(data.items) ? data.items : []

    items.forEach(item => {
      const filename = item.filename || item.name
      const cacheKeys = [item.asset_key, item.path, item.saved_path, filename].filter(Boolean)
      if (cacheKeys.length === 0) return

      const convertedAnnotations = convertBackendBatchAnnotations(item.annotations || [])
      const cacheValue = {
        annotations: convertedAnnotations,
        taskId,
        raw: item
      }
      cacheKeys.forEach(key => aiBatchResultCache.set(key, cacheValue))
    })

    hasMore = Boolean(data.has_more)
    page += 1
    await nextTick()
  }
}

const monitorBackendBatchTask = async (taskId, initialStatus = null) => {
  aiBatchTaskId.value = taskId
  localStorage.setItem(annotationTaskStorageKey(), taskId)
  if (aiBatchCancelRequested.value) await cancelAIAssistAll()
  let consecutiveStatusErrors = 0
  const fetchStatusWithRecovery = async () => {
    while (true) {
      try {
        const status = await fetchBackendBatchAnnotationStatus(taskId)
        consecutiveStatusErrors = 0
        return status
      } catch (error) {
        const statusCode = Number(error?.response?.status || 0)
        const recoverable = statusCode === 0 || statusCode >= 500
        if (!recoverable || consecutiveStatusErrors >= 60) throw error

        consecutiveStatusErrors += 1
        aiBatchStatusMessage.value = `后端服务暂时不可用，正在恢复连接（${consecutiveStatusErrors}/60）`
        await wait(document.hidden ? 5000 : 2000)
      }
    }
  }

  let statusData = initialStatus || await fetchStatusWithRecovery()
  updateBatchProgress(statusData)

  while (annotationViewActive && ['pending', 'running', 'cancelling', 'interrupted'].includes(statusData.status)) {
    await wait(document.hidden ? 5000 : 1000)
    statusData = await fetchStatusWithRecovery()
    updateBatchProgress(statusData)
    await nextTick()
  }

  if (!annotationViewActive) return statusData
  aiBatchStatusMessage.value = '正在完整保存标注草稿'
  if (!await flushAnnotationDraft({ force: true })) throw new Error(annotationDraftWarning.value || '草稿保存失败')
  annotationDraftBatchTaskId.value = taskId
  if (!await flushAnnotationDraft({ force: true })) throw new Error(annotationDraftWarning.value || 'AI 结果尚未完整保存')
  // The draft now owns the complete result, including unloaded pages. Do not re-append the task on export.
  aiBatchResultCache.clear()
  annotationCache.clear()
  aiBatchTaskId.value = ''
  localStorage.removeItem(annotationTaskStorageKey())
  if (annotationViewActive) await fetchImagesPageFromSelectedPath({ page: imagePage.value, silent: true })
  if (statusData.status === 'failed') throw new Error(statusData.error || 'AI 任务失败，已保存可用的部分结果')
  if (statusData.status === 'cancelled') ElMessage.warning('批量 AI 标注任务已取消，已保存完成的结果')
  return statusData
}

// 一键对 ./data/image 全部图片执行后端后台 AI 标注
const runAIAssistAll = async () => {
  if (isBatchAIAssisting.value) return
  if (!validateAIAssistConfig()) return

  isBatchAIAssisting.value = true
  aiBatchCancelRequested.value = false
  aiBatchTaskId.value = ''
  aiBatchStatusMessage.value = '准备创建任务'
  aiBatchProgress.current = 0
  aiBatchProgress.total = imageTotal.value || imageList.value.length || 0
  aiBatchProgress.success = 0
  aiBatchProgress.failed = 0
  aiBatchProgress.added = 0

  try {
    // 不再强制加载全部图片到前端，避免几千张图片时 DOM 和内存压力过大。
    // 后端会直接扫描 ./data/image 并后台顺序推理。
    if (imageList.value.length === 0) {
      await fetchImagesPageFromSelectedPath({ reset: true, silent: true })
    }

    const startData = await startBackendBatchAnnotationTask()
    const taskId = startData.task_id
    updateBatchProgress(startData)

    ElMessage.info(`后端批量 AI 标注任务已启动，共 ${aiBatchProgress.total} 张图片`)

    const statusData = await monitorBackendBatchTask(taskId, startData)
    if (statusData.status === 'cancelled') {
      return
    }

    showAIAssistDialog.value = false

    if (aiBatchProgress.failed > 0) {
      ElMessage.warning(
        `后端批量 AI 标注完成：成功 ${aiBatchProgress.success} 张，失败 ${aiBatchProgress.failed} 张，共添加 ${aiBatchProgress.added} 个标注框`
      )
    } else {
      ElMessage.success(
        `后端批量 AI 标注完成：已处理 ${aiBatchProgress.success} 张图片，共添加 ${aiBatchProgress.added} 个标注框`
      )
    }
  } catch (error) {
    console.error('后端批量 AI 标注失败:', error)
    ElMessage.error(error?.message || '后端批量 AI 标注失败')
  } finally {
    isBatchAIAssisting.value = false
    aiBatchCancelRequested.value = false
    aiBatchStatusMessage.value = ''
  }
}

// 加载可用模型列表
const staticAnnotationModels = [
  {
    path: 'yoloe-26m-seg.pt',
    name: 'yoloe-26m-seg.pt',
    display_name: 'YOLOE-26M-Seg',
    geometry: 'polygon',
    description: '分割模型，使用多边形标注'
  },
  {
    path: 'yolo26m.pt',
    name: 'yolo26m.pt',
    display_name: 'YOLO26M',
    geometry: 'rect',
    description: '检测模型，使用矩形框标注'
  }
]

const inferModelGeometry = (model) => {
  return modelAnnotationGeometry(model)
}

const getTrainedModelClassNames = (model) => {
  const values = model?.trained_class_names || model?.class_names || model?.names
  if (Array.isArray(values)) return values.length > 0 ? normalizeAnnotationClassNames(values) : []
  if (values && typeof values === 'object') {
    const ordered = Object.keys(values)
      .sort((a, b) => Number(a) - Number(b))
      .map(key => values[key])
    return ordered.length > 0 ? normalizeAnnotationClassNames(ordered) : []
  }
  return []
}

const applySelectedModelClasses = async (modelPath = aiAssistConfig.model) => {
  if (annotationClassScope.value === 'model_session') return false
  const model = availableModels.value.find(item => (
    item.path === modelPath || item.name === modelPath || item.display_name === modelPath
  ))
  if (!model || model.source !== 'trained') return false

  const classes = getTrainedModelClassNames(model)
  if (classes.length === 0) {
    ElMessage.warning('所选训练模型未记录训练类别，请手动编辑类别')
    return false
  }
  const bindings = classes.map((name, index) => ({
    page_index: index,
    name,
    model_class_id: index,
    model_class_name: name
  }))
  if (currentDatasetId.value && currentDatasetRole.value === 'original') {
    const response = await saveDatasetAnnotationModel(currentDatasetId.value, {
      model_path: model.path || model.name || modelPath,
      model_name: model.display_name || model.name || modelPath,
      annotation_geometry: model.geometry || annotationGeometry.value,
      model_source: 'trained',
      trained_class_names: classes
    })
    // Apply the server's dataset vocabulary only after binding succeeds.
    applyAnnotationClasses(response.annotation_classes, response.annotation_class_bindings || [])
  } else {
    applyAnnotationClasses(classes, bindings)
    await persistDatasetAnnotationClasses('trained_model')
  }
  ElMessage.success(`已绑定训练模型类别：${classes.join('、')}`)
  return true
}

const handleAIAssistModelChange = async (modelPath) => {
  const model = availableModels.value.find(item => (
    item.path === modelPath || item.name === modelPath || item.display_name === modelPath
  ))
  if (!model || model.source !== 'trained') return
  await applySelectedModelClasses(modelPath)
}

const loadAvailableModels = async () => {
  try {
    const response = await getModels()
    const merged = [...(Array.isArray(response?.models) ? response.models : []), ...staticAnnotationModels]
    const seen = new Set()
    availableModels.value = merged.filter(model => {
      const key = String(model?.path || model?.name || model?.display_name || '').trim()
      if (!key || seen.has(key)) return false
      seen.add(key)
      return true
    }).map(model => ({ ...model, geometry: inferModelGeometry(model) }))

    if (annotationSessionModel.value && !availableModels.value.some(model => (
      model.path === annotationSessionModel.value ||
      model.name === annotationSessionModel.value ||
      model.display_name === annotationSessionModel.value
    ))) {
      availableModels.value.unshift({
        path: annotationSessionModel.value,
        name: selectedAnnotationModel.value || annotationSessionModel.value,
        display_name: selectedAnnotationModel.value || annotationSessionModel.value,
        geometry: annotationGeometry.value,
        type: 'selected'
      })
    }

    if (availableModels.value.length > 0) {
      if (annotationSessionModel.value) {
        aiAssistConfig.model = annotationSessionModel.value
      }
      const currentModelExists = availableModels.value.some(m =>
        m.path === aiAssistConfig.model || m.name === aiAssistConfig.model || m.display_name === aiAssistConfig.model
      )
      if (!currentModelExists) {
        const defaultModel = availableModels.value.find(m => m.name === 'yolo11n.pt') || availableModels.value[0]
        aiAssistConfig.model = defaultModel.path || defaultModel.name || defaultModel.display_name
      }
      // 模型列表加载不改变已持久化的页面类别；只有用户明确切换训练模型时才导入。
    }
  } catch (error) {
    console.error('加载模型列表失败:', error)
    availableModels.value = staticAnnotationModels.map(model => ({ ...model }))
  }
}

const selectImage = async (index) => {
  if (index >= 0 && index < imageList.value.length) {
    if (index !== currentImageIndex.value && !await flushAnnotationDraft({ force: true })) return
    currentImageIndex.value = index
    selectedAnnotation.value = -1
    ensureImageHistory(currentImage.value)

    await nextTick()
    await loadImageToCanvas()
  }
}

// 删除图片
const deleteImage = async (index) => {
  try {
    const result = await ElMessageBox.confirm(
      `确定要删除图片 "${imageList.value[index].name}" 吗？`,
      '删除确认',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    if (result === 'confirm') {
      imageList.value.splice(index, 1)
      
      // 调整当前选中的图片索引
      if (imageList.value.length === 0) {
        currentImageIndex.value = -1
      } else if (currentImageIndex.value >= imageList.value.length) {
        currentImageIndex.value = imageList.value.length - 1
        loadImageToCanvas()
      } else if (currentImageIndex.value === index) {
        loadImageToCanvas()
      }
      
      ElMessage.success('图片已删除')
    }
  } catch (error) {
    // 用户取消删除
  }
}

const loadImageToCanvas = async () => {
  if (!currentImage.value || !canvasRef.value) return

  const canvas = canvasRef.value
  ctx = canvas.getContext('2d')

  try {
    const imageSource = await ensureImageDataUrl(currentImage.value)

    const img = new Image()
    img.onload = () => {
      currentImageObj = img

      resetZoom()

      if (!currentImage.value.width) currentImage.value.width = img.width
      if (!currentImage.value.height) currentImage.value.height = img.height

      nextTick(() => {
        drawCanvas()
      })
    }
    img.onerror = () => {
      currentImageObj = null
      ctx.clearRect(0, 0, canvasWidth.value, canvasHeight.value)
      ElMessage.error(`图片加载失败：${currentImage.value?.name || ''}`)
    }
    img.src = imageSource
  } catch (error) {
    currentImageObj = null
    ctx.clearRect(0, 0, canvasWidth.value, canvasHeight.value)
    console.error('加载图片到画布失败:', error)
    ElMessage.error(error?.message || '加载图片到画布失败')
  }
}

const setCanvasScale = (nextScale) => {
  if (!currentImageObj) return

  const normalizedScale = Math.min(maxZoom, Math.max(minZoom, nextScale))
  const sourceWidth = Number(currentImageObj.naturalWidth || currentImageObj.width) || 1
  const sourceHeight = Number(currentImageObj.naturalHeight || currentImageObj.height) || 1
  // Synthetic demo assets may be 1x1 placeholders; keep a usable canvas so
  // persisted normalized boxes remain visible during UI demonstrations.
  const renderWidth = sourceWidth <= 1 ? 800 : sourceWidth
  const renderHeight = sourceHeight <= 1 ? 600 : sourceHeight
  scale.value = normalizedScale
  canvasWidth.value = Math.max(1, Math.round(renderWidth * normalizedScale))
  canvasHeight.value = Math.max(1, Math.round(renderHeight * normalizedScale))

  nextTick(drawCanvas)
}

const zoomIn = () => setCanvasScale(scale.value + zoomStep)
const zoomOut = () => setCanvasScale(scale.value - zoomStep)
const resetZoom = () => setCanvasScale(1)

const drawCanvas = () => {
  if (!ctx || !currentImageObj) return
  
  // 清空画布
  ctx.clearRect(0, 0, canvasWidth.value, canvasHeight.value)
  
  // 绘制图片
  ctx.drawImage(currentImageObj, 0, 0, canvasWidth.value, canvasHeight.value)
  
  // 绘制标注框
  currentAnnotations.value.forEach((annotation, index) => {
    drawAnnotation(annotation, index === selectedAnnotation.value)
  })
  
  // 绘制正在绘制中的多边形
  if (isDrawingPolygon.value && polygonPoints.value.length > 0) {
    const points = polygonPoints.value.map(p => ({
      x: p.x * canvasWidth.value,
      y: p.y * canvasHeight.value
    }))
    
    ctx.strokeStyle = '#ffff00'
    ctx.fillStyle = 'rgba(255, 255, 0, 0.1)'
    ctx.lineWidth = 2
    
    // 绘制已有的线段
    ctx.beginPath()
    ctx.moveTo(points[0].x, points[0].y)
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y)
    }
    ctx.stroke()
    
    // 如果有足够的点，显示填充预览
    if (points.length >= 3) {
      ctx.fill()
    }
    
    // 绘制顶点
    points.forEach((point, index) => {
      ctx.fillStyle = '#ffff00'
      ctx.beginPath()
      ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI)
      ctx.fill()
      
      // 第一个点用不同颜色标识
      if (index === 0) {
        ctx.strokeStyle = '#ff0000'
        ctx.lineWidth = 2
        ctx.stroke()
      }
    })
  }
}

const drawAnnotation = (annotation, isSelected = false) => {
  if (!ctx) return
  
  // 根据标注类型绘制
  if (annotation.type === 'polygon' && annotation.points && annotation.points.length > 0) {
    // 绘制多边形
    const points = annotation.points.map(p => ({
      x: p.x * canvasWidth.value,
      y: p.y * canvasHeight.value
    }))
    
    ctx.strokeStyle = isSelected ? '#ff0000' : '#00ff00'
    ctx.fillStyle = isSelected ? 'rgba(255, 0, 0, 0.1)' : 'rgba(0, 255, 0, 0.1)'
    ctx.lineWidth = isSelected ? 3 : 2
    
    // 绘制多边形填充
    ctx.beginPath()
    ctx.moveTo(points[0].x, points[0].y)
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y)
    }
    ctx.closePath()
    ctx.fill()
    ctx.stroke()
    
    // 绘制顶点
    points.forEach((point, index) => {
      ctx.fillStyle = isSelected ? '#ff0000' : '#00ff00'
      ctx.beginPath()
      ctx.arc(point.x, point.y, isSelected ? 5 : 3, 0, 2 * Math.PI)
      ctx.fill()
    })
    
    // 绘制类别标签
    const label = getClassDisplayName(classList.value[annotation.class]) || `未定义 (${annotation.class})`
    ctx.fillStyle = isSelected ? '#ff0000' : '#00ff00'
    ctx.font = '14px Arial'
    const textWidth = ctx.measureText(label).width
    
    const firstPoint = points[0]
    ctx.fillRect(firstPoint.x, firstPoint.y - 20, textWidth + 8, 20)
    ctx.fillStyle = 'white'
    ctx.fillText(label, firstPoint.x + 4, firstPoint.y - 6)
    
  } else {
    // 绘制矩形框
    const x = annotation.x * canvasWidth.value
    const y = annotation.y * canvasHeight.value
    const width = annotation.width * canvasWidth.value
    const height = annotation.height * canvasHeight.value
    
    // 绘制边界框
    ctx.strokeStyle = isSelected ? '#ff0000' : '#00ff00'
    ctx.lineWidth = isSelected ? 3 : 2
    ctx.strokeRect(x, y, width, height)
    
    // 绘制类别标签
    const label = getClassDisplayName(classList.value[annotation.class]) || `未定义 (${annotation.class})`
    ctx.fillStyle = isSelected ? '#ff0000' : '#00ff00'
    ctx.font = '14px Arial'
    const textWidth = ctx.measureText(label).width
    
    ctx.fillRect(x, y - 20, textWidth + 8, 20)
    ctx.fillStyle = 'white'
    ctx.fillText(label, x + 4, y - 6)
    
    // 如果被选中，绘制调整手柄
    if (isSelected) {
      const handleSize = 6
      ctx.fillStyle = '#ff0000'
      
      // 四个角
      ctx.fillRect(x - handleSize/2, y - handleSize/2, handleSize, handleSize)
      ctx.fillRect(x + width - handleSize/2, y - handleSize/2, handleSize, handleSize)
      ctx.fillRect(x - handleSize/2, y + height - handleSize/2, handleSize, handleSize)
      ctx.fillRect(x + width - handleSize/2, y + height - handleSize/2, handleSize, handleSize)
      
      // 四条边的中点
      ctx.fillRect(x + width/2 - handleSize/2, y - handleSize/2, handleSize, handleSize)
      ctx.fillRect(x + width/2 - handleSize/2, y + height - handleSize/2, handleSize, handleSize)
      ctx.fillRect(x - handleSize/2, y + height/2 - handleSize/2, handleSize, handleSize)
      ctx.fillRect(x + width - handleSize/2, y + height/2 - handleSize/2, handleSize, handleSize)
    }
  }
}

const getCanvasCoordinates = (event) => {
  const canvas = canvasRef.value
  const rect = canvas.getBoundingClientRect()
  return {
    x: (event.clientX - rect.left) / rect.width,
    y: (event.clientY - rect.top) / rect.height
  }
}

// 检测鼠标是否在标注框的调整手柄上
const getResizeHandle = (coords, annotation) => {
  const handleWidth = 8 / canvasWidth.value
  const handleHeight = 8 / canvasHeight.value
  const x = annotation.x
  const y = annotation.y
  const w = annotation.width
  const h = annotation.height
  
  // 四个角
  if (Math.abs(coords.x - x) < handleWidth && Math.abs(coords.y - y) < handleHeight) return 'tl'
  if (Math.abs(coords.x - (x + w)) < handleWidth && Math.abs(coords.y - y) < handleHeight) return 'tr'
  if (Math.abs(coords.x - x) < handleWidth && Math.abs(coords.y - (y + h)) < handleHeight) return 'bl'
  if (Math.abs(coords.x - (x + w)) < handleWidth && Math.abs(coords.y - (y + h)) < handleHeight) return 'br'
  
  // 四条边
  if (Math.abs(coords.x - x) < handleWidth && coords.y >= y && coords.y <= y + h) return 'l'
  if (Math.abs(coords.x - (x + w)) < handleWidth && coords.y >= y && coords.y <= y + h) return 'r'
  if (Math.abs(coords.y - y) < handleHeight && coords.x >= x && coords.x <= x + w) return 't'
  if (Math.abs(coords.y - (y + h)) < handleHeight && coords.x >= x && coords.x <= x + w) return 'b'
  
  return ''
}

// 检测鼠标是否在标注框内部
const isInsideAnnotation = (coords, annotation) => {
  if (annotation.type === 'polygon' && annotation.points) {
    // 使用射线法判断点是否在多边形内
    return isPointInPolygon(coords, annotation.points)
  } else {
    // 矩形判断
    return coords.x >= annotation.x && 
           coords.x <= annotation.x + annotation.width &&
           coords.y >= annotation.y && 
           coords.y <= annotation.y + annotation.height
  }
}

// 射线法判断点是否在多边形内
const isPointInPolygon = (point, polygonPoints) => {
  let inside = false
  for (let i = 0, j = polygonPoints.length - 1; i < polygonPoints.length; j = i++) {
    const xi = polygonPoints[i].x, yi = polygonPoints[i].y
    const xj = polygonPoints[j].x, yj = polygonPoints[j].y
    
    const intersect = ((yi > point.y) !== (yj > point.y)) &&
      (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi)
    if (intersect) inside = !inside
  }
  return inside
}

// 检测鼠标是否在多边形顶点附近
const getNearbyPolygonPoint = (coords, annotation) => {
  if (annotation.type !== 'polygon' || !annotation.points) return -1
  
  for (let i = 0; i < annotation.points.length; i++) {
    const point = annotation.points[i]
    const distance = Math.sqrt(
      Math.pow((coords.x - point.x) * canvasWidth.value, 2) +
      Math.pow((coords.y - point.y) * canvasHeight.value, 2)
    )
    if (distance < 10) return i
  }
  return -1
}

const findAnnotationAt = (coords) => {
  for (let i = currentAnnotations.value.length - 1; i >= 0; i--) {
    const annotation = currentAnnotations.value[i]
    if (getNearbyPolygonPoint(coords, annotation) >= 0 || isInsideAnnotation(coords, annotation)) {
      return i
    }
  }
  return -1
}

const startAnnotationEdit = (coords, index) => {
  const annotation = currentAnnotations.value[index]
  if (!annotation) return false

  selectedAnnotation.value = index

  if (annotation.type === 'polygon') {
    const pointIndex = getNearbyPolygonPoint(coords, annotation)
    if (pointIndex >= 0) {
      isMovingPolygonPoint.value = true
      movingPolygonPointIndex.value = pointIndex
      drawCanvas()
      return true
    }
  } else {
    const handle = getResizeHandle(coords, annotation)
    if (handle) {
      isResizing.value = true
      resizeHandle.value = handle
      resizeStartAnnotation.value = { ...annotation }
      dragStartPos.value = coords
      drawCanvas()
      return true
    }
  }

  if (isInsideAnnotation(coords, annotation)) {
    isDragging.value = true
    dragStartPos.value = coords
    drawCanvas()
    return true
  }

  drawCanvas()
  return false
}

const handleMouseDown = (event) => {
  if (!currentImage.value) return
  
  const coords = getCanvasCoordinates(event)
  
  if (currentTool.value === 'polygon') {
    // 多边形绘制模式
    if (!isDrawingPolygon.value) {
      // 开始绘制新多边形
      isDrawingPolygon.value = true
      polygonPoints.value = [coords]
      selectedAnnotation.value = -1
    } else {
      // 检查是否点击在起始点附近（完成多边形）
      const firstPoint = polygonPoints.value[0]
      const distance = Math.sqrt(
        Math.pow((coords.x - firstPoint.x) * canvasWidth.value, 2) + 
        Math.pow((coords.y - firstPoint.y) * canvasHeight.value, 2)
      )
      
      if (distance < 10 && polygonPoints.value.length >= 3) {
        // 完成多边形
        const annotation = {
          type: 'polygon',
          class: currentClass.value,
          points: [...polygonPoints.value]
        }
        currentImage.value.annotations.push(annotation)
        selectedAnnotation.value = currentImage.value.annotations.length - 1
        
        // 重置状态
        isDrawingPolygon.value = false
        polygonPoints.value = []
        
        saveHistory()
        markImageDirty(currentImage.value)
        drawCanvas()
        ElMessage.success('多边形标注已完成')
      } else {
        // 添加新点
        polygonPoints.value.push(coords)
        drawCanvas()
      }
    }
  } else if (currentTool.value === 'rect') {
    if (
      selectedAnnotation.value >= 0 &&
      startAnnotationEdit(coords, selectedAnnotation.value)
    ) {
      return
    }

    const hitIndex = findAnnotationAt(coords)
    if (hitIndex >= 0) {
      startAnnotationEdit(coords, hitIndex)
      return
    }
    
    // 空白区域开始绘制新的标注框
    isDrawing.value = true
    startPoint.value = coords
    selectedAnnotation.value = -1
  } else if (currentTool.value === 'select') {
    if (
      selectedAnnotation.value >= 0 &&
      startAnnotationEdit(coords, selectedAnnotation.value)
    ) {
      return
    }

    const hitIndex = findAnnotationAt(coords)
    if (hitIndex >= 0) {
      startAnnotationEdit(coords, hitIndex)
    } else {
      selectedAnnotation.value = -1
      drawCanvas()
    }
  } else if (currentTool.value === 'delete') {
    deleteAnnotationAt(coords)
  }
}

const handleMouseMove = (event) => {
  const coords = getCanvasCoordinates(event)
  
  // 多边形绘制时显示实时预览线
  if (isDrawingPolygon.value && polygonPoints.value.length > 0) {
    drawCanvas()
    
    // 绘制从最后一个点到鼠标位置的预览线
    const lastPoint = polygonPoints.value[polygonPoints.value.length - 1]
    ctx.strokeStyle = '#ffff00'
    ctx.lineWidth = 2
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.moveTo(lastPoint.x * canvasWidth.value, lastPoint.y * canvasHeight.value)
    ctx.lineTo(coords.x * canvasWidth.value, coords.y * canvasHeight.value)
    ctx.stroke()
    ctx.setLineDash([])
    
    // 如果鼠标靠近起始点，高亮起始点
    const firstPoint = polygonPoints.value[0]
    const distance = Math.sqrt(
      Math.pow((coords.x - firstPoint.x) * canvasWidth.value, 2) + 
      Math.pow((coords.y - firstPoint.y) * canvasHeight.value, 2)
    )
    if (distance < 10 && polygonPoints.value.length >= 3) {
      ctx.fillStyle = '#ff0000'
      ctx.beginPath()
      ctx.arc(firstPoint.x * canvasWidth.value, firstPoint.y * canvasHeight.value, 8, 0, 2 * Math.PI)
      ctx.fill()
    }
    
    return
  }

  // 移动已选多边形的单个顶点
  if (
    isMovingPolygonPoint.value &&
    selectedAnnotation.value >= 0 &&
    movingPolygonPointIndex.value >= 0
  ) {
    const annotation = currentAnnotations.value[selectedAnnotation.value]
    if (annotation?.type === 'polygon' && annotation.points?.[movingPolygonPointIndex.value]) {
      annotation.points[movingPolygonPointIndex.value] = {
        x: Math.max(0, Math.min(1, coords.x)),
        y: Math.max(0, Math.min(1, coords.y))
      }
      drawCanvas()
    }
    return
  }
  
  // 处理拖动标注框
  if (isDragging.value && selectedAnnotation.value >= 0) {
    const annotation = currentAnnotations.value[selectedAnnotation.value]
    const dx = coords.x - dragStartPos.value.x
    const dy = coords.y - dragStartPos.value.y
    
    if (annotation.type === 'polygon' && annotation.points) {
      // 拖动多边形
      const minX = Math.min(...annotation.points.map(point => point.x))
      const maxX = Math.max(...annotation.points.map(point => point.x))
      const minY = Math.min(...annotation.points.map(point => point.y))
      const maxY = Math.max(...annotation.points.map(point => point.y))
      const boundedDx = Math.max(-minX, Math.min(1 - maxX, dx))
      const boundedDy = Math.max(-minY, Math.min(1 - maxY, dy))
      annotation.points = annotation.points.map(p => ({
        x: p.x + boundedDx,
        y: p.y + boundedDy
      }))
    } else {
      // 拖动矩形
      annotation.x = Math.max(0, Math.min(1 - annotation.width, annotation.x + dx))
      annotation.y = Math.max(0, Math.min(1 - annotation.height, annotation.y + dy))
    }
    
    dragStartPos.value = coords
    drawCanvas()
    return
  }
  
  // 处理调整标注框大小（仅矩形）
  if (isResizing.value && selectedAnnotation.value >= 0) {
    const annotation = currentAnnotations.value[selectedAnnotation.value]
    const start = resizeStartAnnotation.value
    const dx = coords.x - dragStartPos.value.x
    const dy = coords.y - dragStartPos.value.y
    
    switch (resizeHandle.value) {
      case 'tl':
        annotation.x = start.x + dx
        annotation.y = start.y + dy
        annotation.width = start.width - dx
        annotation.height = start.height - dy
        break
      case 'tr':
        annotation.y = start.y + dy
        annotation.width = start.width + dx
        annotation.height = start.height - dy
        break
      case 'bl':
        annotation.x = start.x + dx
        annotation.width = start.width - dx
        annotation.height = start.height + dy
        break
      case 'br':
        annotation.width = start.width + dx
        annotation.height = start.height + dy
        break
      case 'l':
        annotation.x = start.x + dx
        annotation.width = start.width - dx
        break
      case 'r':
        annotation.width = start.width + dx
        break
      case 't':
        annotation.y = start.y + dy
        annotation.height = start.height - dy
        break
      case 'b':
        annotation.height = start.height + dy
        break
    }
    
    // 确保宽高不为负
    if (annotation.width < 0) {
      annotation.x += annotation.width
      annotation.width = -annotation.width
    }
    if (annotation.height < 0) {
      annotation.y += annotation.height
      annotation.height = -annotation.height
    }
    
    // 限制在画布范围内
    annotation.x = Math.max(0, Math.min(1, annotation.x))
    annotation.y = Math.max(0, Math.min(1, annotation.y))
    annotation.width = Math.max(0.01, Math.min(1 - annotation.x, annotation.width))
    annotation.height = Math.max(0.01, Math.min(1 - annotation.y, annotation.height))
    
    drawCanvas()
    return
  }
  
  // 处理绘制新标注框
  if (isDrawing.value && currentTool.value === 'rect') {
    drawCanvas()
    
    // 绘制预览框
    const x = Math.min(startPoint.value.x, coords.x) * canvasWidth.value
    const y = Math.min(startPoint.value.y, coords.y) * canvasHeight.value
    const width = Math.abs(coords.x - startPoint.value.x) * canvasWidth.value
    const height = Math.abs(coords.y - startPoint.value.y) * canvasHeight.value
    
    ctx.strokeStyle = '#ffff00'
    ctx.lineWidth = 2
    ctx.strokeRect(x, y, width, height)
    return
  }
  
  // 更新鼠标样式
  if (
    (currentTool.value === 'rect' || currentTool.value === 'select') &&
    selectedAnnotation.value >= 0
  ) {
    const annotation = currentAnnotations.value[selectedAnnotation.value]
    const polygonPoint = getNearbyPolygonPoint(coords, annotation)
    const handle = annotation.type === 'polygon' ? '' : getResizeHandle(coords, annotation)
    
    if (polygonPoint >= 0) {
      canvasRef.value.style.cursor = 'grab'
    } else if (handle) {
      const cursors = {
        'tl': 'nw-resize',
        'tr': 'ne-resize',
        'bl': 'sw-resize',
        'br': 'se-resize',
        'l': 'w-resize',
        'r': 'e-resize',
        't': 'n-resize',
        'b': 's-resize'
      }
      canvasRef.value.style.cursor = cursors[handle]
    } else if (isInsideAnnotation(coords, annotation)) {
      canvasRef.value.style.cursor = 'move'
    } else {
      canvasRef.value.style.cursor = currentTool.value === 'rect' ? 'crosshair' : 'default'
    }
  }
}

const handleMouseUp = (event) => {
  // 结束拖动或调整
  if (isDragging.value || isResizing.value || isMovingPolygonPoint.value) {
    isDragging.value = false
    isResizing.value = false
    isMovingPolygonPoint.value = false
    resizeHandle.value = ''
    resizeStartAnnotation.value = null
    movingPolygonPointIndex.value = -1
    saveHistory() // 保存历史记录
    markImageDirty(currentImage.value)
    drawCanvas()
    return
  }
  
  // 处理绘制新标注框
  if (isDrawing.value && currentTool.value === 'rect') {
    const coords = getCanvasCoordinates(event)
    
    const x = Math.min(startPoint.value.x, coords.x)
    const y = Math.min(startPoint.value.y, coords.y)
    const width = Math.abs(coords.x - startPoint.value.x)
    const height = Math.abs(coords.y - startPoint.value.y)
    
    // 只有在有效大小时才创建标注
    if (width > 0.01 && height > 0.01) {
      const annotation = {
        class: currentClass.value,
        x: x,
        y: y,
        width: width,
        height: height
      }
      
      currentImage.value.annotations.push(annotation)
      selectedAnnotation.value = currentImage.value.annotations.length - 1
      saveHistory() // 保存历史记录
      markImageDirty(currentImage.value)
      drawCanvas()
    }
    
    isDrawing.value = false
  }
}

const deleteAnnotationAt = (coords) => {
  const index = findAnnotationAt(coords)
  if (index >= 0) {
    currentImage.value.annotations.splice(index, 1)
    selectedAnnotation.value = -1
    saveHistory()
    markImageDirty(currentImage.value)
    drawCanvas()
  }
}

const selectAnnotation = (index) => {
  selectedAnnotation.value = index
  drawCanvas()
}

const handleSelectedAnnotationClassChange = (nextClassIndex) => {
  const image = currentImage.value
  const index = selectedAnnotation.value
  const annotation = image?.annotations?.[index]
  if (!image || !annotation || index < 0) return

  const targetClass = Number(nextClassIndex)
  if (!Number.isInteger(targetClass) || targetClass === Number(annotation.class)) return

  try {
    // Establish the pre-edit state before replacing the annotation so Ctrl+Z
    // restores the original label instead of creating a no-op history entry.
    ensureImageHistory(image)
    image.annotations.splice(index, 1, relabelAnnotation(annotation, targetClass, classList.value))
    saveHistory(image)
    markImageDirty(image)
    drawCanvas()
    ElMessage.success(`已将标注类别修改为：${getClassDisplayName(classList.value[targetClass])}`)
  } catch (error) {
    console.error('修改标注类别失败:', error)
    ElMessage.error(error?.message || '修改标注类别失败')
  }
}

const deleteAnnotation = (index) => {
  currentImage.value.annotations.splice(index, 1)
  selectedAnnotation.value = -1
  saveHistory() // 保存历史记录
  markImageDirty(currentImage.value)
  drawCanvas()
}

const handleWheel = (event) => {
  event.preventDefault()
  if (!currentImageObj || event.deltaY === 0) return
  setCanvasScale(scale.value + (event.deltaY < 0 ? zoomStep : -zoomStep))
}

// 处理右键点击
const handleRightClick = (event) => {
  event.preventDefault() // 阻止默认右键菜单
  
  // 如果正在绘制多边形，右键取消
  if (isDrawingPolygon.value) {
    cancelPolygon()
  }
}

// 取消多边形绘制
const cancelPolygon = () => {
  isDrawingPolygon.value = false
  polygonPoints.value = []
  drawCanvas()
  ElMessage.info('已取消多边形绘制')
}

const saveAnnotations = () => {
  if (!currentImage.value) return
  
  const annotations = currentImage.value.annotations
  const yoloFormat = annotations.map(ann => {
    const points = normalizeAnnotationPoints(ann.points)
    if (ann.type === 'polygon' && points.length >= 3) {
      const values = points.flatMap(point => [point.x, point.y])
      return `${ann.class} ${values.map(value => value.toFixed(6)).join(' ')}`
    }

    const centerX = clampNumber(ann.x + ann.width / 2, 0, 1)
    const centerY = clampNumber(ann.y + ann.height / 2, 0, 1)
    const width = clampNumber(ann.width, 0, 1)
    const height = clampNumber(ann.height, 0, 1)

    return `${ann.class} ${centerX.toFixed(6)} ${centerY.toFixed(6)} ${width.toFixed(6)} ${height.toFixed(6)}`
  }).join('\n')
  
  const fileName = currentImage.value.name.replace(/\.[^/.]+$/, '') + '.txt'
  const blob = new Blob([yoloFormat], { type: 'text/plain' })
  saveAs(blob, fileName)
  
  ElMessage.success('标注文件已保存')
}

// 数据增强相关方法
const showDataAugmentationDialog = () => {
  if (!currentImage.value) return
  showAugmentationDialog.value = true
  augmentationPreview.value = []
}

const getAugmentationOperations = () => buildAugmentationOperations(augmentationConfig)

const generatePreview = async () => {
  if (!currentImage.value) return

  const operations = getAugmentationOperations()
  if (operations.length === 0) {
    ElMessage.warning('请至少选择一种增强操作')
    return
  }

  isGeneratingPreview.value = true
  try {
    const sourceImageUrl = await ensureImageDataUrl(currentImage.value)
    const previews = []
    for (const operation of operations) {
      const augmentedImage = await applyImageAugmentation(sourceImageUrl, operation)
      previews.push(augmentedImage)
    }
    augmentationPreview.value = previews
    ElMessage.success(`预览生成完成，共 ${previews.length} 张`)
  } catch (error) {
    console.error('预览生成失败:', error)
    ElMessage.error('预览生成失败: ' + error.message)
  } finally {
    isGeneratingPreview.value = false
  }
}

const applyDataAugmentation = async () => {
  if (!currentImage.value) return

  const operations = getAugmentationOperations()
  if (operations.length === 0) {
    ElMessage.warning('请至少选择一种增强操作')
    return
  }

  isApplyingAugmentation.value = true
  try {
    const originalImage = currentImage.value
    const relativePath = String(originalImage.relative_path || '').replace(/\\/g, '/').trim()
    if (!relativePath) {
      throw new Error('当前图片尚未保存为数据集资产，请刷新数据集后重试')
    }
    const sourceWidth = Number(originalImage.width) || Number(currentImageObj?.naturalWidth) || Number(currentImageObj?.width) || 1
    const sourceHeight = Number(originalImage.height) || Number(currentImageObj?.naturalHeight) || Number(currentImageObj?.height) || 1
    markImageDirty(originalImage)
    markDraftMetadataDirty()
    const draftSaved = await flushAnnotationDraft({ force: true })
    if (!draftSaved || !annotationDraftId.value) {
      throw new Error(annotationDraftWarning.value || '数据增强前无法保存标注草稿')
    }
    const startTask = await startDatasetAugmentation(currentDatasetId.value, {
      ...augmentationConfig,
      annotation_draft_id: annotationDraftId.value,
      annotation_kind: annotationDraftKind.value,
      draft_revision: annotationDraftRevision.value,
      source_relative_paths: [relativePath],
      annotation_overrides: [{
        relative_path: relativePath,
        width: sourceWidth,
        height: sourceHeight,
        annotations: cloneAnnotations(originalImage.annotations)
      }]
    })
    const taskId = startTask?.task_id || startTask?.id
    if (!taskId) throw new Error('后端未返回数据增强任务 ID')

    let task = await getJob(taskId)
    while (['pending', 'running', 'cancelling'].includes(task.status)) {
      await wait(document.hidden ? 3000 : 1000)
      task = await getJob(taskId)
    }
    if (task.status !== 'completed') {
      throw new Error(task.error_detail || task.message || '数据增强任务失败')
    }

    annotationDraftRevision.value = Number(task.result?.draft_revision) || annotationDraftRevision.value
    annotationDraftTemporaryDatasetId.value = task.result?.temporary_dataset_id || annotationDraftTemporaryDatasetId.value
    annotationDraftLastSavedAt.value = new Date().toISOString()
    await fetchImagesPageFromSelectedPath({ page: imagePage.value, silent: true })
    const createdImages = Number.isFinite(Number(task.result?.created_images))
      ? Number(task.result.created_images)
      : operations.length
    ElMessage.success(`成功生成 ${createdImages} 张增强图片并加入临时标注，原始数据集未修改`)
    showAugmentationDialog.value = false

  } catch (error) {
    console.error('数据增强失败:', error)
    ElMessage.error('数据增强失败: ' + error.message)
  } finally {
    isApplyingAugmentation.value = false
  }
}

// 图片增强处理函数
const applyImageAugmentation = async (imageUrl, transformParams) => {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'

    img.onload = () => {
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')

      const sourceWidth = img.naturalWidth || img.width
      const sourceHeight = img.naturalHeight || img.height
      const dimensions = getAugmentedImageDimensions(
        sourceWidth,
        sourceHeight,
        transformParams.type === 'rotation' ? transformParams.value : 0
      )

      canvas.width = dimensions.width
      canvas.height = dimensions.height

      ctx.save()

      const filterParts = []
      if (transformParams.type === 'brightness') {
        filterParts.push(`brightness(${transformParams.value})`)
      }
      if (transformParams.type === 'grayscale') {
        filterParts.push(`grayscale(${transformParams.value})`)
      }
      if (transformParams.type === 'contrast') {
        filterParts.push(`contrast(${transformParams.value})`)
      }
      ctx.filter = filterParts.join(' ') || 'none'

      ctx.translate(dimensions.width / 2, dimensions.height / 2)
      if (transformParams.type === 'rotation') {
        const angle = transformParams.value * Math.PI / 180
        ctx.rotate(angle)
      }
      ctx.scale(
        transformParams.type === 'mirror' && transformParams.value === 'horizontal' ? -1 : 1,
        transformParams.type === 'mirror' && transformParams.value === 'vertical' ? -1 : 1
      )
      ctx.drawImage(img, -sourceWidth / 2, -sourceHeight / 2, sourceWidth, sourceHeight)
      ctx.restore()

      resolve(canvas.toDataURL('image/png'))
    }

    img.onerror = () => reject(new Error('无法读取待增强图片'))
    img.src = imageUrl
  })
}

const openExportDatasetDialog = () => {
  if (!currentDatasetId.value || imageList.value.length === 0 || isExportingDataset.value) return

  exportDatasetPreflight.value = null
  exportDatasetConfig.name = `${currentDatasetName.value || currentDatasetId.value}_标注训练集`
  showExportDatasetDialog.value = true
}

const exportDataset = async () => {
  if (!currentDatasetId.value || imageList.value.length === 0 || isCheckingExport.value || isExportingDataset.value) return

  const savedBeforeExport = await flushAnnotationDraft({ force: true })
  if (!savedBeforeExport) {
    ElMessage.warning(annotationDraftWarning.value || '临时标注尚未保存，暂不能导出')
    return
  }
  const exportedDraftId = annotationDraftId.value

  const datasetName = String(exportDatasetConfig.name || '').trim()
  if (!datasetName) {
    ElMessage.warning('\u8bf7\u8f93\u5165\u8bad\u7ec3\u6570\u636e\u96c6\u540d\u79f0')
    return
  }

  // Only send currently loaded images as explicit user-edit overrides. The
  // backend reads the completed AI job for all other images, so large
  // datasets do not need to be rendered or uploaded as one huge JSON body.
  const exportPayload = {
    dataset_id: currentDatasetId.value,
    annotation_draft_id: exportedDraftId || undefined,
    annotation_task_id: aiBatchTaskId.value || undefined,
    classes: classList.value,
    images: (() => {
      cacheCurrentPageAnnotations()
      return [...annotationCache.values()].filter(item => item.relative_path)
    })(),
    class_mode: aiAssistConfig.classMode,
    target_class: aiAssistConfig.classMode === 'manual' ? aiAssistConfig.targetClass : undefined,
    annotation_mode: aiAssistConfig.mode,
    annotation_model_path: annotationSessionModel.value || aiAssistConfig.model,
    annotation_model_name: selectedAnnotationModel.value || annotationSessionModel.value || aiAssistConfig.model,
    annotation_geometry: annotationGeometry.value,
    annotation_kind: annotationDraftKind.value || undefined,
    name: datasetName,
    required_geometry: annotationGeometry.value === 'polygon' ? 'polygon' : 'bbox',
    seed: 42,
    profile: 'phone_badge_focus_v1',
    train_ratio: 0.8,
    test_ratio: 0
  }

  try {
    isCheckingExport.value = true
    const preflight = await preflightAnnotationTrainingDatasetExport(exportPayload)
    exportDatasetPreflight.value = preflight
    if (preflight?.can_export === false) {
      ElMessage.error(exportPreflightDescription.value)
      return
    }
    const unlabeledCount = Math.max(0, Number(preflight?.unlabeled_count) || 0)
    if (unlabeledCount > 0) {
      let result
      try {
        result = await ElMessageBox.confirm(
          `\u5f53\u524d\u5b58\u5728 ${unlabeledCount} \u5f20\u56fe\u7247\u672a\u6807\u6ce8，\u5bfc\u51fa\u540e\u5c06\u5264\u9664\u8fd9\u4e9b\u56fe\u7247`,
          '\u786e\u8ba4\u5bfc\u51fa',
          {
            confirmButtonText: '\u786e\u5b9a',
            cancelButtonText: '\u53d6\u6d88',
            type: 'warning'
          }
        )
      } catch {
        return
      }
      if (result !== 'confirm') return
    }

    const qualityWarnings = preflight?.quality?.warnings || []
    if (qualityWarnings.length > 0) {
      let result
      try {
        result = await ElMessageBox.confirm(
          `预检发现 ${qualityWarnings.length} 项可确认告警；阈值和确认记录会写入导出版本，是否继续？`,
          '确认质量告警',
          {
            confirmButtonText: '继续导出',
            cancelButtonText: '取消',
            type: 'warning'
          }
        )
      } catch {
        return
      }
      if (result !== 'confirm') return
      exportPayload.quality_confirmation = {
        confirmed: true,
        warning_codes: qualityWarnings.map(item => item.code).filter(Boolean).sort()
      }
    }

    isCheckingExport.value = false
    isExportingDataset.value = true
    showExportDatasetDialog.value = false
    exportDatasetProgress.percentage = 0
    exportDatasetProgress.message = '\u6b63\u5728\u521b\u5efa\u540e\u53f0\u5bfc\u51fa\u4efb\u52a1'
    exportPayload.exclude_unlabeled_images = true

    const startData = await startAnnotationTrainingDatasetExport(exportPayload)
    const taskId = startData?.task_id
    if (!taskId) throw new Error('\u540e\u7aef\u672a\u8fd4\u56de\u5bfc\u51fa\u4efb\u52a1 ID')

    let task = await getJob(taskId)
    while (['pending', 'running', 'cancelling'].includes(task.status)) {
      exportDatasetProgress.percentage = Math.max(0, Math.min(100, Number(task.progress) || 0))
      exportDatasetProgress.message = task.message || '\u540e\u53f0\u5bfc\u51fa\u8bad\u7ec3\u6570\u636e\u96c6'
      await wait(document.hidden ? 3000 : 1000)
      task = await getJob(taskId)
    }
    if (task.status !== 'completed') {
      throw new Error(task.error_detail || task.message || '\u8bad\u7ec3\u6570\u636e\u96c6\u5bfc\u51fa\u5931\u8d25')
    }

    exportDatasetProgress.percentage = 100
    exportDatasetProgress.message = task.message || '\u8bad\u7ec3\u6570\u636e\u96c6\u5bfc\u51fa\u5b8c\u6210'
    const data = task.result || task.results || {}
    const info = data.dataset_info || {}
    const split = data.summary?.split_config?.actual_source_group_counts || {}
    ElMessage.info(`版本 ${data.version_id || data.dataset_version_id || data.dataset_id} · 来源组 train/val/test ${split.train || 0}/${split.val || 0}/${split.test || 0}`)
    if (exportedDraftId && annotationDraftKind.value) {
      try {
        await deleteAnnotationDraft(currentDatasetId.value, annotationDraftKind.value)
      } catch (cleanupError) {
        console.warn('导出成功后清理标注草稿失败:', cleanupError)
      }
    }
    resetAnnotationDraftState()
    ElMessage.success(`\u5df2\u4fdd\u5b58\u4e3a\u8bad\u7ec3\u6570\u636e\u96c6：${info.name || data.dataset_id || currentDatasetName.value}`)
  } catch (error) {
    if (error !== 'cancel' && error !== 'close') {
      console.error('\u4fdd\u5b58\u8bad\u7ec3\u6570\u636e\u96c6\u5931\u8d25:', error)
      ElMessage.error(error?.message || '\u4fdd\u5b58\u8bad\u7ec3\u6570\u636e\u96c6\u5931\u8d25')
    }
  } finally {
    isCheckingExport.value = false
    isExportingDataset.value = false
    exportDatasetProgress.percentage = 0
    exportDatasetProgress.message = ''
  }
}
const addClass = () => {
  classForm.classes.push('')
  classForm.labelsZh.push('')
  classForm.ids.push(null)
}

const removeClass = (index) => {
  if (classForm.classes.length > 1) {
    classForm.classes.splice(index, 1)
    classForm.labelsZh.splice(index, 1)
    classForm.ids.splice(index, 1)
  }
}

const openExportSamplePreview = async (sample) => {
  const popup = window.open('', '_blank')
  if (!popup) {
    ElMessage.warning('浏览器阻止了预览窗口，请允许弹出窗口后重试')
    return
  }
  try {
    const rawUrl = sample?.thumbnail_url
    if (!rawUrl) throw new Error('预览链接不可用')
    const response = await authenticatedFetch(buildImageAssetUrl(rawUrl, sample.relative_path), {
      method: 'GET',
      cache: 'no-store'
    })
    if (!response.ok) throw new Error(`预览图片加载失败（HTTP ${response.status}）`)
    const previewUrl = URL.createObjectURL(await response.blob())
    popup.location.href = previewUrl
    window.setTimeout(() => URL.revokeObjectURL(previewUrl), 60_000)
  } catch (error) {
    popup.close()
    ElMessage.error(error?.message || '预览图片加载失败')
  }
}

const saveClasses = async () => {
  const validation = validateAnnotationClassNames(classForm.classes)
  if (!validation.valid) {
    ElMessage.warning(validation.message)
    return
  }

  const previousClasses = [...classList.value]
  const previousInactiveIds = [...inactiveClassIds.value]
  const keptIds = new Set(classForm.ids.filter(Number.isInteger))
  const retiredIds = previousClasses
    .map((_, index) => index)
    .filter(index => !previousInactiveIds.includes(index) && !keptIds.has(index))
  const nextClasses = [...previousClasses]
  classForm.classes.forEach((name, index) => {
    const id = classForm.ids[index]
    if (Number.isInteger(id)) nextClasses[id] = name.trim()
    else nextClasses.push(name.trim())
  })
  if (retiredIds.length && previousClasses.some((name, index) => nextClasses[index] !== name)) {
    ElMessage.warning('删除类别时请单独保存；英文训练名称需保持不变')
    return
  }
  if (new Set(nextClasses).size !== nextClasses.length) {
    ElMessage.warning('类别英文名称不能重复')
    return
  }
  const nextLabelsZh = { ...classLabelsZh.value }
  classForm.classes.forEach((name, index) => {
    const label = String(classForm.labelsZh[index] || '').trim()
    if (label) nextLabelsZh[name.trim()] = label
    else delete nextLabelsZh[name.trim()]
  })
  if (Object.values(nextLabelsZh).some(label => label.length > 80)) {
    ElMessage.warning('中文名称不能超过 80 字')
    return
  }
  const nextBindings = annotationClassBindings.value.map(binding => ({
    ...binding, name: nextClasses[binding.page_index]
  }))
  if (annotationDraftSaving.value || ((annotationDraftMetadataDirty.value || annotationDraftDirtyPaths.size > 0)
      && !await flushAnnotationDraft({ force: true }))) {
    ElMessage.warning('请等待当前标注草稿保存完成后重试')
    return
  }
  try {
    let response
    if (annotationClassScope.value === 'model_session') {
      const inactive = [...new Set([...previousInactiveIds, ...retiredIds])].sort((a, b) => a - b)
      applyAnnotationClasses(nextClasses, nextBindings, nextLabelsZh, inactive)
      markDraftMetadataDirty()
      if (!await flushAnnotationDraft({ force: true })) return
      response = { classes: nextClasses, bindings: nextBindings, labels_zh: nextLabelsZh, inactive_class_ids: inactive }
    } else {
      response = await updateDatasetAnnotationClasses(
        currentDatasetId.value, nextClasses, 'user', nextBindings, nextLabelsZh,
        retiredIds.length ? previousClasses : null, retiredIds,
        retiredIds.length ? previousInactiveIds : null
      )
    }
    applyAnnotationClasses(response.classes || nextClasses, response.bindings || nextBindings,
      response.labels_zh || nextLabelsZh, response.inactive_class_ids || previousInactiveIds)
    if (retiredIds.length) {
      const filtered = annotations => (annotations || []).filter(item =>
        !retiredIds.includes(Number(item?.class ?? item?.page_class_id)))
      imageList.value.forEach(image => { image.annotations = filtered(image.annotations) })
      for (const entry of annotationCache.values()) entry.annotations = filtered(entry.annotations)
      for (const entry of aiBatchResultCache.values()) entry.annotations = filtered(entry.annotations)
      historyByImage.clear()
      datasetClearAllBaseline.clear()
      datasetClearAllRestoredKeys.clear()
      historyRevision.value += 1
      selectedAnnotation.value = -1
      if (annotationDraftId.value) await loadAnnotationDraftMetadata()
      drawCanvas()
      ElMessage.success(annotationClassScope.value === 'model_session'
        ? '已在本次标注中停用类别，其他标注会话不受影响'
        : `已删除 ${response.deleted_annotation_count || 0} 条标注，原训练类别编号已保留`)
    }
    markDraftMetadataDirty()
  } catch (error) {
    ElMessage.error(error?.response?.data?.detail || '保存类别失败')
    return
  }
  showClassDialog.value = false
}

const initializeDatasetFromRoute = () => {
  const datasetId = String(route.query.datasetId || '').trim()
  const datasetName = String(route.query.datasetName || '').trim()
  const datasetRole = String(route.query.datasetRole || 'original').trim().toLowerCase()
  const routeModel = String(route.query.model || '').trim()
  const routeModelName = String(route.query.modelName || '').trim()
  const routeGeometry = String(route.query.annotationGeometry || '').trim().toLowerCase()
  const routeDraftKind = String(route.query.draftKind || '').trim().toLowerCase()

  annotationDraftKind.value = ['multimodal', 'yolo'].includes(routeDraftKind)
    ? routeDraftKind
    : (modelSupportsTextPrompts({}, routeModel) ? 'multimodal' : 'yolo')
  annotationDraftId.value = String(route.query.draftId || '').trim()
  annotationDraftRevision.value = Number(route.query.draftRevision) || 0

  currentDatasetRole.value = datasetRole === 'training' ? 'training' : 'original'
  if (routeModel) {
    annotationSessionModel.value = routeModel
    selectedAnnotationModel.value = routeModelName || routeModel
    aiAssistConfig.model = routeModel
  }
  if (routeGeometry === 'polygon' || routeGeometry === 'rect') {
    annotationGeometry.value = routeGeometry
  } else if (routeModel) {
    annotationGeometry.value = inferModelGeometry({ path: routeModel, name: routeModelName })
  }
  currentTool.value = annotationGeometry.value === 'polygon' ? 'polygon' : 'rect'

  if (!datasetId) return

  currentDatasetId.value = datasetId
  currentDatasetName.value = datasetName || datasetId
  currentDatasetPath.value = ''
  selectedImagePath.value = datasetId ? (datasetName || datasetId) : ''

  const selectedDataset = findDatasetOption(datasetId)
  if (selectedDataset) {
    applyDatasetOption(selectedDataset)
  }
}

const loadAnnotationDraftMetadata = async () => {
  if (!currentDatasetId.value || currentDatasetRole.value !== 'original' || !annotationDraftKind.value || !annotationDraftId.value) return
  try {
    const response = await getAnnotationDraft(currentDatasetId.value, annotationDraftKind.value, annotationDraftId.value)
    annotationDraftId.value = response.id || annotationDraftId.value
    annotationSessionDatasetId.value = response.dataset_id || currentDatasetId.value
    annotationDraftRevision.value = Number(response.revision) || 0
    annotationDraftSourceVersionId.value = response.source_version_id || ''
    annotationDraftTemporaryDatasetId.value = response.temporary_dataset_id || ''
    annotationDraftResumeBatch.value = response.annotation_batch?.synced ? null : response.annotation_batch || null
    annotationSessionModel.value = response.model_path || annotationSessionModel.value
    selectedAnnotationModel.value = response.model_name || response.model_path || selectedAnnotationModel.value
    aiAssistConfig.model = response.model_path || aiAssistConfig.model
    if (response.annotation_geometry === 'polygon' || response.annotation_geometry === 'rect') {
      annotationGeometry.value = response.annotation_geometry
      currentTool.value = annotationGeometry.value === 'polygon' ? 'polygon' : 'rect'
    }
    applyDraftClassContext(response)
  } catch (error) {
    annotationDraftWarning.value = error?.response?.data?.detail || '临时标注已失效，无法恢复原模型；请重新开始标注。'
    annotationDraftConflict.value = true
  }
}

// 键盘事件处理
const handleKeyDown = (event) => {
  if (!currentImage.value) return
  
  // Ctrl+Z 撤销
  if (event.ctrlKey && event.key === 'z') {
    event.preventDefault()
    undo()
    return
  }
  
  // Ctrl+Y 重做
  if (event.ctrlKey && event.key === 'y') {
    event.preventDefault()
    redo()
    return
  }
  
  switch (event.key) {
    case 'a':
    case 'A':
      if (currentImageIndex.value > 0) {
        selectImage(currentImageIndex.value - 1)
      }
      break
    case 'd':
    case 'D':
      if (currentImageIndex.value < imageList.value.length - 1) {
        selectImage(currentImageIndex.value + 1)
      }
      break
    case 'Delete':
      if (selectedAnnotation.value >= 0) {
        deleteAnnotation(selectedAnnotation.value)
      }
      break
    case 'Escape':
      // 取消多边形绘制
      if (isDrawingPolygon.value) {
        cancelPolygon()
      }
      break
    default:
      // 数字键选择类别
      const num = parseInt(event.key)
      if (num >= 1 && num <= classList.value.length && !inactiveClassIds.value.includes(num - 1)) {
        currentClass.value = num - 1
      }
      break
  }
}

const handleDraftBeforeUnload = (event) => {
  if (annotationDraftPending.value || annotationDraftSaving.value) {
    event.preventDefault()
    event.returnValue = ''
    void flushAnnotationDraft({ keepalive: true, force: true })
  }
}

onBeforeRouteLeave(async () => {
  const saved = await flushAnnotationDraft({ force: true })
  if (!saved) ElMessage.warning(annotationDraftWarning.value || '临时标注尚未完整保存，请稍后重试')
  return saved
})

// 生命周期
onMounted(() => {
  window.addEventListener('beforeunload', handleDraftBeforeUnload)
  document.addEventListener('keydown', handleKeyDown)
  annotationDraftVisibilityHandler = () => {
    if (document.visibilityState === 'hidden') void flushAnnotationDraft({ keepalive: true, force: true })
  }
  annotationDraftPagehideHandler = () => {
    void flushAnnotationDraft({ keepalive: true, force: true })
  }
  annotationDraftLogoutHandler = () => {
    void flushAnnotationDraft({ keepalive: true, force: true })
  }
  document.addEventListener('visibilitychange', annotationDraftVisibilityHandler)
  window.addEventListener('pagehide', annotationDraftPagehideHandler)
  window.addEventListener('deepvision:before-logout', annotationDraftLogoutHandler)
  initializeDatasetFromRoute()
  // 类别按原始数据集隔离；训练模型类别在模型列表加载后导入。
  loadAvailableModels()
  loadDatasetOptions().finally(async () => {
    if (currentDatasetId.value) {
      if (annotationDraftId.value) await loadAnnotationDraftMetadata()
      else await loadDatasetAnnotationClasses(currentDatasetId.value)
      await loadImagesFromSelectedPath()
      await restoreAnnotationDraftLocation()
    }
    startAnnotationDraftTimer()
    const batch = annotationDraftResumeBatch.value
    if (batch?.task_id && annotationViewActive) {
      isBatchAIAssisting.value = true
      try { await monitorBackendBatchTask(batch.task_id) }
      catch (error) { annotationDraftWarning.value = error?.message || '恢复 AI 标注结果失败，可重新打开草稿重试' }
      finally { isBatchAIAssisting.value = false }
    }
  })
})

onUnmounted(() => {
  annotationViewActive = false
  window.removeEventListener('beforeunload', handleDraftBeforeUnload)
  if (annotationDraftDebounceTimer) window.clearTimeout(annotationDraftDebounceTimer)
  void flushAnnotationDraft({ keepalive: true, force: true })
  document.removeEventListener('keydown', handleKeyDown)
  if (annotationDraftVisibilityHandler) document.removeEventListener('visibilitychange', annotationDraftVisibilityHandler)
  if (annotationDraftPagehideHandler) window.removeEventListener('pagehide', annotationDraftPagehideHandler)
  if (annotationDraftLogoutHandler) window.removeEventListener('deepvision:before-logout', annotationDraftLogoutHandler)
  stopAnnotationDraftTimer()
  pendingRequestControllers.forEach(controller => controller.abort())
  pendingRequestControllers.clear()
  revokeThumbnailObjectUrls()
})
</script>

<style scoped>
.annotation-page {
  height: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.floating-ai-batch-button {
  position: fixed;
  top: 78px;
  right: 24px;
  z-index: 9999;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
}

.toolbar {
  background: white;
  border-bottom: 1px solid #e4e7ed;
  padding: 10px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
}

.toolbar-left,
.toolbar-center,
.toolbar-right,
.dataset-context-group {
  display: flex;
  align-items: center;
  gap: 10px;
}

.annotation-draft-warning {
  flex: none;
  border-radius: 0;
}

.annotation-draft-saved,
.annotation-draft-saving {
  color: #67c23a;
  font-size: 12px;
  white-space: nowrap;
}

.annotation-draft-saving {
  color: #909399;
}

.path-label {
  color: #606266;
  font-size: 14px;
  white-space: nowrap;
}

.dataset-path-select {
  width: 216px;
}

.dataset-name-display {
  max-width: 320px;
  overflow: hidden;
  color: #303133;
  font-size: 14px;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ai-batch-progress {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 260px;
  color: #606266;
  font-size: 13px;
}

.ai-model-binding-hint {
  display: block;
  margin-top: 5px;
  color: #909399;
  font-size: 12px;
  line-height: 1.4;
}

.main-content {
  flex: 1;
  min-height: 0;
  display: flex;
  overflow: hidden;
}

.sidebar {
  width: 250px;
  background: #f5f7fa;
  border-right: 1px solid #e4e7ed;
  display: flex;
  flex-direction: column;
}

.sidebar-header {
  padding: 15px;
  border-bottom: 1px solid #e4e7ed;
  background: white;
}

.sidebar-header h3 {
  margin: 0;
  color: #303133;
}

.image-source {
  margin-top: 6px;
  color: #909399;
  font-size: 12px;
  word-break: break-all;
}

.image-list {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
}

.image-item {
  display: flex;
  align-items: center;
  padding: 10px;
  background: white;
  border-radius: 4px;
  margin-bottom: 8px;
  cursor: pointer;
  transition: all 0.3s;
  position: relative;
}

.image-item:hover {
  background: #ecf5ff;
}

.image-item:hover .delete-image-btn {
  opacity: 1;
}

.image-item.active {
  background: #409eff;
  color: white;
}

.delete-image-btn {
  position: absolute;
  top: 5px;
  right: 5px;
  opacity: 0;
  transition: opacity 0.3s;
  z-index: 10;
}

.image-item.active .delete-image-btn {
  opacity: 0.8;
}

.image-item.active .delete-image-btn:hover {
  opacity: 1;
}

.image-item img {
  width: 50px;
  height: 50px;
  object-fit: cover;
  border-radius: 4px;
  margin-right: 10px;
}

.image-info {
  flex: 1;
  min-width: 0;
  padding-right: 30px;
}

.image-name {
  font-weight: bold;
  margin-bottom: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 100%;
}

.annotation-count {
  font-size: 12px;
  opacity: 0.8;
}

.canvas-area {
  flex: 1;
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  background: #f0f0f0;
  overflow: auto;
  padding: 16px;
  box-sizing: border-box;
}

.canvas-controls {
  position: sticky;
  top: 0;
  z-index: 3;
  width: max-content;
  margin: 0 0 10px auto;
  padding: 4px;
  display: flex;
  align-items: center;
  gap: 4px;
  background: rgba(255, 255, 255, 0.96);
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
}

.zoom-level {
  min-width: 46px;
  padding: 0 4px;
  color: #606266;
  font-size: 12px;
  line-height: 28px;
  text-align: center;
  font-variant-numeric: tabular-nums;
}

.canvas-container {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  width: max-content;
  min-width: 100%;
  min-height: 100%;
  margin: auto;
}

.polygon-tip {
  width: 100%;
  max-width: 500px;
  order: 2;
}

.canvas-container canvas {
  border: 1px solid #ddd;
  background: white;
  cursor: crosshair;
  order: 1;
}

.empty-canvas {
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #909399;
  background: #fff;
  border: 1px dashed #dcdfe6;
  box-sizing: border-box;
}

.properties-panel {
  width: 300px;
  background: #f5f7fa;
  border-left: 1px solid #e4e7ed;
  display: flex;
  flex-direction: column;
}

.panel-header {
  padding: 15px;
  border-bottom: 1px solid #e4e7ed;
  background: white;
}

.panel-header h3 {
  margin: 0;
  color: #303133;
}

.selected-annotation-editor {
  padding: 10px;
  border-bottom: 1px solid #e4e7ed;
  background: #fff;
}

.selected-annotation-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
  color: #606266;
  font-size: 12px;
}

.selected-annotation-class {
  width: 100%;
}

.annotation-model-origin {
  margin-top: 6px;
  color: #909399;
  font-size: 12px;
  line-height: 1.4;
}

.annotations-list {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
}

.annotation-item {
  background: white;
  border-radius: 4px;
  padding: 10px;
  margin-bottom: 8px;
  cursor: pointer;
  transition: all 0.3s;
}

.annotation-item:hover {
  background: #ecf5ff;
}

.annotation-item.selected {
  background: #409eff;
  color: white;
}

.annotation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 5px;
}

.class-label {
  font-weight: bold;
}

.annotation-coords {
  font-size: 12px;
  opacity: 0.8;
}

.shortcuts-help {
  padding: 15px;
  border-top: 1px solid #e4e7ed;
  background: white;
}

.shortcuts-help h4 {
  margin: 0 0 10px 0;
  color: #303133;
}

.shortcut-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 12px;
}

.shortcut-item kbd {
  background: #f5f7fa;
  border: 1px solid #ddd;
  border-radius: 3px;
  padding: 2px 6px;
  font-family: monospace;
}

.tool-help {
  padding: 15px;
  border-top: 1px solid #e4e7ed;
  background: #fff9e6;
  margin-top: 10px;
}

.tool-help h4 {
  margin: 0 0 10px 0;
  color: #e6a23c;
  font-size: 14px;
}

.tool-help ul {
  margin: 0;
  padding-left: 20px;
}

.tool-help li {
  font-size: 12px;
  color: #606266;
  margin-bottom: 5px;
}

.class-editor .class-list {
  margin-bottom: 10px;
}

.class-editor .add-class-button {
  color: #fff !important;
}

.class-item {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}

@media (max-width: 1024px) {
  .sidebar {
    width: 200px;
  }
  
  .properties-panel {
    width: 250px;
  }
}

@media (max-width: 768px) {
  .toolbar {
    padding: 8px 12px;
    align-items: flex-start;
  }

  .toolbar-left,
  .toolbar-center,
  .toolbar-right,
  .dataset-context-group {
    width: 100%;
    flex-wrap: wrap;
  }

  .toolbar-center :deep(.el-select) {
    flex: 1 1 140px;
  }

  .dataset-path-select {
    flex: 1 1 180px;
  }

  .main-content {
    flex-direction: row;
    overflow: hidden;
  }
  
  .sidebar,
  .properties-panel {
    width: clamp(88px, 27vw, 132px);
    height: auto;
    min-height: 0;
    flex: 0 0 clamp(88px, 27vw, 132px);
  }

  .sidebar-header,
  .panel-header {
    padding: 8px;
  }

  .sidebar-header h3,
  .panel-header h3 {
    font-size: 12px;
  }

  .image-source {
    display: none;
  }

  .image-list,
  .annotations-list {
    padding: 6px;
  }

  .shortcuts-help {
    padding: 8px;
  }

  .shortcut-item {
    margin-bottom: 4px;
    font-size: 11px;
  }

  .sidebar {
    border-right: 1px solid #e4e7ed;
    border-bottom: 0;
  }

  .properties-panel {
    border-left: 1px solid #e4e7ed;
    border-top: 0;
  }
  
  .canvas-area {
    flex: 1 1 0;
    min-width: 0;
    min-height: 0;
    height: auto;
    padding: 8px;
  }
}

/* 数据增强对话框样式 */
.augmentation-config {
  padding: 20px 0;
}

.augmentation-option-group {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.augmentation-help {
  width: 100%;
  margin-top: 6px;
  color: #909399;
  font-size: 12px;
  line-height: 1.5;
}

.augmentation-preview {
  margin-top: 20px;
  padding: 15px;
  background: #f5f7fa;
  border-radius: 8px;
}

.augmentation-preview h4 {
  margin: 0 0 15px 0;
  color: #303133;
  font-size: 16px;
}

.preview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
}

.preview-item {
  text-align: center;
}

.preview-item img {
  max-width: 100%;
  max-height: 120px;
  border-radius: 4px;
  border: 2px solid #e4e7ed;
  object-fit: contain;
}

.el-form-item {
  margin-bottom: 18px;
}

.el-form-item__label {
  font-weight: 500;
}

/* 视频帧提取对话框样式 */
.video-frame-config {
  padding: 10px 0;
}

.video-info-display {
  background: #f5f7fa;
  padding: 15px;
  border-radius: 8px;
  line-height: 1.8;
}

.video-info-display p {
  margin: 5px 0;
  color: #606266;
}

.video-info-display strong {
  color: #303133;
  margin-right: 8px;
}

/* AI辅助标注对话框样式 */
.ai-assist-config {
  padding: 10px 0;
}

.image-list-pagination {
  padding: 10px 8px 14px;
  text-align: center;
}

.image-list-pagination :deep(.el-pagination) {
  justify-content: center;
}

.load-more-images .el-button + .el-button {
  margin-left: 0;
  margin-top: 8px;
}

.export-quality-summary { margin-top: 12px; padding: 10px; background: #f5f7fa; }
.export-quality-metrics, .export-quality-samples { display: flex; flex-wrap: wrap; gap: 8px 16px; }
.export-policy-hint { color: #606266; margin-left: 8px; font-size: 12px; }
</style>


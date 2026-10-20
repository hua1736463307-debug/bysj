// Registry declarations outrank legacy weight basenames. Ignore experiment names.
const weightNames = (model, reference = '') => [
  model.base_model, model.baseModel, model.lineage_root_model,
  model.parent_model, reference, model.path, model.name
].map(value => String(value || '').replaceAll('\\', '/').split('/').pop().toLowerCase())

export const modelSupportsTextPrompts = (model = {}, reference = '') => {
  const family = String(model.model_family || model.modelFamily || '').toLowerCase().trim()
  if (['yolo', 'yoloe'].includes(family)) return family === 'yoloe'
  const profile = String(model.training_profile || model.trainingProfile || '').toLowerCase().trim()
  if (['standard_yolo', 'yoloe_pe_segment'].includes(profile)) return profile === 'yoloe_pe_segment'
  const capability = model.supports_text_prompts ?? model.supportsTextPrompts
  if (typeof capability === 'boolean') return capability
  return weightNames(model, reference).some(name => /^yoloe(?:[-_.]|$)/.test(name))
}

export const modelAnnotationGeometry = (model = {}, reference = '') => {
  const geometry = String(model.geometry || '').toLowerCase().trim()
  if (['rect', 'bbox'].includes(geometry)) return 'rect'
  if (['polygon', 'segmentation', 'segment'].includes(geometry)) return 'polygon'
  const task = String(model.task || '').toLowerCase().trim()
  if (['detect', 'segment'].includes(task)) return task === 'segment' ? 'polygon' : 'rect'
  if (typeof model.supports_segmentation === 'boolean') return model.supports_segmentation ? 'polygon' : 'rect'
  return weightNames(model, reference).some(name => /(?:^|[-_.])seg(?:ment)?(?:[-_.]|$)/.test(name))
    ? 'polygon' : 'rect'
}

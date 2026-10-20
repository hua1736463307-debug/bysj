const normalizeClassNames = (classNames) => {
  if (!Array.isArray(classNames)) return []
  return classNames.map(value => String(value ?? '').trim())
}

/**
 * Return an annotation with a new page/dataset class while preserving geometry
 * and model-origin fields. The input annotation is never mutated.
 */
export const relabelAnnotation = (annotation, nextClassIndex, classNames) => {
  if (!annotation || typeof annotation !== 'object') {
    throw new TypeError('annotation must be an object')
  }

  const names = normalizeClassNames(classNames)
  const classIndex = Number(nextClassIndex)
  if (!Number.isInteger(classIndex) || classIndex < 0 || classIndex >= names.length) {
    throw new RangeError('nextClassIndex is outside classNames')
  }
  if (!names[classIndex]) {
    throw new Error('target class name cannot be empty')
  }

  const className = names[classIndex]
  return {
    ...annotation,
    class: classIndex,
    class_name: className,
    page_class_id: classIndex,
    page_class_name: className
  }
}

/**
 * Validate class-editor input without silently dropping or reindexing values.
 */
export const validateAnnotationClassNames = (classNames) => {
  if (!Array.isArray(classNames)) {
    return { valid: false, classes: [], message: '类别列表格式无效' }
  }

  const classes = normalizeClassNames(classNames)
  if (classes.length === 0 || classes.some(name => !name)) {
    return { valid: false, classes, message: '类别名称不能为空' }
  }

  const seen = new Set()
  for (const name of classes) {
    const key = name.toLocaleLowerCase()
    if (seen.has(key)) {
      return { valid: false, classes, message: `类别名称重复：${name}` }
    }
    seen.add(key)
  }

  return { valid: true, classes, message: '' }
}


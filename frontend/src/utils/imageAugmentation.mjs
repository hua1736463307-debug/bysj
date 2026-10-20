export const ROTATION_OPTIONS = [30, 60, 90, 135, 180, 225, 270, 315]

export const MIRROR_OPTIONS = [
  { value: 'none', label: '不镜像' },
  { value: 'horizontal', label: '水平镜像' },
  { value: 'vertical', label: '垂直镜像' }
]

export const BRIGHTNESS_OPTIONS = [
  { value: 0.8, label: '80%' },
  { value: 0.6, label: '60%' },
  { value: 0.4, label: '40%' }
]

export const GRAYSCALE_OPTIONS = [
  { value: 0.3, label: '30%' },
  { value: 0.6, label: '60%' },
  { value: 1, label: '100%' }
]

export const CONTRAST_OPTIONS = [
  { value: 0.6, label: '60%' },
  { value: 1, label: '100%' },
  { value: 1.4, label: '140%' }
]

export const buildAugmentationOperations = (config = {}) => {
  const operations = []

  ;(config.rotations || []).forEach(rotation => {
    operations.push({ type: 'rotation', value: Number(rotation) })
  })

  if (config.mirror && config.mirror !== 'none') {
    operations.push({ type: 'mirror', value: config.mirror })
  }

  ;(config.brightnessLevels || []).forEach(value => {
    operations.push({ type: 'brightness', value: Number(value) })
  })

  ;(config.grayscaleLevels || []).forEach(value => {
    operations.push({ type: 'grayscale', value: Number(value) })
  })

  ;(config.contrastLevels || []).forEach(value => {
    operations.push({ type: 'contrast', value: Number(value) })
  })

  return operations
}

export const getAugmentationOperationLabel = (operation) => {
  if (!operation) return '变换'
  if (operation.type === 'rotation') return `rotate-${operation.value}`
  if (operation.type === 'mirror') return `mirror-${operation.value}`
  if (operation.type === 'brightness') return `brightness-${Math.round(operation.value * 100)}`
  if (operation.type === 'grayscale') return `grayscale-${Math.round(operation.value * 100)}`
  if (operation.type === 'contrast') return `contrast-${Math.round(operation.value * 100)}`
  return 'transform'
}

export const getAugmentedImageDimensions = (width, height, rotation = 0) => {
  const sourceWidth = Math.max(1, Number(width) || 1)
  const sourceHeight = Math.max(1, Number(height) || 1)
  const radians = (Number(rotation) || 0) * Math.PI / 180
  const cosine = Math.abs(Math.cos(radians))
  const sine = Math.abs(Math.sin(radians))
  const rotatedWidth = cosine * sourceWidth + sine * sourceHeight
  const rotatedHeight = sine * sourceWidth + cosine * sourceHeight

  return {
    width: Math.max(1, Math.ceil(rotatedWidth - 1e-9)),
    height: Math.max(1, Math.ceil(rotatedHeight - 1e-9))
  }
}

const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value))

const transformPoint = (point, sourceWidth, sourceHeight, operation) => {
  const dimensions = getAugmentedImageDimensions(sourceWidth, sourceHeight, operation?.type === 'rotation' ? operation.value : 0)
  const radians = (operation?.type === 'rotation' ? Number(operation.value) : 0) * Math.PI / 180
  const cosine = Math.cos(radians)
  const sine = Math.sin(radians)
  const mirrorX = operation?.type === 'mirror' && operation.value === 'horizontal' ? -1 : 1
  const mirrorY = operation?.type === 'mirror' && operation.value === 'vertical' ? -1 : 1
  const localX = (Number(point?.x) - 0.5) * sourceWidth * mirrorX
  const localY = (Number(point?.y) - 0.5) * sourceHeight * mirrorY
  const rotatedX = localX * cosine - localY * sine
  const rotatedY = localX * sine + localY * cosine

  return {
    x: clamp((dimensions.width / 2 + rotatedX) / dimensions.width),
    y: clamp((dimensions.height / 2 + rotatedY) / dimensions.height)
  }
}

const getBoundingBox = (points) => {
  if (!points.length) return null
  const xs = points.map(point => point.x)
  const ys = points.map(point => point.y)
  const minX = clamp(Math.min(...xs))
  const minY = clamp(Math.min(...ys))
  const maxX = clamp(Math.max(...xs))
  const maxY = clamp(Math.max(...ys))
  if (maxX <= minX || maxY <= minY) return null
  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY
  }
}

export const transformAnnotations = (annotations, sourceWidth, sourceHeight, operation) => {
  const width = Math.max(1, Number(sourceWidth) || 1)
  const height = Math.max(1, Number(sourceHeight) || 1)

  return (annotations || []).map(annotation => {
    if (annotation?.type === 'polygon' && Array.isArray(annotation.points)) {
      const points = annotation.points.map(point => transformPoint(point, width, height, operation))
      const boundingBox = getBoundingBox(points)
      if (!boundingBox || points.length < 3) return null
      return { ...annotation, ...boundingBox, points }
    }

    const x = Number(annotation?.x) || 0
    const y = Number(annotation?.y) || 0
    const boxWidth = Number(annotation?.width) || 0
    const boxHeight = Number(annotation?.height) || 0
    const corners = [
      { x, y },
      { x: x + boxWidth, y },
      { x: x + boxWidth, y: y + boxHeight },
      { x, y: y + boxHeight }
    ].map(point => transformPoint(point, width, height, operation))
    const boundingBox = getBoundingBox(corners)
    return boundingBox ? { ...annotation, ...boundingBox } : null
  }).filter(Boolean)
}

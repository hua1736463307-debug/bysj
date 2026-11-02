const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value))

export function moveBox(box, dx, dy) {
  const result = { ...box, cx: clamp(box.cx + dx, box.w / 2, 1 - box.w / 2),
    cy: clamp(box.cy + dy, box.h / 2, 1 - box.h / 2) }
  if (box.polygon) result.polygon = box.polygon.map(([x, y]) => [x + result.cx - box.cx, y + result.cy - box.cy])
  return result
}

export function resizeBox(box, x, y) {
  const left = box.cx - box.w / 2
  const top = box.cy - box.h / 2
  const right = clamp(x, Math.min(left + .002, 1), 1)
  const bottom = clamp(y, Math.min(top + .002, 1), 1)
  const result = { ...box, cx: (left + right) / 2, cy: (top + bottom) / 2,
    w: right - left, h: bottom - top }
  if (box.polygon) result.polygon = box.polygon.map(([x, y]) => [
    left + (x - left) * result.w / box.w, top + (y - top) * result.h / box.h,
  ])
  return result
}

export function hitBox(boxes, point) {
  return boxes.findLastIndex(box => Math.abs(point.x - box.cx) <= box.w / 2 &&
    Math.abs(point.y - box.cy) <= box.h / 2)
}

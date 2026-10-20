export const poseTaskActive = task => ['pending', 'running', 'cancelling', 'interrupted'].includes(task?.status)
export function createPoseRequestId(crypto = globalThis.crypto) {
  if (typeof crypto?.randomUUID === 'function') return crypto.randomUUID()
  if (typeof crypto?.getRandomValues !== 'function') throw new Error('浏览器不支持生成训练请求 ID，请更换浏览器后重试')
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`
}
export const poseCanEvaluate = task => task?.type === 'pose_training' && task?.status === 'completed' && Number.isFinite(task?.result?.threshold)
export function mergePoseEvents(current, incoming) {
  return [...new Map([...current, ...incoming].map(event => [event.sequence, event])).values()]
    .sort((a, b) => a.sequence - b.sequence).slice(-200)
}
export const poseStatusText = status => ({ pending: '排队中', running: '进行中', cancelling: '正在取消', interrupted: '等待恢复', completed: '已完成', failed: '失败', cancelled: '已取消' }[status] || status)

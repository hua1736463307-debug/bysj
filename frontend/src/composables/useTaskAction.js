import { ref } from 'vue'
export function useTaskAction() {
  const error = ref(''), busy = ref(false)
  let active = 0
  async function act(fn) {
    ++active; busy.value = true; error.value = ''
    try { return await fn() }
    catch (e) { error.value = `${e.message}${e.requestId ? `（请求 ${e.requestId}）` : ''}` }
    finally { busy.value = --active > 0 }
  }
  return { error, busy, act }
}

import { api, clearAuth, getCurrentUser, saveAuth } from './client'

export async function login(username, password) {
  const data = await api.post('/auth/login', { username, password })
  saveAuth(data)
  return data
}

export async function fetchMe() {
  return api.get('/auth/me')
}

export async function logout() {
  try {
    await api.post('/auth/logout')
  } finally {
    clearAuth()
  }
}

export { getCurrentUser }

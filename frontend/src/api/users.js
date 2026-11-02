import { api } from './client'

export const listUsers = params => api.get('/auth/users', { params })
export const createUser = payload => api.post('/auth/users', payload)
export const updateUser = (id, payload) => api.patch(`/auth/users/${id}`, payload)
export const resetPassword = (id, password) => api.post(`/auth/users/${id}/reset-password`, { password })
export const deleteUser = id => api.delete(`/auth/users/${id}`)
export const listSessions = params => api.get('/auth/sessions', { params })
export const revokeSession = id => api.delete(`/auth/sessions/${id}`)

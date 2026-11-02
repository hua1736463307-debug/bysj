import { api } from './client'

export const listDepartments = () => api.get('/system/departments')
export const saveDepartment = (id, data) => id ? api.put('/system/departments/' + id, data) : api.post('/system/departments', data)
export const deleteDepartment = id => api.delete('/system/departments/' + id)
export const listRoles = () => api.get('/system/roles')
export const saveRole = (id, data) => id ? api.put('/system/roles/' + id, data) : api.post('/system/roles', data)
export const deleteRole = id => api.delete('/system/roles/' + id)
export const getProfile = id => api.get('/system/users/' + id + '/profile')
export const saveProfile = (id, data) => api.put('/system/users/' + id + '/profile', data)
export const listLogs = params => api.get('/system/logs', { params })
export const listMenus = () => api.get('/system/menus')
export const saveMenu = data => api.put('/system/menus', data)

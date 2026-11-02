const pathModules = {
  '/raw-dataset': 'datasets', '/train-dataset': 'datasets',
  '/dataset-annotation': 'annotations', '/model-training': 'training',
  '/model-hub': 'models', '/inference-service': 'inference',
}

export function canAccessPath(user, path) {
  if (user?.role === 'admin') return true
  if (path.startsWith('/system/')) return false
  const module = path.startsWith('/cxr/') ? 'cxr' : path.startsWith('/isic/') ? 'isic' : pathModules[path]
  return !module || !Array.isArray(user?.modules) || user.modules.includes(module)
}

export function firstAuthorizedPath(user) {
  return ['/cxr/dashboard', ...Object.keys(pathModules), '/isic/workspace']
    .find(path => canAccessPath(user, path)) || '/access-denied'
}

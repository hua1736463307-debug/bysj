export function saveBlob(response, filename) {
  const url = URL.createObjectURL(response.data), link = document.createElement('a')
  link.href = url; link.download = filename; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

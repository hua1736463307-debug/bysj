const { defineConfig } = require('@playwright/test')
const output = process.env.BYSJ_E2E_OUTPUT || `../artifacts/browser/run-${Date.now()}`
module.exports = defineConfig({
  testDir: './e2e', timeout: 420000, expect: { timeout: 20000 }, workers: 1,
  outputDir: `${output}/results`, reporter: [['list'], ['html', { outputFolder: `${output}/report`, open: 'never' }]],
  use: { baseURL: process.env.BYSJ_E2E_URL || 'http://127.0.0.1:18080',
    channel: process.env.BYSJ_BROWSER_CHANNEL || 'msedge', headless: true,
    viewport: { width: 1440, height: 1000 }, screenshot: 'only-on-failure', trace: 'retain-on-failure' }
})

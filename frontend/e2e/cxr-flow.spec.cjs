/**
 * CXR 五标签辅助分析：真实浏览器闭环。
 *
 * 需要显式隔离的本地部署与真实数据 fixture：
 *   BYSJ_E2E_URL / BYSJ_E2E_API / BYSJ_E2E_USERNAME / BYSJ_E2E_PASSWORD
 *   BYSJ_E2E_CXR_SOURCE_ID  已配置的只读来源 ID（管理员）
 *   BYSJ_E2E_CXR_CSV        来源 CSV 文件名（默认 train.csv）
 *
 * 凭据只进进程环境，trace 不上传。本文件在无部署环境下不执行（NOT_RUN）。
 */
const { test, expect } = require('@playwright/test')

const apiURL = process.env.BYSJ_E2E_API || 'http://127.0.0.1:18004/api/v1'

test('CXR flow: import, freeze, analyze, explain, review, report', async ({ page, request }, info) => {
  if (!['127.0.0.1', 'localhost'].includes(new URL(apiURL).hostname)) {
    throw new Error('E2E only accepts explicitly isolated local deployments')
  }
  const sourceId = process.env.BYSJ_E2E_CXR_SOURCE_ID
  if (!sourceId || !process.env.BYSJ_E2E_PASSWORD) {
    throw new Error('Set BYSJ_E2E_CXR_SOURCE_ID and BYSJ_E2E_PASSWORD for real acceptance')
  }
  const apiRequests = []
  page.on('request', (r) => {
    const url = new URL(r.url())
    if (url.pathname.startsWith('/api')) apiRequests.push({ method: r.method(), path: url.pathname })
  })

  await page.goto('/login')
  await page.getByPlaceholder('请输入用户名').fill(process.env.BYSJ_E2E_USERNAME || 'admin')
  await page.getByPlaceholder('请输入密码').fill(process.env.BYSJ_E2E_PASSWORD)
  await page.getByRole('button', { name: '登 录' }).click()
  await expect(page).toHaveURL(/cxr\/dashboard/)
  await expect(page.getByText('胸片五标签辅助分析').first()).toBeVisible()

  const token = await page.evaluate(() => localStorage.getItem('token'))
  const headers = { Authorization: `Bearer ${token}` }
  async function data(url) {
    const response = await request.get(apiURL + url, { headers })
    expect(response.ok(), url).toBeTruthy()
    return (await response.json()).data
  }
  async function waitJob(id) {
    await expect
      .poll(async () => (await data('/jobs/' + id)).status, { timeout: 600000, intervals: [1000, 3000] })
      .toMatch(/^(completed|failed|cancelled|interrupted)$/)
    const job = await data('/jobs/' + id)
    expect(job.status, JSON.stringify(job.error)).toBe('completed')
    return job
  }

  // 1. 导入（管理员 + 已配置来源）
  await page.getByRole('link', { name: '胸片数据' }).click()
  await page.getByLabel('来源 ID').fill(sourceId)
  await page.getByLabel('CSV 文件').fill(process.env.BYSJ_E2E_CXR_CSV || 'train.csv')
  const importResponse = page.waitForResponse(
    (r) => r.url().endsWith('/api/v1/cxr/datasets/import') && r.request().method() === 'POST'
  )
  await page.getByRole('button', { name: '提交导入任务' }).click()
  const importJob = (await (await importResponse).json()).data
  const imported = await waitJob(importJob.job_id)
  expect(imported.result.asset_count).toBeGreaterThan(0)

  // 2. 冻结四集
  const freezeResponse = page.waitForResponse(
    (r) => /\/api\/v1\/cxr\/datasets\/.+\/versions$/.test(new URL(r.url()).pathname)
  )
  await page.getByRole('button', { name: '冻结四集' }).first().click()
  const freezeJob = (await (await freezeResponse).json()).data
  const frozen = await waitJob(freezeJob.job_id)
  expect(frozen.result.state).toBe('frozen')
  await expect(page.getByText(/train/).first()).toBeVisible()

  // 3. 分析（受控资产）
  await page.getByRole('link', { name: '病例分析' }).click()
  await page.locator('select').first().selectOption({ index: 1 })
  await page.locator('select').nth(1).selectOption({ index: 1 })
  await page.locator('select').nth(2).selectOption({ index: 1 })
  const checkboxes = page.locator('table input[type=checkbox]')
  await checkboxes.nth(0).check()
  const analyzeResponse = page.waitForResponse(
    (r) => r.url().endsWith('/api/v1/cxr/analysis-batches') && r.request().method() === 'POST'
  )
  await page.getByRole('button', { name: /分析所选图/ }).click()
  const analyzeJob = (await (await analyzeResponse).json()).data
  await waitJob(analyzeJob.job_id)
  await expect(page.getByText('肺不张')).toBeVisible()
  await expect(page.getByText('胸腔积液')).toBeVisible()

  // 4. 解释（模型关注区域）
  const explainResponse = page.waitForResponse((r) => r.url().includes('/explanations'))
  await page.getByRole('button', { name: '关注区域' }).first().click()
  await waitJob((await (await explainResponse).json()).data.job_id)
  await expect(page.locator('img.thumb').first()).toBeVisible()

  // 5. 复核：填满五项后提交并批准
  await page.getByRole('button', { name: '发起人工复核' }).first().click()
  await page.getByRole('link', { name: '人工复核' }).click()
  await page.getByRole('button', { name: '打开' }).first().click()
  const selects = page.locator('table select')
  for (let index = 0; index < 5; index += 1) await selects.nth(index).selectOption('supported')
  await page.getByRole('button', { name: '保存草稿' }).click()
  await page.getByRole('button', { name: '提交复核' }).click()
  await page.getByRole('button', { name: '批准' }).click()
  await expect(page.getByText(/approved/)).toBeVisible()

  // 6. 报告：批准后签发最终报告，并导出只读快照
  const reportResponse = page.waitForResponse(
    (r) => r.url().endsWith('/api/v1/cxr/reports') && r.request().method() === 'POST'
  )
  await page.getByRole('button', { name: '生成报告' }).click()
  const report = (await (await reportResponse).json()).data
  expect(report.state).toBe('final')
  await expect(page).toHaveURL(/cxr\/reports/)
  await page.getByRole('button', { name: '查看' }).first().click()
  await expect(page.getByText(/最终报告|待复核研究草稿/)).toBeVisible()
  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: '导出 HTML' }).first().click()
  expect((await downloadPromise).suggestedFilename()).toMatch(/\.html$/)

  // 7. 跨页断言：所有 API 调用走同源 /api
  expect(apiRequests.every((item) => item.path.startsWith('/api'))).toBe(true)
  await info.attach('api-requests', { body: JSON.stringify(apiRequests, null, 2), contentType: 'application/json' })
})

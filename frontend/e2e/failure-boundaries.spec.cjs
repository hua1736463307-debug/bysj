const { test, expect } = require('@playwright/test')
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto')
const api=process.env.BYSJ_E2E_API || 'http://127.0.0.1:18000/api/v1'
async function login(page){await page.goto('/login');await page.getByPlaceholder('请输入用户名').fill(process.env.BYSJ_E2E_USERNAME||'admin');await page.getByPlaceholder('请输入密码').fill(process.env.BYSJ_E2E_PASSWORD);await page.getByRole('button',{name:'登 录'}).click();await expect(page).toHaveURL(/raw-dataset/);return {Authorization:'Bearer '+await page.evaluate(()=>localStorage.getItem('token'))}}
async function fixture(page,request){
  const headers=await login(page),root=process.env.BYSJ_E2E_DATA
  const names=fs.readdirSync(path.join(root,'images')).filter(n=>n.endsWith('.jpg')).sort().slice(0,3)
  const data=new FormData();data.set('name','failure-boundaries-'+Date.now());data.set('class_map',JSON.stringify({0:'RBC',1:'WBC',2:'Platelets'}))
  for(const name of names){data.append('files',new Blob([fs.readFileSync(path.join(root,'images',name))]),name);const label=name.replace('.jpg','.txt');data.append('files',new Blob([fs.readFileSync(path.join(root,'labels',label))]),label)}
  // Node fetch sends the same multipart contract as the browser client.
  const response=await fetch(api+'/datasets/import',{method:'POST',headers,body:data});expect(response.ok).toBeTruthy();const jobId=(await response.json()).data.job_id
  let job;await expect.poll(async()=>{job=(await (await request.get(api+'/jobs/'+jobId,{headers})).json()).data;return job.status},{timeout:60000}).toBe('completed')
  const result=job.result,asset=(await (await request.get(api+`/datasets/${result.dataset_id}/assets`,{headers})).json()).data.items[0]
  return {headers,result,asset,names,root}
}
async function draw(page){await expect(page.locator('.image-board img')).toBeVisible();const r=await page.locator('.image-board').boundingBox();await page.mouse.move(r.x+r.width*.05,r.y+r.height*.05);await page.mouse.down();await page.mouse.move(r.x+r.width*.15,r.y+r.height*.15);await page.mouse.up()}

test('real concurrent tab conflict retains local draft and blocks navigation',async({page,context,request},info)=>{
  const {result,asset}=await fixture(page,request),url=`/dataset-annotation?dataset=${result.dataset_id}&version=${result.version_id}`
  await page.goto(url);await expect(page.locator('.image-board img')).toBeVisible()
  const second=await context.newPage();await second.goto(url);await expect(second.locator('.image-board img')).toBeVisible()
  await draw(page);const firstSave=page.waitForResponse(r=>r.url().endsWith('/annotations/'+asset.asset_id)&&r.request().method()==='PUT');await page.getByRole('button',{name:'保存标注',exact:true}).click();expect((await firstSave).status()).toBe(200)
  await draw(second);const conflict=second.waitForResponse(r=>r.url().endsWith('/annotations/'+asset.asset_id)&&r.status()===409);await second.getByRole('button',{name:'保存标注',exact:true}).click();await conflict
  await expect(second.getByText('保存冲突，本地草稿已保留。',{exact:true})).toBeVisible()
  await second.getByRole('link',{name:'训练数据',exact:true}).click();await expect(second).toHaveURL(/dataset-annotation/)
  expect(await second.evaluate(id=>Object.keys(localStorage).some(key=>key.startsWith('bysj:draft:')&&key.endsWith(':'+id)),asset.asset_id)).toBeTruthy()
  await second.screenshot({path:info.outputPath('conflict.png'),fullPage:true});await second.close()
})

test('network save error keeps draft; 401 and foreign asset are rejected',async({page,request})=>{
  const {headers,result,asset}=await fixture(page,request)
  await page.goto(`/dataset-annotation?dataset=${result.dataset_id}&version=${result.version_id}`)
  await page.route('**/api/v1/annotations/*',route=>route.request().method()==='PUT'?route.abort('timedout'):route.continue())
  await draw(page);await page.getByRole('button',{name:'保存标注',exact:true}).click();await expect(page.getByRole('alert').filter({hasText:/网络异常|请求超时/})).toBeVisible();await expect(page.getByText('本地草稿待保存',{exact:false})).toBeVisible()
  await page.unroute('**/api/v1/annotations/*');await page.getByRole('button',{name:'保存标注',exact:true}).click()
  const noAuth=await request.get(api+`/assets/${asset.asset_id}/image`);expect(noAuth.status()).toBe(401)
  const username='foreign-'+Date.now(),password=crypto.randomBytes(18).toString('hex')
  expect((await request.post(api+'/auth/users',{headers,data:{username,password,role:'researcher'}})).ok()).toBeTruthy()
  const other=(await (await request.post(api+'/auth/login',{data:{username,password}})).json()).data
  expect((await request.get(api+`/assets/${asset.asset_id}/image`,{headers:{Authorization:'Bearer '+other.token}})).status()).toBe(404)
  await page.evaluate(()=>localStorage.setItem('token','expired-invalid'))
  await page.getByRole('link',{name:'原始数据',exact:true}).click();await expect(page).toHaveURL(/login/)
})

test('bad image becomes partial run without discarding successful item',async({page,request})=>{
  const headers=await login(page),root=process.env.BYSJ_E2E_DATA
  const models=(await (await request.get(api+'/models?page_size=200',{headers})).json()).data.items
  const model=models.find(m=>m.trusted_source==='platform');expect(model).toBeTruthy()
  await page.getByRole('link',{name:'推理验证',exact:true}).click();await page.getByLabel('推理模型').selectOption(model.model_id)
  const name=fs.readdirSync(path.join(root,'images')).find(n=>n.endsWith('.jpg'))
  await page.getByLabel('推理文件').setInputFiles([{name,mimeType:'image/jpeg',buffer:fs.readFileSync(path.join(root,'images',name))},{name:'broken.jpg',mimeType:'image/jpeg',buffer:Buffer.from('not an image')}])
  const response=page.waitForResponse(r=>r.url().endsWith('/inference/images')&&r.request().method()==='POST');await page.getByRole('button',{name:'上传图片推理'}).click();const runId=(await (await response).json()).data.run_id
  await expect.poll(async()=>(await (await request.get(api+'/inference/runs/'+runId,{headers})).json()).data.status,{timeout:60000}).toBe('partial')
  await expect(page.getByText('partial · 完成 1 / 2 · 失败 1',{exact:true})).toBeVisible()
})

test('training cancellation is reachable through the real UI',async({page,request},info)=>{
  const {headers,result}=await fixture(page,request)
  const frozen=(await (await request.post(api+`/datasets/${result.dataset_id}/versions`,{headers,data:{base_version_id:result.version_id,split_seed:42,ratios:[.8,.1,.1]}})).json()).data
  await expect.poll(async()=>(await (await request.get(api+'/jobs/'+frozen.job_id,{headers})).json()).data.status,{timeout:60000}).toBe('completed')
  const models=(await (await request.get(api+'/models?page_size=200',{headers})).json()).data.items
  await page.getByRole('link',{name:'模型训练',exact:true}).click()
  await page.getByLabel('数据集',{exact:true}).selectOption(result.dataset_id)
  await page.getByLabel('版本',{exact:true}).selectOption(result.version_id)
  await page.getByLabel('训练模型').selectOption(models.find(m=>m.trusted_source==='base').model_id)
  await page.getByLabel('Epochs',{exact:true}).fill('1000')
  await page.getByRole('button',{name:'训练预检',exact:true}).click()
  await expect(page.getByText('预检通过',{exact:true})).toBeVisible()
  const submitted=page.waitForResponse(r=>r.url().endsWith('/training/jobs')&&r.request().method()==='POST')
  await page.getByRole('button',{name:'提交训练',exact:true}).click()
  const jobId=(await (await submitted).json()).data.job_id
  try{
    await expect.poll(async()=>(await (await request.get(api+'/jobs/'+jobId,{headers})).json()).data.stage,{timeout:60000}).toBe('training')
    await page.getByRole('button',{name:'取消任务',exact:true}).click()
    await expect.poll(async()=>(await (await request.get(api+'/jobs/'+jobId,{headers})).json()).data.status,{timeout:60000}).toBe('cancelled')
    await expect(page.getByText('任务 cancelled · cancelled',{exact:true})).toBeVisible()
    fs.writeFileSync(info.outputPath('cancel-evidence.json'),JSON.stringify({job_id:jobId,status:'cancelled'},null,2))
  }finally{
    const state=(await (await request.get(api+'/jobs/'+jobId,{headers})).json()).data
    if(['pending','running','cancelling','interrupted'].includes(state.status)) await request.post(api+'/jobs/'+jobId+'/cancel',{headers})
  }
})

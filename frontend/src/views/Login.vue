<template>
  <div class="login-page">
    <div class="login-card">
      <div class="login-brand">
        <img src="../assets/logo.svg" alt="图像训推实验平台" />
        <div>
          <h1>图像训推实验平台</h1>
          <p>训练推理一体化管理系统</p>
        </div>
      </div>

      <el-form ref="loginFormRef" :model="form" :rules="rules" @keyup.enter="submitLogin">
        <el-form-item prop="username">
          <el-input v-model.trim="form.username" placeholder="请输入用户名" size="large" autocomplete="username">
            <template #prefix>
              <el-icon><User /></el-icon>
            </template>
          </el-input>
        </el-form-item>

        <el-form-item prop="password">
          <el-input
            v-model="form.password"
            type="password"
            placeholder="请输入密码"
            size="large"
            show-password
            autocomplete="current-password"
          >
            <template #prefix>
              <el-icon><Lock /></el-icon>
            </template>
          </el-input>
        </el-form-item>

        <el-checkbox v-model="rememberPassword" class="remember">记住密码</el-checkbox>

        <el-button type="primary" size="large" class="login-button" :loading="loading" @click="submitLogin">
          登 录
        </el-button>
      </el-form>

      <div class="login-footer">统一模型、数据、训练、推理和行为智能能力</div>
    </div>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { login, saveAuth } from '../utils/api'

const router = useRouter()
const loginFormRef = ref(null)
const loading = ref(false)
const rememberPassword = ref(false)

const form = reactive({
  username: '',
  password: ''
})

const rules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' }
  ]
}

const submitLogin = async () => {
  if (!loginFormRef.value) return

  try {
    await loginFormRef.value.validate()
  } catch {
    return
  }

  loading.value = true
  try {
    const response = await login(form.username, form.password)
    saveAuth(response)

    const user = response.user
    ElMessage.success('登录成功')

    router.replace('/')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  position: relative;
  box-sizing: border-box;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background:
    linear-gradient(rgba(29, 39, 54, 0.46), rgba(29, 39, 54, 0.46)),
    #07192a url('../assets/login-bg.png') center / cover no-repeat;
  padding: 24px;
}

.login-page::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(9, 20, 34, 0.08);
}

.login-card {
  position: relative;
  z-index: 1;
  width: min(400px, calc(100vw - 48px));
  padding: 28px 26px 24px;
  border-radius: 6px;
  background: #fff;
  box-shadow: 0 12px 40px rgba(0, 21, 41, 0.28);
}

.login-brand {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-bottom: 28px;
}

.login-brand img {
  width: 42px;
  height: 42px;
  border-radius: 4px;
  background: #1890ff;
  padding: 4px;
}

.login-brand h1 {
  margin: 0;
  color: #303133;
  font-size: 21px;
  font-weight: 500;
  letter-spacing: 0;
}

.login-brand p {
  margin: 6px 0 0;
  color: #909399;
  font-size: 12px;
}

.login-card :deep(.el-form-item) {
  margin-bottom: 22px;
}

.login-card :deep(.el-input__wrapper) {
  height: 40px;
  border-radius: 2px;
  box-shadow: 0 0 0 1px #dcdfe6 inset;
}

.login-card :deep(.el-input__prefix) {
  color: #c0c7d2;
  font-size: 18px;
}

.remember {
  margin: -4px 0 20px;
  color: #606266;
  font-size: 14px;
}

.login-button {
  width: 100%;
  height: 40px;
  border-radius: 2px;
  font-size: 14px;
  letter-spacing: 4px;
}

.login-footer {
  margin-top: 18px;
  color: #909399;
  font-size: 12px;
  text-align: center;
}

@media (max-width: 640px) {
  .login-page {
    justify-content: center;
    padding: 24px;
  }

  .login-card {
    padding: 28px 22px 22px;
  }

  .login-brand h1 {
    font-size: 20px;
  }
}
</style>

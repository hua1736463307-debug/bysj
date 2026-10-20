#!/usr/bin/env python3
"""
启动 YOLO API 服务。训练由独立 training-worker 容器管理。
"""

import uvicorn
import os
import torch

# Monkeypatch torch.load to fix the mmap bug in torch 2.11.0
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'mmap' not in kwargs or kwargs['mmap'] is None:
        kwargs['mmap'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from yolo_api import app

if __name__ == "__main__":
    # 创建必要目录
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("temp/training_status", exist_ok=True)
    os.makedirs("temp/training_logs", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)
    
    print("🚀 启动YOLO API服务...")
    print("📊 API文档: http://localhost:8000/docs")
    print("🔧 健康检查: http://localhost:8000/health")
    print("按 Ctrl+C 停止服务")
    
    try:
        # 启动API服务
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        print("\n🛑 收到中断信号，正在停止服务...")
    except Exception as e:
        print(f"\n❌ API服务出错: {e}")
    finally:
        print("🛑 API服务已停止")

"""
任务管理工具类
封装任务ID生成、状态管理等操作
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TaskManager:
    """任务管理器"""
    
    def __init__(self):
        """初始化任务管理器"""
        self.tasks = {}
    
    @staticmethod
    def generate_id() -> str:
        """
        生成唯一任务ID
        
        Returns:
            UUID字符串
        """
        return str(uuid.uuid4())
    
    def create_task(self, task_id: str, initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        创建新任务
        
        Args:
            task_id: 任务ID
            initial_data: 初始数据
            
        Returns:
            任务数据字典
        """
        task_data = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0.0,
            "message": "",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "results": None
        }
        
        if initial_data:
            task_data.update(initial_data)
        
        self.tasks[task_id] = task_data
        return task_data
    
    def update_task(self, task_id: str, status: str = None, progress: float = None,
                   message: str = None, results: Dict = None) -> bool:
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 任务状态
            progress: 进度(0-100)
            message: 状态消息
            results: 结果数据
            
        Returns:
            更新成功返回True,任务不存在返回False
        """
        if task_id not in self.tasks:
            logger.warning(f"任务不存在: {task_id}")
            return False
        
        task = self.tasks[task_id]
        task["updated_at"] = datetime.now().isoformat()
        
        if status is not None:
            task["status"] = status
        if progress is not None:
            task["progress"] = progress
        if message is not None:
            task["message"] = message
        if results is not None:
            task["results"] = results
        
        return True
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务数据字典,不存在返回None
        """
        return self.tasks.get(task_id)
    
    def delete_task(self, task_id: str) -> bool:
        """
        删除任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            删除成功返回True,任务不存在返回False
        """
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False
    
    def list_tasks(self, status_filter: str = None) -> list:
        """
        列出所有任务
        
        Args:
            status_filter: 状态过滤器(可选)
            
        Returns:
            任务列表
        """
        tasks = list(self.tasks.values())
        
        if status_filter:
            tasks = [t for t in tasks if t.get("status") == status_filter]
        
        return tasks
    
    def get_task_count(self, status: str = None) -> int:
        """
        获取任务数量
        
        Args:
            status: 状态过滤(可选)
            
        Returns:
            任务数量
        """
        if status:
            return len([t for t in self.tasks.values() if t.get("status") == status])
        return len(self.tasks)
    
    def clear_completed_tasks(self, keep_recent: int = 10):
        """
        清理已完成的任务,保留最近的N个
        
        Args:
            keep_recent: 保留最近的任务数量
        """
        completed_tasks = [
            (tid, t) for tid, t in self.tasks.items() 
            if t.get("status") in ["completed", "failed", "cancelled"]
        ]
        
        # 按更新时间排序
        completed_tasks.sort(key=lambda x: x[1].get("updated_at", ""), reverse=True)
        
        # 删除旧的任务
        for task_id, _ in completed_tasks[keep_recent:]:
            del self.tasks[task_id]
            logger.info(f"清理旧任务: {task_id}")

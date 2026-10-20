"""
文件处理工具类
封装文件上传、临时文件管理等操作
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, BinaryIO
import logging
from fastapi import UploadFile

logger = logging.getLogger(__name__)


class FileManager:
    """文件管理器"""
    
    def __init__(self, temp_dir: str = "temp"):
        """
        初始化文件管理器
        
        Args:
            temp_dir: 临时文件目录
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
    
    def save_upload_file(self, file: UploadFile, target_dir: str, 
                        filename: Optional[str] = None) -> str:
        """
        保存上传的文件
        
        Args:
            file: FastAPI上传文件对象
            target_dir: 目标目录
            filename: 自定义文件名(可选)
            
        Returns:
            保存后的文件路径
        """
        try:
            # 确保目标目录存在
            target_path = Path(target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            
            # 确定文件名
            if filename is None:
                filename = file.filename
            
            file_path = target_path / filename
            
            # 保存文件
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"文件已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            raise
    
    def create_temp_file(self, suffix: str = "", prefix: str = "tmp_") -> str:
        """
        创建临时文件
        
        Args:
            suffix: 文件后缀
            prefix: 文件前缀
            
        Returns:
            临时文件路径
        """
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=suffix, 
            prefix=prefix,
            dir=str(self.temp_dir)
        )
        temp_file.close()
        return temp_file.name
    
    def create_temp_dir(self, prefix: str = "tmp_") -> str:
        """
        创建临时目录
        
        Args:
            prefix: 目录前缀
            
        Returns:
            临时目录路径
        """
        temp_dir = tempfile.mkdtemp(prefix=prefix, dir=str(self.temp_dir))
        return temp_dir
    
    def cleanup_file(self, file_path: str) -> bool:
        """
        删除文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            删除成功返回True
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"文件已删除: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除文件失败: {e}")
            return False
    
    def cleanup_dir(self, dir_path: str) -> bool:
        """
        删除目录及其内容
        
        Args:
            dir_path: 目录路径
            
        Returns:
            删除成功返回True
        """
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.info(f"目录已删除: {dir_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除目录失败: {e}")
            return False
    
    def get_file_extension(self, filename: str) -> str:
        """
        获取文件扩展名
        
        Args:
            filename: 文件名
            
        Returns:
            扩展名(不含点)
        """
        return filename.split('.')[-1] if '.' in filename else ''
    
    def copy_file(self, src: str, dst: str, create_dirs: bool = True) -> bool:
        """
        复制文件
        
        Args:
            src: 源文件路径
            dst: 目标文件路径
            create_dirs: 是否创建目标目录
            
        Returns:
            复制成功返回True
        """
        try:
            if create_dirs:
                dst_dir = os.path.dirname(dst)
                os.makedirs(dst_dir, exist_ok=True)
            
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.error(f"复制文件失败: {e}")
            return False
    
    def copy_dir_contents(self, src_dir: str, dst_dir: str, 
                         pattern: str = "*") -> int:
        """
        复制目录内容
        
        Args:
            src_dir: 源目录
            dst_dir: 目标目录
            pattern: 文件匹配模式
            
        Returns:
            复制的文件数量
        """
        try:
            src_path = Path(src_dir)
            dst_path = Path(dst_dir)
            dst_path.mkdir(parents=True, exist_ok=True)
            
            count = 0
            for file_path in src_path.glob(pattern):
                if file_path.is_file():
                    shutil.copy2(str(file_path), str(dst_path / file_path.name))
                    count += 1
            
            logger.info(f"已复制 {count} 个文件从 {src_dir} 到 {dst_dir}")
            return count
        except Exception as e:
            logger.error(f"复制目录内容失败: {e}")
            return 0
    
    def get_file_size(self, file_path: str) -> int:
        """
        获取文件大小(字节)
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件大小
        """
        try:
            return os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"获取文件大小失败: {e}")
            return 0
    
    def ensure_dir(self, dir_path: str):
        """
        确保目录存在
        
        Args:
            dir_path: 目录路径
        """
        os.makedirs(dir_path, exist_ok=True)


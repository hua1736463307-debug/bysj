"""
结果处理工具类
封装推理结果的复制、格式化等操作
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ResultManager:
    """结果管理器"""
    
    def __init__(self, static_dir: str = "static/results", runs_dir: str = "runs"):
        """
        初始化结果管理器
        
        Args:
            static_dir: 静态文件目录
            runs_dir: 运行结果目录
        """
        self.runs_dir = Path("/app/runs")
        self.static_dir = Path("/app/static")
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.static_dir.mkdir(parents=True, exist_ok=True)
    
    def copy_prediction_results(self, prediction_id: str, 
                               result_type: str = "predict") -> Tuple[List[str], List[Dict]]:
        """
        复制预测结果到静态目录
        
        Args:
            prediction_id: 预测任务ID
            result_type: 结果类型(predict, batch等)
            
        Returns:
            (result_images, annotated_images)元组
        """
        # YOLO保存结果的路径: runs/predict/prediction_{id} 或 runs/predict/batch_{id}
        if result_type == "predict":
            result_path = self.runs_dir / result_type / f"prediction_{prediction_id}"
        else:
            result_path = self.runs_dir / result_type / f"{result_type}_{prediction_id}"
        
        if not result_path.exists():
            logger.warning(f"结果路径不存在: {result_path}")
            return [], []
        
        # 创建静态目录
        static_result_dir = self.static_dir / f"{result_type}_{prediction_id}"
        static_result_dir.mkdir(parents=True, exist_ok=True)
        
        result_images = []
        annotated_images = []
        
        # 复制图片文件
        for img_file in result_path.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                # 复制到静态目录
                dst_path = static_result_dir / img_file.name
                shutil.copy2(str(img_file), str(dst_path))
                
                # 生成URL
                url = f"/static/results/{result_type}_{prediction_id}/{img_file.name}"
                result_images.append(url)
                annotated_images.append({
                    "filename": img_file.name,
                    "url": url,
                    "path": str(dst_path)
                })
        
        logger.info(f"已复制 {len(result_images)} 个结果图片到静态目录")
        return result_images, annotated_images
    
    def cleanup_prediction_results(self, prediction_id: str, 
                                   result_type: str = "predict") -> bool:
        """
        清理预测结果
        
        Args:
            prediction_id: 预测任务ID
            result_type: 结果类型
            
        Returns:
            清理成功返回True
        """
        try:
            # 清理runs目录
            runs_path = self.runs_dir / result_type / f"{result_type}_{prediction_id}"
            if runs_path.exists():
                shutil.rmtree(runs_path)
            
            # 清理静态目录
            static_path = self.static_dir / f"{result_type}_{prediction_id}"
            if static_path.exists():
                shutil.rmtree(static_path)
            
            logger.info(f"已清理预测结果: {prediction_id}")
            return True
        except Exception as e:
            logger.error(f"清理预测结果失败: {e}")
            return False
    
    def list_static_results(self, result_type: str = None) -> List[Dict]:
        """
        列出静态目录中的结果
        
        Args:
            result_type: 结果类型过滤(可选)
            
        Returns:
            结果信息列表
        """
        results = []
        
        try:
            for result_dir in self.static_dir.iterdir():
                if not result_dir.is_dir():
                    continue
                
                dir_name = result_dir.name
                
                # 过滤结果类型
                if result_type and not dir_name.startswith(f"{result_type}_"):
                    continue
                
                # 统计文件
                image_count = len(list(result_dir.glob("*.jpg"))) + \
                             len(list(result_dir.glob("*.png"))) + \
                             len(list(result_dir.glob("*.jpeg")))
                
                # 计算大小
                total_size = sum(f.stat().st_size for f in result_dir.iterdir() if f.is_file())
                
                results.append({
                    "id": dir_name,
                    "path": str(result_dir),
                    "image_count": image_count,
                    "size": total_size,
                    "created_at": result_dir.stat().st_ctime
                })
            
            return results
        except Exception as e:
            logger.error(f"列出静态结果失败: {e}")
            return []
    
    def cleanup_old_results(self, keep_recent: int = 50) -> int:
        """
        清理旧的结果,保留最近的N个
        
        Args:
            keep_recent: 保留最近的结果数量
            
        Returns:
            清理的结果数量
        """
        try:
            results = self.list_static_results()
            
            # 按创建时间排序
            results.sort(key=lambda x: x['created_at'], reverse=True)
            
            # 删除旧结果
            cleaned = 0
            for result in results[keep_recent:]:
                result_path = Path(result['path'])
                if result_path.exists():
                    shutil.rmtree(result_path)
                    cleaned += 1
            
            logger.info(f"已清理 {cleaned} 个旧结果")
            return cleaned
        except Exception as e:
            logger.error(f"清理旧结果失败: {e}")
            return 0
    
    def copy_batch_result_images(self, batch_id: str):
        """
        复制批量推理生成的结果图到静态目录，并返回按原始结果图文件名索引的映射。
        """
        src_dir = self.runs_dir / "batch" / f"batch_{batch_id}"
        if not src_dir.exists():
            logger.warning(f"批量结果目录不存在: {src_dir}")
            return {}

        dst_dir = self.static_dir / "predictions" / f"batch_{batch_id}"
        dst_dir.mkdir(parents=True, exist_ok=True)

        image_map = {}
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for file_path in src_dir.iterdir():
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in image_extensions:
                continue

            # 正确写法：image_2.jpg -> image_2_annotated.jpg
            annotated_filename = f"{file_path.stem}_annotated{file_path.suffix}"

            dst_path = dst_dir / annotated_filename
            shutil.copy2(file_path, dst_path)

            # key 仍然用 YOLO 原始输出名 image_2.jpg，方便和 result.path 匹配
            image_map[file_path.name] = {
                "file_name": annotated_filename,
                "original_result_name": file_path.name,
                "url": f"/static/predictions/batch_{batch_id}/{annotated_filename}"
            }

        logger.info(f"批量结果图片复制完成: {len(image_map)} 张")
        return image_map
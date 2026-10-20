"""
数据集管理工具类
封装数据集上传、验证、结构修复等操作
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, datasets_dir: str = "datasets"):
        """
        初始化数据集管理器
        
        Args:
            datasets_dir: 数据集根目录
        """
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(exist_ok=True)
    
    def validate_structure(self, dataset_path: Path) -> bool:
        """
        验证数据集结构是否符合YOLO格式
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            验证通过返回True
        """
        required_dirs = ['train', 'val']
        for dir_name in required_dirs:
            images_dir = dataset_path / dir_name / 'images'
            labels_dir = dataset_path / dir_name / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                return False
        
        return True
    
    def fix_structure(self, dataset_path: Path, train_ratio: float = 0.8) -> Tuple[int, int]:
        """
        修复数据集结构,自动创建train/val目录并分割数据
        
        Args:
            dataset_path: 数据集路径
            train_ratio: 训练集比例
            
        Returns:
            (训练集数量, 验证集数量)元组
        """
        logger.info("检测到数据集缺少train/val目录结构,正在自动修复...")
        
        # 创建必要的目录
        train_path = dataset_path / 'train'
        val_path = dataset_path / 'val'
        test_path = dataset_path / 'test'
        
        for split_path in [train_path, val_path, test_path]:
            (split_path / 'images').mkdir(parents=True, exist_ok=True)
            (split_path / 'labels').mkdir(parents=True, exist_ok=True)
        
        # 收集所有图片和标注文件
        all_images = []
        
        for root, dirs, files in os.walk(dataset_path):
            # 跳过已创建的train/val/test目录
            if any(subdir in root for subdir in ['train', 'val', 'test']):
                continue
            
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = Path(root) / file_name
                    all_images.append(file_path)
        
        # 按比例分割数据
        random.shuffle(all_images)
        train_count = int(len(all_images) * train_ratio)
        train_images = all_images[:train_count]
        val_images = all_images[train_count:]
        
        # 移动训练集文件
        for img_path in train_images:
            self._move_image_and_label(img_path, train_path)
        
        # 移动验证集文件
        for img_path in val_images:
            self._move_image_and_label(img_path, val_path)
        
        logger.info(f"数据集结构修复完成:训练集{len(train_images)}张,验证集{len(val_images)}张")
        return len(train_images), len(val_images)
    
    def _move_image_and_label(self, img_path: Path, target_dir: Path):
        """
        移动图片及其对应的标注文件
        
        Args:
            img_path: 图片路径
            target_dir: 目标目录
        """
        img_name = img_path.name
        label_name = img_path.stem + '.txt'
        label_path = img_path.parent / label_name
        
        # 移动图片
        target_img = target_dir / 'images' / img_name
        if img_path.exists():
            shutil.move(str(img_path), str(target_img))
        
        # 移动对应的标注文件(如果存在)
        target_label = target_dir / 'labels' / label_name
        if label_path.exists():
            shutil.move(str(label_path), str(target_label))
    
    def fix_yaml_config(self, dataset_path: Path) -> bool:
        """
        修复dataset.yaml文件的路径配置
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            修复成功返回True
        """
        yaml_path = dataset_path / 'dataset.yaml'
        if not yaml_path.exists():
            return False
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_content = yaml.safe_load(f)
            
            # 修复path字段为绝对路径
            if 'path' in yaml_content:
                if yaml_content['path'] in ['.', '/path/to/your/dataset']:
                    yaml_content['path'] = str(dataset_path)
                    
                    # 写回修复后的配置
                    with open(yaml_path, 'w', encoding='utf-8') as f:
                        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
                    
                    logger.info(f"已修复dataset.yaml路径配置: {dataset_path}")
                    return True
        except Exception as e:
            logger.warning(f"修复dataset.yaml失败: {e}")
            return False
        
        return True
    
    def analyze_dataset(self, dataset_path: Path) -> Dict[str, int]:
        """
        分析数据集统计信息
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            包含统计信息的字典
        """
        image_count = 0
        label_count = 0
        dataset_size = 0
        
        # 检查是否是标准YOLO格式
        train_dir = dataset_path / 'train'
        val_dir = dataset_path / 'val'
        
        if train_dir.exists() and val_dir.exists():
            # 标准YOLO格式,分别统计train和val目录
            for subdir in ['train', 'val']:
                subdir_path = dataset_path / subdir
                if subdir_path.exists():
                    # 统计images目录
                    images_path = subdir_path / 'images'
                    if images_path.exists():
                        for img_file in images_path.iterdir():
                            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                image_count += 1
                                try:
                                    dataset_size += img_file.stat().st_size
                                except OSError:
                                    pass
                    
                    # 统计labels目录
                    labels_path = subdir_path / 'labels'
                    if labels_path.exists():
                        for label_file in labels_path.iterdir():
                            if label_file.suffix == '.txt' and not label_file.name.endswith('.cache'):
                                label_count += 1
                                try:
                                    dataset_size += label_file.stat().st_size
                                except OSError:
                                    pass
            
            # 统计其他文件(dataset.yaml, classes.txt等)
            for file_path in dataset_path.iterdir():
                if file_path.is_file():
                    try:
                        dataset_size += file_path.stat().st_size
                    except OSError:
                        pass
        else:
            # 非标准格式,使用递归扫描
            for root, dirs, files in os.walk(dataset_path):
                for file_name in files:
                    file_path = Path(root) / file_name
                    try:
                        dataset_size += file_path.stat().st_size
                        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            image_count += 1
                        elif file_path.suffix == '.txt' and file_name not in ['classes.txt'] and not file_name.endswith('.cache'):
                            label_count += 1
                    except OSError:
                        continue
        
        return {
            "image_count": image_count,
            "label_count": label_count,
            "size": dataset_size
        }
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """
        删除数据集
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            删除成功返回True
        """
        dataset_path = self.datasets_dir / dataset_id
        
        if not dataset_path.exists():
            logger.warning(f"数据集不存在: {dataset_id}")
            return False
        
        try:
            shutil.rmtree(dataset_path)
            logger.info(f"数据集已删除: {dataset_id}")
            return True
        except Exception as e:
            logger.error(f"删除数据集失败: {e}")
            return False
    
    def list_datasets(self) -> list:
        """
        列出所有数据集
        
        Returns:
            数据集ID列表
        """
        if not self.datasets_dir.exists():
            return []
        
        return [d.name for d in self.datasets_dir.iterdir() if d.is_dir()]
    
    def _metadata_path(self, dataset_path: Path) -> Path:
        """
        获取数据集元信息文件路径
        """
        return dataset_path / "metadata.json"

    def load_metadata(self, dataset_id: str) -> Optional[Dict]:
        """
        快速读取数据集 metadata，不扫描大文件目录
        """
        dataset_path = self.datasets_dir / dataset_id
        metadata_path = self._metadata_path(dataset_path)

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"读取 metadata 失败: {dataset_id}, error={e}")
            return None

    def save_metadata(self, dataset_id: str, metadata: Dict) -> bool:
        """
        保存数据集 metadata
        """
        dataset_path = self.datasets_dir / dataset_id
        metadata_path = self._metadata_path(dataset_path)

        try:
            dataset_path.mkdir(parents=True, exist_ok=True)

            metadata["id"] = metadata.get("id") or dataset_id
            metadata["name"] = metadata.get("name") or dataset_id
            metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")

            if "created_at" not in metadata:
                metadata["created_at"] = metadata["updated_at"]

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            logger.error(f"保存 metadata 失败: {dataset_id}, error={e}")
            return False

    def refresh_metadata(self, dataset_id: str) -> Optional[Dict]:
        """
        主动刷新某个数据集的统计信息。

        注意：这个方法会扫描数据集目录，适合在上传完成、导入完成、手动刷新时调用。
        不要在 GET /datasets 列表接口中频繁调用。
        """
        dataset_path = self.datasets_dir / dataset_id

        if not dataset_path.exists() or not dataset_path.is_dir():
            logger.warning(f"数据集不存在，无法刷新 metadata: {dataset_id}")
            return None

        stats = self.analyze_dataset(dataset_path)

        old_metadata = self.load_metadata(dataset_id) or {}

        metadata = {
            "id": dataset_id,
            "name": old_metadata.get("name", dataset_id),
            "image_count": stats.get("image_count", 0),
            "label_count": stats.get("label_count", 0),
            "size": stats.get("size", 0),
            "created_at": old_metadata.get(
                "created_at",
                datetime.fromtimestamp(dataset_path.stat().st_ctime).isoformat(timespec="seconds")
            ),
            "updated_at": datetime.now().isoformat(timespec="seconds")
        }

        self.save_metadata(dataset_id, metadata)
        return metadata

    def get_dataset_summary(self, dataset_id: str, auto_refresh: bool = False) -> Optional[Dict]:
        """
        获取单个数据集摘要信息。

        默认只读 metadata，不扫描目录。
        如果 auto_refresh=True，并且 metadata 不存在，才扫描一次。
        """
        dataset_path = self.datasets_dir / dataset_id

        if not dataset_path.exists() or not dataset_path.is_dir():
            return None

        metadata = self.load_metadata(dataset_id)

        if metadata is not None:
            return {
                "id": metadata.get("id", dataset_id),
                "name": metadata.get("name", dataset_id),
                "image_count": metadata.get("image_count", 0),
                "label_count": metadata.get("label_count", 0),
                "size": metadata.get("size", 0),
                "created_at": metadata.get("created_at"),
                "updated_at": metadata.get("updated_at")
            }

        if auto_refresh:
            return self.refresh_metadata(dataset_id)

        return {
            "id": dataset_id,
            "name": dataset_id,
            "image_count": 0,
            "label_count": 0,
            "size": 0,
            "created_at": datetime.fromtimestamp(dataset_path.stat().st_ctime).isoformat(timespec="seconds"),
            "updated_at": None
        }

    def list_dataset_summaries(
        self,
        page: int = 1,
        page_size: int = 20,
        auto_refresh: bool = False
    ) -> Dict:
        """
        分页获取数据集摘要列表。

        这个方法用于 GET /datasets。
        默认不扫描大目录，只读取 metadata。
        """
        if not self.datasets_dir.exists():
            return {
                "datasets": [],
                "total": 0,
                "page": page,
                "page_size": page_size
            }

        dataset_dirs = [
            d for d in self.datasets_dir.iterdir()
            if d.is_dir()
        ]

        dataset_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        total = len(dataset_dirs)

        start = max((page - 1) * page_size, 0)
        end = start + page_size

        page_dirs = dataset_dirs[start:end]

        datasets = []

        for dataset_dir in page_dirs:
            summary = self.get_dataset_summary(
                dataset_dir.name,
                auto_refresh=auto_refresh
            )

            if summary is not None:
                datasets.append(summary)

        return {
            "datasets": datasets,
            "total": total,
            "page": page,
            "page_size": page_size
        }


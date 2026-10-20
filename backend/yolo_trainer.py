"""
YOLO Model Training Script
使用 Ultralytics YOLO 进行模型训练的完整实现

功能包括:
- 数据集准备和验证
- 模型训练配置
- 训练过程监控
- 模型评估和导出
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("YOLO_CONFIG_DIR", "/app/.ultralytics")
import gc
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cleanup_torch_runtime():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"清理CUDA缓存失败: {e}")
    finally:
        gc.collect()


class YOLOTrainer:
    """YOLO模型训练器"""
    
    def __init__(self, model_name="yolov8n.pt", data_config=None, training_profile="standard_yolo"):
        """
        初始化YOLO训练器
        
        Args:
            model_name (str): 预训练模型名称 (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
            data_config (str): 数据集配置文件路径
        """
        self.model_name = model_name
        self.data_config = data_config
        self.training_profile = str(training_profile or "standard_yolo")
        self.model = None
        self.results = None
        
    def load_model(self, pretrained=True):
        """
        加载YOLO模型
        
        Args:
            pretrained (bool): 是否使用预训练权重
        """
        try:
            cleanup_torch_runtime()
            if pretrained:
                logger.info(f"准备加载预训练模型: {self.model_name}")
                if self.training_profile == "yoloe_pe_segment":
                    from ultralytics import YOLOE

                    self.model = YOLOE(self.model_name)
                else:
                    self.model = YOLO(self.model_name)
                logger.info(f"已加载预训练模型: {self.model_name}")
            else:
                # 加载模型架构但不加载权重
                model_yaml = self.model_name.replace('.pt', '.yaml')
                logger.info(f"准备加载模型架构: {model_yaml}")
                self.model = YOLO(model_yaml)
                logger.info(f"已加载模型架构: {model_yaml}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def create_dataset_config(self, dataset_path, train_split="train", val_split="val", 
                            test_split="test", class_names=None):
        """
        创建数据集配置文件
        
        Args:
            dataset_path (str): 数据集根目录路径
            train_split (str): 训练集子目录名
            val_split (str): 验证集子目录名
            test_split (str): 测试集子目录名
            class_names (list): 类别名称列表，None时使用单一默认类别
        """
        if class_names is None or len(class_names) == 0:
            class_names = ["object"]  # 最小默认类别
            logger.warning("未提供类别信息，使用默认类别: ['object']")
        
        dataset_config = {
            'path': str(Path(dataset_path).absolute()),
            'train': train_split,
            'val': val_split,
            'test': test_split,
            'nc': len(class_names),  # 类别数量
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        config_path = Path(dataset_path) / "dataset.yaml"
        # 确保目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        self.data_config = str(config_path)
        logger.info(f"数据集配置已保存至: {config_path}")
        return config_path
    
    def remap_dataset_classes(self, dataset_path, selected_class_names):
        """
        重映射数据集类别ID，仅保留选中的类别
        
        当用户只选择部分类别进行训练时，需要：
        1. 读取原始classes.txt，获取原始类别ID映射
        2. 创建新的类别ID映射（0-based连续编号）
        3. 遍历所有标注文件，重新映射类别ID或过滤掉未选中的类别
        
        Args:
            dataset_path (str): 数据集根目录路径
            selected_class_names (list): 用户选择的类别名称列表
        
        Returns:
            bool: 是否成功重映射
        """
        dataset_path = Path(dataset_path)
        classes_file = dataset_path / "classes.txt"
        
        if not classes_file.exists():
            logger.warning(f"未找到classes.txt文件: {classes_file}")
            return True  # 如果没有classes.txt，假设标注已经正确
        
        # 读取原始类别映射
        original_class_map = {}  # {class_id: class_name}
        with open(classes_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    class_id = int(parts[0].strip())
                    class_name = parts[1].strip()
                    original_class_map[class_id] = class_name
        
        # 如果选中的类别数量等于原始类别数量，且顺序一致，不需要重映射
        if len(selected_class_names) == len(original_class_map):
            # 检查是否按顺序选择了所有类别
            all_match = True
            for i, class_name in enumerate(selected_class_names):
                if i not in original_class_map or original_class_map[i] != class_name:
                    all_match = False
                    break
            if all_match:
                logger.info("选中的类别与原始类别完全一致，跳过重映射")
                return True
        
        # 创建新的类别ID映射（只包含选中的类别）
        old_to_new_id = {}  # {原始ID: 新ID}
        for new_id, class_name in enumerate(selected_class_names):
            # 找到原始ID
            for old_id, name in original_class_map.items():
                if name == class_name:
                    old_to_new_id[old_id] = new_id
                    break
        
        logger.info(f"类别ID映射: {old_to_new_id}")
        logger.info(f"从 {len(original_class_map)} 个类别映射到 {len(selected_class_names)} 个类别")
        
        # 遍历并重映射标注文件
        remapped_count = 0
        filtered_count = 0
        
        for split in ['train', 'val', 'test']:
            labels_dir = dataset_path / split / 'labels'
            if not labels_dir.exists():
                continue
            
            for label_file in labels_dir.glob('*.txt'):
                try:
                    # 读取原始标注
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    # 如果文件为空，跳过
                    if not lines:
                        continue
                    
                    new_lines = []
                    file_changed = False
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            old_class_id = int(parts[0])
                            
                            # 如果这个类别在选中范围内，进行映射
                            if old_class_id in old_to_new_id:
                                new_class_id = old_to_new_id[old_class_id]
                                if old_class_id != new_class_id:
                                    parts[0] = str(new_class_id)
                                    file_changed = True
                                new_lines.append(' '.join(parts) + '\n')
                                remapped_count += 1
                            else:
                                # 过滤掉未选中的类别
                                filtered_count += 1
                                file_changed = True
                    
                    # 只有当文件发生变化时才写回
                    if file_changed:
                        with open(label_file, 'w') as f:
                            f.writelines(new_lines)
                
                except Exception as e:
                    logger.error(f"处理标注文件失败 {label_file}: {e}")
                    return False
        
        logger.info(f"类别重映射完成: 处理 {remapped_count} 个标注，过滤 {filtered_count} 个未选中类别的标注")
        return True
    
    def validate_dataset_classes(self, dataset_path, config_path=None):
        """
        验证数据集类别配置是否与实际标注文件匹配
        
        Args:
            dataset_path (str): 数据集根目录路径
            
        Returns:
            (is_valid, issues): 验证结果和问题列表
        """
        issues = []
        
        try:
            dataset_path = Path(dataset_path)
            
            yaml_path = Path(config_path) if config_path else dataset_path / "dataset.yaml"
            if not yaml_path.exists():
                return False, ["dataset.yaml文件不存在"]
            
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            nc = config.get('nc', 0)
            if nc == 0:
                return False, ["dataset.yaml中未定义类别数量(nc)"]
            
            # 扫描所有标注文件，找出使用的类别ID
            label_dirs = []
            for split in ['train', 'val', 'test']:
                label_dir = dataset_path / split / 'labels'
                if label_dir.exists():
                    label_dirs.append(label_dir)
            
            if not label_dirs:
                return False, ["未找到标注文件目录(train/labels, val/labels, test/labels)"]
            
            # 收集所有使用的类别ID
            used_class_ids = set()
            invalid_files = []
            
            for label_dir in label_dirs:
                for label_file in label_dir.glob("*.txt"):
                    try:
                        with open(label_file, 'r') as f:
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if not line:
                                    continue
                                
                                parts = line.split()
                                if len(parts) >= 5:
                                    try:
                                        class_id = int(parts[0])
                                        used_class_ids.add(class_id)
                                        
                                        # 检查类别ID是否超出范围
                                        if class_id >= nc:
                                            invalid_files.append({
                                                'file': str(label_file.relative_to(dataset_path)),
                                                'line': line_num,
                                                'class_id': class_id,
                                                'max_allowed': nc - 1
                                            })
                                    except ValueError:
                                        pass  # 忽略无效的类别ID
                    except Exception as e:
                        logger.warning(f"读取标注文件失败: {label_file}, {e}")
            
            # 检查是否有类别ID超出范围
            max_class_id = max(used_class_ids) if used_class_ids else -1
            
            if max_class_id >= nc:
                issues.append(f"发现类别ID超出范围: 最大类别ID为{max_class_id}, 但配置的类别数量(nc)为{nc}")
                issues.append(f"允许的类别ID范围: 0-{nc-1}")
                issues.append(f"问题文件数量: {len(invalid_files)}")
                
                if invalid_files:
                    # 只显示前5个问题文件
                    for issue in invalid_files[:5]:
                        issues.append(
                            f"  - {issue['file']}:{issue['line']} 使用类别ID {issue['class_id']} "
                            f"(超出范围 0-{issue['max_allowed']})"
                        )
                    if len(invalid_files) > 5:
                        issues.append(f"  ... 还有 {len(invalid_files) - 5} 个文件存在同样问题")
                
                return False, issues
            
            logger.info(f"数据集类别验证通过: nc={nc}, 使用的类别ID={sorted(used_class_ids)}")
            return True, []
            
        except Exception as e:
            logger.error(f"验证数据集类别时出错: {e}")
            return False, [f"验证失败: {str(e)}"]
    
    def validate_dataset(self, dataset_path):
        """
        验证数据集结构
        
        Args:
            dataset_path (str): 数据集路径
        """
        dataset_path = Path(dataset_path)
        
        # 检查基本目录结构
        required_dirs = ['train', 'val']
        for dir_name in required_dirs:
            images_dir = dataset_path / dir_name / 'images'
            labels_dir = dataset_path / dir_name / 'labels'
            
            if not images_dir.exists():
                logger.warning(f"图片目录不存在: {images_dir}")
            if not labels_dir.exists():
                logger.warning(f"标签目录不存在: {labels_dir}")
        
        logger.info("数据集结构验证完成")
    
    def keep_only_best_pt(self, save_dir):
        """
        训练完成后只保留 weights/best.pt，删除 last.pt、epoch*.pt 等其他 .pt 文件。
        """
        try:
            save_dir = Path(save_dir)
            weights_dir = save_dir if save_dir.name == "weights" else save_dir / "weights"

            if not weights_dir.exists():
                logger.warning(f"weights 目录不存在，跳过模型清理: {weights_dir}")
                return

            best_pt = weights_dir / "best.pt"
            if not best_pt.exists():
                logger.warning(f"best.pt 不存在，跳过模型清理，避免误删: {best_pt}")
                return

            removed_count = 0
            for pt_file in weights_dir.glob("*.pt"):
                if pt_file.name != "best.pt":
                    pt_file.unlink()
                    removed_count += 1
                    logger.info(f"已删除非最佳模型文件: {pt_file}")

            logger.info(f"模型清理完成，仅保留 best.pt: {best_pt}，删除数量: {removed_count}")

        except Exception as e:
            logger.warning(f"清理模型文件失败: {e}")
    
    def train(self, epochs=None, imgsz=None, batch_size=None, device='auto',
              save_dir='/app/runs/train', name='exp', **kwargs):
        """
        训练YOLO模型
        
        Args:
            epochs (int): 训练轮数
            imgsz (int): 输入图像大小
            batch_size (int): 批次大小
            device (str): 训练设备 ('auto', 'cpu', '0', '1', etc.)
            save_dir (str): 保存目录
            name (str): 实验名称
            **kwargs: 其他训练参数
        """
        if self.model is None:
            self.load_model()
        
        if self.data_config is None:
            raise ValueError("请先设置数据集配置文件路径")
        
        try:
            is_yoloe_pe = self.training_profile == "yoloe_pe_segment"
            epochs = int(epochs or (150 if is_yoloe_pe else 100))
            imgsz = int(imgsz or 640)
            batch_size = int(batch_size or 16)
            # 设置训练参数
            train_args = {
                'data': self.data_config,
                'epochs': epochs,
                'imgsz': imgsz,
                'batch': batch_size,
                'device': device,
                'project': save_dir,
                'name': name,
                'save': True,
                'save_period': -1,  # 不保存epoch过程模型，只保留训练产生的best.pt
                'patience': 30 if is_yoloe_pe else 20,
                'optimizer': 'AdamW',
                'lr0': 0.001 if is_yoloe_pe else 0.01,
                'lrf': 0.01,        # 最终学习率比例
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,         # box loss权重
                'cls': 0.5,         # cls loss权重
                'dfl': 1.5,         # dfl loss权重
                'verbose': True,    # 确保输出详细日志
                'workers': 0,
                'cache': False,
                'amp': False,
                'exist_ok': True,
                **kwargs
            }
            if is_yoloe_pe:
                from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

                train_args["trainer"] = YOLOEPESegTrainer
            
            logger.info("开始训练...")
            logger.info(f"训练参数: {train_args}")
            
            # 确保输出目录存在
            os.makedirs(f"{save_dir}/{name}", exist_ok=True)
            
            # 开始训练
            logger.info(f"=== 开始训练 {epochs} 个epoch ===")
            cleanup_torch_runtime()
            self.results = self.model.train(**train_args)
            
            # Ultralytics默认会生成last.pt；训练完成后清理，只保留best.pt
            result_save_dir = getattr(self.results, 'save_dir', Path(save_dir) / name)
            self.keep_only_best_pt(result_save_dir)
            
            logger.info("=== 训练完成! ===")
            
            # 输出训练结果概要
            if self.results:
                logger.info(f"训练结果已保存到: {save_dir}/{name}")
                logger.info("训练指标概要:")
                # 输出一些关键指标
                try:
                    if hasattr(self.results, 'results_dict'):
                        results_dict = self.results.results_dict
                        if 'metrics/mAP50(B)' in results_dict:
                            logger.info(f"  mAP50: {results_dict['metrics/mAP50(B)']:.4f}")
                        if 'metrics/mAP50-95(B)' in results_dict:
                            logger.info(f"  mAP50-95: {results_dict['metrics/mAP50-95(B)']:.4f}")
                except Exception as e:
                    logger.warning(f"无法获取训练指标: {e}")
            return self.results
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            logger.error(f"错误类型: {type(e).__name__}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise
        finally:
            cleanup_torch_runtime()
    
    def validate(self, data_config=None, save_dir='runs/val', name='exp'):
        """
        验证模型性能
        
        Args:
            data_config (str): 数据集配置文件路径
            save_dir (str): 保存目录
            name (str): 实验名称
        """
        if self.model is None:
            logger.error("请先加载模型")
            return
        
        data_config = data_config or self.data_config
        if data_config is None:
            logger.error("请提供数据集配置文件路径")
            return
        
        try:
            results = self.model.val(
                data=data_config,
                project=save_dir,
                name=name,
                save_json=True,
                save_hybrid=True
            )
            
            logger.info("验证完成!")
            return results
            
        except Exception as e:
            logger.error(f"验证失败: {e}")
            raise
    
    def export_model(self, format='onnx', save_dir='runs/export'):
        """
        导出模型
        
        Args:
            format (str): 导出格式 ('onnx', 'torchscript', 'tflite', etc.)
            save_dir (str): 保存目录
        """
        if self.model is None:
            logger.error("请先加载模型")
            return
        
        try:
            self.model.export(format=format, project=save_dir)
            logger.info(f"模型已导出为 {format} 格式")
        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            raise
    
    def predict(self, source, save_dir='/app/runs/predict', name='exp', conf=0.25, iou=0.7, save_results=True):
        """
        使用训练好的模型进行预测

        Args:
            source (str): 输入源，图片路径、视频路径、目录等
            save_dir (str): YOLO输出父目录。单图推理建议为 /app/runs/predict
            name (str): 实验名称，例如 prediction_xxx
            conf (float): 置信度阈值
            iou (float): IoU阈值
            save_results (bool): 是否保存结果文件
        """
        if self.model is None:
            logger.error("请先加载模型")
            return None

        try:
            if save_results:
                save_project = Path(save_dir)

                # 强制修正旧的错误路径，避免继续保存到 /app/runs/detect/runs/predict
                wrong_suffix = Path("runs/detect/runs/predict")
                if str(save_project).replace("\\", "/").endswith(str(wrong_suffix).replace("\\", "/")):
                    save_project = Path("/app/runs/predict")

                save_project.mkdir(parents=True, exist_ok=True)

                results = self.model.predict(
                    source=source,
                    project=str(save_project),
                    name=name,
                    conf=conf,
                    iou=iou,
                    save=True,
                    save_txt=True,
                    save_conf=True,
                    exist_ok=True
                )

                logger.info(f"预测完成! 结果保存目录: {Path(results[0].save_dir) if results else save_project / name}")

            else:
                # 实时推理模式，不保存文件
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    results = self.model.predict(
                        source=source,
                        project=temp_dir,
                        name='realtime',
                        conf=conf,
                        iou=iou,
                        save=False,
                        save_txt=False,
                        save_conf=False,
                        verbose=False,
                        exist_ok=True
                    )

                logger.info("实时预测完成，不保存结果文件")

            return results

        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise
    
    def predict_realtime(
        self,
        source,
        conf=0.25,
        iou=0.7,
        device=None,
        half=False,
        imgsz=640,
    ):
        """Run an in-memory realtime prediction without filesystem output."""
        if self.model is None:
            logger.error("请先加载模型")
            return None

        predict_args = {
            "source": source,
            "conf": conf,
            "iou": iou,
            "save": False,
            "save_txt": False,
            "save_conf": False,
            "verbose": False,
        }
        normalized_device = str(device or "").strip()
        if normalized_device and normalized_device.lower() != "auto":
            predict_args["device"] = normalized_device
        if imgsz:
            predict_args["imgsz"] = int(imgsz)
        predict_args["half"] = bool(half) and normalized_device.lower() != "cpu"
        return self.model.predict(**predict_args)

    def track(self, source, conf=0.25, iou=0.7, imgsz=None):
        """
        使用训练好的模型进行跟踪
        
        Args:
            source (str): 输入源 (图片路径、视频路径、目录等)
            conf (float): 置信度阈值
            iou (float): IoU阈值
            imgsz (int): 可选的实时推理输入尺寸
        """
        if self.model is None:
            logger.error("请先加载模型")
            return
        
        try:
            track_args = {
                "source": source,
                "conf": conf,
                "iou": iou,
                "persist": True,
                "verbose": False,
            }
            if imgsz:
                track_args["imgsz"] = int(imgsz)
            results = self.model.track(**track_args)
            
            logger.info("跟踪完成!")
            return results
            
        except Exception as e:
            logger.error(f"跟踪失败: {e}")
            raise
    
    def plot_results(self, save_dir='runs/train/exp'):
        """
        绘制训练结果图表
        
        Args:
            save_dir (str): 结果保存目录
        """
        results_csv = Path(save_dir) / 'results.csv'
        
        if not results_csv.exists():
            logger.warning(f"结果文件不存在: {results_csv}")
            return
        
        try:
            import pandas as pd
            
            # 读取结果
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()  # 去除列名空格
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('YOLO Training Results', fontsize=16)
            
            # Loss曲线
            if 'train/box_loss' in df.columns:
                axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
                axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
                axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
                axes[0, 0].set_title('Training Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # mAP曲线
            if 'metrics/mAP50(B)' in df.columns:
                axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
                axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
                axes[0, 1].set_title('Validation mAP')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('mAP')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Precision & Recall
            if 'metrics/precision(B)' in df.columns:
                axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
                axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
                axes[1, 0].set_title('Precision & Recall')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Learning Rate
            if 'lr/pg0' in df.columns:
                axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='Learning Rate')
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('LR')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 保存图表
            plot_path = Path(save_dir) / 'training_results.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"训练结果图表已保存至: {plot_path}")
            
        except Exception as e:
            logger.error(f"绘制结果图表失败: {e}")


def create_sample_dataset_structure(base_path="./dataset"):
    """
    创建示例数据集目录结构
    
    Args:
        base_path (str): 数据集基础路径
    """
    base_path = Path(base_path)
    
    # 创建目录结构
    dirs_to_create = [
        'train/images',
        'train/labels',
        'val/images',
        'val/labels',
        'test/images',
        'test/labels'
    ]
    
    for dir_path in dirs_to_create:
        (base_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    # 创建示例README
    readme_content = """
# 数据集结构说明

## 目录结构
```
dataset/
├── train/
│   ├── images/     # 训练图片
│   └── labels/     # 训练标签 (.txt格式)
├── val/
│   ├── images/     # 验证图片
│   └── labels/     # 验证标签 (.txt格式)
├── test/
│   ├── images/     # 测试图片
│   └── labels/     # 测试标签 (.txt格式)
└── dataset.yaml   # 数据集配置文件

## 标签格式 (YOLO格式)
每个图片对应一个同名的.txt标签文件，格式为:
class_id center_x center_y width height

其中坐标为相对坐标 (0-1之间)

示例:
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
"""
    
    with open(base_path / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    logger.info(f"示例数据集结构已创建: {base_path}")


# 使用示例
if __name__ == "__main__":
    # 创建示例数据集结构
    create_sample_dataset_structure("./my_dataset")
    
    # 初始化训练器
    trainer = YOLOTrainer(model_name="yolov8n.pt")
    
    # 创建数据集配置
    trainer.create_dataset_config(
        dataset_path="./my_dataset",
        class_names=["person", "car", "bicycle", "dog", "cat"]
    )
    
    # 验证数据集结构
    trainer.validate_dataset("./my_dataset")
    
    print("训练器初始化完成!")
    print("请将您的数据放入相应目录，然后调用 trainer.train() 开始训练")
    
    # 训练示例 (需要有实际数据)
    # trainer.train(
    #     epochs=100,
    #     imgsz=640,
    #     batch_size=16,
    #     device='auto',
    #     name='my_experiment'
    # )
    
    # 验证示例
    # trainer.validate(name='my_validation')
    
    # 预测示例
    # trainer.predict(source='./test_images', name='my_prediction')
    
    # 导出模型示例
    # trainer.export_model(format='onnx')

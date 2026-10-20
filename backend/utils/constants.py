"""
常量定义
集中管理系统中使用的常量
"""

# COCO数据集类别名称(80个类别)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# 支持的图片格式
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

# 支持的视频格式
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv']

# 支持的模型格式
MODEL_EXTENSIONS = ['.pt', '.pth', '.onnx', '.tflite', '.mlmodel']

# 默认配置
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_IMG_SIZE = 640
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.01

# 目录配置
TEMP_DIR = "temp"
DATASETS_DIR = "datasets"
MODELS_DIR = "models"
EXPORTS_DIR = "exports"
STATIC_DIR = "static"
RESULTS_DIR = "static/results"
RUNS_DIR = "runs"

# 批量处理限制
MAX_BATCH_IMAGES = 50
MAX_UPLOAD_SIZE_MB = 500

# 任务状态
TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_CANCELLED = "cancelled"

# 数据集状态
DATASET_STATUS_READY = "ready"
DATASET_STATUS_EMPTY = "empty"
DATASET_STATUS_PROCESSING = "processing"
DATASET_STATUS_ERROR = "error"

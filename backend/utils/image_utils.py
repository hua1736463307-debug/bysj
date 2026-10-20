"""
图像处理工具类
封装图像编解码、转换等操作
"""

import base64
import io
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """图像处理器"""

    _FORMAT_MIME_TYPES = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "BMP": "image/bmp",
    }

    @staticmethod
    def _pil_to_rgb_array(pil_image: Image.Image) -> np.ndarray:
        """按上传时的原始像素顺序转换为 RGB 数组，不应用 EXIF 旋转。"""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return np.array(pil_image)
    
    @staticmethod
    def decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
        """
        解码base64图片为numpy数组
        
        Args:
            base64_string: base64编码的图片字符串
            
        Returns:
            numpy数组格式的图片,失败返回None
        """
        try:
            # 移除data URL前缀
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # 解码base64
            image_data = base64.b64decode(base64_string)
            
            # 保持上传时的像素顺序；标注坐标也以该尺寸为基准。
            with Image.open(io.BytesIO(image_data)) as pil_image:
                return ImageProcessor._pil_to_rgb_array(pil_image)
        except Exception as e:
            logger.error(f"解码base64图片失败: {e}")
            return None

    @staticmethod
    def decode_image_file(image_path: str) -> Optional[np.ndarray]:
        """读取磁盘图片并保持上传时的像素顺序返回 RGB 数组。"""
        try:
            with Image.open(image_path) as pil_image:
                return ImageProcessor._pil_to_rgb_array(pil_image)
        except Exception as e:
            logger.error(f"读取图片文件失败: {image_path}, error={e}")
            return None

    @staticmethod
    def get_image_size_from_file(image_path: str) -> Tuple[int, int]:
        """返回上传时的原始图片尺寸 (width, height)。"""
        try:
            with Image.open(image_path) as pil_image:
                return int(pil_image.width), int(pil_image.height)
        except Exception as e:
            logger.warning(f"读取图片尺寸失败: {image_path}, error={e}")
            return 0, 0

    @staticmethod
    def encode_image_file_to_base64(image_path: str) -> Optional[Tuple[str, str]]:
        """按原始像素顺序编码图片，并移除 EXIF 方向标记。"""
        try:
            with Image.open(image_path) as pil_image:
                output_format = str(pil_image.format or '').upper()
                if output_format not in ImageProcessor._FORMAT_MIME_TYPES:
                    output_format = 'JPEG'
                if output_format == 'JPEG' and pil_image.mode not in {'RGB', 'L'}:
                    pil_image = pil_image.convert('RGB')

                buffer = io.BytesIO()
                save_kwargs = {'quality': 95} if output_format == 'JPEG' else {}
                # Do not pass the source EXIF block. This prevents the browser
                # from rotating an otherwise raw-coordinate image on display.
                pil_image.save(buffer, format=output_format, **save_kwargs)

            mime_type = ImageProcessor._FORMAT_MIME_TYPES[output_format]
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
            return f"data:{mime_type};base64,{encoded}", mime_type
        except Exception as e:
            logger.error(f"编码图片文件失败: {image_path}, error={e}")
            return None
    
    @staticmethod
    def encode_image_to_base64(image: np.ndarray, quality: int = 85) -> Optional[str]:
        """
        将numpy图片数组编码为base64
        
        Args:
            image: numpy数组格式的图片
            quality: JPEG质量(1-100)
            
        Returns:
            base64编码的图片字符串,失败返回None
        """
        try:
            # 转换为PIL图片
            pil_image = Image.fromarray(image)
            
            # 保存到字节流
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=quality)
            
            # 编码为base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/jpeg;base64,{image_base64}"
        except Exception as e:
            logger.error(f"编码图片为base64失败: {e}")
            return None
    
    @staticmethod
    def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """
        将BGR格式图片转换为RGB格式
        
        Args:
            image: BGR格式的图片
            
        Returns:
            RGB格式的图片
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        """
        将RGB格式图片转换为BGR格式
        
        Args:
            image: RGB格式的图片
            
        Returns:
            BGR格式的图片
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def get_image_size(image: np.ndarray) -> tuple:
        """
        获取图片尺寸
        
        Args:
            image: numpy数组格式的图片
            
        Returns:
            (height, width)元组
        """
        return image.shape[:2]
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        调整图片大小
        
        Args:
            image: 原始图片
            target_size: 目标大小(width, height)
            
        Returns:
            调整后的图片
        """
        return cv2.resize(image, target_size)

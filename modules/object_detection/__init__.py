"""
目标检测模块
"""

from .yolo_detector import YOLODetector
from .detection_result import DetectionResult, Detection

__all__ = [
    'YOLODetector',
    'DetectionResult',
    'Detection',
]


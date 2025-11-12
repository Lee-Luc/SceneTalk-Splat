"""
检测结果数据结构
定义检测框和检测结果的数据类
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Detection:
    """单个检测结果"""
    
    # 基本信息
    class_name: str          # 类别名称
    class_id: int            # 类别ID
    confidence: float        # 置信度
    
    # 2D边界框 [x1, y1, x2, y2]
    bbox: Tuple[float, float, float, float]
    
    # 中心点 [x, y]
    center: Tuple[float, float]
    
    # 可选的3D信息（后续填充）
    position_3d: Optional[Tuple[float, float, float]] = None  # 3D位置
    size_3d: Optional[Tuple[float, float, float]] = None      # 3D尺寸
    orientation: Optional[Tuple[float, float, float]] = None  # 朝向（欧拉角）
    
    # 关联的Gaussian索引
    gaussian_indices: Optional[List[int]] = None
    
    # 可见性信息
    visible_views: Optional[List[int]] = None  # 可见的视角ID列表
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    def get_area(self) -> float:
        """计算边界框面积"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def get_width_height(self) -> Tuple[float, float]:
        """获取边界框宽高"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1, y2 - y1)
    
    def iou(self, other: 'Detection') -> float:
        """
        计算与另一个检测框的IoU
        
        Args:
            other: 另一个检测对象
            
        Returns:
            IoU值
        """
        x1_min, y1_min, x1_max, y1_max = self.bbox
        x2_min, y2_min, x2_max, y2_max = other.bbox
        
        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # 计算并集
        area1 = self.get_area()
        area2 = other.get_area()
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class DetectionResult:
    """单张图像的检测结果"""
    
    image_name: str                    # 图像名称
    image_path: str                    # 图像路径
    view_id: int                       # 视角ID
    detections: List[Detection]        # 检测列表
    image_size: Tuple[int, int]        # 图像尺寸 (width, height)
    
    def __len__(self) -> int:
        """返回检测数量"""
        return len(self.detections)
    
    def get_detections_by_class(self, class_name: str) -> List[Detection]:
        """
        获取指定类别的所有检测
        
        Args:
            class_name: 类别名称
            
        Returns:
            检测列表
        """
        return [det for det in self.detections if det.class_name == class_name]
    
    def filter_by_confidence(self, threshold: float) -> 'DetectionResult':
        """
        根据置信度过滤检测结果
        
        Args:
            threshold: 置信度阈值
            
        Returns:
            新的DetectionResult对象
        """
        filtered_detections = [
            det for det in self.detections if det.confidence >= threshold
        ]
        
        return DetectionResult(
            image_name=self.image_name,
            image_path=self.image_path,
            view_id=self.view_id,
            detections=filtered_detections,
            image_size=self.image_size
        )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'image_name': self.image_name,
            'image_path': self.image_path,
            'view_id': self.view_id,
            'image_size': self.image_size,
            'num_detections': len(self.detections),
            'detections': [det.to_dict() for det in self.detections]
        }
    
    def get_statistics(self) -> dict:
        """
        获取检测统计信息
        
        Returns:
            统计字典
        """
        class_counts = {}
        for det in self.detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        return {
            'total_detections': len(self.detections),
            'class_distribution': class_counts,
            'mean_confidence': np.mean([det.confidence for det in self.detections]) if self.detections else 0.0,
            'min_confidence': min([det.confidence for det in self.detections]) if self.detections else 0.0,
            'max_confidence': max([det.confidence for det in self.detections]) if self.detections else 0.0,
        }


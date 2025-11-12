"""
边界框投影器
将2D检测框投影到不同视角
"""

import numpy as np
import torch
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from ..utils.camera_utils import CameraUtils
from ..utils.logger import default_logger as logger


@dataclass
class ProjectionResult:
    """投影结果"""
    projected_bbox: Optional[Tuple[float, float, float, float]]  # 投影后的2D框
    visibility_score: float  # 可见性得分 [0-1]
    num_visible_points: int  # 可见点数量
    total_points: int        # 总点数量
    center_2d: Optional[Tuple[float, float]]  # 投影中心点


class BBoxProjector:
    """边界框投影器类"""
    
    def __init__(
        self,
        renderer,
        depth_sample_step: int = 10,
        min_visible_points: int = 4,
        visibility_threshold: float = 0.3
    ):
        """
        初始化投影器
        
        Args:
            renderer: 高斯渲染器
            depth_sample_step: 深度采样步长
            min_visible_points: 最小可见点数
            visibility_threshold: 可见性阈值
        """
        self.renderer = renderer
        self.depth_sample_step = depth_sample_step
        self.min_visible_points = min_visible_points
        self.visibility_threshold = visibility_threshold
    
    def project_bbox(
        self,
        bbox_2d: Tuple[float, float, float, float],
        source_camera,
        target_camera,
        depth_map: Optional[np.ndarray] = None,
        use_gaussian_depth: bool = True
    ) -> ProjectionResult:
        """
        将2D边界框从源视角投影到目标视角
        
        Args:
            bbox_2d: 源视角的2D边界框 [x1, y1, x2, y2]
            source_camera: 源相机
            target_camera: 目标相机
            depth_map: 可选的深度图
            use_gaussian_depth: 是否使用高斯深度图
            
        Returns:
            投影结果
        """
        # 1. 获取或渲染深度图
        if depth_map is None and use_gaussian_depth:
            depth_map = self.renderer.render_depth_map(source_camera)
        
        if depth_map is None:
            logger.warning("无深度图可用，使用默认深度估计")
            # 使用场景平均深度作为估计
            depth_map = np.ones((source_camera.image_height, 
                                source_camera.image_width)) * 5.0
        
        # 2. 在边界框内采样3D点
        points_3d = self._sample_3d_points_from_bbox(
            bbox_2d, depth_map, source_camera
        )
        
        if len(points_3d) == 0:
            return ProjectionResult(
                projected_bbox=None,
                visibility_score=0.0,
                num_visible_points=0,
                total_points=0,
                center_2d=None
            )
        
        # 3. 投影到目标视角
        points_2d = CameraUtils.project_3d_to_2d(points_3d, target_camera)
        
        # 4. 检查可见性
        visible_mask = CameraUtils.check_point_in_view(
            points_2d, points_3d, target_camera
        )
        
        num_visible = visible_mask.sum()
        total_points = len(points_3d)
        visibility_score = num_visible / total_points if total_points > 0 else 0.0
        
        # 5. 计算投影边界框
        if num_visible >= self.min_visible_points:
            visible_points_2d = points_2d[visible_mask]
            
            x_min = visible_points_2d[:, 0].min()
            y_min = visible_points_2d[:, 1].min()
            x_max = visible_points_2d[:, 0].max()
            y_max = visible_points_2d[:, 1].max()
            
            # 添加一些边距
            margin = 5
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(target_camera.image_width, x_max + margin)
            y_max = min(target_camera.image_height, y_max + margin)
            
            projected_bbox = (float(x_min), float(y_min), 
                            float(x_max), float(y_max))
            
            # 计算中心点
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            center_2d = (float(center_x), float(center_y))
        else:
            projected_bbox = None
            center_2d = None
        
        return ProjectionResult(
            projected_bbox=projected_bbox,
            visibility_score=visibility_score,
            num_visible_points=int(num_visible),
            total_points=total_points,
            center_2d=center_2d
        )
    
    def _sample_3d_points_from_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        depth_map: np.ndarray,
        camera
    ) -> np.ndarray:
        """
        从2D边界框和深度图采样3D点
        
        Args:
            bbox: 2D边界框 [x1, y1, x2, y2]
            depth_map: 深度图 [H, W]
            camera: 相机对象
            
        Returns:
            3D点云 [N, 3]
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # 确保在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1], x2)
        y2 = min(depth_map.shape[0], y2)
        
        points_3d = []
        
        # 在边界框内均匀采样
        for y in range(y1, y2, self.depth_sample_step):
            for x in range(x1, x2, self.depth_sample_step):
                if y < depth_map.shape[0] and x < depth_map.shape[1]:
                    depth = depth_map[y, x]
                    
                    # 过滤无效深度
                    if depth > 0 and not np.isnan(depth) and not np.isinf(depth):
                        # 反投影到3D
                        point_3d = CameraUtils.pixel_to_3d(x, y, depth, camera)
                        points_3d.append(point_3d)
        
        # 如果采样点太少，增加边界采样
        if len(points_3d) < self.min_visible_points:
            # 采样边界点
            for x in [x1, x2]:
                for y in range(y1, y2, self.depth_sample_step):
                    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                        depth = depth_map[y, x]
                        if depth > 0 and not np.isnan(depth):
                            point_3d = CameraUtils.pixel_to_3d(x, y, depth, camera)
                            points_3d.append(point_3d)
            
            for y in [y1, y2]:
                for x in range(x1, x2, self.depth_sample_step):
                    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                        depth = depth_map[y, x]
                        if depth > 0 and not np.isnan(depth):
                            point_3d = CameraUtils.pixel_to_3d(x, y, depth, camera)
                            points_3d.append(point_3d)
        
        if len(points_3d) == 0:
            logger.warning(f"在边界框 {bbox} 内未找到有效的3D点")
            return np.array([])
        
        return np.array(points_3d)
    
    def project_detections_to_view(
        self,
        detections: List,
        source_camera,
        target_camera,
        depth_map: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        将多个检测结果投影到目标视角
        
        Args:
            detections: 检测结果列表
            source_camera: 源相机
            target_camera: 目标相机
            depth_map: 深度图
            
        Returns:
            投影结果列表，每个包含原始检测和投影信息
        """
        projected_results = []
        
        for det in detections:
            proj_result = self.project_bbox(
                det.bbox,
                source_camera,
                target_camera,
                depth_map
            )
            
            # 只保留可见度足够的投影
            if proj_result.visibility_score >= self.visibility_threshold:
                projected_results.append({
                    'detection': det,
                    'projection': proj_result,
                    'is_visible': True
                })
            else:
                projected_results.append({
                    'detection': det,
                    'projection': proj_result,
                    'is_visible': False
                })
        
        return projected_results
    
    def compute_projection_quality(
        self,
        projected_bbox: Tuple[float, float, float, float],
        ground_truth_bbox: Tuple[float, float, float, float]
    ) -> Dict[str, float]:
        """
        计算投影质量指标
        
        Args:
            projected_bbox: 投影的边界框
            ground_truth_bbox: 真实边界框
            
        Returns:
            质量指标字典
        """
        # 计算IoU
        x1_min, y1_min, x1_max, y1_max = projected_bbox
        x2_min, y2_min, x2_max, y2_max = ground_truth_bbox
        
        # 交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * \
                    max(0, inter_y_max - inter_y_min)
        
        # 并集
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0.0
        
        # 中心点距离
        center1 = ((x1_min + x1_max) / 2, (y1_min + y1_max) / 2)
        center2 = ((x2_min + x2_max) / 2, (y2_min + y2_max) / 2)
        center_dist = np.sqrt((center1[0] - center2[0])**2 + 
                             (center1[1] - center2[1])**2)
        
        # 尺度误差
        size1 = (x1_max - x1_min) * (y1_max - y1_min)
        size2 = (x2_max - x2_min) * (y2_max - y2_min)
        size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0.0
        
        return {
            'iou': float(iou),
            'center_distance': float(center_dist),
            'size_ratio': float(size_ratio),
        }


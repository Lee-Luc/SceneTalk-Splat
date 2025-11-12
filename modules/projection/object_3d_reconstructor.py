"""
3D物体重建器
从多视角检测重建3D物体
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN
import cv2

from ..object_detection.detection_result import Detection, DetectionResult
from ..utils.camera_utils import CameraUtils
from ..utils.logger import default_logger as logger


@dataclass
class Object3D:
    """3D物体类"""
    object_id: int                                    # 物体ID
    class_name: str                                   # 类别名称
    class_id: int                                     # 类别ID
    confidence: float                                 # 平均置信度
    
    # 3D几何属性
    position: np.ndarray                              # 3D中心位置 [x, y, z]
    bbox_3d_min: np.ndarray                          # 3D包围盒最小点
    bbox_3d_max: np.ndarray                          # 3D包围盒最大点
    size: np.ndarray                                  # 尺寸 [w, h, d]
    
    # 关联的2D检测
    detections_2d: List[Detection] = field(default_factory=list)
    view_ids: List[int] = field(default_factory=list)  # 可见视角ID
    
    # 可选属性
    gaussian_indices: Optional[List[int]] = None      # 关联的Gaussian点索引
    orientation: Optional[np.ndarray] = None          # 朝向（欧拉角）
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'object_id': self.object_id,
            'class_name': self.class_name,
            'class_id': self.class_id,
            'confidence': float(self.confidence),
            'position': self.position.tolist(),
            'bbox_3d_min': self.bbox_3d_min.tolist(),
            'bbox_3d_max': self.bbox_3d_max.tolist(),
            'size': self.size.tolist(),
            'num_views': len(self.view_ids),
            'view_ids': self.view_ids,
        }
    
    def get_bbox_corners(self) -> np.ndarray:
        """
        获取3D包围盒的8个角点
        
        Returns:
            角点数组 [8, 3]
        """
        min_pt = self.bbox_3d_min
        max_pt = self.bbox_3d_max
        
        corners = np.array([
            [min_pt[0], min_pt[1], min_pt[2]],  # 0
            [max_pt[0], min_pt[1], min_pt[2]],  # 1
            [max_pt[0], max_pt[1], min_pt[2]],  # 2
            [min_pt[0], max_pt[1], min_pt[2]],  # 3
            [min_pt[0], min_pt[1], max_pt[2]],  # 4
            [max_pt[0], min_pt[1], max_pt[2]],  # 5
            [max_pt[0], max_pt[1], max_pt[2]],  # 6
            [min_pt[0], max_pt[1], max_pt[2]],  # 7
        ])
        
        return corners


class Object3DReconstructor:
    """3D物体重建器类"""
    
    def __init__(
        self,
        renderer,
        min_views: int = 2,
        clustering_eps: float = 1.0,
        clustering_min_samples: int = 3
    ):
        """
        初始化重建器
        
        Args:
            renderer: 高斯渲染器
            min_views: 最少需要的视角数
            clustering_eps: DBSCAN聚类距离阈值
            clustering_min_samples: DBSCAN最小样本数
        """
        self.renderer = renderer
        self.min_views = min_views
        self.clustering_eps = clustering_eps
        self.clustering_min_samples = clustering_min_samples
    
    def reconstruct_objects_3d(
        self,
        detection_results: List[DetectionResult],
        cameras: List
    ) -> List[Object3D]:
        """
        从多视角检测重建3D物体
        
        Args:
            detection_results: 多个视角的检测结果
            cameras: 对应的相机列表
            
        Returns:
            3D物体列表
        """
        logger.info(f"开始从 {len(detection_results)} 个视角重建3D物体...")
        
        # 1. 收集所有检测的3D位置估计
        all_3d_detections = []
        
        for idx, (det_result, camera) in enumerate(zip(detection_results, cameras)):
            # 渲染深度图
            depth_map = self.renderer.render_depth_map(camera)
            
            for det in det_result.detections:
                # 估计3D位置
                position_3d = self._estimate_3d_position(det, depth_map, camera)
                
                if position_3d is not None:
                    all_3d_detections.append({
                        'detection': det,
                        'position_3d': position_3d,
                        'view_id': idx,
                        'camera': camera
                    })
        
        logger.info(f"收集到 {len(all_3d_detections)} 个3D检测")
        
        # 2. 按类别分组
        detections_by_class = {}
        for det_3d in all_3d_detections:
            class_name = det_3d['detection'].class_name
            if class_name not in detections_by_class:
                detections_by_class[class_name] = []
            detections_by_class[class_name].append(det_3d)
        
        # 3. 对每个类别进行聚类匹配
        objects_3d = []
        object_id = 0
        
        for class_name, dets in detections_by_class.items():
            class_objects = self._cluster_and_merge_detections(dets, class_name)
            
            for obj in class_objects:
                obj.object_id = object_id
                objects_3d.append(obj)
                object_id += 1
        
        logger.info(f"重建完成，共 {len(objects_3d)} 个3D物体")
        
        return objects_3d
    
    def _estimate_3d_position(
        self,
        detection: Detection,
        depth_map: np.ndarray,
        camera
    ) -> Optional[np.ndarray]:
        """
        从2D检测和深度图估计3D位置
        
        Args:
            detection: 2D检测
            depth_map: 深度图
            camera: 相机
            
        Returns:
            3D位置 [x, y, z] 或 None
        """
        if depth_map is None:
            return None
        
        # 使用检测框中心
        center_x, center_y = detection.center
        x, y = int(center_x), int(center_y)
        
        # 检查边界
        if not (0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]):
            return None
        
        # 获取中心深度
        depth = depth_map[y, x]
        
        # 如果中心深度无效，尝试邻域平均
        if depth <= 0 or np.isnan(depth) or np.isinf(depth):
            x1, y1, x2, y2 = map(int, detection.bbox)
            region = depth_map[max(0, y1):min(depth_map.shape[0], y2),
                              max(0, x1):min(depth_map.shape[1], x2)]
            valid_depths = region[region > 0]
            if len(valid_depths) > 0:
                depth = np.median(valid_depths)
            else:
                return None
        
        # 反投影到3D
        position_3d = CameraUtils.pixel_to_3d(center_x, center_y, depth, camera)
        
        return position_3d
    
    def _cluster_and_merge_detections(
        self,
        detections_3d: List[Dict],
        class_name: str
    ) -> List[Object3D]:
        """
        聚类并合并同一物体的多视角检测
        
        Args:
            detections_3d: 3D检测列表
            class_name: 类别名称
            
        Returns:
            合并后的3D物体列表
        """
        if len(detections_3d) < self.min_views:
            return []
        
        # 提取3D位置
        positions = np.array([d['position_3d'] for d in detections_3d])
        
        # DBSCAN聚类
        clustering = DBSCAN(
            eps=self.clustering_eps,
            min_samples=self.clustering_min_samples
        ).fit(positions)
        
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        logger.info(f"类别 {class_name}: 发现 {n_clusters} 个聚类")
        
        # 合并每个聚类
        objects = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_detections = [d for i, d in enumerate(detections_3d) if cluster_mask[i]]
            
            if len(cluster_detections) >= self.min_views:
                obj = self._merge_cluster_to_object(cluster_detections, class_name)
                if obj is not None:
                    objects.append(obj)
        
        return objects
    
    def _merge_cluster_to_object(
        self,
        cluster_detections: List[Dict],
        class_name: str
    ) -> Optional[Object3D]:
        """
        将聚类的检测合并为单个3D物体
        
        Args:
            cluster_detections: 聚类内的检测列表
            class_name: 类别名称
            
        Returns:
            3D物体对象
        """
        if len(cluster_detections) == 0:
            return None
        
        # 计算平均位置
        positions = np.array([d['position_3d'] for d in cluster_detections])
        avg_position = positions.mean(axis=0)
        
        # 计算3D包围盒
        bbox_min = positions.min(axis=0)
        bbox_max = positions.max(axis=0)
        
        # 扩展包围盒（考虑物体尺寸）
        # 使用2D框尺寸的统计来估计
        box_sizes_2d = []
        for d in cluster_detections:
            width, height = d['detection'].get_width_height()
            box_sizes_2d.append((width, height))
        
        avg_size_2d = np.mean(box_sizes_2d, axis=0)
        # 简单估计：假设3D尺寸与2D尺寸成比例
        size_factor = 0.01  # 调整因子
        expansion = avg_size_2d * size_factor
        
        bbox_min -= np.array([expansion[0], expansion[1], expansion[0]])
        bbox_max += np.array([expansion[0], expansion[1], expansion[0]])
        
        size = bbox_max - bbox_min
        
        # 平均置信度
        confidences = [d['detection'].confidence for d in cluster_detections]
        avg_confidence = np.mean(confidences)
        
        # 收集2D检测和视角
        detections_2d = [d['detection'] for d in cluster_detections]
        view_ids = [d['view_id'] for d in cluster_detections]
        
        # 获取类别ID
        class_id = cluster_detections[0]['detection'].class_id
        
        obj = Object3D(
            object_id=-1,  # 稍后分配
            class_name=class_name,
            class_id=class_id,
            confidence=float(avg_confidence),
            position=avg_position,
            bbox_3d_min=bbox_min,
            bbox_3d_max=bbox_max,
            size=size,
            detections_2d=detections_2d,
            view_ids=view_ids
        )
        
        return obj


"""
场景图谱
构建和管理场景的结构化表示
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict

from ..projection.object_3d_reconstructor import Object3D
from ..utils.logger import default_logger as logger


@dataclass
class SpatialRelation:
    """空间关系"""
    subject_id: int          # 主体物体ID
    predicate: str           # 关系类型 (on, near, left_of, etc.)
    object_id: int           # 客体物体ID
    distance: float          # 距离
    confidence: float = 1.0  # 置信度
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SceneGraph:
    """场景图谱类"""
    
    def __init__(self, objects_3d: List[Object3D]):
        """
        初始化场景图谱
        
        Args:
            objects_3d: 3D物体列表
        """
        self.objects = {obj.object_id: obj for obj in objects_3d}
        self.relations: List[SpatialRelation] = []
        
        # 场景边界
        self.scene_bounds = self._compute_scene_bounds()
        
        logger.info(f"场景图谱已创建，包含 {len(self.objects)} 个物体")
    
    def _compute_scene_bounds(self) -> Dict[str, np.ndarray]:
        """计算场景边界"""
        if len(self.objects) == 0:
            # 返回默认值（空场景）
            return {
                'min': np.zeros(3),
                'max': np.zeros(3),
                'center': np.zeros(3),
                'size': np.zeros(3)
            }
        
        all_positions = np.array([obj.position for obj in self.objects.values()])
        
        return {
            'min': all_positions.min(axis=0),
            'max': all_positions.max(axis=0),
            'center': all_positions.mean(axis=0),
            'size': all_positions.max(axis=0) - all_positions.min(axis=0)
        }
    
    def add_relation(self, relation: SpatialRelation):
        """
        添加空间关系
        
        Args:
            relation: 空间关系对象
        """
        self.relations.append(relation)
    
    def get_object_by_id(self, obj_id: int) -> Optional[Object3D]:
        """根据ID获取物体"""
        return self.objects.get(obj_id)
    
    def get_objects_by_class(self, class_name: str) -> List[Object3D]:
        """获取指定类别的所有物体"""
        return [obj for obj in self.objects.values() if obj.class_name == class_name]
    
    def get_relations_for_object(self, obj_id: int) -> List[SpatialRelation]:
        """获取与指定物体相关的所有关系"""
        return [rel for rel in self.relations 
                if rel.subject_id == obj_id or rel.object_id == obj_id]
    
    def find_nearest_object(
        self,
        reference_obj_id: int,
        class_filter: Optional[str] = None
    ) -> Optional[Tuple[Object3D, float]]:
        """
        找到最近的物体
        
        Args:
            reference_obj_id: 参考物体ID
            class_filter: 类别过滤
            
        Returns:
            (最近的物体, 距离) 或 None
        """
        ref_obj = self.objects.get(reference_obj_id)
        if ref_obj is None:
            return None
        
        min_dist = float('inf')
        nearest_obj = None
        
        for obj_id, obj in self.objects.items():
            if obj_id == reference_obj_id:
                continue
            
            if class_filter and obj.class_name != class_filter:
                continue
            
            dist = np.linalg.norm(ref_obj.position - obj.position)
            if dist < min_dist:
                min_dist = dist
                nearest_obj = obj
        
        if nearest_obj:
            return (nearest_obj, min_dist)
        return None
    
    def get_objects_in_radius(
        self,
        center: np.ndarray,
        radius: float,
        class_filter: Optional[str] = None
    ) -> List[Tuple[Object3D, float]]:
        """
        获取半径范围内的物体
        
        Args:
            center: 中心点 [x, y, z]
            radius: 半径
            class_filter: 类别过滤
            
        Returns:
            [(物体, 距离), ...] 列表
        """
        results = []
        
        for obj in self.objects.values():
            if class_filter and obj.class_name != class_filter:
                continue
            
            dist = np.linalg.norm(obj.position - center)
            if dist <= radius:
                results.append((obj, dist))
        
        # 按距离排序
        results.sort(key=lambda x: x[1])
        
        return results
    
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            'objects': [obj.to_dict() for obj in self.objects.values()],
            'relations': [rel.to_dict() for rel in self.relations],
            'scene_bounds': {
                'min': self.scene_bounds['min'].tolist(),
                'max': self.scene_bounds['max'].tolist(),
                'center': self.scene_bounds['center'].tolist(),
                'size': self.scene_bounds['size'].tolist(),
            },
            'statistics': {
                'num_objects': len(self.objects),
                'num_relations': len(self.relations),
                'num_classes': len(set(obj.class_name for obj in self.objects.values())),
            }
        }
    
    def to_text_description(self) -> str:
        """生成文本描述"""
        desc = f"场景包含 {len(self.objects)} 个物体：\n\n"
        
        # 按类别分组
        objects_by_class = {}
        for obj in self.objects.values():
            if obj.class_name not in objects_by_class:
                objects_by_class[obj.class_name] = []
            objects_by_class[obj.class_name].append(obj)
        
        for class_name, objs in objects_by_class.items():
            desc += f"- {len(objs)} 个 {class_name}\n"
        
        desc += f"\n场景尺寸: {self.scene_bounds['size']}\n"
        desc += f"场景中心: {self.scene_bounds['center']}\n"
        
        return desc


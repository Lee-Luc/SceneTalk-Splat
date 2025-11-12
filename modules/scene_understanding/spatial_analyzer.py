"""
空间分析器
计算物体之间的空间关系
"""

import numpy as np
from typing import List, Dict, Tuple
from .scene_graph import SceneGraph, SpatialRelation, Object3D
from ..utils.logger import default_logger as logger


class SpatialAnalyzer:
    """空间分析器类"""
    
    def __init__(
        self,
        distance_threshold: float = 2.0,
        near_threshold: float = 1.0
    ):
        """
        初始化空间分析器
        
        Args:
            distance_threshold: 空间关系距离阈值
            near_threshold: "near"关系的距离阈值
        """
        self.distance_threshold = distance_threshold
        self.near_threshold = near_threshold
    
    def analyze_scene(self, scene_graph: SceneGraph):
        """
        分析场景并添加空间关系
        
        Args:
            scene_graph: 场景图谱对象
        """
        logger.info("开始空间关系分析...")
        
        objects = list(scene_graph.objects.values())
        
        # 计算所有物体对之间的关系
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                relations = self._compute_pairwise_relations(obj1, obj2)
                for rel in relations:
                    scene_graph.add_relation(rel)
        
        logger.info(f"空间关系分析完成，共发现 {len(scene_graph.relations)} 个关系")
    
    def _compute_pairwise_relations(
        self,
        obj1: Object3D,
        obj2: Object3D
    ) -> List[SpatialRelation]:
        """
        计算两个物体之间的空间关系
        
        Args:
            obj1: 物体1
            obj2: 物体2
            
        Returns:
            空间关系列表
        """
        relations = []
        
        # 计算距离
        distance = np.linalg.norm(obj1.position - obj2.position)
        
        # 1. 距离关系 (near/far)
        if distance < self.near_threshold:
            relations.append(SpatialRelation(
                subject_id=obj1.object_id,
                predicate='near',
                object_id=obj2.object_id,
                distance=distance
            ))
        
        # 2. 垂直关系 (above/below/on)
        vertical_dist = obj1.position[1] - obj2.position[1]
        horizontal_dist = np.linalg.norm(obj1.position[[0,2]] - obj2.position[[0,2]])
        
        # 物体1在物体2上方
        if vertical_dist > obj2.size[1] / 2:
            if horizontal_dist < (obj1.size[0] + obj2.size[0]) / 4:
                # 位置对齐，可能是"on"关系
                if abs(vertical_dist - obj2.size[1]/2) < 0.3:
                    relations.append(SpatialRelation(
                        subject_id=obj1.object_id,
                        predicate='on',
                        object_id=obj2.object_id,
                        distance=distance
                    ))
                else:
                    relations.append(SpatialRelation(
                        subject_id=obj1.object_id,
                        predicate='above',
                        object_id=obj2.object_id,
                        distance=distance
                    ))
        
        # 物体1在物体2下方
        elif vertical_dist < -obj1.size[1] / 2:
            relations.append(SpatialRelation(
                subject_id=obj1.object_id,
                predicate='below',
                object_id=obj2.object_id,
                distance=distance
            ))
        
        # 3. 水平方向关系 (left/right/front/back)
        if abs(vertical_dist) < max(obj1.size[1], obj2.size[1]) / 2:
            # 在同一水平面
            dx = obj1.position[0] - obj2.position[0]
            dz = obj1.position[2] - obj2.position[2]
            
            if abs(dx) > abs(dz):
                # X轴方向占主导
                if dx > 0:
                    relations.append(SpatialRelation(
                        subject_id=obj1.object_id,
                        predicate='right_of',
                        object_id=obj2.object_id,
                        distance=distance
                    ))
                else:
                    relations.append(SpatialRelation(
                        subject_id=obj1.object_id,
                        predicate='left_of',
                        object_id=obj2.object_id,
                        distance=distance
                    ))
            else:
                # Z轴方向占主导
                if dz > 0:
                    relations.append(SpatialRelation(
                        subject_id=obj1.object_id,
                        predicate='behind',
                        object_id=obj2.object_id,
                        distance=distance
                    ))
                else:
                    relations.append(SpatialRelation(
                        subject_id=obj1.object_id,
                        predicate='in_front_of',
                        object_id=obj2.object_id,
                        distance=distance
                    ))
        
        return relations
    
    def query_spatial_relation(
        self,
        scene_graph: SceneGraph,
        query: str
    ) -> List[Dict]:
        """
        查询空间关系
        
        Args:
            scene_graph: 场景图谱
            query: 查询字符串，如 "chair on table"
            
        Returns:
            查询结果列表
        """
        # 简单的查询解析
        query_lower = query.lower()
        
        results = []
        
        # 解析查询类型
        if ' on ' in query_lower:
            parts = query_lower.split(' on ')
            if len(parts) == 2:
                subject_class = parts[0].strip()
                object_class = parts[1].strip()
                
                # 查找匹配的关系
                for rel in scene_graph.relations:
                    if rel.predicate == 'on':
                        subj_obj = scene_graph.get_object_by_id(rel.subject_id)
                        obj_obj = scene_graph.get_object_by_id(rel.object_id)
                        
                        if (subj_obj and obj_obj and
                            subject_class in subj_obj.class_name.lower() and
                            object_class in obj_obj.class_name.lower()):
                            results.append({
                                'subject': subj_obj.to_dict(),
                                'relation': rel.predicate,
                                'object': obj_obj.to_dict()
                            })
        
        elif ' near ' in query_lower:
            parts = query_lower.split(' near ')
            if len(parts) == 2:
                subject_class = parts[0].strip()
                object_class = parts[1].strip()
                
                for rel in scene_graph.relations:
                    if rel.predicate == 'near':
                        subj_obj = scene_graph.get_object_by_id(rel.subject_id)
                        obj_obj = scene_graph.get_object_by_id(rel.object_id)
                        
                        if (subj_obj and obj_obj and
                            subject_class in subj_obj.class_name.lower() and
                            object_class in obj_obj.class_name.lower()):
                            results.append({
                                'subject': subj_obj.to_dict(),
                                'relation': rel.predicate,
                                'object': obj_obj.to_dict(),
                                'distance': rel.distance
                            })
        
        return results
    
    def compute_object_statistics(
        self,
        scene_graph: SceneGraph
    ) -> Dict:
        """
        计算物体统计信息
        
        Args:
            scene_graph: 场景图谱
            
        Returns:
            统计字典
        """
        stats = {
            'total_objects': len(scene_graph.objects),
            'total_relations': len(scene_graph.relations),
            'classes': {},
            'relation_types': {},
            'avg_object_size': None,
            'scene_density': None,
        }
        
        # 类别统计
        for obj in scene_graph.objects.values():
            stats['classes'][obj.class_name] = stats['classes'].get(obj.class_name, 0) + 1
        
        # 关系类型统计
        for rel in scene_graph.relations:
            stats['relation_types'][rel.predicate] = stats['relation_types'].get(rel.predicate, 0) + 1
        
        # 平均物体尺寸
        if len(scene_graph.objects) > 0:
            all_sizes = np.array([obj.size for obj in scene_graph.objects.values()])
            stats['avg_object_size'] = all_sizes.mean(axis=0).tolist()
        
        # 场景密度（物体数 / 场景体积）
        scene_volume = np.prod(scene_graph.scene_bounds['size'])
        if scene_volume > 0:
            stats['scene_density'] = len(scene_graph.objects) / scene_volume
        
        return stats


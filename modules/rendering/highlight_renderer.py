"""
高亮渲染器
支持高亮显示特定物体
"""

import torch
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

from .gaussian_renderer import GaussianRendererWrapper
from ..utils.logger import default_logger as logger


class HighlightRenderer:
    """高亮渲染器类"""
    
    def __init__(self, renderer: GaussianRendererWrapper):
        """
        初始化高亮渲染器
        
        Args:
            renderer: 高斯渲染器包装对象
        """
        self.renderer = renderer
    
    def render_with_highlight(
        self,
        camera,
        highlight_mask: Optional[torch.Tensor] = None,
        highlight_color: Tuple[float, float, float] = (1.0, 1.0, 0.0),
        highlight_intensity: float = 0.5
    ) -> np.ndarray:
        """
        渲染并高亮显示指定的高斯点
        
        Args:
            camera: 相机对象
            highlight_mask: 高斯点的mask [N]，True表示需要高亮
            highlight_color: 高亮颜色 (R, G, B)，范围[0, 1]
            highlight_intensity: 高亮强度
            
        Returns:
            高亮后的图像 [H, W, 3]
        """
        # 渲染原始图像
        render_result = self.renderer.render_view(camera)
        rendered_image = render_result['render']  # [3, H, W]
        
        # 转换为numpy
        img_np = rendered_image.permute(1, 2, 0).cpu().numpy()
        
        # 如果有高亮mask，进行高亮处理
        if highlight_mask is not None:
            # TODO: 实现基于mask的高亮渲染
            # 这需要修改渲染过程，暂时使用后处理方式
            pass
        
        # 转换为uint8
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        
        return img_np
    
    def render_with_bbox_overlay(
        self,
        camera,
        bboxes_2d: List[Tuple[float, float, float, float]],
        labels: List[str] = None,
        colors: List[Tuple[int, int, int]] = None,
        thickness: int = 2
    ) -> np.ndarray:
        """
        渲染图像并叠加2D边界框
        
        Args:
            camera: 相机对象
            bboxes_2d: 2D边界框列表 [(x1, y1, x2, y2), ...]
            labels: 标签列表
            colors: 颜色列表 [(B, G, R), ...]
            thickness: 线宽
            
        Returns:
            带标注的图像 [H, W, 3]
        """
        # 渲染基础图像
        render_result = self.renderer.render_view(camera)
        rendered_image = render_result['render']
        
        # 转换为numpy和BGR格式
        img_np = rendered_image.permute(1, 2, 0).cpu().numpy()
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 默认颜色
        if colors is None:
            colors = [(0, 255, 0)] * len(bboxes_2d)  # 绿色
        
        # 绘制每个边界框
        for idx, bbox in enumerate(bboxes_2d):
            x1, y1, x2, y2 = map(int, bbox)
            color = colors[idx] if idx < len(colors) else (0, 255, 0)
            
            # 绘制矩形
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制标签
            if labels and idx < len(labels):
                label = labels[idx]
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness
                )
                
                # 标签背景
                cv2.rectangle(
                    img_bgr,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    color,
                    -1
                )
                
                # 标签文字
                cv2.putText(
                    img_bgr,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    thickness
                )
        
        # 转回RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_rgb
    
    def render_with_3d_bbox(
        self,
        camera,
        bbox_3d_corners: List[np.ndarray],
        colors: List[Tuple[int, int, int]] = None,
        thickness: int = 2
    ) -> np.ndarray:
        """
        渲染图像并叠加3D边界框投影
        
        Args:
            camera: 相机对象
            bbox_3d_corners: 3D边界框角点列表，每个为[8, 3]
            colors: 颜色列表
            thickness: 线宽
            
        Returns:
            带3D框的图像
        """
        from ..utils.camera_utils import CameraUtils
        
        # 渲染基础图像
        img_rgb = self.render_with_bbox_overlay(camera, [], [], [])
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # 默认颜色
        if colors is None:
            colors = [(255, 0, 0)] * len(bbox_3d_corners)  # 红色
        
        # 3D框的边连接关系
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7),  # 竖边
        ]
        
        # 绘制每个3D框
        for idx, corners_3d in enumerate(bbox_3d_corners):
            # 投影到2D
            corners_2d = CameraUtils.project_3d_to_2d(corners_3d, camera)
            
            # 检查可见性
            visible = CameraUtils.check_point_in_view(corners_2d, corners_3d, camera)
            
            if visible.sum() < 4:  # 至少4个点可见
                continue
            
            color = colors[idx] if idx < len(colors) else (255, 0, 0)
            
            # 绘制边
            for edge in edges:
                if visible[edge[0]] and visible[edge[1]]:
                    pt1 = tuple(corners_2d[edge[0]].astype(int))
                    pt2 = tuple(corners_2d[edge[1]].astype(int))
                    cv2.line(img_bgr, pt1, pt2, color, thickness)
        
        # 转回RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_rgb


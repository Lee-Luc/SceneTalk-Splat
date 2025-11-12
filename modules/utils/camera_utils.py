"""
相机工具函数
提供相机相关的计算和转换功能
"""

import numpy as np
import torch
from typing import Tuple, Optional


class CameraUtils:
    """相机工具类"""
    
    @staticmethod
    def pixel_to_3d(x: float, y: float, depth: float, camera) -> np.ndarray:
        """
        将像素坐标和深度转换为3D世界坐标
        
        Args:
            x: 像素x坐标
            y: 像素y坐标
            depth: 深度值
            camera: 相机对象
            
        Returns:
            3D世界坐标 [x, y, z]
        """
        # 获取相机内参
        fx = camera.image_width / (2 * np.tan(camera.FoVx / 2))
        fy = camera.image_height / (2 * np.tan(camera.FoVy / 2))
        cx = camera.image_width / 2
        cy = camera.image_height / 2
        
        # 像素坐标 -> 相机坐标
        x_cam = (x - cx) * depth / fx
        y_cam = (y - cy) * depth / fy
        z_cam = depth
        
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        
        # 相机坐标 -> 世界坐标
        if hasattr(camera, 'world_view_transform'):
            # 使用逆变换矩阵
            w2c = camera.world_view_transform.cpu().numpy()
            c2w = np.linalg.inv(w2c.T)  # 注意转置
            point_world = c2w @ point_cam
            return point_world[:3]
        else:
            return point_cam[:3]
    
    @staticmethod
    def project_3d_to_2d(points_3d: np.ndarray, camera) -> np.ndarray:
        """
        将3D世界坐标投影到2D像素坐标
        
        Args:
            points_3d: 3D点云 [N, 3]
            camera: 相机对象
            
        Returns:
            2D像素坐标 [N, 2]
        """
        if len(points_3d.shape) == 1:
            points_3d = points_3d.reshape(1, -1)
        
        N = points_3d.shape[0]
        points_3d_homo = np.concatenate([points_3d, np.ones((N, 1))], axis=1)
        
        # 世界坐标 -> 相机坐标
        if hasattr(camera, 'world_view_transform'):
            w2c = camera.world_view_transform.cpu().numpy().T
            points_cam = (w2c @ points_3d_homo.T).T
        else:
            points_cam = points_3d_homo
        
        # 相机坐标 -> 图像坐标
        fx = camera.image_width / (2 * np.tan(camera.FoVx / 2))
        fy = camera.image_height / (2 * np.tan(camera.FoVy / 2))
        cx = camera.image_width / 2
        cy = camera.image_height / 2
        
        # 避免除零
        z = points_cam[:, 2]
        z[z == 0] = 1e-6
        
        x_img = fx * points_cam[:, 0] / z + cx
        y_img = fy * points_cam[:, 1] / z + cy
        
        points_2d = np.stack([x_img, y_img], axis=1)
        
        return points_2d
    
    @staticmethod
    def check_point_in_view(points_2d: np.ndarray, points_3d: np.ndarray, 
                           camera, depth_threshold: float = 0.01) -> np.ndarray:
        """
        检查3D点是否在相机视野内
        
        Args:
            points_2d: 2D投影坐标 [N, 2]
            points_3d: 3D世界坐标 [N, 3]
            camera: 相机对象
            depth_threshold: 深度阈值
            
        Returns:
            可见性mask [N]
        """
        # 检查深度（必须在相机前方）
        if hasattr(camera, 'world_view_transform'):
            w2c = camera.world_view_transform.cpu().numpy().T
            N = points_3d.shape[0]
            points_3d_homo = np.concatenate([points_3d, np.ones((N, 1))], axis=1)
            points_cam = (w2c @ points_3d_homo.T).T
            depth_mask = points_cam[:, 2] > depth_threshold
        else:
            depth_mask = points_3d[:, 2] > depth_threshold
        
        # 检查是否在图像范围内
        x_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < camera.image_width)
        y_mask = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < camera.image_height)
        
        visible_mask = depth_mask & x_mask & y_mask
        
        return visible_mask
    
    @staticmethod
    def compute_camera_distance(camera1, camera2) -> float:
        """
        计算两个相机之间的距离
        
        Args:
            camera1: 第一个相机
            camera2: 第二个相机
            
        Returns:
            距离值
        """
        pos1 = camera1.camera_center.cpu().numpy() if hasattr(camera1, 'camera_center') else np.zeros(3)
        pos2 = camera2.camera_center.cpu().numpy() if hasattr(camera2, 'camera_center') else np.zeros(3)
        
        return np.linalg.norm(pos1 - pos2)
    
    @staticmethod
    def get_camera_frustum_corners(camera, depth: float = 1.0) -> np.ndarray:
        """
        获取相机视锥体的8个角点
        
        Args:
            camera: 相机对象
            depth: 视锥体深度
            
        Returns:
            角点坐标 [8, 3]
        """
        # 图像四个角的像素坐标
        corners_2d = np.array([
            [0, 0],
            [camera.image_width, 0],
            [camera.image_width, camera.image_height],
            [0, camera.image_height]
        ])
        
        # 近平面和远平面
        near_depth = camera.znear if hasattr(camera, 'znear') else 0.1
        far_depth = depth
        
        corners_3d = []
        
        for d in [near_depth, far_depth]:
            for corner in corners_2d:
                point_3d = CameraUtils.pixel_to_3d(corner[0], corner[1], d, camera)
                corners_3d.append(point_3d)
        
        return np.array(corners_3d)
    
    @staticmethod
    def interpolate_camera_path(camera_start, camera_end, num_steps: int):
        """
        在两个相机之间插值生成平滑路径
        
        Args:
            camera_start: 起始相机
            camera_end: 结束相机
            num_steps: 插值步数
            
        Returns:
            相机参数列表
        """
        # TODO: 实现相机路径插值
        # 这个功能可以用于生成平滑的视频过渡
        pass


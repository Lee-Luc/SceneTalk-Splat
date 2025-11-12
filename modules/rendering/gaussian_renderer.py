"""
高斯渲染器包装类
封装原始3DGS渲染功能
"""

import torch
import torchvision
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from argparse import Namespace
from ..utils.logger import default_logger as logger


class GaussianRendererWrapper:
    """高斯渲染器包装类"""
    
    def __init__(
        self,
        model_path: str,
        source_path: str = None,
        sh_degree: int = 3,
        load_iteration: int = -1,
        white_background: bool = False
    ):
        """
        初始化渲染器
        
        Args:
            model_path: 训练好的模型路径
            source_path: 源数据集路径
            sh_degree: 球谐函数阶数
            load_iteration: 加载的迭代次数
            white_background: 是否使用白色背景
        """
        self.model_path = Path(model_path)
        self.sh_degree = sh_degree
        self.white_background = white_background
        
        logger.info(f"正在加载3D高斯模型: {model_path}")
        
        # 如果没有提供source_path，尝试从模型路径推断
        if source_path is None:
            cfg_file = self.model_path / "cfg_args"
            if cfg_file.exists():
                import re
                with open(cfg_file, 'r') as f:
                    cfg_content = f.read()
                    # 使用正则表达式提取source_path
                    match = re.search(r"source_path='([^']*)'", cfg_content)
                    if match:
                        source_path = match.group(1)
                    else:
                        raise ValueError("无法从配置文件推断源数据路径，请手动指定source_path参数")
            else:
                raise ValueError("找不到配置文件 cfg_args，请手动指定source_path参数")
        
        self.source_path = source_path
        logger.info(f"使用数据集路径: {source_path}")
        
        # 创建 ModelParams 风格的参数对象
        # Scene 类需要一个具有特定属性的对象，而不是字符串
        dataset_args = Namespace(
            sh_degree=sh_degree,
            source_path=source_path,
            model_path=str(model_path),
            images="images",
            depths="",
            resolution=-1,
            white_background=white_background,
            train_test_exp=False,
            data_device="cuda",
            eval=False
        )
        
        # 加载高斯模型
        self.gaussians = GaussianModel(sh_degree)
        
        # 创建场景（传入 Namespace 对象而不是字符串）
        self.scene = Scene(dataset_args, self.gaussians, 
                         load_iteration=load_iteration, shuffle=False)
        
        # 设置渲染参数
        # PipelineParams 需要 parser 参数，我们直接创建一个包含默认值的对象
        self.pipeline = Namespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
            antialiasing=False
        )
        
        # 设置背景颜色
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 验证模型是否正确加载
        num_gaussians = self.gaussians.get_xyz.shape[0]
        logger.info(f"模型加载完成，共有 {num_gaussians} 个高斯点")
        
        if num_gaussians == 0:
            logger.error("❌ 警告：高斯点数量为0，模型可能未正确加载！")
        else:
            logger.info(f"✓ 高斯模型验证成功")
            # 输出场景统计信息
            xyz_range = self.gaussians.get_xyz.max(dim=0)[0] - self.gaussians.get_xyz.min(dim=0)[0]
            logger.info(f"  场景范围: X={xyz_range[0]:.2f}m, Y={xyz_range[1]:.2f}m, Z={xyz_range[2]:.2f}m")
            opacity = self.gaussians.get_opacity
            logger.info(f"  不透明度: min={opacity.min():.3f}, max={opacity.max():.3f}, mean={opacity.mean():.3f}")
    
    def render_view(
        self,
        camera,
        return_depth: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        渲染单个视角
        
        Args:
            camera: 相机对象
            return_depth: 是否返回深度图
            
        Returns:
            渲染结果字典
        """
        with torch.no_grad():
            render_pkg = render(camera, self.gaussians, self.pipeline, self.background)
        
        rendered_image = render_pkg['render']  # [3, H, W]
        
        # 验证渲染结果（调试用）
        if hasattr(camera, 'original_image'):
            gt_image = camera.original_image.cuda()
            # 检查渲染图和ground truth的差异
            diff = torch.abs(rendered_image - gt_image).mean()
            if diff < 0.001:
                logger.warning(f"⚠️  渲染图与ground truth几乎相同（差异={diff:.6f}），可能有问题")
            else:
                logger.debug(f"✓ 渲染差异正常（diff={diff:.4f}）")
        
        result = {
            'render': rendered_image,
        }
        
        if return_depth:
            result['depth'] = render_pkg.get('depth', None)  # [1, H, W]
        
        return result
    
    def render_train_views(
        self,
        output_dir: str = None,
        return_images: bool = False,
        scale: float = 1.0
    ) -> Optional[List[np.ndarray]]:
        """
        渲染所有训练视角
        
        Args:
            output_dir: 输出目录
            return_images: 是否返回图像列表
            scale: 分辨率缩放比例
            
        Returns:
            如果return_images为True，返回图像列表
        """
        cameras = self.scene.getTrainCameras(scale)
        return self._render_camera_list(cameras, output_dir, "train", return_images)
    
    def render_test_views(
        self,
        output_dir: str = None,
        return_images: bool = False,
        scale: float = 1.0
    ) -> Optional[List[np.ndarray]]:
        """
        渲染所有测试视角
        
        Args:
            output_dir: 输出目录
            return_images: 是否返回图像列表
            scale: 分辨率缩放比例
            
        Returns:
            如果return_images为True，返回图像列表
        """
        cameras = self.scene.getTestCameras(scale)
        return self._render_camera_list(cameras, output_dir, "test", return_images)
    
    def _render_camera_list(
        self,
        cameras: List,
        output_dir: Optional[str],
        split_name: str,
        return_images: bool
    ) -> Optional[List[np.ndarray]]:
        """
        渲染相机列表
        
        Args:
            cameras: 相机列表
            output_dir: 输出目录
            split_name: 数据集划分名称
            return_images: 是否返回图像
            
        Returns:
            图像列表（如果return_images为True）
        """
        images = [] if return_images else None
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"正在渲染 {len(cameras)} 个{split_name}视角...")
        
        for idx, camera in enumerate(tqdm(cameras, desc=f"渲染{split_name}视角")):
            # 渲染
            render_result = self.render_view(camera)
            rendered_image = render_result['render']  # [3, H, W]
            
            # 验证渲染结果
            if rendered_image is None or rendered_image.numel() == 0:
                logger.error(f"  ❌ 视角 {idx}: 渲染失败，输出为空！")
                continue
            
            # 保存或收集图像
            if output_dir:
                filename = f"{idx:05d}_render.png"
                filepath = output_path / filename
                # 确保图像在[0,1]范围内并保存
                rendered_image_clamped = torch.clamp(rendered_image, 0.0, 1.0)
                
                # 保存前验证
                logger.info(f"  视角 {idx}: 保存渲染图 -> {filename}")
                logger.debug(f"    shape={rendered_image.shape}, 值范围=[{rendered_image.min():.3f}, {rendered_image.max():.3f}]")
                
                torchvision.utils.save_image(rendered_image_clamped, filepath)
                
                # 验证文件是否成功保存
                if not filepath.exists():
                    logger.error(f"  ❌ 文件保存失败: {filepath}")
                else:
                    file_size_kb = filepath.stat().st_size / 1024
                    logger.debug(f"  ✓ 保存成功 ({file_size_kb:.1f} KB)")
            
            if return_images:
                # 转换为numpy数组
                img_np = rendered_image.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                images.append(img_np)
        
        logger.info(f"{split_name}视角渲染完成")
        
        return images
    
    def get_train_cameras(self, scale: float = 1.0) -> List:
        """获取训练相机列表"""
        return self.scene.getTrainCameras(scale)
    
    def get_test_cameras(self, scale: float = 1.0) -> List:
        """获取测试相机列表"""
        return self.scene.getTestCameras(scale)
    
    def get_gaussian_positions(self) -> np.ndarray:
        """
        获取所有高斯点的位置
        
        Returns:
            位置数组 [N, 3]
        """
        return self.gaussians.get_xyz.cpu().numpy()
    
    def get_gaussian_colors(self) -> np.ndarray:
        """
        获取所有高斯点的颜色
        
        Returns:
            颜色数组 [N, 3]
        """
        features = self.gaussians.get_features
        # 简化：只取DC分量
        colors = features[:, :, 0].cpu().numpy()
        return colors
    
    def render_depth_map(self, camera) -> np.ndarray:
        """
        渲染深度图
        
        Args:
            camera: 相机对象
            
        Returns:
            深度图 [H, W]
        """
        with torch.no_grad():
            render_pkg = render(camera, self.gaussians, self.pipeline, self.background)
            depth = render_pkg.get('depth', None)
        
        if depth is not None:
            depth_np = depth.squeeze().cpu().numpy()
            return depth_np
        else:
            logger.warning("深度图不可用")
            return None
    
    def get_scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取场景边界
        
        Returns:
            (min_bound, max_bound)，每个为[x, y, z]
        """
        positions = self.get_gaussian_positions()
        min_bound = positions.min(axis=0)
        max_bound = positions.max(axis=0)
        return min_bound, max_bound
    
    def estimate_scene_scale(self) -> float:
        """
        估计场景尺度（最大维度）
        
        Returns:
            场景尺度
        """
        min_bound, max_bound = self.get_scene_bounds()
        scale = np.max(max_bound - min_bound)
        return float(scale)


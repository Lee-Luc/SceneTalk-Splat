"""
3D高斯渲染器 - 第二版
"""

import torch
import torchvision
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional
import math
from tqdm import tqdm

# 导入3DGS核心组件
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene import Scene
from argparse import Namespace

from modules.utils.logger import logger


class PureGaussianRenderer:
    """
    纯3D高斯渲染器
    完全从头构建，不依赖任何可能返回原图的函数
    """
    
    def __init__(
        self,
        model_path: str,
        source_path: str,
        sh_degree: int = 3,
        load_iteration: int = -1,
        white_background: bool = False
    ):
        """
        初始化渲染器
        
        Args:
            model_path: 训练好的模型路径（包含point_cloud/iteration_xxx/）
            source_path: 原始数据路径
            sh_degree: 球谐阶数
            load_iteration: 加载的迭代次数
            white_background: 是否使用白色背景
        """
        self.model_path = Path(model_path)
        self.source_path = source_path
        self.white_background = white_background
        
        logger.info(f"[PureGaussianRenderer] 加载模型: {model_path}")
        logger.info(f"[PureGaussianRenderer] 数据路径: {source_path}")
        
        # 1. 创建高斯模型
        self.gaussians = GaussianModel(sh_degree)
        logger.info(f"[PureGaussianRenderer] 创建高斯模型，sh_degree={sh_degree}")
        
        # 2. 创建场景参数
        dataset_args = Namespace(
            sh_degree=sh_degree,
            source_path=source_path,
            model_path=str(model_path),
            images="images",
            depths="",
            resolution=1,  # 全分辨率
            white_background=white_background,
            train_test_exp=False,
            data_device="cuda",
            eval=False
        )
        
        # 3. 加载场景和相机
        self.scene = Scene(
            dataset_args,
            self.gaussians,
            load_iteration=load_iteration,
            shuffle=False,
            resolution_scales=[1.0]  # 只使用全分辨率
        )
        
        # 4. 设置背景颜色
        bg_color = [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        logger.info(f"[PureGaussianRenderer] 高斯点数量: {self.gaussians.get_xyz.shape[0]}")
        logger.info(f"[PureGaussianRenderer] 背景颜色: {bg_color}")
        logger.info(f"[PureGaussianRenderer] 初始化完成")
    
    def render_single_view(
        self,
        camera,
        scaling_modifier: float = 1.0,
        override_color: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        渲染单个视角 - 完全手动实现，不使用任何可能返回原图的函数
        
        Args:
            camera: 相机对象
            scaling_modifier: 缩放修改器
            override_color: 覆盖颜色
            
        Returns:
            包含渲染结果的字典
        """
        with torch.no_grad():
            # 创建screenspace points
            screenspace_points = torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=False,
                device="cuda"
            )
            
            # 计算视场角的tan值
            tanfovx = math.tan(camera.FoVx * 0.5)
            tanfovy = math.tan(camera.FoVy * 0.5)
            
            # 创建光栅化设置
            raster_settings = GaussianRasterizationSettings(
                image_height=int(camera.image_height),
                image_width=int(camera.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=self.background,
                scale_modifier=scaling_modifier,
                viewmatrix=camera.world_view_transform,
                projmatrix=camera.full_proj_transform,
                sh_degree=self.gaussians.active_sh_degree,
                campos=camera.camera_center,
                prefiltered=False,
                debug=False,
                antialiasing=False
            )
            
            # 创建光栅化器
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            
            # 获取高斯参数
            means3D = self.gaussians.get_xyz
            means2D = screenspace_points
            opacity = self.gaussians.get_opacity
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation
            
            # 获取颜色（通过球谐函数）
            shs = None
            colors_precomp = None
            if override_color is None:
                shs = self.gaussians.get_features
            else:
                colors_precomp = override_color
            
            # 执行光栅化 - 这是真正的3DGS渲染
            rendered_image, radii, depth = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None
            )
            
            # Clamp到[0,1]
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
            
            logger.debug(f"[PureGaussianRenderer] 渲染完成: shape={rendered_image.shape}, "
                        f"range=[{rendered_image.min():.3f}, {rendered_image.max():.3f}]")
            
            return {
                'render': rendered_image,
                'depth': depth,
                'radii': radii
            }
    
    def get_train_cameras(self, scale: float = 1.0):
        """获取训练相机"""
        return self.scene.getTrainCameras(scale)
    
    def get_test_cameras(self, scale: float = 1.0):
        """获取测试相机"""
        return self.scene.getTestCameras(scale)
    
    def render_and_save_all_views(
        self,
        output_dir: str,
        view_type: str = 'train',
        add_watermark: bool = True,
        compare_with_original: bool = True
    ):
        """
        渲染并保存所有视角
        
        Args:
            output_dir: 输出目录
            view_type: 'train' 或 'test'
            add_watermark: 是否添加水印
            compare_with_original: 是否与原图对比
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取相机列表
        if view_type == 'train':
            cameras = self.get_train_cameras()
        else:
            cameras = self.get_test_cameras()
        
        logger.info(f"[PureGaussianRenderer] 开始渲染 {len(cameras)} 个{view_type}视角")
        logger.info(f"[PureGaussianRenderer] 输出目录: {output_dir}")
        
        all_stats = []
        
        for idx, camera in enumerate(tqdm(cameras, desc=f"渲染{view_type}视角")):
            # 1. 使用纯高斯渲染
            render_result = self.render_single_view(camera)
            rendered_image = render_result['render']  # [3, H, W], 在GPU上
            
            # 2. 如果需要，添加水印（明确标识这是渲染图）
            if add_watermark:
                # 左上角20x20红色方块
                rendered_image[:, 0:20, 0:20] = torch.tensor(
                    [1.0, 0.0, 0.0],
                    device=rendered_image.device
                ).view(3, 1, 1)
                
                # 右上角20x20绿色方块（双重验证）
                rendered_image[:, 0:20, -20:] = torch.tensor(
                    [0.0, 1.0, 0.0],
                    device=rendered_image.device
                ).view(3, 1, 1)
            
            # 3. 转换为numpy并保存
            render_np = rendered_image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            render_np = (render_np * 255).astype(np.uint8)
            render_bgr = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)
            
            # 保存渲染图
            render_path = output_path / f"{idx:05d}_pure_render.png"
            cv2.imwrite(str(render_path), render_bgr)
            
            # 4. 如果需要，与原图对比
            if compare_with_original and hasattr(camera, 'original_image'):
                orig_img = camera.original_image  # [3, H, W]
                
                # 移除水印区域后再对比
                if add_watermark:
                    compare_render = rendered_image.clone()
                    compare_render[:, 0:20, 0:20] = orig_img[:, 0:20, 0:20]
                    compare_render[:, 0:20, -20:] = orig_img[:, 0:20, -20:]
                else:
                    compare_render = rendered_image
                
                # 计算差异
                diff = torch.abs(compare_render - orig_img)
                mean_diff = diff.mean().item()
                max_diff = diff.max().item()
                mse = ((compare_render - orig_img) ** 2).mean().item()
                psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100.0
                
                all_stats.append({
                    'view_idx': idx,
                    'mean_diff': mean_diff,
                    'max_diff': max_diff,
                    'psnr': psnr
                })
                
                # 保存对比图
                orig_np = orig_img.permute(1, 2, 0).cpu().numpy()
                orig_np = (orig_np * 255).astype(np.uint8)
                orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
                
                # 创建并排对比
                h, w = orig_bgr.shape[:2]
                comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
                comparison[:, :w] = orig_bgr
                comparison[:, w+20:] = render_bgr
                
                # 添加标签
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 255, 0), 2)
                cv2.putText(comparison, "3DGS Render", (w+30, 30), font, 1, (0, 255, 0), 2)
                cv2.putText(comparison, f"PSNR: {psnr:.2f} dB", (w+30, h-20), font, 0.8, (0, 255, 0), 2)
                
                comp_path = output_path / f"{idx:05d}_comparison.png"
                cv2.imwrite(str(comp_path), comparison)
                
                logger.info(f"  视角{idx}: PSNR={psnr:.2f}dB, 差异={mean_diff:.6f}")
        
        logger.info(f"[PureGaussianRenderer] 渲染完成，共保存 {len(cameras)} 张图像")
        
        if all_stats:
            avg_psnr = np.mean([s['psnr'] for s in all_stats])
            avg_diff = np.mean([s['mean_diff'] for s in all_stats])
            logger.info(f"[PureGaussianRenderer] 平均PSNR: {avg_psnr:.2f} dB")
            logger.info(f"[PureGaussianRenderer] 平均差异: {avg_diff:.6f}")
        
        return all_stats


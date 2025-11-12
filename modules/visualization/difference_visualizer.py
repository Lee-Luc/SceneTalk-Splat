"""
差异可视化模块
用于对比原图和渲染图的差异，验证3DGS渲染质量
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # 非交互后端
import matplotlib.pyplot as plt

from modules.utils.logger import logger


class DifferenceVisualizer:
    """差异可视化器"""
    
    def __init__(self):
        """初始化差异可视化器"""
        pass
    
    def compute_difference(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        normalize: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        计算两张图像的差异
        
        Args:
            img1: 第一张图像 (H, W, 3)
            img2: 第二张图像 (H, W, 3)
            normalize: 是否归一化差异图
            
        Returns:
            差异图和统计信息
        """
        # 确保图像尺寸相同
        if img1.shape != img2.shape:
            logger.warning(f"图像尺寸不匹配: {img1.shape} vs {img2.shape}")
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 转换为float32
        img1_f = img1.astype(np.float32)
        img2_f = img2.astype(np.float32)
        
        # 计算绝对差异
        diff = np.abs(img1_f - img2_f)
        
        # 计算统计信息
        stats = {
            'mae': np.mean(diff),  # 平均绝对误差
            'mse': np.mean((img1_f - img2_f) ** 2),  # 均方误差
            'max_error': np.max(diff),  # 最大误差
            'psnr': self._compute_psnr(img1_f, img2_f)  # 峰值信噪比
        }
        
        # 归一化差异图以便可视化
        if normalize:
            diff_vis = (diff / 255.0 * 10.0)  # 放大10倍以便观察
            diff_vis = np.clip(diff_vis, 0, 255).astype(np.uint8)
        else:
            diff_vis = diff.astype(np.uint8)
        
        return diff_vis, stats
    
    def _compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算PSNR"""
        mse = np.mean((img1 - img2) ** 2)
        if mse < 1e-10:
            return 100.0
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def create_comparison_with_difference(
        self,
        original: np.ndarray,
        rendered: np.ndarray,
        output_path: str,
        title: str = "Original vs Rendered"
    ) -> dict:
        """
        创建包含差异图的三联对比
        
        Args:
            original: 原始图像
            rendered: 渲染图像
            output_path: 输出路径
            title: 标题
            
        Returns:
            差异统计信息
        """
        # 计算差异
        diff_img, stats = self.compute_difference(original, rendered)
        
        # 创建差异热力图
        diff_heatmap = cv2.applyColorMap(
            cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY),
            cv2.COLORMAP_JET
        )
        
        # 创建三联图
        h, w = original.shape[:2]
        canvas = np.ones((h, w * 3 + 40, 3), dtype=np.uint8) * 255
        
        # 放置图像
        canvas[:, :w] = original
        canvas[:, w+20:w*2+20] = rendered
        canvas[:, w*2+40:] = diff_heatmap
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Original", (10, 30), font, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, "Rendered (3DGS)", (w+30, 30), font, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, "Difference (x10)", (w*2+50, 30), font, 0.8, (0, 0, 0), 2)
        
        # 添加统计信息
        y_offset = h - 80
        info_text = [
            f"PSNR: {stats['psnr']:.2f} dB",
            f"MAE: {stats['mae']:.2f}",
            f"Max Error: {stats['max_error']:.0f}"
        ]
        for i, text in enumerate(info_text):
            cv2.putText(
                canvas, text,
                (w*2+50, y_offset + i*25),
                font, 0.6, (0, 0, 0), 2
            )
        
        # 保存
        cv2.imwrite(output_path, canvas)
        logger.info(f"差异对比图已保存: {output_path}")
        logger.info(f"  PSNR: {stats['psnr']:.2f} dB (越高越好，>30dB表示高质量)")
        logger.info(f"  平均误差: {stats['mae']:.2f}/255")
        
        return stats
    
    def create_error_distribution(
        self,
        original: np.ndarray,
        rendered: np.ndarray,
        output_path: str
    ):
        """
        创建误差分布直方图
        
        Args:
            original: 原始图像
            rendered: 渲染图像
            output_path: 输出路径
        """
        # 计算每个像素的误差
        diff = np.abs(original.astype(np.float32) - rendered.astype(np.float32))
        diff_per_pixel = np.mean(diff, axis=2).flatten()  # 每个像素的平均误差
        
        # 创建直方图
        plt.figure(figsize=(10, 6))
        plt.hist(diff_per_pixel, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel('Pixel Error', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Error Distribution (Original vs Rendered)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_error = np.mean(diff_per_pixel)
        median_error = np.median(diff_per_pixel)
        plt.axvline(mean_error, color='red', linestyle='--', 
                   label=f'Mean: {mean_error:.2f}')
        plt.axvline(median_error, color='green', linestyle='--', 
                   label=f'Median: {median_error:.2f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"误差分布图已保存: {output_path}")


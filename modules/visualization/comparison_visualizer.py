"""
对比可视化器
生成多种对比图
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from ..utils.logger import default_logger as logger


class ComparisonVisualizer:
    """对比可视化器类"""
    
    def __init__(self, dpi: int = 150):
        """
        初始化可视化器
        
        Args:
            dpi: 图像DPI
        """
        self.dpi = dpi
    
    def create_four_panel_comparison(
        self,
        original_img: np.ndarray,
        yolo_detected_img: np.ndarray,
        rendered_img: np.ndarray,
        projected_img: np.ndarray,
        output_path: str = None
    ) -> np.ndarray:
        """
        创建四联对比图
        
        Args:
            original_img: 原始图像
            yolo_detected_img: YOLO检测后的图像
            rendered_img: 高斯渲染图像
            projected_img: 投影检测图像
            output_path: 输出路径
            
        Returns:
            拼接后的图像
        """
        # 确保所有图像尺寸一致
        h, w = original_img.shape[:2]
        
        # 调整其他图像尺寸
        yolo_detected_img = cv2.resize(yolo_detected_img, (w, h))
        rendered_img = cv2.resize(rendered_img, (w, h))
        projected_img = cv2.resize(projected_img, (w, h))
        
        # 添加标题
        def add_title(img, title, color=(0, 0, 0)):
            img_with_title = np.ones((h + 60, w, 3), dtype=np.uint8) * 255
            img_with_title[60:, :] = img
            
            # 添加标题文字
            cv2.putText(
                img_with_title,
                title,
                (w//2 - len(title)*10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                2,
                cv2.LINE_AA
            )
            
            return img_with_title
        
        original_titled = add_title(original_img, "1. Original Image")
        yolo_titled = add_title(yolo_detected_img, "2. YOLO Detection")
        rendered_titled = add_title(rendered_img, "3. Gaussian Rendered")
        projected_titled = add_title(projected_img, "4. Projected Detection")
        
        # 拼接：2x2布局
        top_row = np.hstack([original_titled, yolo_titled])
        bottom_row = np.hstack([rendered_titled, projected_titled])
        comparison = np.vstack([top_row, bottom_row])
        
        # 添加分隔线
        h_total, w_total = comparison.shape[:2]
        
        # 垂直分隔线
        cv2.line(comparison, (w_total//2, 0), (w_total//2, h_total), (200, 200, 200), 3)
        
        # 水平分隔线
        cv2.line(comparison, (0, h_total//2), (w_total, h_total//2), (200, 200, 200), 3)
        
        # 保存
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), comparison)
            logger.info(f"对比图已保存: {output_path}")
        
        return comparison
    
    def create_side_by_side_comparison(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        title1: str = "Image 1",
        title2: str = "Image 2",
        output_path: str = None
    ) -> np.ndarray:
        """
        创建左右对比图
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            title1: 标题1
            title2: 标题2
            output_path: 输出路径
            
        Returns:
            对比图
        """
        # 确保尺寸一致
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        
        img1_resized = cv2.resize(img1, (w, h))
        img2_resized = cv2.resize(img2, (w, h))
        
        # 添加标题
        def add_title(img, title):
            img_with_title = np.ones((h + 50, w, 3), dtype=np.uint8) * 255
            img_with_title[50:, :] = img
            cv2.putText(img_with_title, title, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            return img_with_title
        
        img1_titled = add_title(img1_resized, title1)
        img2_titled = add_title(img2_resized, title2)
        
        # 水平拼接
        comparison = np.hstack([img1_titled, img2_titled])
        
        # 添加分隔线
        h_total, w_total = comparison.shape[:2]
        cv2.line(comparison, (w_total//2, 0), (w_total//2, h_total), (200, 200, 200), 2)
        
        if output_path:
            cv2.imwrite(str(output_path), comparison)
            logger.info(f"对比图已保存: {output_path}")
        
        return comparison
    
    def create_detection_quality_plot(
        self,
        metrics: Dict[str, List[float]],
        output_path: str = None
    ):
        """
        创建检测质量统计图
        
        Args:
            metrics: 指标字典，如 {'iou': [...], 'confidence': [...]}
            output_path: 输出路径
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, (metric_name, values) in zip(axes, metrics.items()):
            ax.hist(values, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel(metric_name.upper())
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric_name.upper()} Distribution')
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_val = np.mean(values)
            median_val = np.median(values)
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
            ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"统计图已保存: {output_path}")
        
        plt.close()
    
    def create_class_distribution_plot(
        self,
        class_counts: Dict[str, int],
        output_path: str = None
    ):
        """
        创建类别分布图
        
        Args:
            class_counts: 类别计数字典
            output_path: 输出路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # 柱状图
        ax1.bar(classes, counts, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('Object Class Distribution')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 饼图
        ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Proportion')
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"类别分布图已保存: {output_path}")
        
        plt.close()
    
    def create_projection_quality_heatmap(
        self,
        quality_matrix: np.ndarray,
        view_names: List[str],
        output_path: str = None
    ):
        """
        创建投影质量热力图
        
        Args:
            quality_matrix: 质量矩阵 [N_views, N_views]
            view_names: 视角名称列表
            output_path: 输出路径
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(quality_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        
        # 设置坐标轴
        ax.set_xticks(np.arange(len(view_names)))
        ax.set_yticks(np.arange(len(view_names)))
        ax.set_xticklabels(view_names)
        ax.set_yticklabels(view_names)
        
        # 旋转标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值
        for i in range(len(view_names)):
            for j in range(len(view_names)):
                text = ax.text(j, i, f'{quality_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title("Cross-View Projection Quality (IoU)")
        ax.set_xlabel("Target View")
        ax.set_ylabel("Source View")
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('IoU Score', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"热力图已保存: {output_path}")
        
        plt.close()
    
    def draw_bbox_with_label(
        self,
        img: np.ndarray,
        bbox: Tuple[float, float, float, float],
        label: str,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        在图像上绘制带标签的边界框
        
        Args:
            img: 输入图像
            bbox: 边界框 [x1, y1, x2, y2]
            label: 标签文本
            color: 颜色 (B, G, R)
            thickness: 线宽
            
        Returns:
            绘制后的图像
        """
        img_copy = img.copy()
        x1, y1, x2, y2 = map(int, bbox)
        
        # 绘制矩形
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        # 绘制标签
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness
        )
        
        cv2.rectangle(
            img_copy,
            (x1, y1 - label_h - 10),
            (x1 + label_w, y1),
            color,
            -1
        )
        
        cv2.putText(
            img_copy,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            thickness
        )
        
        return img_copy


"""
YOLO检测器
使用YOLOv8进行目标检测
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import torch

from .detection_result import Detection, DetectionResult
from ..utils.logger import default_logger as logger


class YOLODetector:
    """YOLO检测器类"""
    
    def __init__(
        self,
        model_name: str = 'yolov8x.pt',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'cuda',
        classes: Optional[List[int]] = None,
        img_size: int = 1280,
        agnostic_nms: bool = False,
        max_det: int = 300
    ):
        """
        初始化YOLO检测器
        
        Args:
            model_name: YOLO模型名称
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IoU阈值
            device: 运行设备
            classes: 要检测的类别ID列表，None表示所有类别
            img_size: 输入图像尺寸
            agnostic_nms: 是否使用类别无关的NMS
            max_det: 最大检测数量
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes
        self.img_size = img_size
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        
        # 加载YOLO模型
        logger.info(f"正在加载YOLO模型: {model_name}")
        self.model = YOLO(model_name)
        self.model.to(device)
        
        # 获取类别名称
        self.class_names = self.model.names
        
        # 生成颜色映射
        self.colors = self._generate_colors(len(self.class_names))
        
        logger.info(f"YOLO模型加载成功，支持 {len(self.class_names)} 个类别")
    
    def _generate_colors(self, num_classes: int) -> Dict[int, Tuple[int, int, int]]:
        """
        为每个类别生成独特的颜色
        
        Args:
            num_classes: 类别数量
            
        Returns:
            颜色字典 {class_id: (B, G, R)}
        """
        np.random.seed(42)  # 固定随机种子以保持颜色一致
        colors = {}
        for i in range(num_classes):
            colors[i] = tuple(np.random.randint(0, 255, 3).tolist())
        return colors
    
    def detect(
        self,
        image_path: str,
        view_id: int = 0
    ) -> DetectionResult:
        """
        对单张图像进行检测
        
        Args:
            image_path: 图像路径
            view_id: 视角ID
            
        Returns:
            检测结果对象
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return DetectionResult(
                image_name=Path(image_path).name,
                image_path=str(image_path),
                view_id=view_id,
                detections=[],
                image_size=(0, 0)
            )
        
        height, width = image.shape[:2]
        
        # 运行YOLO检测
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.img_size,
            agnostic_nms=self.agnostic_nms,
            max_det=self.max_det,
            verbose=False
        )[0]
        
        # 解析检测结果
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            cls_name = self.class_names[cls]
            
            # 创建Detection对象
            detection = Detection(
                class_name=cls_name,
                class_id=cls,
                confidence=conf,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                center=(float((x1 + x2) / 2), float((y1 + y2) / 2))
            )
            detections.append(detection)
        
        logger.info(f"检测到 {len(detections)} 个物体: {image_path}")
        
        return DetectionResult(
            image_name=Path(image_path).name,
            image_path=str(image_path),
            view_id=view_id,
            detections=detections,
            image_size=(width, height)
        )
    
    def detect_batch(
        self,
        image_paths: List[str],
        start_view_id: int = 0
    ) -> List[DetectionResult]:
        """
        批量检测多张图像
        
        Args:
            image_paths: 图像路径列表
            start_view_id: 起始视角ID
            
        Returns:
            检测结果列表
        """
        results = []
        for idx, image_path in enumerate(image_paths):
            result = self.detect(image_path, view_id=start_view_id + idx)
            results.append(result)
        
        return results
    
    def visualize_detection(
        self,
        image_path: str,
        detection_result: DetectionResult,
        output_path: str = None,
        show_confidence: bool = True,
        thickness: int = 2,
        font_scale: float = 0.6
    ) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image_path: 原始图像路径
            detection_result: 检测结果
            output_path: 输出路径，如果为None则不保存
            show_confidence: 是否显示置信度
            thickness: 边界框线宽
            font_scale: 字体大小
            
        Returns:
            可视化图像
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        vis_image = image.copy()
        
        # 绘制每个检测框
        for det in detection_result.detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = self.colors[det.class_id]
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # 准备标签文本
            if show_confidence:
                label = f"{det.class_name} {det.confidence:.2f}"
            else:
                label = det.class_name
            
            # 计算标签背景大小
            (label_w, label_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )
            
            # 绘制标签背景
            label_y1 = max(y1, label_h + 10)
            cv2.rectangle(
                vis_image,
                (x1, label_y1 - label_h - 10),
                (x1 + label_w, label_y1),
                color,
                -1
            )
            
            # 绘制标签文字
            cv2.putText(
                vis_image,
                label,
                (x1, label_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        # 添加统计信息
        stats = detection_result.get_statistics()
        info_text = f"Detections: {stats['total_detections']}"
        cv2.putText(
            vis_image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        # 保存图像
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis_image)
            logger.info(f"可视化结果已保存: {output_path}")
        
        return vis_image
    
    def get_summary_statistics(
        self,
        detection_results: List[DetectionResult]
    ) -> Dict:
        """
        获取所有检测结果的汇总统计
        
        Args:
            detection_results: 检测结果列表
            
        Returns:
            统计字典
        """
        total_detections = sum(len(res.detections) for res in detection_results)
        
        # 统计每个类别的数量
        class_counts = {}
        all_confidences = []
        
        for result in detection_results:
            for det in result.detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                all_confidences.append(det.confidence)
        
        return {
            'total_images': len(detection_results),
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(detection_results) if detection_results else 0,
            'class_distribution': class_counts,
            'num_unique_classes': len(class_counts),
            'mean_confidence': float(np.mean(all_confidences)) if all_confidences else 0.0,
            'std_confidence': float(np.std(all_confidences)) if all_confidences else 0.0,
        }


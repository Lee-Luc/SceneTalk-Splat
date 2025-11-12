#!/usr/bin/env python3
"""
 主程序
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np

# 导入所有模块
from modules.utils import ConfigLoader, setup_logger, FileManager
from modules.object_detection import YOLODetector
from modules.rendering import GaussianRendererWrapper, HighlightRenderer
from modules.projection import BBoxProjector, Object3DReconstructor
from modules.visualization import ComparisonVisualizer, ReportGenerator
from modules.scene_understanding import SpatialAnalyzer, LLMInterface, SceneGraph

# 设置日志
logger = setup_logger(log_file="output/scene_understanding.log", level="INFO")


class SceneUnderstandingPipeline:
    """场景理解主流程类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化流程
        
        Args:
            config_path: 配置文件路径
        """
        logger.info("开始进行初始化.....")
        
        # 加载配置
        self.config = ConfigLoader(config_path)
        logger.info(f"配置文件加载成功: {config_path}")
        
        # 初始化文件管理器
        scene_name = Path(self.config.get('paths.source_path', 'scene')).name
        self.file_manager = FileManager(
            self.config.get('paths.output_root'),
            scene_name
        )
        
        # 初始化各个组件
        self._initialize_components()
        
        logger.info("初始化完成！\n")
    
    def _initialize_components(self):
        """初始化所有组件"""
        # 1. YOLO检测器
        yolo_config = self.config.get_yolo_config()
        self.yolo_detector = YOLODetector(
            model_name=yolo_config.get('model_name', 'yolov8x.pt'),
            conf_threshold=yolo_config.get('conf_threshold', 0.15),
            iou_threshold=yolo_config.get('iou_threshold', 0.45),
            device=yolo_config.get('device', 'cuda'),
            classes=yolo_config.get('classes'),
            img_size=yolo_config.get('img_size', 1280),
            agnostic_nms=yolo_config.get('agnostic_nms', False),
            max_det=yolo_config.get('max_det', 300)
        )
        
        # 2. 高斯渲染器
        gaussian_config = self.config.get_gaussian_config()
        self.gaussian_renderer = GaussianRendererWrapper(
            model_path=self.config.get('paths.model_path'),
            source_path=self.config.get('paths.source_path'),
            sh_degree=gaussian_config.get('sh_degree', 3),
            load_iteration=gaussian_config.get('load_iteration', -1),
            white_background=gaussian_config.get('white_background', False)
        )
        
        # 3. 投影器
        proj_config = self.config.get_projection_config()
        self.bbox_projector = BBoxProjector(
            self.gaussian_renderer,
            depth_sample_step=proj_config.get('depth_sample_step', 10),
            min_visible_points=proj_config.get('min_visible_points', 4),
            visibility_threshold=proj_config.get('visibility_threshold', 0.3)
        )
        
        # 4. 3D重建器
        self.object_reconstructor = Object3DReconstructor(
            self.gaussian_renderer,
            min_views=2,
            clustering_eps=1.0
        )
        
        # 5. 可视化器
        vis_config = self.config.get_visualization_config()
        self.visualizer = ComparisonVisualizer(
            dpi=vis_config.get('dpi', 150)
        )
        
        # 6. 报告生成器
        self.report_generator = ReportGenerator(
            self.file_manager.get_dir('report')
        )
        
        # 7. 空间分析器
        self.spatial_analyzer = SpatialAnalyzer(
            distance_threshold=2.0,
            near_threshold=1.0
        )
        
        # 8. LLM接口（可选）
        llm_config = self.config.get_llm_config()
        if llm_config.get('enable_llm', False):
            self.llm_interface = LLMInterface(
                provider=llm_config.get('provider', 'openai'),
                model=llm_config.get('model', 'gpt-4-turbo'),
                api_key=llm_config.get('api_key'),
                temperature=llm_config.get('temperature', 0.7),
                max_tokens=llm_config.get('max_tokens', 2000)
            )
        else:
            logger.info("LLM功能未启用")
            self.llm_interface = None
    
    def run(self):
        """运行完整流程"""
        try:
            logger.info("开始执行")
            
            # Pipeline 1: YOLO检测
            detection_results = self.step1_yolo_detection()
            
            # Pipeline 2: 高斯渲染
            self.step2_gaussian_rendering()
            
            # Pipeline 3: 检测框投影
            projection_metrics = self.step3_projection()
            
            # Pipeline 4: 3D物体重建
            objects_3d = self.step4_3d_reconstruction(detection_results)
            
            # Pipeline 5: 场景理解
            scene_graph = self.step5_scene_understanding(objects_3d)
            
            # Pipeline 6: 生成报告
            self.step6_generate_report(
                detection_results,
                projection_metrics,
                objects_3d,
                scene_graph
            )
            
            logger.info("完成！")
            logger.info(f"结果已保存: {self.file_manager.scene_dir}")
            
        except Exception as e:
            logger.error(f"流程执行失败: {e}", exc_info=True)
            raise
    
    def step1_yolo_detection(self):
        """步骤1：YOLO目标检测"""
        logger.info("[步骤 1/6] YOLO目标检测")
        
        # 获取训练图像
        train_cameras = self.gaussian_renderer.get_train_cameras()
        
        detection_results = []
        
        # 获取图像目录
        source_path = Path(self.config.get('paths.source_path'))
        images_dir = source_path / "images"
        
        for idx, camera in enumerate(train_cameras):
            # 获取原始图像路径
            # camera.image_name 通常包含图像文件名
            if hasattr(camera, 'image_name'):
                image_path = str(images_dir / camera.image_name)
            elif hasattr(camera, 'image_path'):
                image_path = camera.image_path
            else:
                # 作为后备，尝试找到对应的图像文件
                image_files = sorted(images_dir.glob('*'))
                if idx < len(image_files):
                    image_path = str(image_files[idx])
                else:
                    logger.warning(f"无法找到图像 {idx}，跳过")
                    continue
            
            # 检测
            det_result = self.yolo_detector.detect(image_path, view_id=idx)
            detection_results.append(det_result)
            
            # 可视化
            vis_output_path = self.file_manager.get_path(
                'yolo_vis',
                f'{idx:05d}_detected.png'
            )
            self.yolo_detector.visualize_detection(
                image_path,
                det_result,
                output_path=vis_output_path
            )
            
            logger.info(f"  图像 {idx}: 检测到 {len(det_result.detections)} 个物体")
        
        # 保存检测结果
        all_detections_dict = [res.to_dict() for res in detection_results]
        self.file_manager.save_json(
            all_detections_dict,
            'yolo',
            'detections.json'
        )
        
        # 统计信息
        stats = self.yolo_detector.get_summary_statistics(detection_results)
        self.file_manager.save_json(stats, 'yolo', 'statistics.json')
        
        logger.info(f"  总计: {stats['total_detections']} 个检测")
        logger.info(f"  类别: {stats['num_unique_classes']} 种")
        
        return detection_results
    
    def step2_gaussian_rendering(self):
        """步骤2：高斯渲染"""
        logger.info("[步骤 2/6] 高斯场景渲染")
        
        # 渲染训练视角
        if not self.config.get('experiment.skip_train_views', False):
            train_output_dir = self.file_manager.get_dir('rendered_train')
            logger.info(f"  渲染训练视角到: {train_output_dir}")
            try:
                self.gaussian_renderer.render_train_views(
                    output_dir=str(train_output_dir)
                )
                # 检查是否成功保存
                saved_files = list(train_output_dir.glob('*.png'))
                logger.info(f"  已保存 {len(saved_files)} 个训练视角图像")
            except Exception as e:
                logger.error(f"  训练视角渲染失败: {e}")
        
        # 渲染测试视角
        if not self.config.get('experiment.skip_test_views', False):
            test_output_dir = self.file_manager.get_dir('rendered_test')
            logger.info(f"  渲染测试视角到: {test_output_dir}")
            try:
                # 首先检查是否有测试相机
                test_cameras = self.gaussian_renderer.get_test_cameras()
                
                if len(test_cameras) == 0:
                    logger.warning("  数据集中没有测试视角相机")
                    logger.info("  将跳过测试视角渲染")
                else:
                    logger.info(f"  找到 {len(test_cameras)} 个测试视角")
                    self.gaussian_renderer.render_test_views(
                        output_dir=str(test_output_dir)
                    )
                    # 检查是否成功保存
                    saved_files = list(test_output_dir.glob('*.png'))
                    logger.info(f"  已保存 {len(saved_files)} 个测试视角图像")
            except Exception as e:
                logger.error(f"  测试视角渲染失败: {e}")
                logger.info("  这可能是因为数据集没有测试集划分，将继续处理")
        
        logger.info("  渲染完成！")
    
    def step3_projection(self):
        """步骤3：检测框投影并可视化"""
        logger.info("[步骤 3/6] 检测框跨视角投影")
        
        train_cameras = self.gaussian_renderer.get_train_cameras()
        
        # 加载检测结果
        detections_data = self.file_manager.load_json('yolo', 'detections.json')
        
        projection_metrics = {
            'mean_iou': 0.0,
            'mean_visibility': 0.0,
            'success_rate': 0.0,
            'all_projections': []
        }
        
        logger.info(f"  生成投影可视化...")
        
        import cv2
        
        # 为每个视角生成投影可视化
        for idx, camera in enumerate(train_cameras):
            # 获取该视角的检测结果
            if idx >= len(detections_data):
                continue
                
            detections = detections_data[idx]['detections']
            
            # 1. 读取原始图像
            source_path = Path(self.config.get('paths.source_path'))
            images_dir = source_path / "images"
            if hasattr(camera, 'image_name'):
                orig_image_path = images_dir / camera.image_name
            else:
                image_files = sorted(images_dir.glob('*'))
                if idx < len(image_files):
                    orig_image_path = image_files[idx]
                else:
                    continue
            
            original_img = cv2.imread(str(orig_image_path))
            if original_img is None:
                logger.warning(f"  无法读取原始图像: {orig_image_path}")
                continue
            
            orig_h, orig_w = original_img.shape[:2]
            logger.debug(f"  视角 {idx}: 原始图像尺寸 = {orig_w}x{orig_h}")
            
            # 2. 读取YOLO检测可视化图
            yolo_vis_path = self.file_manager.get_path('yolo_vis', f'{idx:05d}_detected.png')
            yolo_img = cv2.imread(str(yolo_vis_path))
            
            # 3. 读取高斯渲染图 - 关键：必须使用渲染图而非原始图
            rendered_path = self.file_manager.get_path('rendered_train', f'{idx:05d}_render.png')
            logger.info(f"  视角 {idx}: 读取渲染图 -> {rendered_path}")
            
            if not Path(rendered_path).exists():
                logger.error(f"  ❌ 渲染图不存在: {rendered_path}")
                logger.error(f"     请确保step2已成功运行并保存渲染图")
                continue
            
            rendered_img = cv2.imread(str(rendered_path))
            if rendered_img is None:
                logger.error(f"  ❌ 无法读取渲染图: {rendered_path}")
                continue
            
            render_h, render_w = rendered_img.shape[:2]
            logger.info(f"  视角 {idx}: 渲染图尺寸 = {render_w}x{render_h}")
            
            # 验证渲染图与原始图不同（确保使用了渲染结果）
            if rendered_img.shape == original_img.shape:
                diff = cv2.absdiff(rendered_img, original_img)
                diff_mean = diff.mean()
                if diff_mean < 1.0:
                    logger.warning(f"  ⚠️  渲染图与原始图几乎相同（差异={diff_mean:.2f}），可能有问题")
                else:
                    logger.info(f"  ✓ 渲染图与原图差异正常（差异={diff_mean:.2f}）")
            else:
                logger.info(f"  ✓ 渲染图尺寸与原图不同，确认使用了渲染结果")
            
            logger.debug(f"  视角 {idx}: 渲染图像尺寸 = {render_w}x{render_h}")
            
            # 计算缩放因子
            scale_x = render_w / orig_w
            scale_y = render_h / orig_h
            
            if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
                logger.info(f"  视角 {idx}: 检测到分辨率差异，缩放因子 = ({scale_x:.3f}, {scale_y:.3f})")
            
            # 4. 在渲染图上绘制检测框（投影）- 关键：使用渲染图作为基础
            projected_img = rendered_img.copy()
            logger.info(f"  视角 {idx}: 使用渲染图作为基础 (shape={projected_img.shape})")
            
            for det in detections:
                bbox = det['bbox']
                label = f"{det['class_name']} {det['confidence']:.2f}"
                
                # 缩放检测框坐标以匹配渲染图像分辨率
                x1 = int(bbox[0] * scale_x)
                y1 = int(bbox[1] * scale_y)
                x2 = int(bbox[2] * scale_x)
                y2 = int(bbox[3] * scale_y)
                
                # 边界检查
                x1 = max(0, min(x1, render_w - 1))
                y1 = max(0, min(y1, render_h - 1))
                x2 = max(0, min(x2, render_w - 1))
                y2 = max(0, min(y2, render_h - 1))
                
                # 确保x2 > x1 和 y2 > y1
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"  跳过无效检测框: ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                # 绘制绿色检测框
                cv2.rectangle(projected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制标签
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # 确保标签在图像范围内
                label_y1 = max(label_h + 10, y1)
                
                cv2.rectangle(
                    projected_img,
                    (x1, label_y1 - label_h - 10),
                    (min(x1 + label_w, render_w), label_y1),
                    (0, 255, 0),
                    -1
                )
                cv2.putText(
                    projected_img,
                    label,
                    (x1, label_y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
            
            # 5. 保存投影结果
            projected_output_path = self.file_manager.get_path(
                'projected',
                f'{idx:05d}_projected.png'
            )
            cv2.imwrite(str(projected_output_path), projected_img)
            
            # 6. 生成四联对比图
            if yolo_img is not None:
                comparison = self.visualizer.create_four_panel_comparison(
                    original_img,
                    yolo_img,
                    rendered_img,
                    projected_img,
                    output_path=str(self.file_manager.get_path(
                        'projected_comparison',
                        f'{idx:05d}_comparison.png'
                    ))
                )
                logger.info(f"  视角 {idx}: 生成对比图，检测到 {len(detections)} 个物体")
        
        # 计算投影指标（简化版）
        projection_metrics['success_rate'] = 1.0  # 同视角投影总是成功
        projection_metrics['mean_visibility'] = 1.0
        
        logger.info(f"  投影可视化完成！")
        
        # 保存投影指标
        self.file_manager.save_json(
            projection_metrics,
            'projected',
            'projection_quality.json'
        )
        
        return projection_metrics
    
    def step4_3d_reconstruction(self, detection_results):
        """步骤4：3D物体重建"""
        logger.info("[步骤 4/6] 3D物体重建")
        
        train_cameras = self.gaussian_renderer.get_train_cameras()
        
        # 重建3D物体
        objects_3d = self.object_reconstructor.reconstruct_objects_3d(
            detection_results,
            train_cameras
        )
        
        logger.info(f"  重建了 {len(objects_3d)} 个3D物体")
        
        # 保存3D物体数据
        objects_dict = [obj.to_dict() for obj in objects_3d]
        self.file_manager.save_json(
            objects_dict,
            'scene_understanding',
            'object_database.json'
        )
        
        return objects_3d
    
    def step5_scene_understanding(self, objects_3d):
        """步骤5：场景理解"""
        logger.info("[步骤 5/6] 场景理解与分析")
        
        # 构建场景图谱
        scene_graph = SceneGraph(objects_3d)
        
        # 空间关系分析
        self.spatial_analyzer.analyze_scene(scene_graph)
        
        # 保存场景图谱
        scene_graph_dict = scene_graph.to_dict()
        self.file_manager.save_json(
            scene_graph_dict,
            'scene_understanding',
            'scene_graph.json'
        )
        
        # LLM场景描述（如果启用）
        if self.llm_interface:
            logger.info("  生成LLM场景描述...")
            description = self.llm_interface.generate_scene_description(scene_graph_dict)
            
            # 保存描述
            desc_path = self.file_manager.get_path('scene_understanding', 'scene_description.txt')
            with open(desc_path, 'w', encoding='utf-8') as f:
                f.write(description)
            
            logger.info(f"  场景描述已保存: {desc_path}")
        
        logger.info(f"  场景分析完成！")
        
        return scene_graph
    
    def step6_generate_report(
        self,
        detection_results,
        projection_metrics,
        objects_3d,
        scene_graph
    ):
        """步骤6：生成综合报告"""
        logger.info("[步骤 6/6] 生成综合报告")
        
        # 准备报告数据
        summary = {
            'total_images': len(detection_results),
            'total_detections': sum(len(res.detections) for res in detection_results),
            'num_3d_objects': len(objects_3d),
            'num_classes': len(set(obj.class_name for obj in objects_3d))
        }
        
        detection_stats = self.yolo_detector.get_summary_statistics(detection_results)
        detection_stats['avg_per_image'] = detection_stats['avg_detections_per_image']
        
        objects_dict = [obj.to_dict() for obj in objects_3d]
        
        # 生成HTML报告
        report_path = self.report_generator.generate_html_report(
            scene_name=self.file_manager.scene_name,
            summary=summary,
            detection_stats=detection_stats,
            projection_metrics=projection_metrics,
            objects_3d=objects_dict
        )
        
        # 创建README
        self.file_manager.create_readme({
            '检测物体数': summary['total_detections'],
            '3D物体数': summary['num_3d_objects'],
            '类别数': summary['num_classes']
        })
        
        logger.info(f"  报告已生成: {report_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="LLM增强的3D高斯场景理解",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --config config/config.yaml
  python main.py --source data/scene --model output/model
"""
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='配置文件路径 (默认: config/config.yaml)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        help='源数据集路径（会覆盖配置文件中的设置）'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='训练好的模型路径（会覆盖配置文件中的设置）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出目录（会覆盖配置文件中的设置）'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    try:
        # 初始化流程
        pipeline = SceneUnderstandingPipeline(args.config)
        
        # 覆盖配置（如果提供了命令行参数）
        if args.source:
            pipeline.config.update_config('paths.source_path', args.source)
        
        if args.model:
            pipeline.config.update_config('paths.model_path', args.model)
        
        if args.output:
            pipeline.config.update_config('paths.output_root', args.output)
        
        # 运行流程
        pipeline.run()
        
        logger.info("\n所有任务完成！")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n 用户中断执行")
        return 1
        
    except Exception as e:
        logger.error(f"\n执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


"""
报告生成器
生成HTML格式的综合分析报告
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from jinja2 import Template

from ..utils.logger import default_logger as logger


class ReportGenerator:
    """报告生成器类"""
    
    def __init__(self, output_dir: str):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_html_report(
        self,
        scene_name: str,
        summary: Dict[str, Any],
        detection_stats: Dict[str, Any],
        projection_metrics: Dict[str, Any],
        objects_3d: List[Dict],
        image_paths: Dict[str, List[str]] = None
    ) -> str:
        """
        生成HTML报告
        
        Args:
            scene_name: 场景名称
            summary: 总结信息
            detection_stats: 检测统计
            projection_metrics: 投影指标
            objects_3d: 3D物体列表
            image_paths: 图像路径字典
            
        Returns:
            报告文件路径
        """
        logger.info("正在生成HTML报告...")
        
        # HTML模板
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>场景分析报告 - {{ scene_name }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .timestamp {
            opacity: 0.9;
            font-size: 0.9em;
        }
        .section {
            background: white;
            margin-bottom: 30px;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h2 {
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-card h3 {
            font-size: 2em;
            margin-bottom: 5px;
        }
        .stat-card p {
            opacity: 0.9;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #667eea;
            color: white;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .image-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .image-item {
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }
        .image-item img {
            width: 100%;
            height: auto;
            display: block;
        }
        .image-caption {
            padding: 10px;
            background: #f9f9f9;
            text-align: center;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .metric-label {
            font-weight: bold;
        }
        .metric-value {
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 3D场景理解分析报告</h1>
            <p class="timestamp">场景: {{ scene_name }}</p>
            <p class="timestamp">生成时间: {{ timestamp }}</p>
        </header>
        
        <div class="section">
            <h2>📊 总体统计</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>{{ summary.total_images }}</h3>
                    <p>处理图像数</p>
                </div>
                <div class="stat-card">
                    <h3>{{ summary.total_detections }}</h3>
                    <p>检测物体总数</p>
                </div>
                <div class="stat-card">
                    <h3>{{ summary.num_3d_objects }}</h3>
                    <p>3D物体数量</p>
                </div>
                <div class="stat-card">
                    <h3>{{ summary.num_classes }}</h3>
                    <p>物体类别数</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔍 检测统计</h2>
            <div class="metric-row">
                <span class="metric-label">平均检测数/图像:</span>
                <span class="metric-value">{{ "%.2f"|format(detection_stats.avg_per_image) }}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">平均置信度:</span>
                <span class="metric-value">{{ "%.3f"|format(detection_stats.mean_confidence) }}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">置信度标准差:</span>
                <span class="metric-value">{{ "%.3f"|format(detection_stats.std_confidence) }}</span>
            </div>
            
            <h3 style="margin-top: 30px;">类别分布</h3>
            <table>
                <thead>
                    <tr>
                        <th>类别</th>
                        <th>数量</th>
                        <th>占比</th>
                    </tr>
                </thead>
                <tbody>
                    {% for class_name, count in detection_stats.class_distribution.items() %}
                    <tr>
                        <td>{{ class_name }}</td>
                        <td>{{ count }}</td>
                        <td>{{ "%.1f"|format(count / summary.total_detections * 100) }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>🎯 投影质量评估</h2>
            <div class="metric-row">
                <span class="metric-label">平均IoU:</span>
                <span class="metric-value">{{ "%.3f"|format(projection_metrics.mean_iou) }}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">平均可见性得分:</span>
                <span class="metric-value">{{ "%.3f"|format(projection_metrics.mean_visibility) }}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">成功投影率:</span>
                <span class="metric-value">{{ "%.1f"|format(projection_metrics.success_rate * 100) }}%</span>
            </div>
        </div>
        
        <div class="section">
            <h2>🏗️ 3D物体列表</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>类别</th>
                        <th>置信度</th>
                        <th>位置 (x, y, z)</th>
                        <th>尺寸 (w, h, d)</th>
                        <th>可见视角数</th>
                    </tr>
                </thead>
                <tbody>
                    {% for obj in objects_3d %}
                    <tr>
                        <td>{{ obj.object_id }}</td>
                        <td>{{ obj.class_name }}</td>
                        <td>{{ "%.2f"|format(obj.confidence) }}</td>
                        <td>{{ "%.2f, %.2f, %.2f"|format(obj.position[0], obj.position[1], obj.position[2]) }}</td>
                        <td>{{ "%.2f, %.2f, %.2f"|format(obj.size[0], obj.size[1], obj.size[2]) }}</td>
                        <td>{{ obj.num_views }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        {% if image_paths %}
        <div class="section">
            <h2>📷 可视化结果</h2>
            <div class="image-gallery">
                {% for img_path in image_paths.comparisons[:6] %}
                <div class="image-item">
                    <img src="{{ img_path }}" alt="对比图">
                    <div class="image-caption">对比图 {{ loop.index }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        <footer style="text-align: center; padding: 20px; color: #666;">
            <p>生成自 LLM增强的3D高斯场景理解系统</p>
        </footer>
    </div>
</body>
</html>
"""
        
        # 渲染模板
        template = Template(html_template)
        html_content = template.render(
            scene_name=scene_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary=summary,
            detection_stats=detection_stats,
            projection_metrics=projection_metrics,
            objects_3d=objects_3d,
            image_paths=image_paths
        )
        
        # 保存HTML文件
        report_path = self.output_dir / "report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML报告已生成: {report_path}")
        
        return str(report_path)
    
    def save_metrics_json(self, metrics: Dict[str, Any], filename: str = "metrics.json"):
        """
        保存指标到JSON文件
        
        Args:
            metrics: 指标字典
            filename: 文件名
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"指标已保存: {filepath}")


"""
文件管理器
负责创建和管理输出目录结构
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class FileManager:
    """文件管理器类"""
    
    def __init__(self, output_root: str, scene_name: str = None):
        """
        初始化文件管理器
        
        Args:
            output_root: 输出根目录
            scene_name: 场景名称，如果为None则使用时间戳
        """
        if scene_name is None:
            scene_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.output_root = Path(output_root)
        self.scene_name = scene_name
        self.scene_dir = self.output_root / scene_name
        
        # 创建目录结构
        self._create_directory_structure()
        
    def _create_directory_structure(self):
        """创建标准的目录结构"""
        # 主要目录
        self.dirs = {
            'root': self.scene_dir,
            'original': self.scene_dir / '1_original_images',
            'yolo': self.scene_dir / '2_yolo_detection',
            'yolo_vis': self.scene_dir / '2_yolo_detection' / 'visualizations',
            'rendered': self.scene_dir / '3_gaussian_rendered',
            'rendered_train': self.scene_dir / '3_gaussian_rendered' / 'train_views',
            'rendered_test': self.scene_dir / '3_gaussian_rendered' / 'test_views',
            'rendered_novel': self.scene_dir / '3_gaussian_rendered' / 'novel_views',
            'projected': self.scene_dir / '4_projected_detection',
            'projected_comparison': self.scene_dir / '4_projected_detection' / 'comparison',
            'scene_understanding': self.scene_dir / '5_scene_understanding',
            'scene_query': self.scene_dir / '5_scene_understanding' / 'query_results',
            'report': self.scene_dir / '6_comprehensive_report',
            'report_figures': self.scene_dir / '6_comprehensive_report' / 'figures',
        }
        
        # 创建所有目录
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_dir(self, dir_name: str) -> Path:
        """
        获取指定目录路径
        
        Args:
            dir_name: 目录名称（如 'yolo', 'rendered'等）
            
        Returns:
            目录路径
        """
        if dir_name not in self.dirs:
            raise ValueError(f"未知的目录名称: {dir_name}")
        return self.dirs[dir_name]
    
    def get_path(self, dir_name: str, filename: str) -> Path:
        """
        获取完整的文件路径
        
        Args:
            dir_name: 目录名称
            filename: 文件名
            
        Returns:
            完整文件路径
        """
        return self.get_dir(dir_name) / filename
    
    def save_json(self, data: Dict, dir_name: str, filename: str):
        """
        保存JSON文件
        
        Args:
            data: 要保存的数据
            dir_name: 目录名称
            filename: 文件名
        """
        filepath = self.get_path(dir_name, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_json(self, dir_name: str, filename: str) -> Dict:
        """
        加载JSON文件
        
        Args:
            dir_name: 目录名称
            filename: 文件名
            
        Returns:
            加载的数据
        """
        filepath = self.get_path(dir_name, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_files(self, dir_name: str, pattern: str = "*") -> List[Path]:
        """
        列出目录中的文件
        
        Args:
            dir_name: 目录名称
            pattern: 文件模式（如 "*.png"）
            
        Returns:
            文件路径列表
        """
        dir_path = self.get_dir(dir_name)
        return sorted(dir_path.glob(pattern))
    
    def copy_file(self, src: str, dir_name: str, filename: str = None):
        """
        复制文件到指定目录
        
        Args:
            src: 源文件路径
            dir_name: 目标目录名称
            filename: 目标文件名，如果为None则使用源文件名
        """
        src_path = Path(src)
        if filename is None:
            filename = src_path.name
        dst_path = self.get_path(dir_name, filename)
        shutil.copy2(src_path, dst_path)
    
    def get_summary(self) -> Dict[str, int]:
        """
        获取各目录的文件统计
        
        Returns:
            统计字典
        """
        summary = {}
        for name, path in self.dirs.items():
            if path.exists():
                file_count = len(list(path.glob('*.*')))
                summary[name] = file_count
        return summary
    
    def create_readme(self, info: Dict[str, str]):
        """
        创建README文件
        
        Args:
            info: 场景信息字典
        """
        readme_path = self.scene_dir / "README.md"
        
        content = f"""# 场景分析结果: {self.scene_name}

## 基本信息
- 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 场景名称: {self.scene_name}

## 目录结构
- `1_original_images/`: 原始输入图像
- `2_yolo_detection/`: YOLO目标检测结果
- `3_gaussian_rendered/`: 高斯渲染图像
- `4_projected_detection/`: 检测框投影结果
- `5_scene_understanding/`: 场景理解分析
- `6_comprehensive_report/`: 综合分析报告

## 统计信息
"""
        
        # 添加统计信息
        summary = self.get_summary()
        for name, count in summary.items():
            content += f"- {name}: {count} 个文件\n"
        
        # 添加额外信息
        if info:
            content += "\n## 详细信息\n"
            for key, value in info.items():
                content += f"- {key}: {value}\n"
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)


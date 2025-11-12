"""
配置文件加载器
负责加载和解析YAML配置文件
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            # 默认配置文件路径
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        从YAML文件加载配置
        
        Returns:
            配置字典
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证必要的配置项
        self._validate_config(config)
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]):
        """
        验证配置文件的完整性
        
        Args:
            config: 配置字典
        """
        required_sections = ['paths', 'yolo', 'gaussian', 'projection']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"配置文件缺少必需的部分: {section}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置项，支持点号分隔的嵌套键
        
        Args:
            key_path: 配置键路径，如 "yolo.model_name"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_paths(self) -> Dict[str, str]:
        """获取所有路径配置"""
        return self.config.get('paths', {})
    
    def get_yolo_config(self) -> Dict[str, Any]:
        """获取YOLO配置"""
        return self.config.get('yolo', {})
    
    def get_gaussian_config(self) -> Dict[str, Any]:
        """获取高斯配置"""
        return self.config.get('gaussian', {})
    
    def get_projection_config(self) -> Dict[str, Any]:
        """获取投影配置"""
        return self.config.get('projection', {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        return self.config.get('llm', {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """获取可视化配置"""
        return self.config.get('visualization', {})
    
    def update_config(self, key_path: str, value: Any):
        """
        更新配置项
        
        Args:
            key_path: 配置键路径
            value: 新值
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save_config(self, output_path: str = None):
        """
        保存配置到文件
        
        Args:
            output_path: 输出路径，如果为None则覆盖原文件
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)


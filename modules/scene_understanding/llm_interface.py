"""
LLM接口
与大语言模型交互
"""

import json
from typing import Dict, List, Optional, Any
from openai import OpenAI
from ..utils.logger import default_logger as logger


class LLMInterface:
    """LLM接口类"""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4-turbo",
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        base_url: str = None
    ):
        """
        初始化LLM接口
        
        Args:
            provider: 提供商 (openai, deepseek, anthropic)
            model: 模型名称
            api_key: API密钥
            temperature: 温度参数
            max_tokens: 最大token数
            base_url: API基础URL（用于DeepSeek等兼容OpenAI API的服务）
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if provider in ["openai", "deepseek"]:
            if api_key:
                # 如果提供了base_url，使用自定义URL（如DeepSeek）
                if base_url:
                    self.client = OpenAI(api_key=api_key, base_url=base_url)
                    logger.info(f"已初始化{provider}客户端，模型: {model}, URL: {base_url}")
                else:
                    self.client = OpenAI(api_key=api_key)
                    logger.info(f"已初始化{provider}客户端，模型: {model}")
            else:
                logger.warning("未提供API密钥，LLM功能将不可用")
                self.client = None
        else:
            logger.warning(f"不支持的LLM提供商: {provider}")
            self.client = None
    
    def generate_scene_description(
        self,
        scene_graph_dict: Dict
    ) -> str:
        """
        生成场景描述
        
        Args:
            scene_graph_dict: 场景图谱字典
            
        Returns:
            场景描述文本
        """
        if self.client is None:
            return self._generate_fallback_description(scene_graph_dict)
        
        # 构建prompt
        system_prompt = """你是一个3D场景理解助手。你会收到一个场景的结构化数据，包括物体和它们的空间关系。
请生成一个自然、详细的场景描述，包括：
1. 场景中有哪些主要物体
2. 这些物体的空间布局
3. 物体之间的关系
4. 场景的整体特征

用中文回答，语言要自然流畅。"""
        
        user_prompt = f"""场景数据：
{json.dumps(scene_graph_dict, ensure_ascii=False, indent=2)}

请描述这个场景。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            description = response.choices[0].message.content
            logger.info("场景描述生成成功")
            return description
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return self._generate_fallback_description(scene_graph_dict)
    
    def answer_query(
        self,
        scene_graph_dict: Dict,
        query: str
    ) -> Dict[str, Any]:
        """
        回答关于场景的查询
        
        Args:
            scene_graph_dict: 场景图谱字典
            query: 用户查询
            
        Returns:
            回答字典
        """
        if self.client is None:
            return {
                'answer': '抱歉，LLM功能未启用。',
                'highlight_objects': [],
                'camera_suggestion': None
            }
        
        system_prompt = """你是一个3D场景查询助手。用户会询问关于场景的问题，你需要：
1. 基于场景数据回答问题
2. 指出需要高亮显示的物体ID（如果适用）
3. 建议最佳观察视角（如果适用）

以JSON格式回答，包含以下字段：
{
  "answer": "自然语言回答",
  "highlight_objects": [物体ID列表],
  "reasoning": "推理过程",
  "camera_suggestion": {
    "position": [x, y, z],
    "target": [x, y, z],
    "description": "视角描述"
  }
}"""
        
        user_prompt = f"""场景数据：
{json.dumps(scene_graph_dict, ensure_ascii=False, indent=2)}

用户问题：{query}

请回答。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"查询回答成功: {query}")
            return result
            
        except Exception as e:
            logger.error(f"LLM查询失败: {e}")
            return {
                'answer': f'查询处理出错: {str(e)}',
                'highlight_objects': [],
                'camera_suggestion': None
            }
    
    def suggest_viewpoint(
        self,
        scene_graph_dict: Dict,
        focus_object_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        建议观察视角
        
        Args:
            scene_graph_dict: 场景图谱字典
            focus_object_id: 焦点物体ID
            
        Returns:
            视角建议
        """
        # 简单的规则based方法
        scene_bounds = scene_graph_dict['scene_bounds']
        center = scene_bounds['center']
        size = scene_bounds['size']
        
        # 默认：从场景中心斜上方俯瞰
        camera_distance = max(size) * 1.5
        
        suggestion = {
            'position': [
                center[0] + camera_distance * 0.5,
                center[1] + camera_distance * 0.7,
                center[2] + camera_distance * 0.5
            ],
            'target': center,
            'description': '从斜上方俯瞰整个场景'
        }
        
        # 如果指定了焦点物体，调整视角
        if focus_object_id is not None:
            for obj in scene_graph_dict['objects']:
                if obj['object_id'] == focus_object_id:
                    obj_pos = obj['position']
                    suggestion['target'] = obj_pos
                    suggestion['position'] = [
                        obj_pos[0] + 2.0,
                        obj_pos[1] + 1.5,
                        obj_pos[2] + 2.0
                    ]
                    suggestion['description'] = f'聚焦于{obj["class_name"]}物体'
                    break
        
        return suggestion
    
    def _generate_fallback_description(self, scene_graph_dict: Dict) -> str:
        """
        生成后备描述（当LLM不可用时）
        
        Args:
            scene_graph_dict: 场景图谱字典
            
        Returns:
            描述文本
        """
        objects = scene_graph_dict['objects']
        stats = scene_graph_dict['statistics']
        
        desc = f"这个场景包含 {stats['num_objects']} 个物体，分属 {stats['num_classes']} 个类别。\n\n"
        
        # 统计每个类别的数量
        class_counts = {}
        for obj in objects:
            class_name = obj['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        desc += "物体分布：\n"
        for class_name, count in class_counts.items():
            desc += f"- {count} 个 {class_name}\n"
        
        desc += f"\n场景尺寸约为 {scene_graph_dict['scene_bounds']['size']} 米。"
        
        return desc


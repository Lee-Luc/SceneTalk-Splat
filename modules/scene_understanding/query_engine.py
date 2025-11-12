"""
场景查询引擎
支持自然语言查询3D场景信息
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from ..utils.logger import default_logger as logger


class QueryEngine:
    """场景查询引擎"""
    
    def __init__(
        self,
        scene_data: Dict[str, Any],
        llm_interface,
        scene_name: str = "unknown"
    ):
        """
        初始化查询引擎
        
        Args:
            scene_data: 场景数据字典，包含：
                - scene_graph: 场景图谱
                - detections: YOLO检测结果
                - statistics: 检测统计信息
                - object_database: 3D对象数据库（可选）
            llm_interface: LLM接口实例
            scene_name: 场景名称
        """
        self.scene_data = scene_data
        self.llm = llm_interface
        self.scene_name = scene_name
        self.conversation_history = []
        
        # 提取关键信息
        self.scene_graph = scene_data.get('scene_graph', {})
        self.detections = scene_data.get('detections', [])
        self.statistics = scene_data.get('statistics', {})
        self.object_database = scene_data.get('object_database', {})
        
        logger.info(f"查询引擎已初始化，场景: {scene_name}")
        self._log_scene_info()
    
    def _log_scene_info(self):
        """记录场景基本信息"""
        total_images = self.statistics.get('total_images', 0)
        total_detections = self.statistics.get('total_detections', 0)
        num_classes = self.statistics.get('num_unique_classes', 0)
        
        logger.info(f"  图像数: {total_images}")
        logger.info(f"  检测数: {total_detections}")
        logger.info(f"  类别数: {num_classes}")
    
    def query(self, question: str, use_history: bool = True) -> str:
        """
        执行查询
        
        Args:
            question: 用户问题
            use_history: 是否使用对话历史（支持多轮对话）
            
        Returns:
            LLM生成的回答
        """
        if self.llm.client is None:
            return "❌ LLM未初始化或API Key无效，无法进行查询。请检查配置文件。"
        
        try:
            # 1. 构建场景上下文
            context = self._build_context()
            
            # 2. 构建完整的提示词
            messages = self._build_messages(question, context, use_history)
            
            # 3. 调用LLM
            logger.info(f"正在查询LLM: {question}")
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            # 4. 记录到对话历史
            self.conversation_history.append({
                "question": question,
                "answer": answer
            })
            
            logger.info("查询完成")
            return answer
            
        except Exception as e:
            error_msg = f"查询失败: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
    
    def _build_context(self) -> str:
        """
        构建场景上下文信息
        
        Returns:
            格式化的场景上下文文本
        """
        context_parts = []
        
        # 1. 3D场景中的真实物体（优先级最高）
        sg = self.scene_graph
        if sg and sg.get('objects'):
            # 统计真实的3D物体类别分布
            real_objects = {}
            for obj in sg['objects']:
                class_name = obj.get('class_name', 'unknown')
                real_objects[class_name] = real_objects.get(class_name, 0) + 1
            
            context_parts.append("【场景中的真实物体】（重要：回答物体数量问题请使用这个数据）")
            context_parts.append(f"- 3D物体总数: {len(sg['objects'])}个")
            context_parts.append(f"- 物体类别数: {len(real_objects)}种")
            context_parts.append("")
            context_parts.append("各类别物体数量：")
            for class_name, count in sorted(real_objects.items(), key=lambda x: x[1], reverse=True):
                context_parts.append(f"  - {class_name}: {count}个")
            
            # 空间关系
            if sg.get('relations'):
                context_parts.append(f"\n空间关系: {len(sg['relations'])}条")
            
            # 场景边界
            bounds = sg.get('scene_bounds', {})
            if bounds and bounds.get('size'):
                size = bounds['size']
                context_parts.append(f"场景尺寸: {size[0]:.2f}m × {size[1]:.2f}m × {size[2]:.2f}m")
            context_parts.append("")
        
        # 2. 2D检测统计（仅作参考，不要用于回答物体数量）
        stats = self.statistics
        if stats:
            context_parts.append("【2D检测统计】（参考信息：这是跨所有图像的检测累计次数，不是真实物体数量）")
            context_parts.append(f"- 分析图像数: {stats.get('total_images', 0)}张")
            context_parts.append(f"- 累计检测次数: {stats.get('total_detections', 0)}次")
            context_parts.append(f"- 平均每图检测: {stats.get('avg_detections_per_image', 0):.1f}次")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _build_messages(
        self,
        question: str,
        context: str,
        use_history: bool
    ) -> List[Dict[str, str]]:
        """
        构建LLM的消息列表
        
        Args:
            question: 用户问题
            context: 场景上下文
            use_history: 是否使用对话历史
            
        Returns:
            消息列表
        """
        messages = []
        
        # 系统提示词
        system_prompt = """你是一个3D场景理解助手。用户会问你关于一个3D场景的问题，你需要根据提供的场景数据来回答。

【重要】数据优先级说明：
1. 【场景中的真实物体】= 场景中实际存在的物体数量（最高优先级）
   - 回答"有多少个XX"、"场景里有什么"等问题时，必须使用这个数据
   - 这是通过3D重建得到的真实物体数量
   
2. 【2D检测统计】= 跨所有图像的检测累计次数（仅供参考）
   - 这不是真实物体数量，而是检测次数
   - 同一物体在多张图像中被检测，会被计数多次
   - 仅用于了解检测过程，不要用于回答物体数量问题

回答要求：
1. 准确：严格基于【场景中的真实物体】数据回答物体数量问题
2. 简洁：直接回答问题，避免冗长的解释
3. 友好：使用自然、友好的语气
4. 中文：使用中文回答（除非用户用英文提问）
5. 具体：提供具体的数字和细节

如果数据中没有相关信息，请诚实地告诉用户"数据中没有这方面的信息"。"""
        
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # 添加对话历史（如果启用）
        if use_history and self.conversation_history:
            for item in self.conversation_history[-3:]:  # 只保留最近3轮对话
                messages.append({
                    "role": "user",
                    "content": item["question"]
                })
                messages.append({
                    "role": "assistant",
                    "content": item["answer"]
                })
        
        # 当前问题（包含场景上下文）
        user_message = f"""【场景数据】
{context}

【用户问题】
{question}

请基于以上场景数据回答用户的问题。"""
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def get_scene_summary(self) -> str:
        """
        获取场景摘要信息
        
        Returns:
            格式化的场景摘要
        """
        summary_parts = [
            f"📊 场景摘要：{self.scene_name}",
            "=" * 50
        ]
        
        # 优先显示3D真实物体信息
        sg = self.scene_graph
        if sg and sg.get('objects'):
            # 统计真实的3D物体
            real_objects = {}
            for obj in sg['objects']:
                class_name = obj.get('class_name', 'unknown')
                real_objects[class_name] = real_objects.get(class_name, 0) + 1
            
            summary_parts.append(f"场景中的真实物体数: {len(sg['objects'])}个")
            summary_parts.append(f"物体类别数: {len(real_objects)}种")
            summary_parts.append("")
            summary_parts.append("各类别物体数量：")
            
            if real_objects:
                for class_name, count in sorted(real_objects.items(), key=lambda x: x[1], reverse=True):
                    summary_parts.append(f"  - {class_name}: {count}个")
            else:
                summary_parts.append("  （无数据）")
        else:
            summary_parts.append("⚠️  3D场景图谱数据不可用")
            summary_parts.append("")
        
        # 附加2D检测统计（参考信息）
        stats = self.statistics
        if stats:
            summary_parts.append("")
            summary_parts.append("─" * 50)
            summary_parts.append("2D检测统计（仅供参考）：")
            summary_parts.append(f"  分析图像数: {stats.get('total_images', 0)}张")
            summary_parts.append(f"  累计检测次数: {stats.get('total_detections', 0)}次")
        
        return "\n".join(summary_parts)
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        logger.info("对话历史已清空")
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Returns:
            对话历史列表
        """
        return self.conversation_history.copy()
    
    def save_history(self, output_path: str):
        """
        保存对话历史到文件
        
        Args:
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'scene_name': self.scene_name,
                    'conversation_history': self.conversation_history,
                    'total_queries': len(self.conversation_history)
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"对话历史已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存对话历史失败: {str(e)}")
    
    def load_history(self, input_path: str) -> bool:
        """
        从文件加载对话历史
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            是否加载成功
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.conversation_history = data.get('conversation_history', [])
            logger.info(f"对话历史已加载: {len(self.conversation_history)}条")
            return True
            
        except Exception as e:
            logger.error(f"加载对话历史失败: {str(e)}")
            return False


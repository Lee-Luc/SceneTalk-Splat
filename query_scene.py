#!/usr/bin/env python3
"""
场景交互查询工具
用自然语言查询3D场景信息
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.scene_understanding.query_engine import QueryEngine
from modules.scene_understanding.llm_interface import LLMInterface
from modules.utils.config_loader import ConfigLoader
from modules.utils.logger import default_logger as logger


class InteractiveQueryCLI:
    """交互式查询命令行界面"""
    
    def __init__(self, scene_dir: Path, config_loader: ConfigLoader):
        """
        初始化交互式查询CLI
        
        Args:
            scene_dir: 场景数据目录
            config_loader: 配置加载器
        """
        self.scene_dir = Path(scene_dir)
        self.config_loader = config_loader
        self.scene_name = self.scene_dir.name
        
        # 加载场景数据
        self.scene_data = self._load_scene_data()
        
        # 初始化LLM接口
        self.llm_interface = self._initialize_llm()
        
        # 初始化查询引擎
        self.query_engine = QueryEngine(
            scene_data=self.scene_data,
            llm_interface=self.llm_interface,
            scene_name=self.scene_name
        )
        
        # 对话历史保存路径
        self.history_path = self.scene_dir / "query_history.json"
    
    def _load_scene_data(self) -> Dict[str, Any]:
        """
        加载场景数据
        
        Returns:
            场景数据字典
        """
        logger.info(f"正在加载场景数据: {self.scene_dir}")
        
        scene_data = {}
        
        # 1. 加载场景图谱
        scene_graph_path = self.scene_dir / "5_scene_understanding" / "scene_graph.json"
        if scene_graph_path.exists():
            with open(scene_graph_path, 'r', encoding='utf-8') as f:
                scene_data['scene_graph'] = json.load(f)
            logger.info("✓ 已加载场景图谱")
        else:
            logger.warning(f"⚠ 场景图谱文件不存在: {scene_graph_path}")
            scene_data['scene_graph'] = {}
        
        # 2. 加载检测统计
        stats_path = self.scene_dir / "2_yolo_detection" / "statistics.json"
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                scene_data['statistics'] = json.load(f)
            logger.info("✓ 已加载检测统计")
        else:
            logger.warning(f"⚠ 检测统计文件不存在: {stats_path}")
            scene_data['statistics'] = {}
        
        # 3. 加载详细检测结果
        detections_path = self.scene_dir / "2_yolo_detection" / "detections.json"
        if detections_path.exists():
            with open(detections_path, 'r', encoding='utf-8') as f:
                scene_data['detections'] = json.load(f)
            logger.info("✓ 已加载详细检测结果")
        else:
            logger.warning(f"⚠ 检测结果文件不存在: {detections_path}")
            scene_data['detections'] = []
        
        # 4. 加载3D对象数据库（如果存在）
        obj_db_path = self.scene_dir / "5_scene_understanding" / "object_database.json"
        if obj_db_path.exists():
            with open(obj_db_path, 'r', encoding='utf-8') as f:
                scene_data['object_database'] = json.load(f)
            logger.info("✓ 已加载3D对象数据库")
        else:
            logger.info("ℹ 3D对象数据库文件不存在（可选）")
            scene_data['object_database'] = {}
        
        return scene_data
    
    def _initialize_llm(self) -> LLMInterface:
        """
        初始化LLM接口
        
        Returns:
            LLM接口实例
        """
        llm_config = self.config_loader.get_llm_config()
        
        if not llm_config.get('enable_llm', False):
            logger.error("❌ LLM功能未启用！请在配置文件中设置 enable_llm: true")
            print("\n" + "="*60)
            print("⚠️  错误：LLM功能未启用")
            print("="*60)
            print("\n请编辑配置文件 config/config.yaml，设置：")
            print("\nllm:")
            print("  enable_llm: true           # 改为 true")
            print("  provider: \"deepseek\"")
            print("  model: \"deepseek-chat\"")
            print("  api_key: \"your-api-key\"   # 填入你的API Key")
            print("  base_url: \"https://api.deepseek.com/v1\"")
            print("\n" + "="*60 + "\n")
            sys.exit(1)
        
        return LLMInterface(
            provider=llm_config.get('provider', 'openai'),
            model=llm_config.get('model', 'gpt-4-turbo'),
            api_key=llm_config.get('api_key'),
            base_url=llm_config.get('base_url'),
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 2000)
        )
    
    def print_welcome(self):
        """打印欢迎信息"""
        stats = self.scene_data.get('statistics', {})
        scene_graph = self.scene_data.get('scene_graph', {})
        
        #print("\n" + "="*70)
        #print("🔍  3D场景交互查询系统")
        #print("="*70)
        #print(f"\n📁 场景名称: {self.scene_name}")
        #print(f"📂 数据目录: {self.scene_dir}")
        
        # 优先显示3D真实物体信息
        if scene_graph and scene_graph.get('objects'):
            # 统计真实的3D物体
            real_objects = {}
            for obj in scene_graph['objects']:
                class_name = obj.get('class_name', 'unknown')
                real_objects[class_name] = real_objects.get(class_name, 0) + 1
            
            print("\n场景中的真实物体:")
            print(f"   - 物体总数: {len(scene_graph['objects'])}个")
            print(f"   - 物体类别: {len(real_objects)}种")
            
            if real_objects:
                print("\n🏷️  各类别物体数量:")
                for i, (class_name, count) in enumerate(
                    sorted(real_objects.items(), key=lambda x: x[1], reverse=True)[:5]
                ):
                    print(f"   {i+1}. {class_name}: {count}个")
                if len(real_objects) > 5:
                    print(f"   ... 还有 {len(real_objects) - 5} 种类别")
        else:
            print("\n⚠️  3D场景图谱数据不可用")
        
        # 附加2D检测统计
        if stats:
            #print(f"\n📸 2D检测统计（参考）:")
            print(f"   - 分析图像数: {stats.get('total_images', 0)}张")
            #print(f"   - 累计检测次数: {stats.get('total_detections', 0)}次")
        
        print(f"\nLLM模型: {self.llm_interface.model}")
        #print(f"🌐 提供商: {self.llm_interface.provider}")
        
        print("\n")
        print("- 请输入你的问题")
        print("\n")
    
    def print_scene_info(self):
        """打印详细场景信息"""
        print("\n" + self.query_engine.get_scene_summary())
        print()
    
    def print_history(self):
        """打印对话历史"""
        history = self.query_engine.get_history()
        
        if not history:
            print("\n📝 对话历史为空\n")
            return
        
        print("\n" + "="*70)
        print(f"📝 对话历史（共 {len(history)} 条）")
        print("="*70 + "\n")
        
        for i, item in enumerate(history, 1):
            print(f"{i}. 👤 问: {item['question']}")
            print(f"   🤖 答: {item['answer'][:100]}..." if len(item['answer']) > 100 else f"   🤖 答: {item['answer']}")
            print()
    
    def save_conversation(self):
        """保存对话历史"""
        history = self.query_engine.get_history()
        
        if not history:
            print("\n⚠️  对话历史为空，无需保存\n")
            return
        
        self.query_engine.save_history(str(self.history_path))
        print(f"\n💾 对话历史已保存到: {self.history_path}\n")
    
    def run(self):
        """运行交互式查询循环"""
        self.print_welcome()
        
        # 尝试加载之前的对话历史
        if self.history_path.exists():
            load = input("📂 发现历史对话记录，是否加载？(y/n): ").strip().lower()
            if load == 'y':
                if self.query_engine.load_history(str(self.history_path)):
                    print("✓ 对话历史已加载\n")
        
        # 主循环
        while True:
            try:
                # 获取用户输入
                question = input("👤 你的问题 > ").strip()
                
                # 处理特殊命令
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit', 'q']:
                    self._handle_exit()
                    break
                
                if question.lower() == 'info':
                    self.print_scene_info()
                    continue
                
                if question.lower() == 'clear':
                    self.query_engine.clear_history()
                    print("\n✓ 对话历史已清空\n")
                    continue
                
                if question.lower() == 'history':
                    self.print_history()
                    continue
                
                if question.lower() == 'save':
                    self.save_conversation()
                    continue
                
                if question.lower() == 'help':
                    self.print_welcome()
                    continue
                
                # 执行查询
                print("\n🤖 正在思考...\n")
                answer = self.query_engine.query(question)
                
                # 显示答案
                print("🤖 回答:")
                print("-" * 70)
                print(answer)
                print("-" * 70)
                print()
                
            except KeyboardInterrupt:
                print("\n\n⚠️  检测到中断信号")
                self._handle_exit()
                break
            
            except Exception as e:
                print(f"\n❌ 发生错误: {str(e)}\n")
                logger.error(f"查询过程中发生错误: {str(e)}")
    
    def _handle_exit(self):
        """处理退出"""
        history = self.query_engine.get_history()
        
        if history:
            save = input("\n💾 是否保存对话历史？(y/n): ").strip().lower()
            if save == 'y':
                self.save_conversation()
        
        print("\n👋 感谢使用！再见！\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="3D场景交互查询工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 查询lerf场景
  python query_scene.py --scene lerf
  
  # 指定场景数据目录
  python query_scene.py --scene_dir output/lerf_analysis/lerf
  
  # 使用自定义配置文件
  python query_scene.py --scene lerf --config my_config.yaml
"""
    )
    
    parser.add_argument(
        '--scene',
        type=str,
        help='场景名称（如 lerf），将自动查找 output/<scene>_analysis/<scene>/'
    )
    
    parser.add_argument(
        '--scene_dir',
        type=str,
        help='场景数据目录的完整路径'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='配置文件路径（默认: config/config.yaml）'
    )
    
    args = parser.parse_args()
    
    # 确定场景目录
    if args.scene_dir:
        scene_dir = Path(args.scene_dir)
    elif args.scene:
        scene_dir = Path(f"output/{args.scene}_analysis/{args.scene}")
    else:
        parser.error("请提供 --scene 或 --scene_dir 参数")
    
    # 检查场景目录是否存在
    if not scene_dir.exists():
        print(f"\n❌ 错误：场景目录不存在: {scene_dir}")
        print("\n请确认：")
        print("  1. 场景名称是否正确")
        print("  2. 是否已经运行过 main.py 生成场景数据")
        print(f"\n预期目录结构：{scene_dir}/")
        print("  ├── 2_yolo_detection/")
        print("  │   ├── detections.json")
        print("  │   └── statistics.json")
        print("  └── 5_scene_understanding/")
        print("      └── scene_graph.json")
        print()
        sys.exit(1)
    
    # 加载配置
    try:
        config_loader = ConfigLoader(args.config)
    except Exception as e:
        print(f"\n❌ 加载配置文件失败: {str(e)}\n")
        sys.exit(1)
    
    # 创建并运行CLI
    try:
        cli = InteractiveQueryCLI(scene_dir, config_loader)
        cli.run()
    except Exception as e:
        logger.error(f"程序运行失败: {str(e)}")
        print(f"\n❌ 程序运行失败: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Qwen3模型升级脚本
验证和升级到Qwen3系列模型
"""
import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger("upgrade.qwen3")

class Qwen3Upgrader:
    """Qwen3升级器"""
    
    def __init__(self):
        self.qwen3_models = {
            "LLM": "Qwen/Qwen3-8B",
            "Embedding": "Qwen/Qwen3-Embedding-8B", 
            "Reranker": "Qwen/Qwen3-Reranker-8B"
        }
        
        self.old_models = {
            "LLM": "Qwen/Qwen2.5-8B-Instruct",
            "Embedding": "Qwen/Qwen2.5-Embedding-8B",
            "Reranker": "Qwen/Qwen2.5-Reranker-4B"
        }
        
        logger.info("Qwen3升级器初始化完成")
    
    def check_model_availability(self, model_name: str) -> bool:
        """检查模型在Hugging Face上的可用性"""
        try:
            from huggingface_hub import model_info
            info = model_info(model_name)
            return True
        except Exception as e:
            logger.error(f"模型 {model_name} 不可用: {e}")
            return False
    
    def check_all_models(self) -> Dict[str, bool]:
        """检查所有Qwen3模型的可用性"""
        logger.info("检查Qwen3模型可用性...")
        
        results = {}
        for model_type, model_name in self.qwen3_models.items():
            logger.info(f"检查 {model_type} 模型: {model_name}")
            results[model_type] = self.check_model_availability(model_name)
        
        return results
    
    def test_model_loading(self, model_name: str, model_type: str) -> bool:
        """测试模型加载"""
        try:
            logger.info(f"测试加载 {model_type} 模型: {model_name}")
            
            if model_type == "LLM":
                from transformers import AutoTokenizer, AutoModelForCausalLM
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                # 只加载tokenizer，不加载完整模型以节省时间和内存
                logger.info(f"✅ {model_type} tokenizer 加载成功")
                return True
                
            elif model_type == "Embedding":
                from sentence_transformers import SentenceTransformer
                # 只检查模型配置，不实际加载
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                logger.info(f"✅ {model_type} 配置加载成功")
                return True
                
            elif model_type == "Reranker":
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                logger.info(f"✅ {model_type} 配置加载成功")
                return True
                
        except Exception as e:
            logger.error(f"❌ {model_type} 模型加载失败: {e}")
            return False
    
    def backup_current_config(self):
        """备份当前配置"""
        logger.info("备份当前配置...")
        
        config_files = [".env", "config/settings.py"]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                backup_file = f"{config_file}.backup"
                try:
                    import shutil
                    shutil.copy2(config_file, backup_file)
                    logger.info(f"✅ 备份 {config_file} -> {backup_file}")
                except Exception as e:
                    logger.error(f"❌ 备份 {config_file} 失败: {e}")
    
    def update_env_file(self):
        """更新.env文件"""
        logger.info("更新.env文件...")
        
        env_file = ".env"
        if not os.path.exists(env_file):
            logger.warning(".env文件不存在，跳过更新")
            return
        
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 更新模型路径
            replacements = {
                "Qwen/Qwen2.5-8B-Instruct": "Qwen/Qwen3-8B",
                "Qwen/Qwen2.5-Embedding-8B": "Qwen/Qwen3-Embedding-8B", 
                "Qwen/Qwen2.5-Reranker-4B": "Qwen/Qwen3-Reranker-8B",
                "BAAI/bge-large-zh-v1.5": "Qwen/Qwen3-Embedding-8B",
                "BAAI/bge-reranker-large": "Qwen/Qwen3-Reranker-8B"
            }
            
            for old_model, new_model in replacements.items():
                if old_model in content:
                    content = content.replace(old_model, new_model)
                    logger.info(f"✅ 更新: {old_model} -> {new_model}")
            
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("✅ .env文件更新完成")
            
        except Exception as e:
            logger.error(f"❌ 更新.env文件失败: {e}")
    
    def verify_upgrade(self) -> bool:
        """验证升级结果"""
        logger.info("验证升级结果...")
        
        try:
            # 重新加载配置
            from config.settings import settings
            
            current_models = {
                "LLM": settings.QWEN_MODEL_PATH,
                "Embedding": settings.EMBEDDING_MODEL_PATH,
                "Reranker": settings.RERANKER_MODEL_PATH
            }
            
            logger.info("当前模型配置:")
            for model_type, model_name in current_models.items():
                logger.info(f"  {model_type}: {model_name}")
            
            # 检查是否使用了Qwen3模型
            qwen3_count = 0
            for model_type, model_name in current_models.items():
                if "Qwen3" in model_name or "Qwen/Qwen3" in model_name:
                    qwen3_count += 1
                    logger.info(f"✅ {model_type} 已升级到Qwen3")
                else:
                    logger.warning(f"⚠️ {model_type} 未使用Qwen3: {model_name}")
            
            success_rate = qwen3_count / len(current_models) * 100
            logger.info(f"Qwen3升级率: {success_rate:.1f}%")
            
            return qwen3_count >= 2  # 至少2个模型使用Qwen3
            
        except Exception as e:
            logger.error(f"验证升级失败: {e}")
            return False
    
    def show_upgrade_benefits(self):
        """显示升级优势"""
        print("\n" + "="*60)
        print("🚀 Qwen3系列模型优势")
        print("="*60)
        
        benefits = [
            "🧠 更强的推理能力 - 支持思维链推理模式",
            "🎯 更高的准确性 - 中文理解能力提升7-8%", 
            "🔍 更好的检索效果 - 嵌入模型精度提升6%",
            "⚡ 更快的推理速度 - 优化的模型架构",
            "🔄 统一的模型家族 - 更好的兼容性",
            "📈 持续的更新支持 - 阿里巴巴最新技术"
        ]
        
        for benefit in benefits:
            print(f"  {benefit}")
        
        print("="*60)
    
    def show_migration_guide(self):
        """显示迁移指南"""
        print("\n" + "="*60)
        print("📋 Qwen3迁移指南")
        print("="*60)
        
        print("1. 模型对应关系:")
        print("   Qwen2.5-8B-Instruct    -> Qwen3-8B")
        print("   Qwen2.5-Embedding-8B   -> Qwen3-Embedding-8B")
        print("   Qwen2.5-Reranker-4B    -> Qwen3-Reranker-8B")
        print("   BGE-large-zh-v1.5      -> Qwen3-Embedding-8B")
        print("   BGE-reranker-large     -> Qwen3-Reranker-8B")
        
        print("\n2. 配置更新:")
        print("   QWEN_MODEL_PATH=Qwen/Qwen3-8B")
        print("   EMBEDDING_MODEL_PATH=Qwen/Qwen3-Embedding-8B")
        print("   RERANKER_MODEL_PATH=Qwen/Qwen3-Reranker-8B")
        
        print("\n3. 注意事项:")
        print("   - 首次使用会自动下载模型（约15-20GB）")
        print("   - 建议使用16GB+内存的机器")
        print("   - 支持CPU和GPU推理")
        print("   - 兼容现有的API接口")
        
        print("="*60)
    
    async def run_upgrade(self):
        """运行升级流程"""
        logger.info("开始Qwen3升级流程...")
        
        # 1. 显示升级优势
        self.show_upgrade_benefits()
        
        # 2. 检查模型可用性
        availability = self.check_all_models()
        
        available_count = sum(availability.values())
        total_count = len(availability)
        
        print(f"\n📊 模型可用性检查结果: {available_count}/{total_count}")
        for model_type, available in availability.items():
            status = "✅ 可用" if available else "❌ 不可用"
            print(f"  {model_type}: {status}")
        
        if available_count < total_count:
            logger.error("部分Qwen3模型不可用，请检查网络连接或稍后重试")
            return False
        
        # 3. 测试模型加载
        print("\n🧪 测试模型加载...")
        loading_results = {}
        for model_type, model_name in self.qwen3_models.items():
            loading_results[model_type] = self.test_model_loading(model_name, model_type)
        
        loading_success = sum(loading_results.values())
        print(f"模型加载测试: {loading_success}/{len(loading_results)}")
        
        # 4. 询问用户是否继续
        print(f"\n❓ 是否继续升级到Qwen3系列模型？")
        print("   - 将备份当前配置")
        print("   - 更新模型路径到Qwen3系列")
        print("   - 首次使用时会自动下载模型")
        
        response = input("继续升级？(y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            logger.info("用户取消升级")
            return False
        
        # 5. 执行升级
        logger.info("执行升级...")
        
        # 备份配置
        self.backup_current_config()
        
        # 更新配置文件
        self.update_env_file()
        
        # 6. 验证升级
        if self.verify_upgrade():
            logger.info("✅ Qwen3升级成功！")
            self.show_migration_guide()
            
            print("\n🎉 升级完成！")
            print("请重启系统以使用新的Qwen3模型。")
            print("运行命令: python scripts/start_services.py")
            
            return True
        else:
            logger.error("❌ 升级验证失败")
            return False


async def main():
    """主函数"""
    print("🚀 企业私有知识库系统 - Qwen3升级工具")
    print("="*60)
    
    try:
        upgrader = Qwen3Upgrader()
        success = await upgrader.run_upgrade()
        
        if success:
            print("\n✅ 升级成功完成！")
            sys.exit(0)
        else:
            print("\n❌ 升级失败或被取消")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断升级")
        sys.exit(1)
    except Exception as e:
        logger.error(f"升级过程出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

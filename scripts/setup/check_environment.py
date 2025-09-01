#!/usr/bin/env python3
"""
环境检查脚本
验证epkbs conda环境是否正确配置
"""
import sys
import os
import subprocess
from pathlib import Path


def print_colored(text, color="white"):
    """打印彩色文本"""
    colors = {
        "red": "\033[0;31m",
        "green": "\033[0;32m",
        "yellow": "\033[1;33m",
        "blue": "\033[0;34m",
        "white": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{text}\033[0m")


def check_python_environment():
    """检查Python环境"""
    print_colored("🐍 检查Python环境", "blue")
    print("=" * 40)

    # Python版本
    python_version = sys.version
    print(f"Python版本: {python_version}")

    # Python路径
    python_path = sys.executable
    print(f"Python路径: {python_path}")

    # 检查是否在epkbs环境中
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "未知")
    print(f"Conda环境: {conda_env}")

    if conda_env == "epkbs":
        print_colored("✅ 正确使用epkbs环境", "green")
    else:
        print_colored(f"❌ 当前环境是 {conda_env}，应该是 epkbs", "red")
        print_colored(
            "💡 请运行: source scripts/setup/activate_environment.sh", "yellow")
        return False

    return True


def check_required_packages():
    """检查必需的包"""
    print_colored("\n📦 检查必需的包", "blue")
    print("=" * 40)

    # 核心包列表
    required_packages = {
        # LangGraph和LangChain
        "langgraph": "LangGraph工作流框架",
        "langchain": "LangChain核心框架",
        "langchain_community": "LangChain社区包",

        # MCP框架
        "fastmcp": "FastMCP框架",

        # Web框架
        "fastapi": "FastAPI Web框架",
        "streamlit": "Streamlit UI框架",

        # 数据库
        "sqlalchemy": "SQLAlchemy ORM",
        "asyncpg": "PostgreSQL异步驱动",
        "redis": "Redis客户端",

        # 向量数据库
        "pymilvus": "Milvus向量数据库",

        # 文档处理
        "unstructured": "Unstructured文档解析",
        "pypdf": "PDF处理",
        "docx": "Word文档处理",

        # 表格提取
        "camelot": "Camelot表格提取",
        "pdfplumber": "PDFPlumber表格提取",

        # 机器学习
        "transformers": "Transformers模型库",
        "sentence_transformers": "句子嵌入模型",
        "torch": "PyTorch深度学习框架",

        # 工具库
        "pydantic": "数据验证",
        "numpy": "数值计算",
        "pandas": "数据处理"
    }

    missing_packages = []
    installed_packages = []

    for package, description in required_packages.items():
        try:
            # 尝试导入包
            __import__(package)

            # 获取版本信息
            try:
                module = __import__(package)
                version = getattr(module, '__version__', '未知版本')
            except:
                version = "未知版本"

            print_colored(f"✅ {package} ({version}) - {description}", "green")
            installed_packages.append(package)

        except ImportError:
            print_colored(f"❌ {package} - {description}", "red")
            missing_packages.append(package)

    print(f"\n📊 包检查结果:")
    print(f"  已安装: {len(installed_packages)}/{len(required_packages)}")
    print(f"  缺失: {len(missing_packages)}")

    if missing_packages:
        print_colored(f"\n⚠️  缺失的包: {', '.join(missing_packages)}", "yellow")
        print_colored("💡 请运行: pip install -r requirements.txt", "yellow")
        return False
    else:
        print_colored("🎉 所有必需的包都已安装！", "green")
        return True


def check_project_structure():
    """检查项目结构"""
    print_colored("\n📁 检查项目结构", "blue")
    print("=" * 40)

    # 关键目录和文件
    required_paths = [
        "src/agent/core.py",
        "src/agent/langgraph_agent.py",
        "src/agent/workflows.py",
        "src/agent/states.py",
        "src/epkbs_mcp/server_manager.py",
        "src/rag/core.py",
        "config/settings.py",
        "config/langgraph_settings.py",
        "requirements.txt"
    ]

    missing_paths = []

    for path in required_paths:
        if Path(path).exists():
            print_colored(f"✅ {path}", "green")
        else:
            print_colored(f"❌ {path}", "red")
            missing_paths.append(path)

    if missing_paths:
        print_colored(f"\n⚠️  缺失的文件: {len(missing_paths)} 个", "yellow")
        return False
    else:
        print_colored("🎉 项目结构完整！", "green")
        return True


def check_configuration():
    """检查配置文件"""
    print_colored("\n⚙️ 检查配置文件", "blue")
    print("=" * 40)

    try:
        # 检查主配置
        sys.path.insert(0, str(Path.cwd() / "src"))

        from config.settings import settings
        print_colored("✅ 主配置文件加载成功", "green")

        # 检查LangGraph配置
        from config.langgraph_settings import get_langgraph_config
        langgraph_config = get_langgraph_config("development")
        print_colored("✅ LangGraph配置文件加载成功", "green")

        # 检查MCP配置
        from config.mcp_settings import get_mcp_config
        mcp_config = get_mcp_config("development")
        print_colored("✅ MCP配置文件加载成功", "green")

        return True

    except Exception as e:
        print_colored(f"❌ 配置文件检查失败: {e}", "red")
        return False


def main():
    """主检查函数"""
    print_colored("🔍 EPKBS环境检查", "blue")
    print("=" * 50)

    checks = [
        ("Python环境", check_python_environment),
        ("必需包", check_required_packages),
        ("项目结构", check_project_structure),
        ("配置文件", check_configuration)
    ]

    all_passed = True

    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print_colored(f"❌ {check_name}检查失败: {e}", "red")
            all_passed = False

    print_colored("\n" + "=" * 50, "blue")

    if all_passed:
        print_colored("🎉 所有检查通过！epkbs环境配置正确", "green")
        print_colored("\n🚀 可以开始使用EPKBS系统了！", "green")
        print_colored("\n💡 运行示例:", "blue")
        print("   python examples/advanced/langgraph_showcase.py")
        print("   python tests/integration/test_langgraph_integration.py")
    else:
        print_colored("❌ 环境检查失败，请解决上述问题", "red")
        print_colored("\n💡 解决步骤:", "yellow")
        print("1. 确保激活epkbs环境: source scripts/setup/activate_environment.sh")
        print("2. 安装缺失的包: pip install -r requirements.txt")
        print("3. 重新运行检查: python scripts/setup/check_environment.py")


if __name__ == "__main__":
    main()

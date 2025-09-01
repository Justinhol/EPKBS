#!/usr/bin/env python3
"""
项目清理脚本
清理不必要的文件和目录，保持项目整洁
"""
import os
import shutil
from pathlib import Path
import glob


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


def clean_python_cache():
    """清理Python缓存文件"""
    print_colored("🧹 清理Python缓存文件...", "blue")
    
    # 清理__pycache__目录
    pycache_dirs = list(Path(".").rglob("__pycache__"))
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            print_colored(f"  ✅ 删除: {pycache_dir}", "green")
        except Exception as e:
            print_colored(f"  ❌ 删除失败: {pycache_dir} - {e}", "red")
    
    # 清理.pyc文件
    pyc_files = list(Path(".").rglob("*.pyc"))
    for pyc_file in pyc_files:
        try:
            pyc_file.unlink()
            print_colored(f"  ✅ 删除: {pyc_file}", "green")
        except Exception as e:
            print_colored(f"  ❌ 删除失败: {pyc_file} - {e}", "red")
    
    print_colored(f"Python缓存清理完成: 删除了 {len(pycache_dirs)} 个目录和 {len(pyc_files)} 个文件", "green")


def clean_log_files():
    """清理日志文件"""
    print_colored("\n📝 清理日志文件...", "blue")
    
    log_patterns = ["*.log", "logs/*.log", "*.log.*"]
    deleted_count = 0
    
    for pattern in log_patterns:
        log_files = glob.glob(pattern, recursive=True)
        for log_file in log_files:
            try:
                os.remove(log_file)
                print_colored(f"  ✅ 删除: {log_file}", "green")
                deleted_count += 1
            except Exception as e:
                print_colored(f"  ❌ 删除失败: {log_file} - {e}", "red")
    
    print_colored(f"日志文件清理完成: 删除了 {deleted_count} 个文件", "green")


def clean_temp_files():
    """清理临时文件"""
    print_colored("\n🗑️ 清理临时文件...", "blue")
    
    # 临时文件模式
    temp_patterns = [
        "*.tmp", "*.temp", ".tmp*", "temp/*", "tmp/*",
        "=*", ">=*", "<=*",  # 奇怪的文件
        ".DS_Store", "Thumbs.db", "Desktop.ini"
    ]
    
    deleted_count = 0
    
    for pattern in temp_patterns:
        temp_files = glob.glob(pattern, recursive=True)
        for temp_file in temp_files:
            try:
                if os.path.isfile(temp_file):
                    os.remove(temp_file)
                    print_colored(f"  ✅ 删除文件: {temp_file}", "green")
                    deleted_count += 1
                elif os.path.isdir(temp_file):
                    shutil.rmtree(temp_file)
                    print_colored(f"  ✅ 删除目录: {temp_file}", "green")
                    deleted_count += 1
            except Exception as e:
                print_colored(f"  ❌ 删除失败: {temp_file} - {e}", "red")
    
    print_colored(f"临时文件清理完成: 删除了 {deleted_count} 个项目", "green")


def clean_virtual_environments():
    """清理虚拟环境（保留conda环境）"""
    print_colored("\n🐍 检查虚拟环境...", "blue")
    
    # 检查是否有venv目录
    venv_dirs = ["venv", ".venv", "env", ".env"]
    
    for venv_dir in venv_dirs:
        if os.path.exists(venv_dir) and os.path.isdir(venv_dir):
            print_colored(f"  ⚠️ 发现虚拟环境目录: {venv_dir}", "yellow")
            print_colored(f"     项目使用conda环境'epkbs'，此目录可能不需要", "yellow")
            
            # 询问是否删除
            response = input(f"     是否删除 {venv_dir} 目录? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                try:
                    shutil.rmtree(venv_dir)
                    print_colored(f"  ✅ 已删除: {venv_dir}", "green")
                except Exception as e:
                    print_colored(f"  ❌ 删除失败: {venv_dir} - {e}", "red")
            else:
                print_colored(f"  ⏭️ 跳过: {venv_dir}", "blue")


def clean_build_artifacts():
    """清理构建产物"""
    print_colored("\n🔨 清理构建产物...", "blue")
    
    build_dirs = ["build", "dist", "*.egg-info"]
    deleted_count = 0
    
    for pattern in build_dirs:
        items = glob.glob(pattern, recursive=True)
        for item in items:
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                    print_colored(f"  ✅ 删除目录: {item}", "green")
                else:
                    os.remove(item)
                    print_colored(f"  ✅ 删除文件: {item}", "green")
                deleted_count += 1
            except Exception as e:
                print_colored(f"  ❌ 删除失败: {item} - {e}", "red")
    
    print_colored(f"构建产物清理完成: 删除了 {deleted_count} 个项目", "green")


def check_project_structure():
    """检查项目结构完整性"""
    print_colored("\n📁 检查项目结构...", "blue")
    
    required_dirs = [
        "src/agent",
        "src/epkbs_mcp",  # 重命名后的MCP目录
        "src/rag", 
        "src/api",
        "src/frontend",
        "config",
        "tests",
        "docs",
        "examples",
        "scripts"
    ]
    
    required_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "main.py",
        "src/agent/langgraph_agent.py",
        "src/agent/workflows.py",
        "src/agent/states.py",
        "config/langgraph_settings.py"
    ]
    
    missing_items = []
    
    # 检查目录
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_items.append(f"目录: {dir_path}")
            print_colored(f"  ❌ 缺失目录: {dir_path}", "red")
        else:
            print_colored(f"  ✅ 目录存在: {dir_path}", "green")
    
    # 检查文件
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_items.append(f"文件: {file_path}")
            print_colored(f"  ❌ 缺失文件: {file_path}", "red")
        else:
            print_colored(f"  ✅ 文件存在: {file_path}", "green")
    
    if missing_items:
        print_colored(f"\n⚠️ 发现 {len(missing_items)} 个缺失项目:", "yellow")
        for item in missing_items:
            print_colored(f"  - {item}", "yellow")
    else:
        print_colored("\n🎉 项目结构完整！", "green")
    
    return len(missing_items) == 0


def generate_project_summary():
    """生成项目摘要"""
    print_colored("\n📊 生成项目摘要...", "blue")
    
    # 统计代码行数
    python_files = list(Path(".").rglob("*.py"))
    total_lines = 0
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                total_lines += lines
        except:
            pass
    
    # 统计文件数量
    file_counts = {
        "Python文件": len(list(Path(".").rglob("*.py"))),
        "配置文件": len(list(Path(".").rglob("*.yaml"))) + len(list(Path(".").rglob("*.yml"))) + len(list(Path(".").rglob("*.toml"))),
        "文档文件": len(list(Path(".").rglob("*.md"))),
        "测试文件": len(list(Path("tests").rglob("*.py"))) if Path("tests").exists() else 0,
        "示例文件": len(list(Path("examples").rglob("*.py"))) if Path("examples").exists() else 0
    }
    
    print_colored("项目统计:", "green")
    print_colored(f"  📝 总代码行数: {total_lines:,}", "white")
    for file_type, count in file_counts.items():
        print_colored(f"  📄 {file_type}: {count}", "white")


def main():
    """主清理函数"""
    print_colored("🧹 EPKBS项目清理工具", "blue")
    print_colored("=" * 50, "blue")
    
    # 确认是否在正确的目录
    if not os.path.exists("README.md") or not os.path.exists("src"):
        print_colored("❌ 请在EPKBS项目根目录运行此脚本", "red")
        return
    
    print_colored("📍 当前目录: " + os.getcwd(), "white")
    print_colored("🎯 开始清理项目...\n", "white")
    
    # 执行清理步骤
    try:
        clean_python_cache()
        clean_log_files()
        clean_temp_files()
        clean_virtual_environments()
        clean_build_artifacts()
        
        print_colored("\n" + "=" * 50, "blue")
        
        # 检查项目结构
        structure_ok = check_project_structure()
        
        # 生成项目摘要
        generate_project_summary()
        
        print_colored("\n" + "=" * 50, "blue")
        
        if structure_ok:
            print_colored("🎉 项目清理完成！项目结构完整，可以正常使用。", "green")
        else:
            print_colored("⚠️ 项目清理完成，但发现一些缺失的文件或目录。", "yellow")
        
        print_colored("\n💡 建议:", "blue")
        print_colored("  1. 使用 'conda activate epkbs' 激活正确的环境", "white")
        print_colored("  2. 运行 'python scripts/setup/check_environment.py' 验证环境", "white")
        print_colored("  3. 定期运行此清理脚本保持项目整洁", "white")
        
    except Exception as e:
        print_colored(f"❌ 清理过程中发生错误: {e}", "red")


if __name__ == "__main__":
    main()

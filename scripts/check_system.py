#!/usr/bin/env python3
"""
系统状态检查脚本
检查所有组件的运行状态和配置
"""
import asyncio
import sys
import subprocess
import socket
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger("system.check")

class SystemChecker:
    """系统状态检查器"""
    
    def __init__(self):
        self.results = {}
        logger.info("系统状态检查器初始化完成")
    
    def check_port(self, host: str, port: int, timeout: int = 5) -> bool:
        """检查端口是否可访问"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def run_command(self, command: str) -> Tuple[bool, str]:
        """运行系统命令"""
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0, result.stdout.strip()
        except Exception as e:
            return False, str(e)
    
    def check_python_environment(self) -> Dict[str, Any]:
        """检查Python环境"""
        logger.info("检查Python环境...")
        
        result = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "virtual_env": os.getenv('VIRTUAL_ENV'),
            "packages": {}
        }
        
        # 检查关键包
        key_packages = [
            'fastapi', 'streamlit', 'langchain', 'sqlalchemy',
            'redis', 'milvus', 'transformers', 'torch'
        ]
        
        for package in key_packages:
            try:
                __import__(package)
                result["packages"][package] = "✅ 已安装"
            except ImportError:
                result["packages"][package] = "❌ 未安装"
        
        return result
    
    def check_docker_services(self) -> Dict[str, Any]:
        """检查Docker服务"""
        logger.info("检查Docker服务...")
        
        result = {
            "docker_available": False,
            "docker_compose_available": False,
            "containers": {}
        }
        
        # 检查Docker
        docker_available, docker_output = self.run_command("docker --version")
        result["docker_available"] = docker_available
        if docker_available:
            result["docker_version"] = docker_output
        
        # 检查Docker Compose
        compose_available, compose_output = self.run_command("docker-compose --version")
        result["docker_compose_available"] = compose_available
        if compose_available:
            result["docker_compose_version"] = compose_output
        
        # 检查容器状态
        if docker_available and compose_available:
            containers_available, containers_output = self.run_command("docker-compose ps")
            if containers_available:
                result["containers_status"] = containers_output
                
                # 检查特定容器
                key_containers = ["postgres", "redis", "milvus", "etcd", "minio"]
                for container in key_containers:
                    container_running, _ = self.run_command(f"docker-compose ps {container}")
                    result["containers"][container] = "✅ 运行中" if container_running else "❌ 未运行"
        
        return result
    
    def check_network_ports(self) -> Dict[str, Any]:
        """检查网络端口"""
        logger.info("检查网络端口...")
        
        ports_to_check = {
            "PostgreSQL": ("localhost", 5432),
            "Redis": ("localhost", 6379),
            "Milvus": ("localhost", 19530),
            "API服务": ("localhost", 8000),
            "Streamlit": ("localhost", 8501),
            "Etcd": ("localhost", 2379),
            "MinIO": ("localhost", 9000)
        }
        
        result = {}
        for service, (host, port) in ports_to_check.items():
            is_open = self.check_port(host, port)
            result[service] = {
                "host": host,
                "port": port,
                "status": "✅ 可访问" if is_open else "❌ 不可访问"
            }
        
        return result
    
    def check_file_system(self) -> Dict[str, Any]:
        """检查文件系统"""
        logger.info("检查文件系统...")
        
        result = {
            "project_root": str(Path.cwd()),
            "directories": {},
            "config_files": {}
        }
        
        # 检查关键目录
        key_directories = [
            "src", "config", "scripts", "data", "logs",
            "data/uploads", "data/models", "data/vector_store"
        ]
        
        for directory in key_directories:
            path = Path(directory)
            result["directories"][directory] = {
                "exists": path.exists(),
                "is_directory": path.is_dir() if path.exists() else False,
                "permissions": oct(path.stat().st_mode)[-3:] if path.exists() else "N/A"
            }
        
        # 检查配置文件
        config_files = [
            ".env", ".env.example", "docker-compose.yml", "requirements.txt",
            "config/settings.py", "scripts/start_services.py"
        ]
        
        for config_file in config_files:
            path = Path(config_file)
            result["config_files"][config_file] = {
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
                "readable": os.access(path, os.R_OK) if path.exists() else False
            }
        
        return result
    
    async def check_database_connection(self) -> Dict[str, Any]:
        """检查数据库连接"""
        logger.info("检查数据库连接...")
        
        result = {
            "postgresql": "❌ 连接失败",
            "redis": "❌ 连接失败",
            "milvus": "❌ 连接失败"
        }
        
        try:
            from src.api.database import check_database_health
            health_status = await check_database_health()
            
            for service, status in health_status.items():
                if "healthy" in status.lower():
                    result[service] = "✅ 连接正常"
                else:
                    result[service] = f"❌ {status}"
                    
        except Exception as e:
            result["error"] = f"检查异常: {str(e)}"
        
        return result
    
    def check_environment_variables(self) -> Dict[str, Any]:
        """检查环境变量"""
        logger.info("检查环境变量...")
        
        key_env_vars = [
            "DATABASE_URL", "REDIS_URL", "SECRET_KEY",
            "MILVUS_HOST", "MILVUS_PORT", "MODEL_PATH",
            "EMBEDDING_MODEL", "RERANKER_MODEL"
        ]
        
        result = {}
        for var in key_env_vars:
            value = os.getenv(var)
            if value:
                # 隐藏敏感信息
                if any(sensitive in var.lower() for sensitive in ['password', 'secret', 'key']):
                    display_value = f"{value[:8]}..." if len(value) > 8 else "***"
                else:
                    display_value = value
                result[var] = f"✅ {display_value}"
            else:
                result[var] = "❌ 未设置"
        
        return result
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """运行所有检查"""
        logger.info("开始系统状态检查...")
        
        checks = [
            ("Python环境", self.check_python_environment),
            ("Docker服务", self.check_docker_services),
            ("网络端口", self.check_network_ports),
            ("文件系统", self.check_file_system),
            ("环境变量", self.check_environment_variables),
        ]
        
        results = {}
        
        for check_name, check_func in checks:
            logger.info(f"执行检查: {check_name}")
            try:
                if asyncio.iscoroutinefunction(check_func):
                    results[check_name] = await check_func()
                else:
                    results[check_name] = check_func()
            except Exception as e:
                results[check_name] = {"error": str(e)}
        
        # 数据库连接检查
        logger.info("执行检查: 数据库连接")
        try:
            results["数据库连接"] = await self.check_database_connection()
        except Exception as e:
            results["数据库连接"] = {"error": str(e)}
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """打印检查结果"""
        print("\n" + "=" * 80)
        print("🔍 系统状态检查报告")
        print("=" * 80)
        
        for category, data in results.items():
            print(f"\n📋 {category}")
            print("-" * 40)
            
            if isinstance(data, dict):
                if "error" in data:
                    print(f"❌ 检查失败: {data['error']}")
                else:
                    self._print_dict(data, indent=2)
            else:
                print(f"  {data}")
        
        print("\n" + "=" * 80)
        print("检查完成")
        print("=" * 80)
    
    def _print_dict(self, data: Dict[str, Any], indent: int = 0):
        """递归打印字典"""
        for key, value in data.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_dict(value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}{key}: {', '.join(map(str, value))}")
            else:
                print(f"{prefix}{key}: {value}")


async def main():
    """主函数"""
    try:
        checker = SystemChecker()
        results = await checker.run_all_checks()
        checker.print_results(results)
        
        # 生成JSON报告
        report_file = Path("logs/system_check_report.json")
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        
    except Exception as e:
        logger.error(f"系统检查失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

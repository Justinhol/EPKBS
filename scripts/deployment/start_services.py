#!/usr/bin/env python3
"""
系统服务启动脚本
同时启动API服务器和Streamlit前端
"""
import asyncio
import subprocess
import sys
import time
import signal
import os
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("system.startup")

class ServiceManager:
    """服务管理器"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.running = True
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("服务管理器初始化完成")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"接收到信号 {signum}，开始关闭服务...")
        self.running = False
        self._stop_all_services()
        sys.exit(0)
    
    async def wait_for_service(self, host: str, port: int, timeout: int = 60) -> bool:
        """等待服务启动"""
        import aiohttp
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://{host}:{port}/health", timeout=5) as response:
                        if response.status == 200:
                            logger.info(f"服务 {host}:{port} 启动成功")
                            return True
            except Exception:
                pass
            
            await asyncio.sleep(2)
        
        logger.error(f"服务 {host}:{port} 启动超时")
        return False
    
    def _start_api_server(self) -> Optional[subprocess.Popen]:
        """启动API服务器"""
        try:
            logger.info("启动API服务器...")
            
            cmd = [
                sys.executable, "-m", "uvicorn",
                "src.api.app:app",
                "--host", settings.API_HOST,
                "--port", str(settings.API_PORT),
                "--log-level", settings.LOG_LEVEL.lower(),
                "--access-log" if settings.DEBUG else "--no-access-log"
            ]
            
            if settings.DEBUG:
                cmd.append("--reload")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            logger.info(f"API服务器进程启动: PID={process.pid}")
            return process
            
        except Exception as e:
            logger.error(f"启动API服务器失败: {e}")
            return None
    
    def _start_streamlit_app(self) -> Optional[subprocess.Popen]:
        """启动Streamlit应用"""
        try:
            logger.info("启动Streamlit应用...")
            
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                "src/frontend/app.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false"
            ]
            
            if not settings.DEBUG:
                cmd.extend([
                    "--logger.level", "warning",
                    "--client.showErrorDetails", "false"
                ])
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            logger.info(f"Streamlit应用进程启动: PID={process.pid}")
            return process
            
        except Exception as e:
            logger.error(f"启动Streamlit应用失败: {e}")
            return None
    
    def _monitor_process(self, process: subprocess.Popen, name: str):
        """监控进程输出"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    logger.info(f"[{name}] {line.strip()}")
                
                if not self.running:
                    break
        except Exception as e:
            logger.error(f"监控进程 {name} 失败: {e}")
    
    def _stop_all_services(self):
        """停止所有服务"""
        logger.info("正在停止所有服务...")
        
        for process in self.processes:
            try:
                if process.poll() is None:  # 进程仍在运行
                    logger.info(f"终止进程 PID={process.pid}")
                    process.terminate()
                    
                    # 等待进程优雅退出
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"强制杀死进程 PID={process.pid}")
                        process.kill()
                        process.wait()
            except Exception as e:
                logger.error(f"停止进程失败: {e}")
        
        logger.info("所有服务已停止")
    
    async def initialize_system(self):
        """初始化系统"""
        logger.info("开始系统初始化...")
        
        try:
            # 初始化数据库
            from src.api.database import init_database
            await init_database()
            logger.info("数据库初始化完成")
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise
    
    async def start_services(self):
        """启动所有服务"""
        logger.info("=" * 60)
        logger.info("企业私有知识库系统启动中...")
        logger.info("=" * 60)
        
        try:
            # 系统初始化
            await self.initialize_system()
            
            # 启动API服务器
            api_process = self._start_api_server()
            if api_process:
                self.processes.append(api_process)
            else:
                raise Exception("API服务器启动失败")
            
            # 等待API服务器启动
            logger.info("等待API服务器启动...")
            api_ready = await self.wait_for_service(settings.API_HOST, settings.API_PORT)
            if not api_ready:
                raise Exception("API服务器启动超时")
            
            # 启动Streamlit应用
            streamlit_process = self._start_streamlit_app()
            if streamlit_process:
                self.processes.append(streamlit_process)
            else:
                raise Exception("Streamlit应用启动失败")
            
            # 等待Streamlit应用启动
            logger.info("等待Streamlit应用启动...")
            await asyncio.sleep(10)  # Streamlit需要更长的启动时间
            
            logger.info("=" * 60)
            logger.info("🎉 系统启动完成！")
            logger.info(f"📊 API服务器: http://{settings.API_HOST}:{settings.API_PORT}")
            logger.info(f"🌐 Web界面: http://localhost:8501")
            logger.info(f"📖 API文档: http://{settings.API_HOST}:{settings.API_PORT}/docs")
            logger.info("=" * 60)
            
            # 监控进程
            await self._monitor_services()
            
        except Exception as e:
            logger.error(f"服务启动失败: {e}")
            self._stop_all_services()
            sys.exit(1)
    
    async def _monitor_services(self):
        """监控服务状态"""
        logger.info("开始监控服务状态...")
        
        while self.running:
            try:
                # 检查进程状态
                for i, process in enumerate(self.processes):
                    if process.poll() is not None:
                        logger.error(f"进程 {i} (PID={process.pid}) 意外退出")
                        self.running = False
                        break
                
                # 定期健康检查
                if self.running:
                    await asyncio.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                logger.error(f"服务监控异常: {e}")
                break
        
        logger.info("服务监控结束")


async def main():
    """主函数"""
    try:
        # 检查环境
        logger.info(f"Python版本: {sys.version}")
        logger.info(f"工作目录: {os.getcwd()}")
        logger.info(f"项目配置: {settings.PROJECT_NAME} v{settings.PROJECT_VERSION}")
        
        # 创建服务管理器
        service_manager = ServiceManager()
        
        # 启动服务
        await service_manager.start_services()
        
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在退出...")
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

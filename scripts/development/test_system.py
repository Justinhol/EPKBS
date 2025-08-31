#!/usr/bin/env python3
"""
系统集成测试脚本
测试整个系统的各个组件是否正常工作
"""
import asyncio
import sys
import time
import aiohttp
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("test.system")


class SystemTester:
    """系统测试器"""
    
    def __init__(self):
        self.api_base_url = f"http://{settings.API_HOST}:{settings.API_PORT}"
        self.streamlit_url = "http://localhost:8501"
        self.access_token = None
        
        logger.info("系统测试器初始化完成")
    
    async def test_api_health(self) -> Tuple[bool, str]:
        """测试API健康状态"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data.get("status", "unknown")
                        return True, f"API健康状态: {status}"
                    else:
                        return False, f"API健康检查失败: HTTP {response.status}"
        except Exception as e:
            return False, f"API连接失败: {str(e)}"
    
    async def test_api_info(self) -> Tuple[bool, str]:
        """测试API系统信息"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base_url}/info") as response:
                    if response.status == 200:
                        data = await response.json()
                        project_name = data.get("project_name", "Unknown")
                        version = data.get("version", "Unknown")
                        return True, f"系统信息: {project_name} v{version}"
                    else:
                        return False, f"系统信息获取失败: HTTP {response.status}"
        except Exception as e:
            return False, f"系统信息获取异常: {str(e)}"
    
    async def test_user_registration(self) -> Tuple[bool, str]:
        """测试用户注册"""
        try:
            user_data = {
                "username": f"test_user_{int(time.time())}",
                "email": f"test_{int(time.time())}@example.com",
                "password": "test123456",
                "full_name": "测试用户"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/api/v1/auth/register",
                    json=user_data
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        username = data.get("username", "Unknown")
                        return True, f"用户注册成功: {username}"
                    else:
                        error_data = await response.json()
                        error_msg = error_data.get("message", f"HTTP {response.status}")
                        return False, f"用户注册失败: {error_msg}"
        except Exception as e:
            return False, f"用户注册异常: {str(e)}"
    
    async def test_user_login(self) -> Tuple[bool, str]:
        """测试用户登录"""
        try:
            # 使用默认管理员账户
            login_data = {
                "username": "admin",
                "password": "admin123"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/api/v1/auth/login",
                    data=login_data  # OAuth2使用form data
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.access_token = data.get("access_token")
                        user = data.get("user", {})
                        username = user.get("username", "Unknown")
                        return True, f"用户登录成功: {username}"
                    else:
                        error_data = await response.json()
                        error_msg = error_data.get("detail", f"HTTP {response.status}")
                        return False, f"用户登录失败: {error_msg}"
        except Exception as e:
            return False, f"用户登录异常: {str(e)}"
    
    async def test_conversation_creation(self) -> Tuple[bool, str]:
        """测试对话创建"""
        if not self.access_token:
            return False, "需要先登录"
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            conversation_data = {
                "title": "测试对话",
                "use_rag": True,
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/api/v1/chat/conversations",
                    json=conversation_data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        conv_id = data.get("id", "Unknown")
                        title = data.get("title", "Unknown")
                        return True, f"对话创建成功: {title} (ID: {conv_id})"
                    else:
                        error_data = await response.json()
                        error_msg = error_data.get("message", f"HTTP {response.status}")
                        return False, f"对话创建失败: {error_msg}"
        except Exception as e:
            return False, f"对话创建异常: {str(e)}"
    
    async def test_search_functionality(self) -> Tuple[bool, str]:
        """测试搜索功能"""
        if not self.access_token:
            return False, "需要先登录"
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            search_data = {
                "query": "人工智能",
                "top_k": 5,
                "retriever_type": "hybrid",
                "reranker_type": "ensemble"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/api/v1/search/",
                    json=search_data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        result_count = data.get("total_results", 0)
                        execution_time = data.get("execution_time", 0)
                        return True, f"搜索成功: {result_count} 个结果, 耗时 {execution_time:.3f}秒"
                    else:
                        error_data = await response.json()
                        error_msg = error_data.get("message", f"HTTP {response.status}")
                        return False, f"搜索失败: {error_msg}"
        except Exception as e:
            return False, f"搜索异常: {str(e)}"
    
    async def test_streamlit_app(self) -> Tuple[bool, str]:
        """测试Streamlit应用"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.streamlit_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        if "企业私有知识库系统" in content:
                            return True, "Streamlit应用运行正常"
                        else:
                            return False, "Streamlit应用内容异常"
                    else:
                        return False, f"Streamlit应用访问失败: HTTP {response.status}"
        except Exception as e:
            return False, f"Streamlit应用连接失败: {str(e)}"
    
    async def test_database_connection(self) -> Tuple[bool, str]:
        """测试数据库连接"""
        try:
            from src.api.database import check_database_health
            
            health_status = await check_database_health()
            
            db_status = health_status.get("database", "unknown")
            redis_status = health_status.get("redis", "unknown")
            
            if "healthy" in db_status and "healthy" in redis_status:
                return True, f"数据库连接正常: DB={db_status}, Redis={redis_status}"
            else:
                return False, f"数据库连接异常: DB={db_status}, Redis={redis_status}"
        except Exception as e:
            return False, f"数据库连接测试异常: {str(e)}"
    
    async def run_all_tests(self) -> Dict[str, Tuple[bool, str]]:
        """运行所有测试"""
        tests = [
            ("API健康检查", self.test_api_health),
            ("API系统信息", self.test_api_info),
            ("数据库连接", self.test_database_connection),
            ("用户注册", self.test_user_registration),
            ("用户登录", self.test_user_login),
            ("对话创建", self.test_conversation_creation),
            ("搜索功能", self.test_search_functionality),
            ("Streamlit应用", self.test_streamlit_app),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"执行测试: {test_name}")
            try:
                success, message = await test_func()
                results[test_name] = (success, message)
                
                if success:
                    logger.info(f"✅ {test_name}: {message}")
                else:
                    logger.error(f"❌ {test_name}: {message}")
            except Exception as e:
                error_msg = f"测试异常: {str(e)}"
                results[test_name] = (False, error_msg)
                logger.error(f"❌ {test_name}: {error_msg}")
            
            # 测试间隔
            await asyncio.sleep(1)
        
        return results


async def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始系统集成测试")
    logger.info("=" * 60)
    
    tester = SystemTester()
    
    # 运行所有测试
    results = await tester.run_all_tests()
    
    # 统计结果
    total_tests = len(results)
    passed_tests = sum(1 for success, _ in results.values() if success)
    failed_tests = total_tests - passed_tests
    
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    for test_name, (success, message) in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"{test_name}: {status}")
        if not success:
            logger.info(f"  错误信息: {message}")
    
    logger.info("=" * 60)
    logger.info(f"总测试数: {total_tests}")
    logger.info(f"通过: {passed_tests}")
    logger.info(f"失败: {failed_tests}")
    logger.info(f"成功率: {passed_tests/total_tests*100:.1f}%")
    logger.info("=" * 60)
    
    if failed_tests == 0:
        logger.info("🎉 所有测试通过！系统运行正常")
        return True
    else:
        logger.error(f"❌ {failed_tests} 个测试失败，请检查系统配置")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

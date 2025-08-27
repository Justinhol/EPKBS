"""
FastAPI主应用
"""
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import time

from src.api.database import init_database, close_database, check_database_health
from src.api.routes import auth, chat, search
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("正在启动API服务...")
    
    try:
        # 初始化数据库
        await init_database()
        logger.info("数据库初始化完成")
        
        # 这里可以添加其他初始化逻辑
        # 比如预加载模型、初始化缓存等
        
        logger.info("API服务启动完成")
        
    except Exception as e:
        logger.error(f"API服务启动失败: {e}")
        raise
    
    yield
    
    # 关闭时执行
    logger.info("正在关闭API服务...")
    
    try:
        await close_database()
        logger.info("数据库连接已关闭")
        
    except Exception as e:
        logger.warning(f"关闭API服务时出现警告: {e}")
    
    logger.info("API服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="企业私有知识库系统 - RAG + Agent + MCP架构",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加可信主机中间件
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", settings.API_HOST]
    )


# 请求处理时间中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加请求处理时间头"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# 全局异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail} - {request.url}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
            "path": str(request.url),
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    logger.error(f"未处理的异常: {exc} - {request.url}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "status_code": 500,
            "message": "内部服务器错误" if not settings.DEBUG else str(exc),
            "path": str(request.url),
            "timestamp": time.time()
        }
    )


# 注册路由
app.include_router(auth.router, prefix=settings.API_PREFIX)
app.include_router(chat.router, prefix=settings.API_PREFIX)
app.include_router(search.router, prefix=settings.API_PREFIX)


# 根路径
@app.get("/", summary="根路径")
async def root():
    """API根路径"""
    return {
        "message": f"欢迎使用{settings.PROJECT_NAME}",
        "version": settings.PROJECT_VERSION,
        "docs_url": "/docs" if settings.DEBUG else None,
        "api_prefix": settings.API_PREFIX
    }


# 健康检查
@app.get("/health", summary="健康检查")
async def health_check():
    """系统健康检查"""
    try:
        # 检查数据库连接
        db_health = await check_database_health()
        
        # 检查系统状态
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.PROJECT_VERSION,
            "database": db_health.get("database", "unknown"),
            "redis": db_health.get("redis", "unknown"),
            "debug_mode": settings.DEBUG
        }
        
        # 如果数据库不健康，返回503状态
        if db_health.get("database", "").startswith("unhealthy"):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={**health_status, "status": "unhealthy"}
            )
        
        return health_status
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }
        )


# 系统信息
@app.get("/info", summary="系统信息")
async def system_info():
    """获取系统信息"""
    return {
        "project_name": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "debug": settings.DEBUG,
        "api_host": settings.API_HOST,
        "api_port": settings.API_PORT,
        "api_prefix": settings.API_PREFIX,
        "database_url": settings.DATABASE_URL.split("@")[-1] if "@" in settings.DATABASE_URL else "配置中",
        "redis_url": settings.REDIS_URL.split("@")[-1] if "@" in settings.REDIS_URL else "配置中",
        "features": {
            "rag_enabled": True,
            "agent_enabled": True,
            "cache_enabled": settings.ENABLE_CACHE,
            "metrics_enabled": settings.ENABLE_METRICS
        }
    }


# 自定义OpenAPI文档
def custom_openapi():
    """自定义OpenAPI文档"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.PROJECT_NAME,
        version=settings.PROJECT_VERSION,
        description="""
        ## 企业私有知识库系统API
        
        基于RAG + Agent + MCP架构的智能知识库系统，提供以下功能：
        
        ### 🔐 认证功能
        - 用户注册、登录、令牌管理
        - JWT认证和权限控制
        
        ### 💬 聊天功能  
        - 智能对话和问答
        - ReAct Agent推理
        - 流式响应支持
        
        ### 🔍 搜索功能
        - 混合检索（向量+稀疏+全文）
        - 智能重排序
        - 搜索历史和统计
        
        ### 📊 系统功能
        - 健康检查和监控
        - 系统信息和状态
        
        ---
        
        **技术栈**: FastAPI + PostgreSQL + Redis + Milvus + Qwen3
        """,
        routes=app.routes,
    )
    
    # 添加安全定义
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# 自定义Swagger UI
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """自定义Swagger UI"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API文档",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


# 启动函数
async def start_server():
    """启动服务器"""
    import uvicorn
    
    config = uvicorn.Config(
        app=app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.DEBUG,
        access_log=settings.DEBUG
    )
    
    server = uvicorn.Server(config)
    
    logger.info(f"启动API服务器: http://{settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"API文档: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    
    await server.serve()


if __name__ == "__main__":
    asyncio.run(start_server())

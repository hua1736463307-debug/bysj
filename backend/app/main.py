"""应用工厂。

装配顺序：日志 → 中间件 → 异常映射 → 路由白名单。模块导入不产生副作用；
建目录、连数据库、初始化管理员只发生在 lifespan。
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app import __version__
from app.api.v1 import annotations as annotations_api
from app.api.v1 import auth as auth_api
from app.api.v1 import artifacts as artifacts_api
from app.api.v1 import datasets as datasets_api
from app.api.v1 import evaluations as evaluations_api
from app.api.v1 import experiments as experiments_api
from app.api.v1 import health as health_api
from app.api.v1 import inference as inference_api
from app.api.v1 import jobs as jobs_api
from app.api.v1 import models as models_api
from app.api.v1 import reviews as reviews_api
from app.api.v1 import training as training_api
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import RequestContextMiddleware, configure_logging
from app.db.session import get_engine, get_session_factory

logger = logging.getLogger("bysj.main")


def _bootstrap(settings: Settings) -> None:
    settings.ensure_runtime_dirs()
    try:
        factory = get_session_factory()
        with factory() as db:
            db.execute(text("SELECT 1"))
            from app.domains.auth_service import ensure_bootstrap_admin

            ensure_bootstrap_admin(db, settings)
    except Exception as exc:  # 数据库未就绪时保持 live 可用，ready 报告降级
        logger.warning("startup_database_unavailable error=%s", type(exc).__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging("DEBUG" if settings.debug else "INFO")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        _bootstrap(settings)
        yield
        try:
            get_engine().dispose()
        except Exception:  # pragma: no cover
            pass

    app = FastAPI(
        title="医学图像训推平台 API",
        version=__version__,
        description="毕业设计医学图像检测平台（P0 + 人工复核闭环）。",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    # 路由白名单：只挂载医学域。监控/行为/人脸/录机/姿态不进入新入口。
    for module in (
        health_api,
        auth_api,
        jobs_api,
        datasets_api,
        annotations_api,
        training_api,
        models_api,
        evaluations_api,
        inference_api,
        artifacts_api,
        reviews_api,
        experiments_api,
    ):
        app.include_router(module.router, prefix=settings.api_prefix)

    return app


app = create_app()

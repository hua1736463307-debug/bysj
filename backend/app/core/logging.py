"""请求 ID、访问日志与日志配置。"""
from __future__ import annotations

import contextvars
import logging
import sys
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


def get_request_id() -> str:
    return _request_id.get() or "-"


def set_request_id(value: str) -> contextvars.Token:
    return _request_id.set(value)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s")
        )
        root.addHandler(handler)
    root.setLevel(level.upper())

    class _RequestIdFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.request_id = get_request_id()
            return True

    for handler in root.handlers:
        handler.addFilter(_RequestIdFilter())


class RequestContextMiddleware(BaseHTTPMiddleware):
    """为每个请求生成/透传 request_id，并写入响应头。"""

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("X-Request-ID", "").strip()
        request_id = incoming[:64] if incoming else str(uuid.uuid4())
        token = set_request_id(request_id)
        try:
            response = await call_next(request)
        finally:
            _request_id.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response

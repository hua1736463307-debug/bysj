"""统一 JSON envelope、业务异常与错误处理。

成功：``{"code": "ok", "message": "...", "data": {...}, "request_id": "..."}``
失败：code 为稳定业务字符串，HTTP 状态码保持真实语义（CONTRACTS C01）。
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_request_id


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.data = data or {}
        self.headers = headers or {}


def envelope(
    data: Any = None,
    *,
    code: str = "ok",
    message: str = "操作成功",
    request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "data": data,
        "request_id": request_id or get_request_id(),
    }


def ok(data: Any = None, message: str = "操作成功", status_code: int = 200) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=envelope(data, message=message))


def created(data: Any = None, message: str = "创建成功") -> JSONResponse:
    return JSONResponse(status_code=201, content=envelope(data, message=message))


def accepted(data: Any = None, message: str = "任务已接收") -> JSONResponse:
    return JSONResponse(status_code=202, content=envelope(data, message=message))


def nocontent() -> JSONResponse:
    return JSONResponse(status_code=204, content=None)


def error_response(
    status_code: int,
    code: str,
    message: str,
    data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=envelope(data, code=code, message=message),
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message, exc.data, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = []
        for item in exc.errors():
            fields.append(
                {
                    "loc": [str(part) for part in item.get("loc", ())],
                    "msg": item.get("msg", "invalid"),
                    "type": item.get("type", "value_error"),
                }
            )
        return error_response(422, "validation_error", "请求字段校验失败", {"fields": fields})

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {401: "unauthenticated", 403: "forbidden", 404: "not_found", 405: "method_not_allowed"}.get(
            exc.status_code, "http_error"
        )
        return error_response(
            exc.status_code,
            code,
            str(exc.detail) if exc.detail else code,
            headers=getattr(exc, "headers", None) or None,
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # 不返回 traceback / 连接串 / 绝对路径；细节只进日志（含 request_id）。
        import logging

        logging.getLogger("bysj.error").exception("unhandled error: %s", exc)
        return error_response(500, "internal_error", "服务器内部错误，请稍后重试")

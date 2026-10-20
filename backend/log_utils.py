"""系统日志写入工具。"""
from collections import Counter
import json
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import Request
from sqlalchemy.orm import Session

from database import LoginLog, OperationLog


logger = logging.getLogger(__name__)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
AUTH_AUDIT_FAILURE_COUNTS = Counter()


def _count_auth_audit_failure(kind: str) -> int:
    AUTH_AUDIT_FAILURE_COUNTS[kind] += 1
    count = AUTH_AUDIT_FAILURE_COUNTS[kind]
    logger.error("认证审计失败计数告警: kind=%s count=%s", kind, count)
    return count


def get_auth_audit_failure_counts() -> dict:
    return dict(AUTH_AUDIT_FAILURE_COUNTS)


def get_client_ip(request: Optional[Request]) -> str:
    if not request:
        return ""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else ""


def get_request_id(request: Optional[Request]) -> str:
    """Return the bounded request correlation ID assigned by the HTTP middleware."""
    if not request:
        return ""
    current = str(getattr(getattr(request, "state", None), "request_id", "") or "").strip()
    if current:
        return current[:128]
    candidate = str(request.headers.get("x-request-id", "") or "").strip()
    if not REQUEST_ID_PATTERN.fullmatch(candidate):
        candidate = uuid.uuid4().hex
    request.state.request_id = candidate
    return candidate


def parse_user_agent(user_agent: str) -> tuple[str, str]:
    value = user_agent or ""

    if "Edg/" in value:
        browser = f"Edge {value.split('Edg/')[-1].split('.')[0]}"
    elif "Chrome/" in value:
        browser = f"Chrome {value.split('Chrome/')[-1].split('.')[0]}"
    elif "Firefox/" in value:
        browser = f"Firefox {value.split('Firefox/')[-1].split('.')[0]}"
    elif "Safari/" in value:
        browser = "Safari"
    else:
        browser = "未知"

    if "Windows NT 10" in value:
        os_name = "Windows10"
    elif "Windows" in value:
        os_name = "Windows"
    elif "Mac OS X" in value:
        os_name = "macOS"
    elif "Android" in value:
        os_name = "Android"
    elif "iPhone" in value or "iPad" in value:
        os_name = "iOS"
    elif "Linux" in value:
        os_name = "Linux"
    else:
        os_name = "未知"

    return browser, os_name


def record_login_log(db: Session, username: str, status: str, message: str, request: Optional[Request]) -> None:
    browser, os_name = parse_user_agent(request.headers.get("user-agent", "") if request else "")
    try:
        db.add(LoginLog(
            username=username or "",
            ip_address=get_client_ip(request),
            location="内网IP",
            browser=browser,
            os=os_name,
            status=status,
            message=message,
            login_time=datetime.now(),
        ))
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("写入登录日志失败: username=%s status=%s", username or "", status)


def record_operation_log(
    db: Session,
    *,
    title: str,
    operation_type: str,
    operator: str,
    status: str,
    detail: str,
    request: Optional[Request],
    started_at: Optional[float] = None,
    response_data=None,
    error_message: str = "",
    session_id: str = "",
    request_id: str = "",
    reason: str = "",
    user_agent: str = "",
    event_key: str = "",
    commit: bool = True,
    raise_on_error: bool = False,
) -> Optional[OperationLog]:
    cost_ms = int((time.perf_counter() - started_at) * 1000) if started_at else 0
    try:
        request_params = ""
        request_method = ""
        request_url = ""
        if request:
            request_method = request.method
            request_url = str(request.url)
            request_params = json.dumps(dict(request.query_params), ensure_ascii=False)
            request_id = request_id or get_request_id(request)
            user_agent = user_agent or request.headers.get("user-agent", "")
        if response_data is None:
            response_text = ""
        elif isinstance(response_data, str):
            response_text = response_data
        else:
            response_text = json.dumps(response_data, ensure_ascii=False, default=str)

        entry = OperationLog(
            title=title,
            operation_type=operation_type,
            operator=operator or "",
            ip_address=get_client_ip(request),
            location="内网IP",
            status=status,
            detail=detail,
            request_method=request_method,
            request_url=request_url,
            request_params=request_params,
            response_data=response_text,
            error_message=error_message or "",
            session_id=session_id or "",
            request_id=request_id or "",
            reason=reason or "",
            user_agent=user_agent or "",
            event_key=event_key or None,
            cost_ms=cost_ms,
            operation_time=datetime.now(),
        )
        db.add(entry)
        if commit:
            db.commit()
        return entry
    except Exception:
        db.rollback()
        if title == "认证":
            _count_auth_audit_failure("write")
        logger.exception(
            "写入操作日志失败: title=%s type=%s operator=%s event_key=%s",
            title,
            operation_type,
            operator or "",
            event_key or "",
        )
        if raise_on_error:
            raise
        return None


def record_auth_audit(
    db: Session,
    *,
    operator: str,
    session_id: str,
    operation_type: str,
    reason: str,
    detail: str,
    request: Optional[Request],
    status: str = "成功",
    event_key: str = "",
    error_message: str = "",
    commit: bool = False,
) -> Optional[OperationLog]:
    """Append an authentication audit event to the same transaction as session state."""
    if event_key and db.query(OperationLog.id).filter(OperationLog.event_key == event_key).first():
        return None
    return record_operation_log(
        db,
        title="认证",
        operation_type=operation_type,
        operator=operator,
        status=status,
        detail=detail,
        request=request,
        session_id=session_id,
        reason=reason,
        event_key=event_key,
        error_message=error_message,
        commit=commit,
        raise_on_error=True,
    )


def commit_auth_transition(db: Session, context: str) -> None:
    """Commit a session state change and its audit row atomically, with visible failure."""
    try:
        db.commit()
    except Exception:
        db.rollback()
        _count_auth_audit_failure("transaction")
        logger.exception("认证状态与审计日志事务失败: %s", context)
        raise

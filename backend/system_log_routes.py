"""平台管理日志查询与导出接口。"""
import csv
import io
from datetime import datetime, time as datetime_time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy.orm import Session

from auth_utils import require_any_menu
from database import LoginLog, OperationLog, User, get_db
from log_utils import record_operation_log


require_login_log_menu = require_any_menu("/system/log/logininfor")
require_operation_log_menu = require_any_menu("/system/log/operlog")


router = APIRouter(tags=["平台管理日志"])


def _admin_name(admin) -> str:
    return str((admin or {}).get("username") or "admin")


def _ids(value: str):
    result = []
    for item in str(value or "").split(","):
        try:
            result.append(int(item.strip()))
        except (TypeError, ValueError):
            continue
    return list(dict.fromkeys(result))


def _parse_datetime(value: Optional[str], *, end_of_day: bool = False) -> Optional[datetime]:
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    normalized = text.replace("/", "-")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(normalized, fmt)
                break
            except ValueError:
                continue

    if not parsed:
        return None

    if end_of_day and parsed.time() == datetime_time.min:
        return parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return parsed


def _is_ascending(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "asc", "ascending"}


def _apply_order(query, order_by: str, is_asc: str, columns, default_column, tie_breaker):
    column = columns.get(str(order_by or "").strip())
    if column is None:
        direction = default_column.desc()
    else:
        direction = column.asc() if _is_ascending(is_asc) else column.desc()
    return query.order_by(direction, tie_breaker.desc())


def _login_log_query(db: Session, *, ip: str = "", user_name: str = "", status: str = "", start_time: str = "", end_time: str = ""):
    query = db.query(LoginLog)
    if ip:
        query = query.filter(LoginLog.ip_address.contains(ip.strip()))
    if user_name:
        query = query.filter(LoginLog.username.contains(user_name.strip()))
    if status:
        query = query.filter(LoginLog.status == status.strip())
    start_dt = _parse_datetime(start_time)
    end_dt = _parse_datetime(end_time, end_of_day=True)
    if start_dt:
        query = query.filter(LoginLog.login_time >= start_dt)
    if end_dt:
        query = query.filter(LoginLog.login_time <= end_dt)
    return query


def _operation_log_query(db: Session, *, ip: str = "", title: str = "", operator: str = "", operation_type: str = "", status: str = "", session_id: str = "", reason: str = "", start_time: str = "", end_time: str = ""):
    query = db.query(OperationLog)
    operation_type = operation_type if isinstance(operation_type, str) else ""
    if ip:
        query = query.filter(OperationLog.ip_address.contains(ip.strip()))
    if title:
        query = query.filter(OperationLog.title.contains(title.strip()))
    if operator:
        query = query.filter(OperationLog.operator.contains(operator.strip()))
    if operation_type:
        query = query.filter(OperationLog.operation_type == operation_type.strip())
    if status:
        query = query.filter(OperationLog.status == status.strip())
    if session_id:
        query = query.filter(OperationLog.session_id.contains(session_id.strip()))
    if reason:
        query = query.filter(OperationLog.reason == reason.strip())
    start_dt = _parse_datetime(start_time)
    end_dt = _parse_datetime(end_time, end_of_day=True)
    if start_dt:
        query = query.filter(OperationLog.operation_time >= start_dt)
    if end_dt:
        query = query.filter(OperationLog.operation_time <= end_dt)
    return query


def _csv_response(filename: str, headers, rows) -> Response:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return Response(
        content=output.getvalue().encode("utf-8-sig"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/system/login-logs")
@router.get("/api/system/login-logs")
def list_login_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=200),
    ip: str = "",
    user_name: str = "",
    status: str = "",
    start_time: str = "",
    end_time: str = "",
    order_by: str = "",
    is_asc: str = "",
    admin=Depends(require_login_log_menu),
    db: Session = Depends(get_db),
):
    query = _login_log_query(db, ip=ip, user_name=user_name, status=status, start_time=start_time, end_time=end_time)
    total = query.count()
    query = _apply_order(
        query,
        order_by,
        is_asc,
        {"id": LoginLog.id, "userName": LoginLog.username, "user_name": LoginLog.username, "loginTime": LoginLog.login_time, "login_time": LoginLog.login_time},
        LoginLog.login_time,
        LoginLog.id,
    )
    rows = query.offset((page - 1) * page_size).limit(page_size).all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "rows": [row.to_dict() for row in rows],
    }


@router.get("/system/operation-logs")
@router.get("/api/system/operation-logs")
def list_operation_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=200),
    ip: str = "",
    title: str = "",
    operator: str = "",
    operation_type: str = Query("", alias="type"),
    status: str = "",
    session_id: str = "",
    reason: str = "",
    start_time: str = "",
    end_time: str = "",
    order_by: str = "",
    is_asc: str = "",
    admin=Depends(require_operation_log_menu),
    db: Session = Depends(get_db),
):
    query = _operation_log_query(db, ip=ip, title=title, operator=operator, operation_type=operation_type, status=status, session_id=session_id, reason=reason, start_time=start_time, end_time=end_time)
    total = query.count()
    query = _apply_order(
        query,
        order_by,
        is_asc,
        {"id": OperationLog.id, "operator": OperationLog.operator, "operTime": OperationLog.operation_time, "operation_time": OperationLog.operation_time, "cost": OperationLog.cost_ms, "costTime": OperationLog.cost_ms, "cost_ms": OperationLog.cost_ms},
        OperationLog.operation_time,
        OperationLog.id,
    )
    rows = query.offset((page - 1) * page_size).limit(page_size).all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "rows": [row.to_dict() for row in rows],
    }


@router.get("/system/login-logs/export")
@router.get("/api/system/login-logs/export")
def export_login_logs(
    ip: str = "",
    user_name: str = "",
    status: str = "",
    start_time: str = "",
    end_time: str = "",
    order_by: str = "",
    is_asc: str = "",
    admin=Depends(require_login_log_menu),
    db: Session = Depends(get_db),
):
    query = _login_log_query(db, ip=ip, user_name=user_name, status=status, start_time=start_time, end_time=end_time)
    query = _apply_order(query, order_by, is_asc, {"id": LoginLog.id, "userName": LoginLog.username, "user_name": LoginLog.username, "loginTime": LoginLog.login_time, "login_time": LoginLog.login_time}, LoginLog.login_time, LoginLog.id)
    rows = [item.to_dict() for item in query.all()]
    return _csv_response(
        f"login_logs_{datetime.now():%Y%m%d_%H%M%S}.csv",
        ["访问编号", "用户名称", "登录地址", "登录地点", "浏览器", "操作系统", "登录状态", "操作信息", "登录日期"],
        [[row["id"], row["userName"], row["ip"], row["location"], row["browser"], row["os"], row["status"], row["message"], row["loginTime"]] for row in rows],
    )


@router.get("/system/operation-logs/export")
@router.get("/api/system/operation-logs/export")
def export_operation_logs(
    ip: str = "",
    title: str = "",
    operator: str = "",
    operation_type: str = Query("", alias="type"),
    status: str = "",
    session_id: str = "",
    reason: str = "",
    start_time: str = "",
    end_time: str = "",
    order_by: str = "",
    is_asc: str = "",
    admin=Depends(require_operation_log_menu),
    db: Session = Depends(get_db),
):
    query = _operation_log_query(db, ip=ip, title=title, operator=operator, operation_type=operation_type, status=status, session_id=session_id, reason=reason, start_time=start_time, end_time=end_time)
    query = _apply_order(query, order_by, is_asc, {"id": OperationLog.id, "operator": OperationLog.operator, "operTime": OperationLog.operation_time, "operation_time": OperationLog.operation_time, "cost": OperationLog.cost_ms, "costTime": OperationLog.cost_ms, "cost_ms": OperationLog.cost_ms}, OperationLog.operation_time, OperationLog.id)
    rows = [item.to_dict() for item in query.all()]
    return _csv_response(
        f"operation_logs_{datetime.now():%Y%m%d_%H%M%S}.csv",
        ["日志编号", "系统模块", "操作类型", "操作人员", "操作地址", "操作地点", "操作状态", "退出原因", "会话编号", "请求编号", "操作日期", "消耗时间"],
        [[row["id"], row["title"], row["type"], row["operator"], row["ip"], row["location"], row["status"], row["reason"], row["sessionId"], row["requestId"], row["operTime"], row["cost"]] for row in rows],
    )


@router.delete("/system/login-logs/{log_ids}")
@router.delete("/api/system/login-logs/{log_ids}")
def delete_login_logs(
    log_ids: str,
    request: Request,
    admin=Depends(require_login_log_menu),
    db: Session = Depends(get_db),
):
    ids = _ids(log_ids)
    if not ids:
        raise HTTPException(status_code=400, detail="请选择要删除的登录日志")
    deleted = db.query(LoginLog).filter(LoginLog.id.in_(ids)).delete(synchronize_session=False)
    db.commit()
    record_operation_log(db, title="登录日志", operation_type="删除", operator=_admin_name(admin), status="成功", detail=f"删除 {deleted} 条登录日志", request=request)
    return {"message": "登录日志删除成功", "deleted": deleted}


@router.delete("/system/login-logs")
@router.delete("/api/system/login-logs")
def clean_login_logs(request: Request, admin=Depends(require_login_log_menu), db: Session = Depends(get_db)):
    deleted = db.query(LoginLog).delete(synchronize_session=False)
    db.commit()
    record_operation_log(db, title="登录日志", operation_type="清空", operator=_admin_name(admin), status="成功", detail=f"清空 {deleted} 条登录日志", request=request)
    return {"message": "登录日志已清空", "deleted": deleted}


@router.put("/system/login-logs/unlock/{username}")
@router.put("/api/system/login-logs/unlock/{username}")
def unlock_user(username: str, request: Request, admin=Depends(require_login_log_menu), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    user.failed_login_attempts = 0
    user.locked_until = None
    db.commit()
    record_operation_log(db, title="登录日志", operation_type="解锁", operator=_admin_name(admin), status="成功", detail=f"解锁用户：{username}", request=request)
    return {"message": f"用户 {username} 已解锁"}


@router.delete("/system/operation-logs/{log_ids}")
@router.delete("/api/system/operation-logs/{log_ids}")
def delete_operation_logs(
    log_ids: str,
    request: Request,
    admin=Depends(require_operation_log_menu),
    db: Session = Depends(get_db),
):
    ids = _ids(log_ids)
    if not ids:
        raise HTTPException(status_code=400, detail="请选择要删除的操作日志")
    deleted = db.query(OperationLog).filter(OperationLog.id.in_(ids)).delete(synchronize_session=False)
    db.commit()
    record_operation_log(db, title="操作日志", operation_type="删除", operator=_admin_name(admin), status="成功", detail=f"删除 {deleted} 条操作日志", request=request)
    return {"message": "操作日志删除成功", "deleted": deleted}


@router.delete("/system/operation-logs")
@router.delete("/api/system/operation-logs")
def clean_operation_logs(request: Request, admin=Depends(require_operation_log_menu), db: Session = Depends(get_db)):
    deleted = db.query(OperationLog).delete(synchronize_session=False)
    db.commit()
    record_operation_log(db, title="操作日志", operation_type="清空", operator=_admin_name(admin), status="成功", detail=f"清空 {deleted} 条操作日志", request=request)
    return {"message": "操作日志已清空", "deleted": deleted}

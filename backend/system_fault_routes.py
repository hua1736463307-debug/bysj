"""Fault log query and bounded browser-error ingestion."""
import time
import threading
from collections import deque

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from auth_utils import require_any_menu
from system_faults import list_faults, record_fault

router = APIRouter(tags=["系统故障日志"])
require_fault_menu = require_any_menu("/system/log/faults")
_client_events = deque()
_client_lock = threading.Lock()


class ClientFault(BaseModel):
    category: str = Field(max_length=40)
    message: str = Field(max_length=1200)
    details: str = Field(default="", max_length=4000)
    path: str = Field(default="", max_length=250)


@router.get("/system/faults")
@router.get("/api/system/faults")
def get_faults(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
               service: str = "", severity: str = "", start: str = "", end: str = "",
               _admin=Depends(require_fault_menu)):
    return list_faults(page=page, page_size=page_size, service=service,
                       severity=severity, start=start, end=end)


@router.post("/system/faults/client", status_code=204)
@router.post("/api/system/faults/client", status_code=204)
def report_client_fault(body: ClientFault):
    # The application's existing HTTP auth middleware requires a valid login.
    now = time.monotonic()
    with _client_lock:
        while _client_events and now - _client_events[0] > 60:
            _client_events.popleft()
        if len(_client_events) >= 30:
            return
        _client_events.append(now)
    record_fault(service="frontend", severity="ERROR", category=body.category,
                 message=f"{body.path}: {body.message}", details=body.details, source="browser")

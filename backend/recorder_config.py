"""Owner-scoped recorder configuration and authenticated connection tests."""
import ipaddress
import re
import copy
import threading
from datetime import datetime, timezone
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from auth_utils import require_any_menu
from database import RecorderConfig, get_db
from credential_crypto import DeviceCredentialError, decrypt_device_credential, encrypt_device_credential
from recorder_connection import ProbeFailure, connect_recorder, hikvision_channel_inventory
from recorder_playback import validate_template, playback_mode

router = APIRouter(prefix="/monitoring/recorders", tags=["recorder-configuration"])
access = require_any_menu("/monitoring/config")
probe_slots = threading.BoundedSemaphore(2)
probe_lock = threading.Lock()
active_probes = set()


class Channel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    number: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    name: str = Field(min_length=1, max_length=100)
    stream: Literal["main", "sub"] = "sub"
    enabled: bool = True
    rtsp_path: str = Field(default='', max_length=1024, pattern=r'^(?:/[^\s#]*)?$')
    algorithms: list[str] = Field(default_factory=list, max_length=100)


class RecorderInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100)
    vendor: str = Field(default="generic", min_length=1, max_length=64)
    connection_protocol: Literal['rtsp', 'onvif', 'hikvision_isapi'] = 'rtsp'
    onvif_path: str = Field(default='/onvif/device_service', max_length=256, pattern=r'^/[A-Za-z0-9_/-]+$')
    host: str = Field(min_length=1, max_length=253)
    rtsp_port: int = Field(default=554, ge=1, le=65535)
    management_port: int = Field(default=443, ge=1, le=65535)
    management_scheme: Literal["http", "https"] = "https"
    timezone: str = "Asia/Shanghai"
    playback_mode: Literal['disabled', 'hikvision', 'custom'] = 'disabled'
    playback_path: str = Field(default='', max_length=1024)
    username: str = Field(min_length=1, max_length=128)
    password: SecretStr | None = None
    enabled: bool = True
    channels: list[Channel] = Field(default_factory=list, max_length=256)

    @field_validator("host")
    @classmethod
    def valid_host(cls, value):
        try:
            ipaddress.ip_address(value)
            return value
        except ValueError:
            if not all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", part)
                       for part in value.split(".")):
                raise ValueError("只填写 IP 或主机名，不包含协议、端口、路径或账号")
            return value.lower()

    @field_validator('playback_path')
    @classmethod
    def valid_playback_path(cls, value):
        return validate_template(value)

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("请选择有效的 IANA 时区")
        return value

    @field_validator("channels")
    @classmethod
    def unique_channels(cls, value):
        if len({item.number for item in value}) != len(value):
            raise ValueError("通道编号不能重复")
        return value

    @field_validator("password")
    @classmethod
    def bounded_password(cls, value):
        if value is not None and len(value.get_secret_value()) > 1024:
            raise ValueError("密码过长")
        return value


def visible_query(db, user):
    query = db.query(RecorderConfig)
    if user.get("role") != "admin":
        query = query.filter(RecorderConfig.owner_id == str(user["id"]))
    return query


def find(db, user, recorder_id):
    row = visible_query(db, user).filter(RecorderConfig.id == recorder_id).first()
    if row is None:
        raise HTTPException(404, "录像机配置不存在")
    return row


def snapshot(row):
    connection = (row.config_json or {}).get('connection_test') or {}
    return {**row.config_json, 'playback_mode':playback_mode(row.config_json), "id": row.id, "password_configured": bool(row.password_ciphertext),
            "connection_status": connection.get('status', 'not_tested'), "recording_storage": "nvr",
            "capabilities": {"live": False, "playback": False, "channel_discovery": connection.get("status") == "connected" and connection.get('protocol') != 'rtsp'},
            "adapter_status": row.config_json.get('connection_protocol', 'rtsp')}


def apply_config(row, body):
    password = body.password.get_secret_value() if body.password is not None else ""
    if not password and not row.password_ciphertext:
        raise HTTPException(422, "首次配置必须填写密码")
    if password:
        try:
            row.password_ciphertext = encrypt_device_credential(password)
        except DeviceCredentialError:
            raise HTTPException(503, "服务器凭据加密尚未正确配置") from None
    value = body.model_dump(exclude={"password"})
    previous = {c['number']: c.get('algorithms', []) for c in (row.config_json or {}).get('channels', [])}
    for channel in value['channels']:
        channel['algorithms'] = previous.get(channel['number'], [])
    row.config_json = value


@router.get("")
def list_recorders(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                   db=Depends(get_db), user=Depends(access)):
    query = visible_query(db, user)
    return {"total": query.count(), "items": [snapshot(row) for row in query.order_by(
        RecorderConfig.created_at.desc(), RecorderConfig.id).offset((page - 1) * page_size).limit(page_size).all()]}


@router.post("", status_code=201)
def create_recorder(body: RecorderInput, db=Depends(get_db), user=Depends(access)):
    row = RecorderConfig(owner_id=str(user["id"]))
    apply_config(row, body)
    db.add(row)
    db.commit()
    return snapshot(row)


@router.get("/{recorder_id}")
def get_recorder(recorder_id: str, db=Depends(get_db), user=Depends(access)):
    return snapshot(find(db, user, recorder_id))


@router.get("/{recorder_id}/channel-inventory")
def channel_inventory(recorder_id: str, db=Depends(get_db), user=Depends(access)):
    row = find(db, user, recorder_id)
    if not row.config_json.get('enabled'):
        raise HTTPException(422, '录像机配置已停用')
    if not probe_slots.acquire(blocking=False):
        raise HTTPException(429, '通道读取繁忙，请稍后重试')
    key = f'inventory:{recorder_id}'
    with probe_lock:
        if key in active_probes:
            probe_slots.release()
            raise HTTPException(429, '该录像机正在读取通道配置')
        active_probes.add(key)
    try:
        try:
            password = decrypt_device_credential(row.password_ciphertext)
        except DeviceCredentialError:
            raise HTTPException(503, '录像机密码无法解密，请检查加密配置') from None
        try:
            channels = hikvision_channel_inventory(copy.deepcopy(row.config_json), password)
        except ProbeFailure as exc:
            raise HTTPException(422, exc.message) from None
        return {'checked_at': datetime.now(timezone.utc).isoformat(), 'channels': channels}
    finally:
        with probe_lock:
            active_probes.discard(key)
        probe_slots.release()


@router.put("/{recorder_id}")
def update_recorder(recorder_id: str, body: RecorderInput, db=Depends(get_db), user=Depends(access)):
    row = find(db, user, recorder_id)
    ensure_stopped(db, row.owner_id)
    apply_config(row, body)
    db.commit()
    return snapshot(row)


@router.delete("/{recorder_id}", status_code=204)
def delete_recorder(recorder_id: str, db=Depends(get_db), user=Depends(access)):
    row = find(db, user, recorder_id)
    ensure_stopped(db, row.owner_id)
    db.delete(row)
    db.commit()


def ensure_stopped(db, owner_id):
    from continuous_monitoring import admission_lock
    from database import ContinuousMonitor
    admission_lock(db)
    task = db.query(ContinuousMonitor).filter(ContinuousMonitor.owner_id == str(owner_id),
        ((ContinuousMonitor.desired_state == 'running') | (ContinuousMonitor.status == 'stopping'))).first()
    if task:
        raise HTTPException(409, '请先停止监控任务，再修改或删除录像机配置')


@router.post("/{recorder_id}/connect")
def test_connection(recorder_id: str, db=Depends(get_db), user=Depends(access)):
    row = find(db, user, recorder_id)
    original = copy.deepcopy(row.config_json)
    original_password = row.password_ciphertext
    checked_at = (original.get('connection_test') or {}).get('checked_at')
    if checked_at:
        try:
            if (datetime.now(timezone.utc) - datetime.fromisoformat(checked_at)).total_seconds() < 10:
                raise HTTPException(429, '请等待 10 秒后再次测试，避免设备频繁认证')
        except (TypeError, ValueError):
            pass
    if not probe_slots.acquire(blocking=False):
        raise HTTPException(429, '连接测试繁忙，请稍后重试')
    with probe_lock:
        if recorder_id in active_probes:
            probe_slots.release()
            raise HTTPException(429, '该录像机正在测试连接')
        active_probes.add(recorder_id)
    try:
        try:
            password = decrypt_device_credential(original_password)
        except DeviceCredentialError:
            raise HTTPException(503, '录像机密码无法解密，请检查加密配置') from None
        db.rollback()
        result = connect_recorder(original, password)
        row = visible_query(db, user).filter(RecorderConfig.id == recorder_id).with_for_update().first()
        if row is None:
            raise HTTPException(404, '录像机配置不存在')
        if row.config_json != original or row.password_ciphertext != original_password:
            raise HTTPException(409, '配置已改变，请重新测试连接')
        row.config_json = {**original, 'connection_test': result}
        db.commit()
        return result
    finally:
        with probe_lock:
            active_probes.discard(recorder_id)
        probe_slots.release()

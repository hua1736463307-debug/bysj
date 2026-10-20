"""Authenticated on-demand RTSP paths for the live WebRTC preview."""
import hashlib
import json
import os
import re
import shlex
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from fastapi import APIRouter, Depends, HTTPException
from jose import JWTError, jwt
from pydantic import BaseModel

from auth_utils import ALGORITHM, SECRET_KEY, require_any_menu
from database import RecorderConfig, get_db
from password_crypto import decrypt_password
from recorder_connection import validated_host

router = APIRouter(prefix="/monitoring", tags=["monitoring-webrtc"])
live_access = require_any_menu("/monitoring/live")
API_URL = os.getenv("MEDIAMTX_API_URL", "http://mediamtx:9997").rstrip("/")
TOKEN_PURPOSE = "monitoring-webrtc-read"
API_PURPOSE = "monitoring-webrtc-api"
_codec_cache = {}
_codec_lock = threading.Lock()


class MediaAuth(BaseModel):
    token: str = ""
    action: str = ""
    path: str = ""
    protocol: str = ""
    ip: str = ""


def media_path(owner_id, recorder_id, channel_number):
    digest = hashlib.sha256(f"{owner_id}:{recorder_id}:{channel_number}".encode()).hexdigest()[:32]
    return f"camera_{digest}"


def source_codec(source):
    now = time.monotonic()
    with _codec_lock:
        cached = _codec_cache.get(source)
        if cached and now - cached[0] < 300:
            return cached[1]
    result = subprocess.run([
        "ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-rw_timeout", "5000000",
        "-select_streams", "v:0", "-show_entries", "stream=codec_name",
        "-of", "default=nokey=1:noprint_wrappers=1", source,
    ], capture_output=True, text=True, timeout=8)
    codec = result.stdout.strip().lower() if result.returncode == 0 else ""
    if codec not in {"h264", "hevc"}:
        raise ValueError("Unsupported or unavailable video codec")
    with _codec_lock:
        _codec_cache[source] = (now, codec)
    return codec


def configure_path(path, source, codec):
    """Keep credentials on the server; only this backend can reach the Control API."""
    if codec == "hevc":
        command = ("ffmpeg -re -nostdin -hide_banner -loglevel error -rtsp_transport tcp "
                   f"-i {shlex.quote(source)} -an -c:v libx264 -preset ultrafast "
                   "-tune zerolatency -profile:v baseline -pix_fmt yuv420p "
                   "-b:v 3M -maxrate 3M -bufsize 6M -bf 0 -g 25 "
                   "-f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/$MTX_PATH")
        settings = {"source": "publisher", "runOnDemand": command,
                    "runOnDemandStartTimeout": "20s", "runOnDemandCloseAfter": "2s",
                    "runOnDemandRestart": True}
    else:
        settings = {"source": source, "rtspTransport": "tcp",
                    "sourceOnDemand": True, "sourceOnDemandCloseAfter": "2s"}
    body = json.dumps(settings).encode()
    now = datetime.now(timezone.utc)
    api_token = jwt.encode({"purpose": API_PURPOSE, "iat": now,
                            "exp": now + timedelta(minutes=1)}, SECRET_KEY, algorithm=ALGORITHM)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_token}"}
    url = f"{API_URL}/v3/config/paths/add/{path}"
    try:
        with urlopen(Request(url, data=body, headers=headers, method="POST"), timeout=3):
            return
    except HTTPError as exc:
        if exc.code != 400:
            raise
    # Existing path: update the source after a recorder configuration change.
    with urlopen(Request(f"{API_URL}/v3/config/paths/patch/{path}", data=body,
                         headers=headers, method="PATCH"), timeout=3):
        pass


@router.post("/webrtc-session/{recorder_id}/{channel_number}")
def create_session(recorder_id: str, channel_number: str, db=Depends(get_db), user=Depends(live_access)):
    row = db.query(RecorderConfig).filter(RecorderConfig.id == recorder_id,
                                          RecorderConfig.owner_id == str(user["id"])).first()
    if row is None:
        raise HTTPException(404, "摄像头不存在")
    cfg = row.config_json or {}
    channel = next((c for c in cfg.get("channels", []) if str(c.get("number")) == channel_number), None)
    path = (channel or {}).get("rtsp_path") or ""
    if not cfg.get("enabled") or not channel or not channel.get("enabled") or not path.startswith("/") or "#" in path or any(c.isspace() for c in path):
        raise HTTPException(409, "摄像头未启用或 RTSP 地址无效")
    try:
        host = validated_host(cfg["host"])
        password = decrypt_password(row.password_ciphertext)
        source = (f"rtsp://{quote(cfg['username'], safe='')}:{quote(password, safe='')}@"
                  f"{host}:{cfg['rtsp_port']}{path}")
        name = media_path(user["id"], recorder_id, channel_number)
        codec = source_codec(source)
        configure_path(name, source, codec)
    except (HTTPError, URLError, TimeoutError, subprocess.TimeoutExpired, OSError, ValueError, KeyError):
        raise HTTPException(503, "视频网关暂时不可用") from None
    now = datetime.now(timezone.utc)
    token = jwt.encode({"purpose": TOKEN_PURPOSE, "path": name, "sub": str(user["id"]),
                        "iat": now, "exp": now + timedelta(minutes=15)}, SECRET_KEY, algorithm=ALGORITHM)
    return {"url": f"/webrtc/{name}/whep", "token": token}


@router.post("/webrtc-auth")
def authorize_media(body: MediaAuth):
    if (body.action in {"publish", "read"} and body.protocol == "rtsp" and
            body.ip in {"127.0.0.1", "::1"} and re.fullmatch(r"camera_[0-9a-f]{32}", body.path)):
        return {"ok": True}
    if not body.token:
        raise HTTPException(403, "Forbidden")
    try:
        claims = jwt.decode(body.token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(403, "Forbidden") from None
    if body.action == "api":
        if claims.get("purpose") != API_PURPOSE:
            raise HTTPException(403, "Forbidden")
        return {"ok": True}
    if body.protocol != "webrtc" or body.action != "read" or claims.get("purpose") != TOKEN_PURPOSE or claims.get("path") != body.path:
        raise HTTPException(403, "Forbidden")
    return {"ok": True}

"""Reuse grid decode/inference/evidence with a bounded, browser-independent sink."""
import asyncio
import copy
import logging
import time
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException
from database import SessionLocal
from monitoring_policy import resolve_source, public_event

logger = logging.getLogger(__name__)


class MonitorSink:
    preview_max_fps = 25

    def __init__(self, policies=None, behavior=None):
        self.cells = {}
        self.frames = {}
        self.last_view = 0
        self.policies = policies or []
        self.behavior = behavior or {}
        self.alarms = []
        self.generations = {}

    @property
    def no_preview(self):
        return time.monotonic() - self.last_view > 10

    def preview(self):
        self.last_view = time.monotonic()
        now = time.time()
        return {'cells': {position: copy.deepcopy(frame) for position, frame in self.frames.items()
                          if now - frame['received_at'] < 10}, 'alarms': copy.deepcopy(self.alarms)}

    async def send_json(self, payload):
        position = payload.get("grid_position")
        if not isinstance(position, int) or not 0 <= position < 9:
            return
        cell = self.cells.setdefault(position, {})
        run_id = self.generations.get(position, f'legacy-{position}')
        cell['run_id'] = run_id
        if payload.get('type') == 'alarm_event':
            event = payload.get('event') or {}
            events = self.policies[position] if position < len(self.policies) else []
            if event.get('event_type') in events and event.get('should_alert') and event.get('event_id'):
                self.alarms = ([{**public_event(event), 'position': position, 'run_id': run_id}] +
                               [e for e in self.alarms if e.get('event_id') != event['event_id']])[:50]
        if payload.get("type") == "monitor_preview_frame":
            if payload.get('run_id') != run_id or not payload.get('annotated_image'):
                return
            self.frames[position] = {
                'run_id': run_id, 'annotated_image': payload['annotated_image'],
                'frame_sequence': payload.get('frame_sequence'), 'received_at': time.time(),
                'timestamp': payload.get('timestamp'), 'capture_timestamp': payload.get('capture_timestamp'),
                'preview': {k: v for k, v in (payload.get('preview') or {}).items() if k in ('width','height','source_width','source_height')},
                'alarm_events': [],
            }
        elif payload.get("type") == "grid_cell_result":
            events = self.policies[position] if position < len(self.policies) else []
            if payload.get('annotated_image') and len(payload['annotated_image']) <= 700000:
                self.frames[position] = {
                    'run_id': run_id,
                    'annotated_image': payload['annotated_image'], 'frame_sequence': payload.get('frame_sequence'),
                    'received_at': time.time(), 'timestamp': payload.get('timestamp'),
                    'preview': {k: v for k, v in (payload.get('preview') or {}).items() if k in ('width','height','source_width','source_height')},
                    'alarm_events': [public_event(e) for e in payload.get('alarm_events', []) if e.get('event_type') in events][:20],
                }
            stats = payload.get("stats") or {}
            for key in ("frames_processed", "total_detections", "total_alarm_events", "current_fps"):
                value = stats.get(key)
                if isinstance(value, (int, float)):
                    cell[key] = value
            cell["status"] = "error" if payload.get("error") or payload.get("behavior_error") or payload.get("behavior_persistence_warning") else payload.get("stream_status", "waiting")
            cell["updated_at"] = datetime.now().isoformat()
        elif payload.get("type") == "grid_cell_status":
            cell["status"] = payload.get("status", "waiting")
        # Frames stay in bounded memory only; cells is the persisted statistics snapshot.


class GridMonitorAdapter:
    def __init__(self, api):
        self.api = api
        self.sinks = {}

    def validate(self, value, db, user):
        a = self.api
        sources = value.get("video_sources") or []
        if not isinstance(sources, list) or not 1 <= len(sources) <= 9:
            raise HTTPException(400, "请选择 1–9 路服务器视频源")
        recorder_mode = value.get('recorder_monitor') is True
        if recorder_mode:
            sources = [resolve_source(db, source, user['id']) for source in sources]
        published_paths = {p for source in sources for p in ((source.get('algorithm_bindings') or {}).get('detectors', []) + [(source.get('algorithm_bindings') or {}).get('pose')]) if p} if recorder_mode else set()
        normalized = []
        for source in sources:
            result = a.normalize_grid_source_config(source)
            if not result or result["type"] not in {"rtsp", "http"}:
                raise HTTPException(400, "持续监控仅支持有效的 RTSP/HTTP 视频流，请检查地址与视频源类型是否匹配")
            if recorder_mode and result["type"] != "rtsp":
                raise HTTPException(400, "摄像头告警任务仅支持 RTSP 视频流")
            if len(result["url"]) > 2048:
                raise HTTPException(400, "视频源地址过长")
            if result["url"] in {s["url"] for s in normalized}:
                raise HTTPException(400, "同一任务不能重复配置同一个视频源")
            normalized.append({"type": result["type"], "url": result["url"], "name": str(source.get("name") or "视频源")[:100],
                               **({k: source[k] for k in ('recorder_id','channel_number','events')} if recorder_mode else {})})

        def model(path):
            raw = str(path or "")
            if raw in published_paths:
                return raw  # Resolved from administrator-owned catalog IDs, never request paths.
            record = a._find_visible_model_record(db, raw, user)
            if record is not None:
                if str(record.source) not in {"pretrained", "trained"}:
                    raise HTTPException(400, "持续监控仅支持预训练或平台训练的模型")
                return str(a.resolve_managed_model_path(raw, record))
            # A known basename alone does not authorize an arbitrary external path.
            for root in (a.MODELS_DIR, a.PRETRAINED_MODELS_DIR):
                candidate = root / Path(raw).name
                if Path(raw).name.lower() in a.PRETRAINED_MODEL_NAME_SET and candidate.is_file():
                    if raw == candidate.name or Path(raw).resolve() == candidate.resolve():
                        return str(candidate.resolve())
            raise HTTPException(403, "模型不在当前用户可用模型列表中")

        behavior = value.get("shared_behavior_config") or {}
        if not isinstance(behavior, dict):
            raise HTTPException(400, "告警配置格式不正确")
        events = behavior.get("enabled_events") or []
        if recorder_mode:
            events = list(dict.fromkeys(event for source in sources for event in source['events']))
        if not isinstance(events, list) or not events or any(not isinstance(event, str) for event in events) or not set(events) <= {"phone_use", "missing_badge", "fall_down"}:
            raise HTTPException(400, "请选择有效告警类型")
        try:
            fps = int(value.get("max_fps", 8))
            confidence = float(value.get("shared_confidence", .25))
            iou = float(value.get("shared_iou_threshold", .7))
            behavior_fps = int(behavior.get("behavior_fps", 6))
            if not 1 <= fps <= 25 or not 0 <= confidence <= 1 or not 0 <= iou <= 1 or not 1 <= behavior_fps <= 10:
                raise ValueError()
        except (TypeError, ValueError, OverflowError):
            raise HTTPException(400, "帧率或检测阈值超出范围")
        shared_model = model(value.get("shared_model"))
        if recorder_mode:
            # Store references only. Credentials are resolved again inside the runtime.
            normalized = [{k: v for k, v in source.items() if k != 'url'} for source in normalized]
        return {
            'recorder_monitor': recorder_mode,
            "video_sources": normalized, "shared_model": shared_model,
            "max_fps": fps, "shared_confidence": confidence, "shared_iou_threshold": iou,
            "shared_behavior_config": a.normalize_behavior_config({
                **behavior,
                "enabled": True, "enabled_events": events, "behavior_fps": behavior_fps,
                **({'phone_classes': ['phone', 'cell phone', 'cellphone', 'mobilephone', 'mobile phone', 'smartphone', '手机']} if recorder_mode else {}),
                "pose_model_path": model(behavior.get("pose_model_path")),
                "object_model_path": model(behavior.get("object_model_path") or shared_model),
                "evidence": {"snapshot": True, "clip_pre_seconds": 2, "clip_post_seconds": 3, "retention_days": 90},
            }),
        }

    async def start(self, task_id, value, owner):
        a = self.api
        grid_id = "monitor_" + task_id
        sink = MonitorSink([[] for _ in range(9)], value['shared_behavior_config'])
        self.sinks[task_id] = sink
        a.grid_connections[grid_id] = sink
        a.grid_send_locks[grid_id] = asyncio.Lock()
        a.grid_stop_events[grid_id] = asyncio.Event()
        a.grid_sessions[grid_id] = {"is_active": True, "sessions": {}, "owner_id": owner["id"], "owner_username": owner["username"]}
        a.grid_stream_tasks[grid_id] = {}
        a.grid_inference_schedulers[grid_id] = a.GridBatchInferenceScheduler(
            lambda images, model_path, confidence, iou: a.process_realtime_batch(
                images, model_path, confidence, iou, render_annotations=False),
            max_batch_size=9, max_wait_ms=4,
        )
        await self.reconcile(task_id, value, owner)

    async def stop_cell(self, task_id, position):
        """Cooperative per-camera drain; never close the shared batch scheduler."""
        a = self.api
        grid_id = 'monitor_' + task_id
        sessions = a.grid_sessions[grid_id]['sessions']
        cell = sessions.get(position)
        if cell is None:
            return True
        cell['stop_requested'] = True
        session_id = cell['session_id']
        task = a.grid_stream_tasks[grid_id].get(session_id)
        if task and not task.done():
            return False
        if task:
            # Retrieve failures after completion without cancelling native inference.
            await asyncio.gather(task, return_exceptions=True)
        if not await asyncio.to_thread(a.cleanup_video_capture, session_id, .05):
            return False
        if not await a.stop_behavior_frame_scheduler(session_id, .05):
            return False
        await asyncio.to_thread(a.close_behavior_session_sync, session_id, time.time())
        a.realtime_frame_delivery.pop(session_id, None)
        a.realtime_stats.pop(session_id, None)
        a.behavior_session_contexts.pop(session_id, None)
        a.grid_stream_tasks[grid_id].pop(session_id, None)
        sessions.pop(position, None)
        return True

    async def reconcile(self, task_id, value, owner):
        a = self.api
        grid_id = 'monitor_' + task_id
        sink = self.sinks[task_id]
        sessions = a.grid_sessions[grid_id]['sessions']
        for position, reference in enumerate(value['video_sources']):
            run_id = reference.get('run_id', f'legacy-{position}')
            wanted = reference.get('desired_state', 'running') == 'running'
            cell = sessions.get(position)
            stream = a.grid_stream_tasks[grid_id].get(cell['session_id']) if cell else None
            if cell and (not wanted or cell['run_id'] != run_id or cell.get('stop_requested') or not stream or stream.done()):
                try:
                    stopped = await self.stop_cell(task_id, position)
                except Exception as exc:
                    logger.warning('Camera cleanup deferred: position=%s error_type=%s', position, type(exc).__name__)
                    stopped = False
                if not stopped:
                    sink.frames.pop(position, None)
                    sink.cells[position] = {'run_id': cell['run_id'], 'status': 'stopping'}
                    continue
                cell = None
            if cell:
                continue
            sink.frames.pop(position, None)
            sink.alarms = [event for event in sink.alarms if event['position'] != position]
            sink.generations[position] = run_id
            sink.policies[position] = []
            sink.cells[position] = {'run_id': run_id, 'status': 'starting' if wanted else 'stopped'}
            if not wanted:
                continue
            # A bad/offline source must not tear down other cameras.
            try:
                if value.get('recorder_monitor'):
                    source = await asyncio.to_thread(self.resolve_reference, reference, owner['id'])
                else:
                    source = reference
                self.start_cell(task_id, value, owner, position, source, run_id)
            except Exception as exc:
                logger.warning('Camera start deferred: position=%s error_type=%s', position, type(exc).__name__)
                sink.cells[position] = {'run_id': run_id, 'status': 'error', 'reason': '摄像头启动失败，请检查通道配置与服务日志'}

    @staticmethod
    def resolve_reference(reference, owner_id):
        with SessionLocal() as db:
            return resolve_source(db, reference, owner_id)

    def start_cell(self, task_id, value, owner, position, source, run_id):
        a = self.api
        grid_id = 'monitor_' + task_id
        sink = self.sinks[task_id]
        events = source.get('events', value['shared_behavior_config']['enabled_events'])
        bindings = source.get('algorithm_bindings') or {}
        detectors = list(dict.fromkeys(p or value['shared_model'] for p in bindings.get('detectors', []))) or [value['shared_model']]
        model_ids = list(dict.fromkeys(bindings.get('model_ids', [])))
        if len(model_ids) != len(detectors) and value.get('recorder_monitor'):
            raise RuntimeError('所选算法模型未加载，请检查推理服务配置')
        settings = bindings.get('runtime') or {}
        sink.policies[position] = events
        behavior = copy.deepcopy(value['shared_behavior_config'])
        behavior['enabled_events'] = events
        behavior['independent_snapshot_alerts'] = bool(value.get('recorder_monitor'))
        behavior['require_face_for_missing_badge'] = bool(value.get('recorder_monitor'))
        if settings: behavior['behavior_fps'] = settings['behavior_fps']
        if bindings.get('pose'): behavior['pose_model_path'] = bindings['pose']
        if any(bindings.get('detectors', [])): behavior['object_model_path'] = detectors[0]
        behavior['pose_model_path'] = bindings.get('pose') or behavior.get('pose_model_path')
        behavior['pose_model_id'] = bindings.get('pose_model_id')
        behavior['scene_model_ids'] = model_ids
        behavior['monitoring_models'] = model_ids
        behavior['object_model_id'] = model_ids[0] if model_ids else None
        behavior['inference_revision'] = bindings.get('revision', 0)
        session_id = f"{grid_id}_cell_{position}"
        config = a.RealtimeConfig(
            model_path=detectors[0], confidence=settings.get("confidence", value["shared_confidence"]),
            iou_threshold=settings.get("iou_threshold", value["shared_iou_threshold"]), max_fps=settings.get("detection_fps", value["max_fps"]), frame_skip=1,
            source_type=source["type"], source_url=source["url"], loop_video=True,
            client_side_overlay=False, behavior_config=behavior,
        )
        a.grid_sessions[grid_id]["sessions"][position] = {"session_id": session_id, "config": config, 'run_id': run_id}
        started_at = datetime.now().isoformat()
        a.realtime_stats[session_id] = {
            "session_id": session_id, "frames_processed": 0, "total_detections": 0,
            "inference_times": [], "current_fps": 0, "start_time": started_at,
            "last_update": started_at, "is_active": True,
        }
        a.behavior_session_contexts[session_id] = {
            "owner_id": owner["id"], "owner_username": owner["username"], "mode": "grid",
            "source_id": (f"recorder:{source['recorder_id']}:channel:{source['channel_number']}" if value.get('recorder_monitor') else source['url']), "session_id": session_id,
            "grid_id": grid_id, "grid_position": str(position),
            **({'monitoring_algorithms': events} if value.get('recorder_monitor') else {}),
            **({'monitoring_models': model_ids} if value.get('recorder_monitor') else {}),
            **({'face_required_algorithms': list(bindings.get('face_required_algorithms', []))} if value.get('recorder_monitor') else {}),
            **({'inference_revision': bindings.get('revision'), 'alarm_flows': copy.deepcopy(bindings.get('flows', [])), 'run_id': run_id} if value.get('recorder_monitor') else {}),
        }
        a.configure_realtime_frame_delivery(session_id, supports_ack=False)
        a.grid_stream_tasks[grid_id][session_id] = asyncio.create_task(a.process_grid_stream(
            grid_id, sink, session_id, config, position))

    def finished(self, task_id):
        return task_id not in self.sinks

    def snapshot(self, task_id):
        sink = self.sinks.get(task_id)
        return {"cells": copy.deepcopy(sink.cells) if sink else {}}

    def preview(self, task_id):
        sink = self.sinks.get(task_id)
        return sink.preview() if sink else {'cells': {}}

    async def stop(self, task_id):
        a = self.api
        grid_id = "monitor_" + task_id
        if not await a.stop_grid_tasks(grid_id):
            return False
        for cell in a.grid_sessions.get(grid_id, {}).get("sessions", {}).values():
            session_id = cell["session_id"]
            a.realtime_stats.pop(session_id, None)
            a.behavior_session_contexts.pop(session_id, None)
        for mapping in (a.grid_sessions, a.grid_connections, a.grid_send_locks, a.grid_stop_events):
            mapping.pop(grid_id, None)
        self.sinks.pop(task_id, None)
        return True

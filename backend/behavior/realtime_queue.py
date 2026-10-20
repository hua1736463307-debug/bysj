"""Bounded latest-frame scheduling for asynchronous realtime behavior evaluation."""

import asyncio
import copy
import os
import time
from typing import Any, Callable, Dict, List, Optional


def resolve_behavior_queue_process_interval_ms() -> int:
    """Return the legacy Alarm Center refresh interval."""
    try:
        configured = int(os.getenv("REALTIME_BEHAVIOR_QUEUE_PROCESS_INTERVAL_MS", "2000"))
    except (TypeError, ValueError):
        configured = 2000
    return max(200, min(configured, 60_000))


def resolve_behavior_scheduler_interval_ms() -> int:
    """Return the bounded realtime behavior scheduling interval.

    The historical queue interval was also reused as a UI refresh interval and
    defaulted to two seconds. Keep that compatibility field while letting the
    behavior consumer run at a useful cadence by default.
    """
    raw = os.getenv("REALTIME_BEHAVIOR_SCHEDULER_INTERVAL_MS")
    if raw is None:
        raw = os.getenv("REALTIME_BEHAVIOR_QUEUE_PROCESS_INTERVAL_MS", "200")
    try:
        configured = int(raw)
    except (TypeError, ValueError):
        configured = 200
    return max(100, min(configured, 5_000))


def resolve_behavior_max_frame_age_ms() -> int:
    """Return the maximum capture-to-dequeue age accepted by the scheduler."""
    try:
        configured = int(os.getenv("REALTIME_BEHAVIOR_MAX_FRAME_AGE_MS", "2_000"))
    except (TypeError, ValueError):
        configured = 2_000
    return max(250, min(configured, 60_000))


def _capture_age_ms(timestamp: float, now: Optional[float] = None) -> Optional[float]:
    """Compute epoch timestamp age; ignore synthetic/video-relative timestamps."""
    try:
        value = float(timestamp)
    except (TypeError, ValueError):
        return None
    # Realtime callers use Unix seconds. Small values are test/video PTS values
    # and cannot be compared with wall-clock time safely.
    if value < 1_000_000_000:
        return None
    current = time.time() if now is None else float(now)
    return max(0.0, (current - value) * 1000.0)


class BehaviorFifoFrameScheduler:
    """Process the freshest bounded frame without blocking primary inference."""

    def __init__(
        self,
        session_key: str,
        processor: Callable[..., Dict[str, Any]],
        *,
        maxsize: int = 8,
        process_interval_ms: Optional[int] = None,
        max_frame_age_ms: Optional[int] = None,
    ):
        self.session_key = session_key
        self.processor = processor
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(maxsize)))
        interval = resolve_behavior_scheduler_interval_ms() if process_interval_ms is None else int(process_interval_ms)
        self.process_interval_ms = max(100, min(interval, 5_000))
        age_limit = resolve_behavior_max_frame_age_ms() if max_frame_age_ms is None else int(max_frame_age_ms)
        self.max_frame_age_ms = max(250, min(age_limit, 60_000))
        self._last_dequeue_started_at: Optional[float] = None
        self.completed: List[Dict[str, Any]] = []
        self.accepting = True
        self.dropped_frames = 0
        self.replaced_frames = 0
        self.stale_frames = 0
        self.enqueued_frames = 0
        self.processed_frames = 0
        self.failed_frames = 0
        self.last_active_events: List[Dict[str, Any]] = []
        self.last_stats: Dict[str, Any] = {}
        self.last_dequeue_age_ms: Optional[float] = None
        self.max_observed_age_ms = 0.0
        self.task: Optional[asyncio.Task] = asyncio.create_task(self._run())

    def submit(
        self,
        image: Any,
        result: Dict[str, Any],
        config: Dict[str, Any],
        context: Dict[str, Any],
        timestamp: float,
    ) -> Dict[str, Any]:
        """Submit without waiting; a full queue replaces its oldest frame."""
        if self.accepting:
            capture_age_ms = _capture_age_ms(timestamp)
            if capture_age_ms is not None and capture_age_ms > self.max_frame_age_ms:
                self.dropped_frames += 1
                self.stale_frames += 1
                return self.drain()
            payload = (
                image.copy(),
                {"detections": copy.deepcopy(result.get("detections") or [])},
                copy.deepcopy(config),
                dict(context),
                float(timestamp),
            )
            try:
                self.queue.put_nowait(payload)
                self.enqueued_frames += 1
            except asyncio.QueueFull:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self.queue.put_nowait(payload)
                    self.enqueued_frames += 1
                    self.replaced_frames += 1
                    self.dropped_frames += 1
                except asyncio.QueueFull:
                    self.dropped_frames += 1
        return self.drain()

    def _queue_stats(self) -> Dict[str, Any]:
        return {
            "queue_capacity": self.queue.maxsize,
            "queue_depth": self.queue.qsize(),
            "queue_dropped_frames": self.dropped_frames,
            "queue_replaced_frames": self.replaced_frames,
            "queue_stale_frames": self.stale_frames,
            "queue_enqueued_frames": self.enqueued_frames,
            "queue_processed_frames": self.processed_frames,
            "queue_failed_frames": self.failed_frames,
            "queue_process_interval_ms": self.process_interval_ms,
            "queue_max_frame_age_ms": self.max_frame_age_ms,
            "queue_last_dequeue_age_ms": self.last_dequeue_age_ms,
            "queue_max_observed_age_ms": round(self.max_observed_age_ms, 3),
            # Preserve the previous response field while exposing FIFO-specific names.
            "scheduler_dropped_frames": self.dropped_frames,
        }

    def drain(self) -> Dict[str, Any]:
        results = self.completed
        self.completed = []
        transitions: List[Dict[str, Any]] = []
        evidence_updates: List[Dict[str, Any]] = []
        temporal_shadow_events: List[Dict[str, Any]] = []
        temporal_score_snapshots: List[Dict[str, Any]] = []
        temporal_model: Dict[str, Any] = {}
        active_events: List[Dict[str, Any]] = copy.deepcopy(self.last_active_events)
        stats: Dict[str, Any] = {**self.last_stats, **self._queue_stats()}
        response: Dict[str, Any] = {
            "action_events": transitions,
            "active_events": active_events,
            "temporal_shadow_events": temporal_shadow_events,
            "temporal_score_snapshots": temporal_score_snapshots,
            "temporal_model": temporal_model,
            "behavior_stats": stats,
            "evidence_updates": evidence_updates,
        }
        for item in results:
            if item.get('_monitor_preview'):
                response['_monitor_preview'] = item['_monitor_preview']
            transitions.extend(item.get("action_events") or [])
            evidence_updates.extend(item.get("evidence_updates") or [])
            temporal_shadow_events.extend(item.get("temporal_shadow_events") or [])
            temporal_score_snapshots.extend(item.get("temporal_score_snapshots") or [])
            if item.get("temporal_model"):
                temporal_model = dict(item.get("temporal_model") or {})
            if "active_events" in item:
                active_events = item.get("active_events") or []
            stats.update(item.get("behavior_stats") or {})
            for key in ("error", "persistence_warning", "persistence_status"):
                if item.get(key):
                    response[key] = item[key]
        stats.update(self._queue_stats())
        self.last_active_events = copy.deepcopy(active_events)
        self.last_stats = dict(stats)
        response["active_events"] = active_events
        response["temporal_shadow_events"] = temporal_shadow_events
        response["temporal_score_snapshots"] = temporal_score_snapshots
        response["temporal_model"] = temporal_model
        return response

    async def _run(self) -> None:
        while True:
            if self._last_dequeue_started_at is not None:
                elapsed = time.monotonic() - self._last_dequeue_started_at
                remaining = (self.process_interval_ms / 1000) - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
            payload = await self.queue.get()
            dequeue_started_at = time.time()
            self._last_dequeue_started_at = time.monotonic()
            try:
                image, result, config, context, timestamp = payload
                dequeue_age_ms = _capture_age_ms(timestamp, dequeue_started_at)
                self.last_dequeue_age_ms = None if dequeue_age_ms is None else round(dequeue_age_ms, 3)
                if dequeue_age_ms is not None:
                    self.max_observed_age_ms = max(self.max_observed_age_ms, dequeue_age_ms)
                    if dequeue_age_ms > self.max_frame_age_ms:
                        self.dropped_frames += 1
                        self.stale_frames += 1
                        continue
                process_context = dict(context or {})
                process_context.update({
                    "capture_timestamp": float(timestamp),
                    "dequeue_timestamp": dequeue_started_at,
                    "dequeue_age_ms": self.last_dequeue_age_ms,
                    "sample_reason": "latest_frame",
                })
                processed = await asyncio.to_thread(
                    self.processor,
                    self.session_key,
                    image,
                    result,
                    config,
                    context=process_context,
                    timestamp=timestamp,
                )
                self.completed.append(processed)
                self.processed_frames += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.failed_frames += 1
                self.completed.append({
                    "action_events": [],
                    "active_events": copy.deepcopy(self.last_active_events),
                    "behavior_stats": self._queue_stats(),
                    "evidence_updates": [],
                    "error": f"behavior queue processing failed: {exc}",
                })
            finally:
                self.queue.task_done()

    async def close(self, timeout: float = 15.0) -> bool:
        """Stop accepting frames, drain the FIFO, then stop its idle consumer."""
        self.accepting = False
        if self.task is None:
            return True
        try:
            await asyncio.wait_for(self.queue.join(), timeout=max(0.1, timeout))
        except asyncio.TimeoutError:
            return False

        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        self.task = None
        return True

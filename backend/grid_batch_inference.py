"""Async micro-batching for independent nine-grid inference streams."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple


BatchProcessor = Callable[[List[Any], str, float, float], List[Dict[str, Any]]]
GroupKey = Tuple[str, float, float]


@dataclass
class _BatchRequest:
    image: Any
    future: asyncio.Future
    submitted_at: float


@dataclass
class _BatchGroup:
    requests: List[_BatchRequest] = field(default_factory=list)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


class GridBatchInferenceScheduler:
    """Collect one pending frame per stream into short, model-compatible batches."""

    def __init__(
        self,
        processor: BatchProcessor,
        *,
        max_batch_size: int = 9,
        max_wait_ms: float = 4.0,
    ) -> None:
        self.processor = processor
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_seconds = max(0.0, float(max_wait_ms)) / 1000.0
        self._groups: Dict[GroupKey, _BatchGroup] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self.batches_processed = 0
        self.frames_processed = 0
        self._init_inference_client()

    def _init_inference_client(self):
        from inference_service.client import inference_client
        self.inference_client = inference_client

    async def infer_monitoring(self, image, models, confidence, iou, *, tracking=False,
                               enabled_events=None, revision=None):
        self._init_inference_client()
        return await self.inference_client.infer_monitoring(image, models, confidence, iou,
                                                            tracking=tracking, enabled_events=enabled_events,
                                                            revision=revision)

    @property
    def revision(self):
        return self.inference_client.revision

    async def infer(
        self,
        image: Any,
        model_path: str,
        confidence: float,
        iou_threshold: float,
    ) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("grid batch scheduler is closed")

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        key = (str(model_path), round(float(confidence), 6), round(float(iou_threshold), 6))

        async with self._lock:
            if self._closed:
                raise RuntimeError("grid batch scheduler is closed")
            group = self._groups.get(key)
            if group is None:
                group = _BatchGroup()
                self._groups[key] = group
            group.requests.append(_BatchRequest(
                image=image,
                future=future,
                submitted_at=time.monotonic(),
            ))
            if group.task is None or group.task.done():
                group.task = asyncio.create_task(self._run_group(key, group))
            if len(group.requests) >= self.max_batch_size:
                group.wake.set()

        return await future

    async def _run_group(self, key: GroupKey, group: _BatchGroup) -> None:
        try:
            while True:
                if not group.wake.is_set() and self.max_wait_seconds > 0:
                    try:
                        await asyncio.wait_for(group.wake.wait(), timeout=self.max_wait_seconds)
                    except asyncio.TimeoutError:
                        pass
                group.wake.clear()

                async with self._lock:
                    if self._closed:
                        return
                    requests = group.requests[:self.max_batch_size]
                    del group.requests[:len(requests)]

                if not requests:
                    return

                images = [request.image for request in requests]
                dispatched_at = time.monotonic()
                try:
                    results = await asyncio.to_thread(
                        self.processor,
                        images,
                        key[0],
                        key[1],
                        key[2],
                    )
                    if len(results) != len(requests):
                        raise RuntimeError(
                            f"batch result count mismatch: expected {len(requests)}, got {len(results)}"
                        )
                    self.batches_processed += 1
                    self.frames_processed += len(requests)
                    for request, result in zip(requests, results):
                        if not request.future.done():
                            result.setdefault(
                                "batch_queue_wait_ms",
                                max(0.0, dispatched_at - request.submitted_at) * 1000.0,
                            )
                            request.future.set_result(result)
                except asyncio.CancelledError:
                    for request in requests:
                        if not request.future.done():
                            request.future.cancel()
                    raise
                except Exception as exc:
                    for request in requests:
                        if not request.future.done():
                            request.future.set_exception(exc)

                async with self._lock:
                    if not group.requests:
                        if self._groups.get(key) is group:
                            self._groups.pop(key, None)
                        return
                    if len(group.requests) >= self.max_batch_size:
                        group.wake.set()
        finally:
            async with self._lock:
                if self._groups.get(key) is group and not group.requests:
                    self._groups.pop(key, None)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            groups = list(self._groups.values())
            self._groups.clear()
            tasks = [group.task for group in groups if group.task is not None]
            requests = [request for group in groups for request in group.requests]
            for group in groups:
                group.requests = []
                group.wake.set()

        for request in requests:
            if not request.future.done():
                request.future.cancel()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def snapshot(self) -> Dict[str, int]:
        pending = sum(len(group.requests) for group in self._groups.values())
        return {
            "max_batch_size": self.max_batch_size,
            "pending_frames": pending,
            "batches_processed": self.batches_processed,
            "frames_processed": self.frames_processed,
        }

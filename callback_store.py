"""In-memory store for Suno webhook callbacks."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional


class CallbackStore:
    """Stores webhook payloads keyed by Suno task ID."""

    def __init__(self) -> None:
        self._callbacks: Dict[str, Dict[str, Any]] = {}
        self._events: Dict[str, asyncio.Event] = {}

    def record(self, task_id: str, payload: Dict[str, Any]) -> None:
        self._callbacks[task_id] = payload
        event = self._events.get(task_id)
        if event is not None:
            event.set()

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._callbacks.get(task_id)

    def create_waiter(self, task_id: str) -> asyncio.Event:
        event = asyncio.Event()
        self._events[task_id] = event
        if task_id in self._callbacks:
            event.set()
        return event

    def clear_waiter(self, task_id: str) -> None:
        self._events.pop(task_id, None)

    async def wait_for(self, task_id: str, timeout: float) -> Optional[Dict[str, Any]]:
        if task_id in self._callbacks:
            return self._callbacks[task_id]

        event = self.create_waiter(task_id)
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.clear_waiter(task_id)

        return self._callbacks.get(task_id)

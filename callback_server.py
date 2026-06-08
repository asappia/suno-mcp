"""Embedded HTTP server for Suno webhook callbacks."""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple
from urllib.parse import urlparse

from aiohttp import web

from callback_store import CallbackStore
from suno_response import extract_callback_type, extract_task_id_from_callback

logger = logging.getLogger(__name__)


def resolve_callback_url(user_provided: Optional[str] = None) -> str:
    """
    Resolve the callback URL sent to Suno API.

    Priority:
    1. Explicit tool parameter
    2. SUNO_CALLBACK_PUBLIC_URL (+ /api/suno/callback)
    3. Built-in local server URL
    """
    if user_provided:
        return user_provided

    public_url = os.getenv("SUNO_CALLBACK_PUBLIC_URL", "").strip()
    if public_url:
        return public_url.rstrip("/") + "/api/suno/callback"

    host = os.getenv("SUNO_CALLBACK_HOST", "127.0.0.1")
    port = int(os.getenv("SUNO_CALLBACK_PORT", "8090"))
    return f"http://{host}:{port}/api/suno/callback"


def validate_callback_url(callback_url: str) -> None:
    parsed = urlparse(callback_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            f"Invalid callback_url format: '{callback_url}'. "
            "Must be a valid URL like 'https://example.com/webhook'"
        )
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid callback_url scheme: '{parsed.scheme}'. "
            "Must use http:// or https://"
        )


async def start_callback_server(
    store: CallbackStore,
    host: str = "0.0.0.0",
    port: Optional[int] = None,
) -> Tuple[web.AppRunner, str]:
    """Start the webhook server and return the runner plus resolved callback URL."""
    listen_port = port or int(os.getenv("SUNO_CALLBACK_PORT", "8090"))

    async def handle_callback(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"status": "invalid_json"}, status=400)

        task_id = extract_task_id_from_callback(payload)
        if task_id:
            store.record(task_id, payload)
            callback_type = extract_callback_type(payload) or "unknown"
            logger.debug("Received Suno callback for task %s (%s)", task_id, callback_type)
        else:
            logger.debug("Received Suno callback without task ID")

        return web.json_response({"status": "received"})

    async def handle_health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    app = web.Application()
    app.router.add_post("/api/suno/callback", handle_callback)
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, listen_port)
    await site.start()

    callback_url = resolve_callback_url()
    logger.debug("Callback server listening on %s:%s -> %s", host, listen_port, callback_url)
    return runner, callback_url

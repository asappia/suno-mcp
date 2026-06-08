"""Helpers for discovering ngrok public URLs from the local ngrok API."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional


def wait_for_public_url(
    api_url: str = "http://127.0.0.1:4040/api/tunnels",
    timeout: float = 30.0,
    poll_interval: float = 1.0,
) -> Optional[str]:
    """Return the first HTTPS public URL exposed by ngrok, if available."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(api_url, timeout=2) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(poll_interval)
            continue

        for tunnel in payload.get("tunnels", []):
            public_url = tunnel.get("public_url", "")
            if isinstance(public_url, str) and public_url.startswith("https://"):
                return public_url.rstrip("/")

        time.sleep(poll_interval)

    return None


if __name__ == "__main__":
    url = wait_for_public_url()
    if url:
        print(url)

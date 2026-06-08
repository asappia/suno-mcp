"""Tests for ngrok tunnel URL discovery."""

import json
from unittest.mock import patch

from ngrok_tunnel import wait_for_public_url


def test_wait_for_public_url_parses_https_tunnel():
    payload = {
        "tunnels": [
            {"public_url": "http://example.ngrok-free.app"},
            {"public_url": "https://abc123.ngrok-free.app"},
        ]
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    with patch("ngrok_tunnel.urllib.request.urlopen", return_value=FakeResponse()):
        assert wait_for_public_url(timeout=1, poll_interval=0) == "https://abc123.ngrok-free.app"


if __name__ == "__main__":
    test_wait_for_public_url_parses_https_tunnel()
    print("ngrok_tunnel tests passed")

#!/bin/sh
set -e

# MCP uses stdio on stdout. Log startup messages to stderr only.
if [ -n "$NGROK_AUTHTOKEN" ] && [ -z "$SUNO_CALLBACK_PUBLIC_URL" ]; then
  CALLBACK_PORT="${SUNO_CALLBACK_PORT:-8090}"

  ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null 2>&1 || true
  ngrok http "$CALLBACK_PORT" --log=stdout --log-format=logfmt >/tmp/ngrok.log 2>&1 &

  PUBLIC_URL="$(python ngrok_tunnel.py || true)"
  if [ -n "$PUBLIC_URL" ]; then
    export SUNO_CALLBACK_PUBLIC_URL="$PUBLIC_URL"
    echo "ngrok tunnel ready: ${PUBLIC_URL}/api/suno/callback" >&2
  else
    echo "Warning: ngrok started but no public URL was detected; using polling fallback" >&2
  fi
elif [ -n "$NGROK_AUTHTOKEN" ] && [ -n "$SUNO_CALLBACK_PUBLIC_URL" ]; then
  echo "Using configured SUNO_CALLBACK_PUBLIC_URL (ngrok autostart skipped)" >&2
fi

exec python server.py

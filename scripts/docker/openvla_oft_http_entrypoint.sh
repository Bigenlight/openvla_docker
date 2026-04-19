#!/usr/bin/env bash
# Entrypoint for the openvla-oft-http container.
# Source is bind-mounted at /app, so `git pull/push` on the host is reflected
# immediately without rebuilding the image.

set -euo pipefail

cd /app
export PYTHONPATH="/app:${PYTHONPATH:-}"

exec python scripts/serve_openvla_oft_http.py \
    --port "${OPENVLA_OFT_HTTP_PORT:-8700}" ${OPENVLA_OFT_HTTP_ARGS:-}

#!/usr/bin/env bash
# Entrypoint for the openvla-http container.
#
# The openvla source tree is bind-mounted at /app at runtime (see
# openvla_http_compose.yml). This script simply execs the FastAPI server from
# the live source — no in-container patching required (unlike openpi).

set -euo pipefail

cd /app
export PYTHONPATH="/app:${PYTHONPATH:-}"

exec python scripts/serve_openvla_http.py \
    --port "${OPENVLA_HTTP_PORT:-8600}" ${OPENVLA_HTTP_ARGS:-}

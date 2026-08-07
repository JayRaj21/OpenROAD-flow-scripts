#!/usr/bin/env bash
# Launch the OpenROAD Flow web UI.
#
# Usage:
#   bash flow/run_ui.sh           -- run directly on the host (requires python3 deps)
#   bash flow/run_ui.sh --docker  -- run inside Docker (no host Python install needed)
#
# Open http://localhost:5000 after startup.

set -e
FLOW_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ "${1:-}" == "--docker" ]]; then
    IMAGE="orfs-ui"
    DOCKERFILE="$FLOW_DIR/Dockerfile.ui"

    if ! docker image inspect "$IMAGE" &>/dev/null; then
        echo "Building $IMAGE (first run may take a few minutes)..."
        docker build -t "$IMAGE" -f "$DOCKERFILE" "$FLOW_DIR"
    fi

    echo "Starting UI at http://localhost:5000 ..."
    exec docker run --rm -it \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v "$FLOW_DIR:/work" \
        -e ORFS_HOST_FLOW_DIR="$FLOW_DIR" \
        -p 5000:5000 \
        "$IMAGE"
else
    cd "$FLOW_DIR"
    exec python3 ui/app.py
fi

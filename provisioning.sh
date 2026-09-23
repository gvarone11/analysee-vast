#!/usr/bin/env bash
# =============================================================================
# vast/provisioning.sh
# GPU Inference Service Deployment Script for Vast.ai
#
# Purpose: Deploy YOLO GPU inference microservice on Vast.ai GPU instance
# Usage:   bash vast/provisioning.sh
#
# Environment Variables:
#   YOLO_MODELS      - Comma-separated list of model filenames to pre-download
#                      (default: "yolov8n.pt")
#                      Example: "yolov8n.pt,yolov8s.pt"
#   MODEL_CACHE_DIR  - Host dir where models are stored (default: /opt/yolo_models)
#   UVICORN_WORKERS  - Number of Uvicorn workers (default: 1)
#
# No AWS/S3 credentials required — models are standard Ultralytics releases
# downloaded directly from https://github.com/ultralytics/assets
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
DOCKER_IMAGE="gms-gpu-inference:latest"
CUDA_VERSION="12.1"
CONTAINER_PORT=8000
HOST_PORT=8000
CONTAINER_NAME="gpu_inference_service"

# Environment defaults
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/opt/yolo_models}"
YOLO_MODELS="${YOLO_MODELS:-yolov8n.pt}"
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"
RABBITMQ_URL="${RABBITMQ_URL:-amqp://guest:guest@rabbitmq:5672/%2F}"

# Logging
LOG_DIR="/var/log/gpu-inference"
LOG_FILE="$LOG_DIR/service.log"

# =============================================================================
# 1. Verify environment
# =============================================================================
echo "🔍 [provisioning] Verifying environment..."

if ! command -v docker &>/dev/null; then
    echo "❌ [provisioning] ERROR: Docker not found." >&2
    exit 1
fi
echo "✅ [provisioning] Docker: $(docker --version)"

if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi -L &>/dev/null 2>&1; then
    echo "❌ [provisioning] ERROR: No NVIDIA GPU detected (nvidia-smi failed)." >&2
    exit 1
fi
echo "✅ [provisioning] GPU: $(nvidia-smi -L | head -1)"

if ! docker run --rm --gpus all nvidia/cuda:${CUDA_VERSION}.0-runtime-ubuntu22.04 nvidia-smi &>/dev/null 2>&1; then
    echo "❌ [provisioning] ERROR: NVIDIA Docker runtime not available." >&2
    exit 1
fi
echo "✅ [provisioning] NVIDIA Docker runtime OK"

# =============================================================================
# 2. Create log directory
# =============================================================================
echo "📁 [provisioning] Creating log directory: $LOG_DIR"
mkdir -p "$LOG_DIR"
chmod 777 "$LOG_DIR"

# =============================================================================
# 3. Build Docker image
# =============================================================================
echo "🔨 [provisioning] Building Docker image: $DOCKER_IMAGE"
cd "$SCRIPT_DIR/gpu_inference"

if docker build \
    --tag "$DOCKER_IMAGE" \
    --build-arg CUDA_VERSION="$CUDA_VERSION" \
    --file Dockerfile \
    . >> "$LOG_FILE" 2>&1; then
    echo "✅ [provisioning] Docker image built: $DOCKER_IMAGE"
else
    echo "❌ [provisioning] ERROR: Docker build failed. See $LOG_FILE" >&2
    exit 1
fi

# =============================================================================
# 4. Pre-download YOLO models (Ultralytics auto-download, no S3 needed)
#
# Standard models are downloaded from:
#   https://github.com/ultralytics/assets/releases
#
# Models are saved to MODEL_CACHE_DIR on the HOST so they persist
# across container restarts (volume-mounted read-only inside the container).
#
# Override via:  YOLO_MODELS="yolov8n.pt,yolov8s.pt" bash provisioning.sh
# =============================================================================
echo "📥 [provisioning] Pre-downloading YOLO models to $MODEL_CACHE_DIR..."
echo "   Models: $YOLO_MODELS"

mkdir -p "$MODEL_CACHE_DIR"

python3 - <<PYEOF
import os, sys, shutil
from pathlib import Path

cache_dir = Path("$MODEL_CACHE_DIR")
cache_dir.mkdir(parents=True, exist_ok=True)
models_env = "$YOLO_MODELS"

for model_name in [m.strip() for m in models_env.split(",") if m.strip()]:
    dst = cache_dir / model_name
    if dst.exists():
        print(f"  ✅ {model_name} already cached ({dst.stat().st_size/1e6:.1f} MB)")
        continue

    print(f"  📥 Downloading {model_name} ...")
    try:
        from ultralytics import YOLO
        # YOLO() downloads the model to its own cache (~/.ultralytics/assets/)
        m = YOLO(model_name)

        # Locate the downloaded file and copy to MODEL_CACHE_DIR
        import glob
        search_dirs = [
            Path.home() / ".ultralytics" / "assets",
            Path.home() / "ultralytics",
            Path.cwd(),
        ]
        found = None
        for d in search_dirs:
            candidate = d / model_name
            if candidate.exists():
                found = candidate
                break
        # Broader glob fallback
        if not found:
            hits = glob.glob(str(Path.home() / "**" / model_name), recursive=True)
            if hits:
                found = Path(hits[0])

        if found:
            shutil.copy2(found, dst)
            print(f"  ✅ {model_name} saved to {dst} ({dst.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"  ⚠️  {model_name} downloaded to Ultralytics cache but not copied to {cache_dir}.")
            print(f"     It will be auto-downloaded by the container on first inference.")
    except Exception as e:
        print(f"  ❌ Failed to download {model_name}: {e}")
        sys.exit(1)
PYEOF

MODEL_COUNT=$(find "$MODEL_CACHE_DIR" \( -name "*.pt" -o -name "*.pth" \) 2>/dev/null | wc -l)
echo "✅ [provisioning] $MODEL_COUNT .pt/.pth file(s) in $MODEL_CACHE_DIR:"
find "$MODEL_CACHE_DIR" \( -name "*.pt" -o -name "*.pth" \) -exec ls -lh {} \; | awk '{print "   " $5, $9}'


# =============================================================================
# 5. Stop existing container (if running)
# =============================================================================
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🛑 [provisioning] Stopping existing container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    sleep 2
fi

# =============================================================================
# 6. Launch container with GPU
#
# The container is stateless: no AWS credentials, no DB connection.
# It reads YOLO models from MODEL_CACHE_DIR (volume-mounted, pre-populated above).
# Images are received via HTTP POST /infer and results returned via HTTP.
# =============================================================================
echo "🚀 [provisioning] Launching GPU inference container..."
echo "   Container: $CONTAINER_NAME"
echo "   Image: $DOCKER_IMAGE"
echo "   Port: $HOST_PORT:$CONTAINER_PORT"
echo "   GPU Count: $GPU_COUNT"
echo "   GPU Memory: $GPU_MEMORY"
echo "   Model Cache: $MODEL_CACHE_DIR (read-only in container)"
echo "   RabbitMQ URL: ${RABBITMQ_URL%:*}:***@${RABBITMQ_URL##*@}"

docker run \
    --name "$CONTAINER_NAME" \
    --detach \
    --restart unless-stopped \
    --gpus "all" \
    --memory 16g \
    --memory-swap 16g \
    --cpus 4 \
    --publish "$HOST_PORT:$CONTAINER_PORT" \
    --volume "$MODEL_CACHE_DIR:/tmp/yolo_models:ro" \
    --volume "$LOG_DIR:/var/log/gpu-inference:rw" \
    --env MODEL_CACHE_DIR="/tmp/yolo_models" \
    --env UVICORN_PORT="$CONTAINER_PORT" \
    --env UVICORN_WORKERS="$UVICORN_WORKERS" \
    --env RABBITMQ_URL="$RABBITMQ_URL" \
    --env CUDA_VISIBLE_DEVICES="0" \
    --env PYTHONUNBUFFERED=1 \
    --log-driver json-file \
    --log-opt max-size=100m \
    --log-opt max-file=5 \
    "$DOCKER_IMAGE" \
    >> "$LOG_FILE" 2>&1

if [[ $? -eq 0 ]]; then
    echo "✅ [provisioning] Container started: $CONTAINER_NAME"
else
    echo "❌ [provisioning] ERROR: Failed to start container. See $LOG_FILE for details." >&2
    docker logs "$CONTAINER_NAME" 2>&1 | tail -20
    exit 1
fi

# =============================================================================
# 7. Wait for service to be ready
# =============================================================================
echo "⏳ [provisioning] Waiting for GPU inference service to be ready..."

HEALTH_URL="http://localhost:$HOST_PORT/health"
TIMEOUT=120
ELAPSED=0

while [[ $ELAPSED -lt $TIMEOUT ]]; do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        echo "✅ [provisioning] GPU inference service is healthy!"
        
        # Get health details
        HEALTH_INFO=$(curl -s "$HEALTH_URL" | python3 -m json.tool 2>/dev/null || echo "")
        echo "📊 [provisioning] Service Info:"
        echo "$HEALTH_INFO" | sed 's/^/   /'
        
        break
    fi
    
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    echo "   Waiting... ($ELAPSED/$TIMEOUT seconds)"
done

if [[ $ELAPSED -ge $TIMEOUT ]]; then
    echo "❌ [provisioning] ERROR: Service did not respond within $TIMEOUT seconds." >&2
    echo "Container logs:"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -30
    exit 1
fi

# =============================================================================
# 8. Verify GPU access
# =============================================================================
echo ""
echo "🎮 [provisioning] Verifying GPU access in container..."

if docker exec "$CONTAINER_NAME" nvidia-smi -L &>/dev/null 2>&1; then
    GPU_NAME=$(docker exec "$CONTAINER_NAME" nvidia-smi -L 2>/dev/null | head -1 || echo "Unknown GPU")
    echo "✅ [provisioning] GPU accessible: $GPU_NAME"
else
    echo "⚠️  [provisioning] WARNING: Could not verify GPU access in container"
fi

# =============================================================================
# 9. Summary
# =============================================================================
echo ""
echo "╔═════════════════════════════════════════════════════════════════════╗"
echo "║                  GPU INFERENCE SERVICE DEPLOYED                    ║"
echo "╚═════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📍 Service URL:        http://localhost:$HOST_PORT"
echo "📍 Health Check:       http://localhost:$HOST_PORT/health"
echo "📍 Inference Endpoint: http://localhost:$HOST_PORT/infer"
echo "📍 Cache Stats:        http://localhost:$HOST_PORT/cache-stats"
echo ""
echo "📊 Container:     $CONTAINER_NAME"
echo "🖼️  Image:         $DOCKER_IMAGE"
echo "📂 Logs:          $LOG_FILE"
echo "💾 Model Cache:   $MODEL_CACHE_DIR"
echo ""
echo "🔧 Useful commands:"
echo "   View logs:        docker logs -f $CONTAINER_NAME"
echo "   Shell access:     docker exec -it $CONTAINER_NAME bash"
echo "   Stop service:     docker stop $CONTAINER_NAME"
echo "   Start service:    docker start $CONTAINER_NAME"
echo "   Remove container: docker rm $CONTAINER_NAME"
echo ""
echo "✅ [provisioning] Done. Service is ready for inference."
echo ""

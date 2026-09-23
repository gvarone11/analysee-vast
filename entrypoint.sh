#!/bin/bash
##############################################################################
# GPU Inference Service Entrypoint
#
# Runs both FastAPI service and RabbitMQ worker concurrently:
# 1. FastAPI (uvicorn) on port 8002 in background
# 2. RabbitMQ worker consumer in foreground
#
# If either process dies, the container exits.
# Handles signals (SIGTERM, SIGINT) for graceful shutdown.
##############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Trap to handle signals for graceful shutdown
trap_handler() {
    log_info "Received signal. Shutting down gracefully..."
    log_info "Shutdown complete"
    exit 0
}

# Set up signal handlers
trap trap_handler SIGTERM SIGINT

##############################################################################
# Main Startup
##############################################################################

log_info "========================================================================"
log_info "🚀 GPU Inference Service (FastAPI + RabbitMQ Worker)"
log_info "========================================================================"

# Validate environment
log_info "Environment configuration:"
log_info "  PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"
log_info "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log_info "  MODEL_CACHE_DIR=${MODEL_CACHE_DIR}"
log_info "  UVICORN_HOST=${UVICORN_HOST}"
log_info "  UVICORN_PORT=${UVICORN_PORT}"
log_info "  RABBITMQ_URL=${RABBITMQ_URL:-amqp://guest:guest@localhost:5672/%2F}"

# Check Python
PYTHON_VERSION=$(python --version 2>&1)
log_info "Python version: $PYTHON_VERSION"

# Check model cache directory
if [[ ! -d "$MODEL_CACHE_DIR" ]]; then
    log_warn "Model cache directory not found: $MODEL_CACHE_DIR"
    log_info "Creating model cache directory..."
    mkdir -p "$MODEL_CACHE_DIR"
fi

##############################################################################
# Start Service (uvicorn in foreground — i worker RabbitMQ partono come
# thread daemon dentro FastAPI via lifespan, stessa cache del modello)
##############################################################################

log_info "=========================================="
log_info "🚀 Starting GPU Inference Service (uvicorn + RabbitMQ worker threads)..."
log_info "=========================================="
log_info "  RABBITMQ_WORKER_THREADS=${RABBITMQ_WORKER_THREADS:-2}"
log_info ""

exec python -m uvicorn main:app \
    --host "${UVICORN_HOST}" \
    --port "${UVICORN_PORT}" \
    --workers 1 \
    --log-level info

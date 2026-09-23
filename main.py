# File: vast/main.py
"""
FastAPI application for GPU-accelerated YOLO inference.

Stateless microservice:
- Receives image bytes via HTTP POST /infer
- Runs YOLO inference (GPU/CPU)
- Returns detections + processed image (base64 JPEG)
- NO S3 access, NO DB access, NO side effects

Models are pre-loaded from MODEL_CACHE_DIR (populated by provisioning.sh).
The Django backend handles all S3 uploads, thumbnails and DB writes.

Endpoints:
- GET /health - Service health and device status
- POST /infer - Image inference
- GET /cache-stats - Model cache statistics
"""

import logging
import os
import json
import threading
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Tuple
import sys

from inference import run_inference, get_device_info
from model_cache import get_cache_stats, preload_all
from worker_watchdog import WorkerWatchdog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Module-level storage for lifespan objects (accessible to debug endpoints)
_worker_threads_store: List[Any] = []
_shutdown_event_store: threading.Event = None
_watchdog_store: Any = None

# ============================================================================
# Lifespan: pre-warm models + avvia worker RabbitMQ come thread interni
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: carica modelli + avvia thread consumer RabbitMQ. Shutdown: automatico (daemon threads)."""
    logger.info("🚀 GPU Inference Service startup...")

    # Pre-load tutti i modelli in VRAM una sola volta
    preload_all()

    # Importa qui per evitare import circolare (rabbitmq_worker importa inference)
    from rabbitmq_worker import RabbitMqInferenceWorker
    import pika

    num_workers = int(os.environ.get("RABBITMQ_WORKER_THREADS", "2"))
    logger.info(f"🐰 Starting {num_workers} RabbitMQ worker thread(s)...")

    # Crea una connessione RabbitMQ DEDICATA per ogni worker thread.
    # Ogni thread ottiene il proprio canale sulla propria connessione.
    # Questo permette a RabbitMQ di vedere consumer individuali —
    # quando un worker muore e viene riavviato, la GUI mostra
    # il consumer temporaneamente scomparire e ricomparire.
    # Tutti i thread condividono comunque i pesi YOLO in VRAM (1×);
    # ciascuno ha un Predictor ultralytics indipendente (thread-local).
    worker_threads: List[Tuple[threading.Thread, Any]] = []
    for i in range(num_workers):
        worker = RabbitMqInferenceWorker(
            shared_connection=None,  # Ogni worker crea la propria connessione
            skip_prewarm=(i > 0)  # Solo il primo thread pre-warma i modelli
        )
        t = threading.Thread(
            target=worker.start,
            daemon=True,  # muore insieme al processo principale
            name=f"rabbitmq-worker-{i}",
        )
        t.start()
        worker_threads.append((t, worker))
        logger.info(f"✅ RabbitMQ worker thread {i} started (own connection)")

    # Start watchdog to monitor and restart dead worker threads
    shutdown_event = threading.Event()
    watchdog = WorkerWatchdog(
        workers=worker_threads,
        shared_connection=None,  # Each worker has its own connection
        shutdown_event=shutdown_event,
    )
    watchdog_thread = threading.Thread(
        target=watchdog.run,
        daemon=True,
        name="worker-watchdog",
    )
    watchdog_thread.start()
    logger.info("✅ Worker watchdog thread started")

    # Store in module-level globals for debug endpoints access
    global _worker_threads_store, _shutdown_event_store, _watchdog_store
    _worker_threads_store = worker_threads
    _shutdown_event_store = shutdown_event
    _watchdog_store = watchdog

    yield

    # Cleanup: signal watchdog to stop, then stop all workers
    logger.info("🛑 Signaling shutdown to watchdog...")
    shutdown_event.set()
    watchdog_thread.join(timeout=10)
    logger.info("✅ Watchdog thread stopped")

    logger.info("🛑 Stopping all worker threads...")
    for i, (t, worker) in enumerate(worker_threads):
        worker.stop()
        t.join(timeout=5)
        if t.is_alive():
            logger.warning(f"⚠️  Worker {i} did not stop in time")
    logger.info("✅ All worker threads stopped")
    logger.info("✅ GPU Inference Service shutdown")


# FastAPI app
app = FastAPI(
    title="GMS GPU Inference Service",
    description="GPU-accelerated YOLO inference microservice on Vast.ai",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Pydantic Models
# ============================================================================

class ModelConfig(BaseModel):
    """Configuration for a single YOLO model"""
    model_id: str = Field(..., description="Unique model ID")
    model_url_s3_or_path: str = Field(..., description="S3 URI or local path to model")
    model_name: str = Field(..., description="Human-readable model name")
    detection_classes: List[str] = Field(default=[], description="Classes to detect (empty = all)")
    is_greyscale: bool = Field(default=False, description="Convert image to greyscale")
    confidence_threshold: float = Field(default=0.5, description="Minimum confidence score to accept a detection")
    input_size: int = Field(default=640, description="YOLO inference image size (imgsz). Must match the training size of the model (e.g. 1536).")
    class_confidence_overrides: Dict[str, float] = Field(default={}, description="Per-class confidence overrides (class_name -> threshold)")
    class_display_names: Dict[str, str] = Field(default={}, description="Per-class display names (class_name -> Italian label); injected by Django after response")
    model_blur: bool = Field(default=False, description="Whether this model uses blur; injected by Django after response")


class InferRequest(BaseModel):
    """Inference request (multipart form data)"""
    model_configs: List[ModelConfig] = Field(..., description="List of models to run")
    crop_polygon: Optional[List[List[float]]] = Field(default=None, description="Polygon coordinates to limit detection area")


class Detection(BaseModel):
    """Single detection result"""
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    conf: float = Field(..., description="Confidence score 0-1")
    class_name: str = Field(..., description="Detected class")
    model_id: str = Field(..., description="Model ID that made detection")


class InferResponse(BaseModel):
    """Inference response"""
    event_detected: bool = Field(..., description="Whether any detections were made")
    detections: List[Dict[str, Any]] = Field(..., description="List of detections")
    processed_image_b64: Optional[str] = Field(default=None, description="Base64-encoded image with drawn boxes")
    device: str = Field(..., description="Device used ('cuda' or 'cpu')")
    device_name: str = Field(..., description="Device name")
    models_run: List[Dict[str, Any]] = Field(..., description="Models that ran and result")
    error: Optional[str] = Field(default=None, description="Error message if inference failed")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """
    Health check endpoint.
    
    Returns current service status and device info.
    """
    try:
        device_name, device_type = get_device_info()
        cache_stats = get_cache_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "device": device_type,
                "device_name": device_name,
                "cache": cache_stats,
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": str(e)}
        )


@app.post("/infer", response_model=InferResponse)
async def infer(
    image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
    model_configs_json: str = Form(..., description="JSON array of model configs"),
    crop_polygon_json: Optional[str] = Form(default=None, description="JSON array of polygon coordinates"),
):
    """
    Run YOLO inference on image.
    
    Request (multipart form data):
    - image: Image file (JPEG/PNG)
    - model_configs_json: JSON string with array of model configs
    - crop_polygon_json: Optional JSON string with polygon coordinates
    
    Response: InferResponse with detections and processed image
    
    Example:
    ```bash
    curl -X POST http://localhost:8000/infer \
      -F "image=@test.jpg" \
      -F 'model_configs_json=[{
        "model_id": "model_1",
        "model_url_s3_or_path": "s3://bucket/yolov8n.pt",
        "model_name": "YOLOv8 Nano",
        "detection_classes": ["cow", "person"],
        "is_greyscale": false
      }]' \
      -F 'crop_polygon_json=[[0,0], [1920,0], [1920,1080], [0,1080]]'
    ```
    """
    try:
        logger.info(f"📨 Received inference request: image={image.filename}")
        
        # Parse model configs from JSON
        try:
            model_configs_raw = json.loads(model_configs_json)
            model_configs = [ModelConfig(**cfg) for cfg in model_configs_raw]
            logger.info(f"✅ Parsed {len(model_configs)} model configs")
        except Exception as e:
            logger.error(f"❌ Failed to parse model_configs_json: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid model_configs_json: {e}"}
            )
        
        # Parse polygon from JSON
        crop_polygon = None
        if crop_polygon_json:
            try:
                crop_polygon = json.loads(crop_polygon_json)
                logger.info(f"✅ Parsed crop polygon with {len(crop_polygon)} vertices")
            except Exception as e:
                logger.error(f"❌ Failed to parse crop_polygon_json: {e}")
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid crop_polygon_json: {e}"}
                )
        
        # Read image bytes
        image_bytes = await image.read()
        logger.info(f"✅ Read image: {len(image_bytes)} bytes from {image.filename}")
        
        # Run inference
        result = run_inference(
            image_bytes=image_bytes,
            model_configs=[cfg.dict() for cfg in model_configs],
            crop_polygon_coords=crop_polygon,
            model_cache_dir=os.environ.get("MODEL_CACHE_DIR", "/tmp/yolo_models"),
        )
        
        logger.info(
            f"✅ Inference result: event_detected={result['event_detected']}, "
            f"detections={len(result['detections'])}, device={result['device']}"
        )
        
        return InferResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Inference endpoint error: {e}", exc_info=True)
        
        # Return error response with safe fallback payload
        return JSONResponse(
            status_code=500,
            content={
                "event_detected": False,
                "detections": [],
                "processed_image_b64": None,
                "device": "unknown",
                "device_name": "unknown",
                "models_run": [],
                "error": str(e),
            }
        )


@app.get("/cache-stats")
async def cache_stats():
    """
    Get current model cache statistics.
    
    Returns cache size, TTL, and list of cached models.
    """
    try:
        stats = get_cache_stats()
        return JSONResponse(status_code=200, content=stats)
    except Exception as e:
        logger.error(f"Cache stats endpoint error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("🚀 GPU Inference Service Starting")
    
    try:
        device_name, device_type = get_device_info()
        logger.info(f"✅ Device: {device_name} ({device_type})")
        
        model_cache_dir = os.environ.get("MODEL_CACHE_DIR", "/tmp/yolo_models")
        
        if not os.path.isdir(model_cache_dir):
            logger.warning(f"⚠️ Model cache directory not found: {model_cache_dir}")
        else:
            import glob
            model_files = glob.glob(os.path.join(model_cache_dir, "*.pt")) + glob.glob(os.path.join(model_cache_dir, "*.pth"))
            logger.info(f"✅ Model cache directory: {model_cache_dir} ({len(model_files)} .pt/.pth files)")
            for mf in model_files:
                size_mb = os.path.getsize(mf) / 1e6
                logger.info(f"   - {os.path.basename(mf)} ({size_mb:.1f} MB)")
        
        logger.info("✅ GPU Inference Service Ready")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}", exc_info=True)
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 GPU Inference Service Shutting Down")


# Debug endpoints (temporary — for development only, remove before merge)
@app.get("/debug/worker-status")
async def debug_worker_status():
    """
    Return status of all RabbitMQ worker threads.
    Used by test_watchdog.py integration tests.
    """
    global _worker_threads_store, _shutdown_event_store
    alive = [t.is_alive() for (t, _w) in _worker_threads_store]
    return JSONResponse({
        "total_workers": len(alive),
        "workers_alive": alive,
    })


@app.post("/debug/kill-worker")
async def debug_kill_worker(index: int = 0):
    """
    Forcibly kill a worker thread by index.
    Used by test_watchdog.py integration tests.
    """
    global _worker_threads_store
    workers = _worker_threads_store
    if index < 0 or index >= len(workers):
        return JSONResponse(
            {"error": f"index out of range (0-{len(workers)-1})"},
            status_code=400
        )
    thread, worker = workers[index]
    if thread.is_alive():
        # Signal the worker to stop via stop_event (breaks the consumer loop)
        try:
            worker.stop()
        except Exception as e:
            logger.warning(f"Error stopping worker: {e}")
        return JSONResponse({
            "index": index,
            "status": "signalled_stop",
            "thread_name": thread.name,
            "was_alive": True,
        })
    else:
        return JSONResponse({
            "index": index,
            "status": "already_dead",
            "thread_name": thread.name,
            "was_alive": False,
        })


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("UVICORN_HOST", "0.0.0.0")
    port = int(os.environ.get("UVICORN_PORT", 8000))
    workers = int(os.environ.get("UVICORN_WORKERS", 1))

    logger.info(f"Starting server on {host}:{port} with {workers} worker(s)")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        access_log=True,
    )

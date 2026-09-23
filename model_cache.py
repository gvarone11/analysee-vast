# File: vast/gpu_inference/model_cache.py
"""
Thread-safe model cache with TTL expiration.

Supports both Ultralytics YOLO checkpoints (.pt) and RF-DETR
models (.pth). Prevents unbounded memory growth and model reloading overhead.
- Max 10 models in cache (~50-100 MB each)
- 1-hour TTL per model (auto-evict on expiration)
- Double-checked locking pattern for thread safety
"""

import threading
import logging
from pathlib import Path
from cachetools import TTLCache
from ultralytics import YOLO
from typing import Optional

# RF-DETR (Roboflow) — lightweight detection transformer
_RFDETR_AVAILABLE = False
_RF_CLASSES = []
try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase
    _RF_CLASSES = [RFDETRBase, RFDETRSmall, RFDETRNano]
    _RFDETR_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Thread-safe YOLO model cache with TTL and size limits
# Prevents unbounded memory growth: max 10 models, 1 hour TTL
# Each model is ~50-100 MB, so max cache size ~500-1000 MB
_model_cache = TTLCache(maxsize=10, ttl=3600)
_cache_lock = threading.Lock()


def _load_rfdetr_model(model_path: str, num_classes: int = 6):
    """
    Load an RF-DETR model from a checkpoint file.

    RF-DETR checkpoints are loaded via the `pretrain_weights` constructor
    parameter (the library handles state_dict loading internally).
    We try all available architectures (Base → Small → Nano) and return
    the first one that loads successfully (no exception).

    Args:
        model_path: Path to .pth checkpoint.
        num_classes: Number of classes in the checkpoint (default 6 for
                     calving model). Must match the checkpoint or class
                     embeddings will be misaligned → garbage predictions.
    """
    if not _RFDETR_AVAILABLE:
        raise ImportError("rfdetr is not installed. Run: pip install rfdetr")

    errors = []
    for RFCls in _RF_CLASSES:
        try:
            model = RFCls(pretrain_weights=model_path, num_classes=num_classes)
            # Fuse ops for ~30% speedup (like torch.compile for this model family)
            try:
                model.optimize_for_inference()
            except Exception:
                pass  # non-fatal; model still works without it
            logger.info(
                f"Loaded RF-DETR model as {RFCls.__name__} "
                f"({num_classes} classes): {model_path}"
            )
            return model
        except Exception as e:
            errors.append(f"{RFCls.__name__}: {e}")
            continue

    raise ValueError(
        f"Could not load {model_path} with any RF-DETR architecture. "
        f"Errors: {'; '.join(errors)}"
    )


def detect_model_type(model_path: str) -> str:
    ext = Path(model_path).suffix.lower()

    if ext == ".pt":
        return "yolo"

    if ext == ".pth":
        return "rfdetr"

    raise ValueError(
        f"Unsupported model format: {ext}. "
        f"Supported formats are .pt (Ultralytics YOLO) and .pth (RF-DETR)."
    )


def load_model(model_key: str, model_path: str):
    """
    Thread-safe model loader with caching.

    Supports:
    - .pt  -> Ultralytics YOLO instance
    - .pth -> RF-DETR model
    """
    if model_key in _model_cache:
        logger.debug(f"♻️ Using cached model: {model_key}")
        return _model_cache[model_key]

    with _cache_lock:
        if model_key in _model_cache:
            logger.debug(f"♻️ Using cached model: {model_key}")
            return _model_cache[model_key]

        model_type = detect_model_type(model_path)
        logger.info(f"Loading {model_type} model: {model_key} from {model_path}")

        if model_type == "yolo":
            model = YOLO(model_path)
        elif model_type == "rfdetr":
            model = _load_rfdetr_model(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        _model_cache[model_key] = model
        logger.info(
            f"Model loaded and cached: {model_key} | Cache size: {len(_model_cache)}/10"
        )
        return model


# ── Shared-weights thread-local model ────────────────────────────────────
# YOLO weights are loaded ONCE in VRAM (shared). Each thread gets its own
# YOLO wrapper with an independent Predictor so concurrent calls do not
# corrupt the mutable Detect-head state stored on the shared module.
#
# RF-DETR (.pth) models are loaded per-thread.
# ──────────────────────────────────────────────────────────────────────────
_base_yolo_cache: dict = {}  # {model_path: YOLO} — shared weights in VRAM
_base_yolo_lock = threading.Lock()
_thread_local = threading.local()  # per-thread wrapper storage


def _make_thread_yolo_wrapper(base_yolo: YOLO) -> YOLO:
    """Create a per-thread YOLO wrapper sharing the base model weights."""
    wrapper = base_yolo.__class__.__new__(base_yolo.__class__)
    wrapper.__dict__ = base_yolo.__dict__.copy()
    wrapper.predictor = None

    if hasattr(wrapper, "overrides") and wrapper.overrides is not None:
        wrapper.overrides = wrapper.overrides.copy()

    return wrapper


def load_model_thread_local(model_key: str, model_path: str):
    """
    Return a per-thread wrapper usable by inference.py.

    Wrapper shape:
    {
        "type": "yolo" | "rfdetr",
        "model": <YOLO wrapper> | <rfdetr model>,
    }
    """
    if not hasattr(_thread_local, "models"):
        _thread_local.models = {}

    if model_key in _thread_local.models:
        return _thread_local.models[model_key]

    model_type = detect_model_type(model_path)

    if model_type == "yolo":
        with _base_yolo_lock:
            if model_path not in _base_yolo_cache:
                logger.info(f"Loading base YOLO model (shared VRAM weights): {model_path}")
                _base_yolo_cache[model_path] = YOLO(model_path)
                logger.info(f"Base YOLO model loaded: {model_path}")

        base_model = _base_yolo_cache[model_path]
        model_obj = _make_thread_yolo_wrapper(base_model)

    elif model_type == "rfdetr":
        # RF-DETR models are stateless during forward() — safe to share.
        # RF-DETR objects are NOT torch.nn.Module, so no .eval() needed.
        logger.info(f"Loading RF-DETR model per-thread: {model_path}")
        model_obj = load_model(model_key, model_path)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    wrapper = {"type": model_type, "model": model_obj}
    _thread_local.models[model_key] = wrapper
    return wrapper


def clear_cache():
    """
    Clear all cached models. Use only for debugging or emergency cleanup.
    """
    with _cache_lock:
        _model_cache.clear()
        _base_yolo_cache.clear()
        logger.warning("🧹 Model cache cleared")


def get_cache_stats() -> dict:
    """
    Get cache statistics for monitoring.

    Returns:
        Dict with cache size, TTL info, etc.
    """
    with _cache_lock:
        return {
            "cache_size": len(_model_cache),
            "max_size": _model_cache.maxsize,
            "ttl": _model_cache.ttl,
            "cached_models": list(_model_cache.keys()),
            "base_yolo_models": list(_base_yolo_cache.keys()),
        }


def preload_all(model_dir: Optional[str] = None) -> None:
    """
    Pre-load all *.pt and *.pth models found in a directory into the cache.

    Scans model_dir for all supported model files and loads them into _model_cache.
    If a model fails to load, logs an error and continues with the next one.
    Useful for eliminating cold-start latency on worker startup.

    Args:
        model_dir: Directory to scan for model files.
                   If None, uses MODEL_CACHE_DIR env var or defaults to /tmp/yolo_models

    Returns:
        None (modifies _model_cache in-place)
    """
    import os

    if model_dir is None:
        model_dir = os.environ.get("MODEL_CACHE_DIR", "/tmp/yolo_models")

    if not os.path.isdir(model_dir):
        logger.warning(f"⚠️  preload_all: directory not found: {model_dir}")
        return

    model_files = [
        f
        for f in os.listdir(model_dir)
        if f.endswith(".pt") or f.endswith(".pth")
    ]
    if not model_files:
        logger.warning(f"⚠️  preload_all: no .pt or .pth files found in {model_dir}")
        return

    logger.info(f"🔥 Pre-loading {len(model_files)} model(s) from {model_dir}...")

    for fname in model_files:
        model_path = os.path.join(model_dir, fname)
        try:
            load_model(model_key=fname, model_path=model_path)
            logger.info(f"✅ Pre-loaded: {fname}")
        except Exception as e:
            logger.error(
                f"❌ preload_all: failed to load {fname}: {e} "
                f"(will be loaded lazily on first use)"
            )

    logger.info(f"🔥 Model pre-loading completed")

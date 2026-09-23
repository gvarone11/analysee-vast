# File: vast/inference.py
"""
Core YOLO inference orchestration logic.

Handles:
1. Model loading from local filesystem (MODEL_CACHE_DIR)
2. Image inference (GPU or CPU fallback)
3. Bounding box drawing on detected objects
4. Response encoding (base64 JPEG)

NOTE: This service is stateless.
- Receives image bytes via HTTP
- Returns detections + processed image via HTTP
- No S3 access, no DB access, no side effects
- Models must be pre-loaded in MODEL_CACHE_DIR by provisioning.sh
"""

import logging
import threading
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any, Optional
from model_cache import load_model, load_model_thread_local

# Serialise concurrent model.forward() calls on the shared nn.Module.
# ultralytics' Detect head stores mutable per-call state (self.shape,
# self.anchors, self.strides) directly on the shared module.  Two threads
# writing these simultaneously while reading them from a third thread causes
# non-deterministic NMS results (same image sometimes detects, sometimes not).
# The lock has negligible throughput cost because GPU kernels are already
# serialised on CUDA's null stream across Python threads.
_gpu_lock = threading.Lock()

from utils import (
    resolve_model_path,
    filter_detections_by_polygon,
    draw_bounding_boxes,
    draw_polygon_outline,
    image_to_base64,
    image_bytes_to_array,
    polygon_from_coordinates,
)

logger = logging.getLogger(__name__)


def get_device_info() -> Tuple[str, str]:
    """
    Get current device info (GPU/CPU).
    
    Returns:
        Tuple of (device_name, device_type) where:
        - device_name: Human-readable name (e.g., "NVIDIA A100")
        - device_type: "cuda" or "cpu"
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_type = "cuda"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU Available: {device_name} ({gpu_memory:.1f} GB)")
    else:
        device_name = "CPU (CUDA not available)"
        device_type = "cpu"
        logger.warning("⚠️ GPU not available, using CPU (expect slower inference)")
    
    return device_name, device_type

class TorchDetection:
    """Minimal detection wrapper for model outputs (YOLO / RF-DETR)."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


def normalize_output(model_type, output):
    if model_type == "yolo":
        return output[0].boxes

    if model_type == "rfdetr":
        return postprocess_rfdetr_output(output)

    raise ValueError(f"Unsupported model type: {model_type}")


def postprocess_rfdetr_output(result):
    """
    Convert RF-DETR predict() result to a list of TorchDetection objects.

    RF-DETR's predict() returns a Detections object with:
    - .xyxy: torch.Tensor [N, 4]
    - .confidence: torch.Tensor [N]
    - .class_id: torch.Tensor [N]
    - .is_empty(): bool
    """
    if result.is_empty():
        return []

    detections = []
    # RF-DETR returns numpy arrays; convert to torch tensors so downstream
    # code (line 353: detection.xyxy[0].cpu().numpy()) works uniformly.
    for xyxy, conf, cls in zip(result.xyxy, result.confidence, result.class_id):
        detections.append(TorchDetection(
            xyxy=torch.from_numpy(xyxy).unsqueeze(0),  # [4] -> [1, 4]
            conf=torch.tensor(float(conf)),
            cls=torch.tensor(int(cls)),
        ))
    return detections


def run_inference(
    image_bytes: bytes,
    model_configs: List[Dict[str, Any]],
    crop_polygon_coords: Optional[List[List[float]]] = None,
    model_cache_dir: str = "/tmp/yolo_models",
) -> Dict[str, Any]:
    """
    Run YOLO inference on image with specified models.
    
    Stateless: receives image bytes, returns results. No I/O side effects.
    
    Args:
        image_bytes: Raw image bytes (JPEG/PNG) received via HTTP
        model_configs: List of model configs with:
            - model_id: Unique model ID (used as cache key)
            - model_url_s3_or_path: Model filename or key (resolved against model_cache_dir)
            - model_name: Human-readable name
            - detection_classes: List of class names to detect (empty = all classes)
            - is_greyscale: Whether to convert to greyscale before inference
        crop_polygon_coords: Optional polygon [[x,y],...] to limit detection area
        model_cache_dir: Local directory where model .pt/.pth files are stored
        
    Returns:
        Dict with:
        - event_detected: bool - whether any detections were found
        - detections: List of detection dicts (bbox, conf, class, model info)
        - processed_image_b64: Base64-encoded JPEG with drawn bounding boxes
        - device: Device used ("cuda" or "cpu")
        - models_run: List of models run and their detection counts
        - error: None or error string (caller should handle gracefully)
    """
    step_start = None
    try:
        import time
        step_start = time.time()
        
        # Get device info
        device_name, device_type = get_device_info()
        
        # Convert image bytes to numpy array
        logger.info(f"📷 Decoding image from bytes")
        image_array = image_bytes_to_array(image_bytes)
        original_height, original_width = image_array.shape[:2]
        logger.info(f"✅ Image decoded: {original_width}x{original_height}")
        
        # Create polygon if provided
        crop_polygon = polygon_from_coordinates(crop_polygon_coords)
        if crop_polygon:
            logger.info(f"🔍 Crop polygon defined with {len(crop_polygon_coords)} vertices")
        
        # Run inference with all configured models
        all_detections = []
        models_run = []

        logger.info(f"MODEL_CONFIG_ORDER: {model_configs}")
        for config in model_configs:
            try:
                model_id = config.get("model_id")
                model_path = config.get("model_url_s3_or_path")
                model_name = config.get("model_name", "Unknown")
                detection_classes = config.get("detection_classes", [])
                is_greyscale = config.get("is_greyscale", False)
                confidence_threshold = float(config.get("confidence_threshold", 0.5))
                input_size = int(config.get("input_size", 640))
                # Per-class confidence overrides: {class_name: threshold}
                class_confidence_overrides = config.get("class_confidence_overrides", {})
                # Display names (Italian translations) sent by Django
                class_display_names = config.get("class_display_names", {})
                
                logger.info(f"🔄 Running model: {model_name} (ID: {model_id})")
                model_start = time.time()
                
                # Resolve model path from local cache (no S3 access)
                model_path = resolve_model_path(model_path, model_cache_dir)
                
                # Modello: pesi VRAM condivisi, Predictor per-thread (no lock)
                model_cache_key = f"{model_id}_{model_path}"
                # yolo_model = load_model_thread_local(model_cache_key, model_path)
                model_wrapper = load_model_thread_local(model_cache_key, model_path)

                # Prepare inference input
                inference_image = image_array
                if is_greyscale:
                    import cv2
                    logger.debug(f"Converting to greyscale for model {model_name}")
                    grey = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                    inference_image = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

                # Serialize GPU forward() — the shared Detect head stores mutable
                # per-call state (anchors, strides, shape) that causes non-deterministic
                # results when two threads call forward() concurrently.
                logger.info(f"🚀 Running inference on {device_type} (imgsz={input_size})")
                # with _gpu_lock:
                #     results = yolo_model(inference_image, imgsz=input_size, verbose=False)
                model_type = model_wrapper["type"]
                model = model_wrapper["model"]

                with _gpu_lock:
                    if model_type == "yolo":
                        results = model(inference_image, imgsz=input_size, verbose=False)

                    elif model_type == "rfdetr":
                        # RF-DETR expects PIL Image, not numpy array
                        pil_img = Image.fromarray(
                            inference_image[..., ::-1]  # BGR -> RGB
                        )
                        results = model.predict(
                            images=pil_img,
                            threshold=confidence_threshold,
                        )

                    else:
                        raise ValueError(f"Unsupported model type: {model_type}")

                # Check for detections: YOLO uses .boxes, rfdetr uses normalize_output
                if model_type == "yolo":
                    has_detections = bool(results and results[0].boxes and len(results[0].boxes) > 0)
                else:
                    has_detections = bool(results is not None and len(normalize_output(model_type, results)) > 0)

                if not has_detections:
                    logger.debug(f"No detections from model {model_name}")
                    models_run.append({"name": model_name, "detections": 0})
                    continue
                
                # Parse detections
                # detections = results[0].boxes
                detections = normalize_output(model_type, results)
                model_detections = []
                
                for detection in detections:
                    try:
                        conf = detection.conf.item()
                        
                        # Get class name first (needed for per-class threshold lookup)
                        class_id = int(detection.cls)
                        if model_type == "yolo":
                            class_name = model.names.get(class_id, f"unknown_{class_id}")
                        elif detection_classes:
                            if class_id < len(detection_classes):
                                class_name = detection_classes[class_id]
                            else:
                                class_name = f"class_{class_id}"
                        else:
                            class_name = f"class_{class_id}"
                        class_name_lower = class_name.lower()

                        # Apply confidence threshold: per-class override takes priority
                        effective_threshold = class_confidence_overrides.get(
                            class_name_lower,
                            class_confidence_overrides.get(class_name, confidence_threshold)
                        )
                        if conf < effective_threshold:
                            continue
                        
                        bbox = detection.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        # Clamp coordinates
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(original_width, x2), min(original_height, y2)
                        
                        if x1 >= x2 or y1 >= y2:
                            continue
                        
                        # Check if class is in detection_classes filter
                        if detection_classes and class_name_lower not in [c.lower() for c in detection_classes]:
                            continue
                        
                        detection_data = {
                            "bbox": [x1, y1, x2, y2],
                            "conf": float(conf),
                            "class": class_name_lower,
                            "raw_class": class_name,
                            "display_class": class_display_names.get(class_name_lower,
                                             class_display_names.get(class_name, class_name)),
                            "yolo_model_id": model_id,
                            "yolo_model_name": model_name,
                            "source": "yolo",
                            "is_cow_detection": len(detection_classes) > 0,
                        }
                        
                        model_detections.append(detection_data)
                    except Exception as e:
                        logger.warning(f"Error parsing detection: {e}")
                        continue
                
                # Filter by polygon if provided
                if crop_polygon:
                    model_detections = filter_detections_by_polygon(model_detections, crop_polygon)
                
                all_detections.extend(model_detections)
                model_time = time.time() - model_start
                logger.info(f"✅ Model {model_name} complete: {len(model_detections)} detections in {model_time:.2f}s")
                models_run.append({"name": model_name, "detections": len(model_detections)})
                
            except Exception as model_e:
                logger.error(f"❌ Error running model {model_name}: {model_e}", exc_info=True)
                models_run.append({"name": model_name, "error": str(model_e)})
                continue
        
        event_detected = len(all_detections) > 0
        total_time = time.time() - step_start
        
        logger.info(
            f"✅ Inference complete: {len(all_detections)} total detections, "
            f"event_detected={event_detected}, device={device_type}, time={total_time:.2f}s"
        )
        
        # processed_image_b64 is intentionally omitted: the backend draws bounding
        # boxes locally using the returned detections. Sending a ~1MB base64-encoded
        # image over the RabbitMQ RPC channel would dominate latency.
        return {
            "event_detected": event_detected,
            "detections": all_detections,
            "processed_image_b64": None,
            "device": device_type,
            "device_name": device_name,
            "models_run": models_run,
            "error": None,
        }
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}", exc_info=True)
        
        # Return safe no-detection payload on any error
        return {
            "event_detected": False,
            "detections": [],
            "processed_image_b64": None,
            "device": "unknown",
            "models_run": [],
            "error": str(e),
        }

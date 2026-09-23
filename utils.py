# File: vast/gpu_inference/utils.py
"""
Utility functions for GPU inference service:
- Polygon-based detection area masking
- Bounding box drawing
- Image encoding/decoding

NOTE: This service is stateless - no S3 or DB access.
Images are received via HTTP and results returned via HTTP.
Model files must be pre-loaded in MODEL_CACHE_DIR by provisioning.sh.
"""

import logging
import base64
import cv2
import numpy as np
import os
from io import BytesIO
from typing import List, Optional, Dict, Any
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

try:
    # pillow-avif-plugin registers the AVIF codec with PIL on import.
    # Camera images arriving from the backend may now be AVIF (the R2 upload
    # format); OpenCV has no AVIF decoder, so PIL is the fallback path.
    import pillow_avif  # noqa: F401
    _AVIF_PLUGIN_AVAILABLE = True
except ImportError:
    _AVIF_PLUGIN_AVAILABLE = False


def _decode_bytes_with_avif_fallback(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes to a BGR array.

    OpenCV handles JPEG/PNG (the historical formats and the GPU service's own
    output). AVIF bytes (OpenCV returns None) are decoded via PIL +
    pillow-avif-plugin. Raises ValueError when the bytes are undecodable.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    if image_bytes[:2] == b"\xff\xd8":
        # Definitively JPEG bytes that OpenCV could not decode: no fallback
        # will help, fail fast with the historical error.
        raise ValueError("Failed to decode image from bytes")

    if not _AVIF_PLUGIN_AVAILABLE:
        logger.error(
            "Non-JPEG image bytes failed OpenCV decode and pillow-avif-plugin "
            "is not installed (AVIF input unsupported). Add 'pillow-avif-plugin' "
            "to requirements."
        )
        raise ValueError("Failed to decode image from bytes")

    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as img:
            rgb = np.asarray(img.convert("RGB"))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        logger.debug(f"AVIF/PIL fallback decode succeeded: shape={bgr.shape}")
        return bgr
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"PIL fallback decode failed: {e}")
        raise ValueError("Failed to decode image from bytes")


def resolve_model_path(model_key: str, model_cache_dir: str) -> str:
    """
    Resolve a model key to a file path or model name.

    Resolution order:
    1. Absolute path (already exists on disk) → return as-is
    2. MODEL_CACHE_DIR/<filename> exists → return absolute path
    3. Fallback: return just the filename so Ultralytics auto-downloads
       it from the official release assets (yolov8n.pt, yolov8s.pt, …)

    Args:
        model_key: Model filename (e.g. "yolov8n.pt") or s3:// URI
        model_cache_dir: Local directory where pre-cached models live

    Returns:
        Absolute path if found locally, otherwise bare filename for
        Ultralytics auto-download.
    """
    # 1. Strip accidental s3:// prefix
    if model_key.startswith("s3://"):
        model_key = model_key.split("/", 3)[-1]

    # 2. Already absolute and exists (Unix path only — Windows paths sent to Linux container won't match)
    if os.path.isabs(model_key) and os.path.exists(model_key):
        logger.debug(f"✅ Model found at absolute path: {model_key}")
        return model_key

    # 3. Use only the filename — handle both Unix ('/') and Windows ('\\') separators.
    #    os.path.basename('C:\\Users\\...\\v1.2.0.pt') on Linux returns the whole string,
    #    so we split on both separators explicitly.
    filename = model_key.replace("\\", "/").split("/")[-1]

    # 4. Check MODEL_CACHE_DIR
    if model_cache_dir:
        local_path = os.path.join(model_cache_dir, filename)
        if os.path.exists(local_path):
            logger.debug(f"✅ Model found in cache: {local_path}")
            return local_path

    # 5. Fallback: let Ultralytics download the standard model automatically
    logger.info(
        f"📥 Model '{filename}' not in MODEL_CACHE_DIR='{model_cache_dir}'. "
        f"Ultralytics will auto-download it on first use."
    )
    return filename


def crop_to_polygon(image: np.ndarray, polygon: Optional[Polygon]) -> np.ndarray:
    """
    Mask image to only show detections within polygon area.
    
    Args:
        image: Input image (numpy array, BGR format)
        polygon: Shapely Polygon defining detection area (None = use full image)
        
    Returns:
        Masked image (same size, areas outside polygon are black)
    """
    if polygon is None or polygon.is_empty:
        return image
    
    try:
        # Create mask from polygon
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Get polygon exterior coordinates
        coords = np.array(polygon.exterior.coords[:-1], dtype=np.int32)
        
        # Draw filled polygon on mask
        cv2.fillPoly(mask, [coords], 255)
        
        # Apply mask to image
        masked = cv2.bitwise_and(image, image, mask=mask)
        logger.debug(f"✅ Applied polygon mask to image")
        return masked
    except Exception as e:
        logger.error(f"❌ Failed to crop to polygon: {e}", exc_info=True)
        return image


def filter_detections_by_polygon(
    detections: List[Dict[str, Any]],
    polygon: Optional[Polygon]
) -> List[Dict[str, Any]]:
    """
    Filter detections to only include those within polygon.
    
    Args:
        detections: List of detection dicts with 'bbox' key
        polygon: Shapely Polygon defining detection area
        
    Returns:
        Filtered detections list
    """
    if polygon is None or polygon.is_empty:
        return detections
    
    filtered = []
    for det in detections:
        try:
            bbox = det.get("bbox", [])
            if len(bbox) < 4:
                continue
            
            x1, y1, x2, y2 = bbox
            bbox_center = Point((x1 + x2) / 2, (y1 + y2) / 2)
            
            if polygon.contains(bbox_center):
                filtered.append(det)
        except Exception as e:
            logger.warning(f"Error checking detection against polygon: {e}")
            filtered.append(det)  # Include on error (safer)
    
    return filtered


def draw_polygon_outline(
    image: np.ndarray,
    polygon: Optional[Polygon],
    color: tuple = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """
    Draw the crop polygon outline on the image (green border).

    Args:
        image: Input image (numpy array, BGR format)
        polygon: Shapely Polygon defining detection area (None = no draw)
        color: Line color in BGR (default green)
        thickness: Line thickness in pixels

    Returns:
        Image with polygon outline drawn (copy).
    """
    if polygon is None or polygon.is_empty:
        logger.debug(f"🎨 Skipping polygon draw: polygon={'None' if polygon is None else 'empty'}")
        return image

    try:
        logger.debug(f"🎨 Drawing polygon outline with {len(polygon.exterior.coords)-1} vertices")
        draw_image = image.copy()
        coords = np.array(polygon.exterior.coords[:-1], dtype=np.int32)
        cv2.polylines(draw_image, [coords], isClosed=True, color=color, thickness=thickness)
        logger.debug(f"✅ Polygon outline drawn successfully")
        return draw_image
    except Exception as e:
        logger.warning(f"❌ Error drawing polygon outline: {e}")
        return image


def draw_bounding_boxes(
    image: np.ndarray,
    detections: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.
    
    Args:
        image: Input image (numpy array, BGR format)
        detections: List of detection dicts with bbox, class, conf, etc.
        
    Returns:
        Image with drawn bounding boxes
    """
    draw_image = image.copy()
    img_h, img_w = draw_image.shape[:2]
    
    for idx, det in enumerate(detections):
        try:
            bbox = det.get("bbox", [])
            if len(bbox) < 4:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            # Color based on detection type
            color = (0, 165, 255)  # Orange for all bounding boxes
            color_name = "ORANGE"
            
            logger.debug(f"  📦 Detection {idx}: {det.get('class', 'Unknown')} → {color_name}")
            
            # Draw rectangle
            thickness = 2
            cv2.rectangle(draw_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            model_name = det.get("yolo_model_name", "Model")
            display_class = det.get("display_class", det.get("class", "Unknown"))
            confidence = det.get("conf", 0.0)
            label = f"{model_name}: {display_class}: {confidence:.2f}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            label_y = y1 - 10 if y1 - lh - 10 > 0 else y1 + lh + 10
            label_x = x1
            
            cv2.rectangle(
                draw_image,
                (label_x, label_y - lh - baseline),
                (label_x + lw, label_y + baseline),
                (255, 255, 255),
                cv2.FILLED
            )
            cv2.putText(
                draw_image,
                label,
                (label_x, label_y),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
                cv2.LINE_AA
            )
        except Exception as e:
            logger.warning(f"Error drawing detection: {e}")
            continue
    
    return draw_image


def image_to_base64(image: np.ndarray) -> str:
    """
    Encode image to base64 JPEG string.
    
    Args:
        image: Input image (numpy array, BGR format)
        
    Returns:
        Base64-encoded JPEG string
    """
    try:
        # Encode to JPEG
        success, jpeg_bytes = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise ValueError("Failed to encode image to JPEG")
        
        # Convert to base64
        b64_string = base64.b64encode(jpeg_bytes).decode("utf-8")
        return b64_string
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}", exc_info=True)
        raise


def base64_to_image(b64_string: str) -> np.ndarray:
    """
    Decode base64 JPEG string to image.
    
    Args:
        b64_string: Base64-encoded JPEG string
        
    Returns:
        Image as numpy array (BGR format)
    """
    try:
        # Decode from base64
        jpeg_bytes = base64.b64decode(b64_string)
        
        # Decode from JPEG
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image from JPEG bytes")
        
        return image
    except Exception as e:
        logger.error(f"Failed to decode image from base64: {e}", exc_info=True)
        raise


def image_bytes_to_array(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to numpy array.
    
    Args:
        image_bytes: Raw image bytes (JPEG/PNG/AVIF/etc.)
        
    Returns:
        Image as numpy array (BGR format)
    """
    try:
        return _decode_bytes_with_avif_fallback(image_bytes)
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to convert image bytes to array: {e}", exc_info=True)
        raise


def array_to_image_bytes(image: np.ndarray, format: str = "jpg") -> bytes:
    """
    Convert image array to bytes.
    
    Args:
        image: Image as numpy array (BGR format)
        format: Output format ("jpg", "png", etc.)
        
    Returns:
        Image bytes
    """
    try:
        if format.lower() == "jpg":
            success, image_bytes = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif format.lower() == "png":
            success, image_bytes = cv2.imencode(".png", image)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if not success:
            raise ValueError(f"Failed to encode image to {format}")
        
        return image_bytes.tobytes()
    except Exception as e:
        logger.error(f"Failed to convert image array to bytes: {e}", exc_info=True)
        raise


def polygon_from_coordinates(coords: Optional[List[List[float]]]) -> Optional[Polygon]:
    """
    Create Shapely Polygon from coordinate list.
    
    Args:
        coords: List of [x, y] coordinate pairs (None = no polygon)
        
    Returns:
        Shapely Polygon or None
    """
    if not coords or len(coords) < 3:
        return None
    
    try:
        polygon = Polygon(coords)
        if not polygon.is_valid:
            logger.warning("Created polygon is not valid")
            return None
        return polygon
    except Exception as e:
        logger.error(f"Failed to create polygon: {e}")
        return None

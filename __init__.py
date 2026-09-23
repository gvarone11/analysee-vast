# GPU Inference Microservice
"""
FastAPI microservice for GPU-accelerated YOLO inference on Vast.ai.

This module provides a stateless HTTP service that:
1. Receives original JPEG images + model configurations
2. Runs YOLO inference on GPU (CUDA 12.1)
3. Returns detections + processed image with bounding boxes
4. Handles errors gracefully with fallback to CPU if needed

Key endpoints:
- GET /health - Service health status
- POST /infer - Image inference with model configs

Environment variables:
- MODEL_CACHE_DIR: Directory for cached YOLO models (default: /tmp/yolo_models)
- AWS_ACCESS_KEY_ID: AWS credentials for S3 model downloads
- AWS_SECRET_ACCESS_KEY: AWS credentials for S3 model downloads
- AWS_DEFAULT_REGION: AWS region (default: eu-west-1)
- GPU_MEMORY_FRACTION: GPU memory allocation (default: 0.8)

Performance characteristics:
- Thread-safe model cache with 1-hour TTL
- Max 10 concurrent models in memory (~500-1000 MB total)
- Model loading: ~2-5s (first time), <100ms (cached)
- Inference: 100-500ms per image depending on model and GPU
"""

__version__ = "1.0.0"
__author__ = "GMS GPU Team"

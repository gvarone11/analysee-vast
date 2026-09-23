"""
RabbitMQ AMQP Consumer for GPU Inference Jobs

This worker:
1. Connects to RabbitMQ broker
2. Listens to 'inference_requests' queue for incoming jobs
3. Decodes image from base64 and parses model configs
4. Runs YOLO inference using run_inference()
5. Publishes response back to reply_to queue with correlation_id
6. Implements fair dispatch (basic_qos prefetch_count=1)
7. Has reconnection logic with exponential backoff

Environment variables:
- RABBITMQ_URL: amqp://user:pass@host:port/vhost (default: amqp://guest:guest@localhost:5672/%2F)
- MODEL_CACHE_DIR: Directory with pre-loaded YOLO models (default: /tmp/yolo_models)

Run as:
    python rabbitmq_worker.py
    
Or in Docker as entrypoint alongside FastAPI server (using process manager).
"""

import pika
import os
import json
import base64
import logging
import time
import threading
from typing import Dict, Any, Optional

from inference import run_inference, get_device_info
from model_cache import preload_all

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RabbitMqInferenceWorker:
    """
    RabbitMQ AMQP consumer for GPU inference jobs.
    
    Implements RPC-over-AMQP pattern:
    - Receives requests on exclusive callback queues
    - Processes synchronously (one job per worker at a time)
    - Sends responses back on reply_to queue with correlation_id matching
    
    Can either:
    - Own its own connection (default, legacy mode)
    - Use a shared connection passed in (optimized for multiple threads)
    """
    
    def __init__(
        self,
        shared_connection: Optional[pika.BlockingConnection] = None,
        skip_prewarm: bool = False,
    ):
        """Initialize worker configuration from environment.
        
        Args:
            shared_connection: If provided, use this RabbitMQ connection instead of creating one.
                              Worker will NOT close this connection on shutdown.
            skip_prewarm: If True, skip model pre-warming (useful if another thread already did it).
        """
        self.connection = shared_connection  # Can be None (create own) or a pika connection
        self.channel = None
        self.owns_connection = shared_connection is None  # Only close if we own it
        self.skip_prewarm = skip_prewarm
        self._stop_event = threading.Event()  # For graceful shutdown / test kill
        
        # RabbitMQ connection details
        self.rabbitmq_url = os.environ.get(
            'RABBITMQ_URL',
            'amqp://guest:guest@localhost:5672/%2F'
        )
        
        # Queue names
        self.request_queue = 'inference_requests'
        
        # Model cache directory
        self.model_cache_dir = os.environ.get(
            'MODEL_CACHE_DIR',
            '/tmp/yolo_models'
        )
        
        # Retry configuration
        self.max_connect_retries = 3
        self.connect_retry_delay = 2
        self.max_reconnect_attempts = 10
        self.reconnect_base_delay = 5
        
        logger.info("🚀 RabbitMQ Inference Worker initialized")
        if shared_connection:
            logger.info("  Mode: shared connection (N threads, 1 connection)")
        else:
            logger.info("  Mode: own connection (legacy)")
        logger.info(f"  Model Cache Dir: {self.model_cache_dir}")
    
    def _create_channel(self, connection: "pika.BlockingConnection") -> None:
        """Crea un canale su una connessione esistente e dichiara la coda."""
        self.channel = connection.channel()
        self.channel.queue_declare(
            queue=self.request_queue,
            durable=True,
            auto_delete=False
        )
        self.channel.basic_qos(prefetch_count=1)

    def connect(self) -> None:
        """
        Connect to RabbitMQ with retry logic.

        Prima tenta la connessione condivisa (se disponibile e aperta).
        Se quella è chiusa (drop della connessione), crea una connessione propria
        in modo da sopravvivere ai reconnect senza bloccarsi.

        Raises:
            Exception: If connection fails after max_connect_retries attempts
        """
        # Prova la connessione condivisa solo se è ancora aperta
        if self.connection is not None and not self.connection.is_closed:
            try:
                logger.info("📡 Creating channel on shared RabbitMQ connection")
                self._create_channel(self.connection)
                logger.info("✅ Channel created on shared connection")
                return
            except Exception as e:
                logger.warning(
                    f"⚠️  Shared connection channel failed ({e}), "
                    f"falling back to own connection"
                )
                # La connessione condivisa non è più usabile:
                # da qui in poi questo thread gestisce la propria connessione
                self.connection = None
                self.owns_connection = True

        # Crea una connessione propria (modalità legacy o fallback dopo drop)
        for attempt in range(self.max_connect_retries):
            try:
                logger.info(
                    f"📡 Connecting to RabbitMQ ({attempt + 1}/{self.max_connect_retries}): "
                    f"{self.rabbitmq_url.split('@')[0]}@..."
                )
                
                # Create connection
                self.connection = pika.BlockingConnection(
                    pika.URLParameters(self.rabbitmq_url)
                )
                self.channel = self.connection.channel()
                
                # Declare queue as durable (survives broker restart)
                self.channel.queue_declare(
                    queue=self.request_queue,
                    durable=True,
                    auto_delete=False
                )
                
                # Fair dispatch: don't give a worker more than 1 job at a time
                # This ensures balanced load across multiple workers
                self.channel.basic_qos(prefetch_count=1)
                
                logger.info("✅ Connected to RabbitMQ successfully")
                return
                
            except Exception as e:
                logger.error(
                    f"❌ Connection attempt {attempt + 1} failed: {e}"
                )
                
                if attempt < self.max_connect_retries - 1:
                    wait_time = self.connect_retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        raise Exception(
            f"Failed to connect to RabbitMQ after {self.max_connect_retries} attempts"
        )
    
    def _prewarm_models(self) -> None:
        """
        Pre-load all YOLO models from MODEL_CACHE_DIR into cache.
        
        Called on worker startup to eliminate cold-start latency for the first
        inference request. Delegates to model_cache.preload_all() for centralized logic.
        """
        preload_all(self.model_cache_dir)
    
    def on_request(
        self,
        ch: pika.adapters.blocking_connection.BlockingChannel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes
    ) -> None:
        """
        Process incoming inference request from queue.
        
        Args:
            ch: Channel
            method: Delivery method (includes delivery_tag, redelivered, exchange, routing_key)
            properties: Message properties (includes correlation_id, reply_to)
            body: Message body (JSON)
        """
        import time
        t_start = time.perf_counter()
        ts_arrival = time.time()
        
        correlation_id = properties.correlation_id
        reply_to = properties.reply_to
        
        logger.info(
            f"📨 Received inference request: "
            f"correlation_id={correlation_id}, reply_to={reply_to}"
        )
        
        result = None
        
        try:
            # --- Binary protocol (application/octet-stream) ---
            # Image bytes are the raw message body; metadata is in AMQP headers.
            # Fallback: accept legacy application/json + base64 for backward compat.
            t_decode_start = time.perf_counter()
            content_type = (properties.content_type or '').lower()

            if content_type == 'application/octet-stream':
                # New binary protocol: body IS the raw JPEG bytes
                image_bytes = body
                headers = properties.headers or {}
                model_configs_json = headers.get('x-model-configs', '[]')
                crop_polygon_json = headers.get('x-crop-polygon', 'null')
                request_id = headers.get('x-request-id', 'unknown')
                logger.info(
                    f"✅ Binary protocol: received {len(image_bytes)} raw image bytes"
                )
            else:
                # Legacy JSON + base64 protocol (backward compat)
                payload = json.loads(body.decode('utf-8'))
                image_b64 = payload.get('image_bytes')
                if not image_b64:
                    raise ValueError("Missing 'image_bytes' in payload")
                image_bytes = base64.b64decode(image_b64)
                model_configs_json = payload.get('model_configs_json', '[]')
                crop_polygon_json = payload.get('crop_polygon_json', 'null')
                request_id = payload.get('request_id', 'unknown')
                logger.info(
                    f"✅ JSON/base64 protocol (legacy): decoded {len(image_bytes)} bytes"
                )

            t_decode_end = time.perf_counter()
            decode_ms = (t_decode_end - t_decode_start) * 1000
            logger.info(f"✅ Image ready: {len(image_bytes)} bytes [decode: {decode_ms:.1f}ms]")

            # Parse metadata JSON strings (common for both protocols)
            model_configs = json.loads(model_configs_json)
            crop_polygon = json.loads(crop_polygon_json)
            
            logger.info(
                f"📊 Inference job: "
                f"request_id={request_id}, "
                f"models={len(model_configs)}, "
                f"crop_polygon={'yes' if crop_polygon else 'no'}"
            )
            
            # Run inference with timing
            t_gpu_start = time.perf_counter()
            logger.info("🚀 Running inference...")
            result = run_inference(
                image_bytes=image_bytes,
                model_configs=model_configs,
                crop_polygon_coords=crop_polygon,
                model_cache_dir=self.model_cache_dir,
            )
            t_gpu_end = time.perf_counter()
            gpu_ms = (t_gpu_end - t_gpu_start) * 1000
            
            logger.info(
                f"✅ Inference completed: "
                f"event_detected={result.get('event_detected')}, "
                f"detections={len(result.get('detections', []))} "
                f"[GPU: {gpu_ms:.1f}ms]"
            )
            
        except Exception as e:
            logger.error(
                f"❌ Error processing request (correlation_id={correlation_id}): {e}",
                exc_info=True
            )
            result = self._get_error_response(str(e))
        
        # Publish response
        if result and reply_to:
            try:
                response_body = json.dumps(result).encode('utf-8')
                
                logger.info(
                    f"📤 Publishing response: "
                    f"correlation_id={correlation_id}, "
                    f"size={len(response_body)} bytes"
                )
                
                # Publish to reply_to queue with correlation_id for RPC matching
                self.channel.basic_publish(
                    exchange='',
                    routing_key=reply_to,
                    properties=pika.BasicProperties(
                        correlation_id=correlation_id,
                        delivery_mode=pika.spec.TRANSIENT_DELIVERY_MODE,  # In-memory only: no disk fsync on broker
                    ),
                    body=response_body,
                )
                
                logger.info(
                    f"✅ Response published successfully: "
                    f"correlation_id={correlation_id}"
                )
                
            except Exception as e:
                logger.error(
                    f"❌ Error publishing response (correlation_id={correlation_id}): {e}",
                    exc_info=True
                )
        
        # Acknowledge original message (remove from queue)
        try:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.debug(f"✅ Message acknowledged: delivery_tag={method.delivery_tag}")
        except Exception as e:
            logger.error(
                f"❌ Error acknowledging message (delivery_tag={method.delivery_tag}): {e}",
                exc_info=True
            )
    
    def _get_error_response(self, error_msg: str) -> Dict[str, Any]:
        """
        Return error response payload for graceful degradation.
        
        Args:
            error_msg: Human-readable error message
            
        Returns:
            Response dict that matches InferResponse structure
        """
        logger.info(f"🚨 Returning error response: {error_msg}")
        
        try:
            device_name, device_type = get_device_info()
        except Exception:
            device_name = "Unknown"
            device_type = "error"
        
        return {
            "event_detected": False,
            "detections": [],
            "processed_image_b64": None,
            "device": device_type,
            "device_name": device_name,
            "models_run": [],
            "error": f"Inference failed: {error_msg}"
        }
    
    def start(self) -> None:
        """
        Start consuming messages from queue with reconnection logic.
        
        Implements exponential backoff for reconnection attempts.
        Blocks indefinitely until KeyboardInterrupt or fatal error.
        """
        logger.info("🎯 Starting RabbitMQ consumer...")
        
        reconnect_attempts = 0
        max_reconnect_attempts = self.max_reconnect_attempts
        reconnect_base_delay = self.reconnect_base_delay
        
        # Check stop event first (allows clean shutdown from close())
        if self._stop_event.is_set():
            logger.info("🛑 Stop event set, exiting consumer loop")
            self.close()
            return

        while True:
            try:
                # Connect to RabbitMQ
                self.connect()
                reconnect_attempts = 0  # Reset on successful connection

                # Check stop event after connecting too (in case it was set during pre-warm)
                if self._stop_event.is_set():
                    logger.info("🛑 Stop event set after connect, exiting consumer loop")
                    self.close()
                    return
                
                # Pre-warm all models from cache directory (only first thread does it)
                if not self.skip_prewarm:
                    self._prewarm_models()
                else:
                    logger.info("⏭️  Skipping pre-warm (already done by another thread)")
                
                # Set up message handler
                self.channel.basic_consume(
                    queue=self.request_queue,
                    on_message_callback=self.on_request,
                    auto_ack=False,  # Manual acknowledgment in callback
                )
                
                logger.info(
                    f"✅ Listening for messages on queue '{self.request_queue}'..."
                )
                logger.info("⏳ Press Ctrl+C to stop")
                
                # Start consuming (blocking)
                self.channel.start_consuming()
                
            except KeyboardInterrupt:
                logger.info("🛑 Received keyboard interrupt (Ctrl+C)")
                break
                
            except Exception as e:
                logger.error(
                    f"❌ Consumer loop error: {e}",
                    exc_info=True
                )
                
                reconnect_attempts += 1
                
                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error(
                        f"❌ Max reconnection attempts ({max_reconnect_attempts}) reached. "
                        f"Exiting..."
                    )
                    break
                
                # Exponential backoff: 5s, 10s, 20s, 40s (capped at ~1min)
                wait_time = reconnect_base_delay * (2 ** min(reconnect_attempts - 1, 3))
                logger.info(
                    f"⏳ Reconnection attempt {reconnect_attempts}/{max_reconnect_attempts}. "
                    f"Retrying in {wait_time}s..."
                )
                
                time.sleep(wait_time)
                self.close()
        
        # Graceful shutdown
        self.close()
        logger.info("🏁 Consumer stopped")
    
    def close(self) -> None:
        """Signal the worker to stop and close RabbitMQ connection.

        Sets the stop event (to break the consumer loop) and closes
        the connection if this worker owns it (not shared).
        Thread-safe: can be called multiple times safely.
        """
        # Always signal stop — this breaks the while True loop in start()
        self._stop_event.set()
        logger.info("🛑 Stop event signaled to worker")

        if not self.owns_connection:
            logger.info("⏭️  Skipping connection close (shared connection, managed externally)")
            return
            
        if self.connection and not self.connection.is_closed:
            try:
                self.connection.close()
                logger.info("✅ Connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        
        self.connection = None
        self.channel = None

    def stop(self) -> None:
        """Alias for close(). Signal the worker to stop gracefully."""
        self.close()


def main():
    """Entry point: create and start worker."""
    logger.info("=" * 70)
    logger.info("🚀 GPU Inference Worker (RabbitMQ AMQP Consumer)")
    logger.info("=" * 70)
    
    try:
        worker = RabbitMqInferenceWorker()
        worker.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

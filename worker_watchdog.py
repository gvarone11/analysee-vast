"""
WorkerWatchdog: monitors RabbitMQ worker threads and recreates dead ones.

Importable without torch - only requires pika and threading.
Designed to be testable: accepts an optional worker_factory callable.
"""

import logging
import os
import threading
import time
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


class WorkerWatchdog:
    """
    Background thread that monitors RabbitMQ worker threads.
    If a worker thread dies, it recreates it with the same configuration.

    Testable: inject worker_factory to avoid importing RabbitMqInferenceWorker
    (which pulls in torch). In production, worker_factory defaults to
    RabbitMqInferenceWorker.
    """

    def __init__(
        self,
        workers: List[Tuple[threading.Thread, Any]],
        shared_connection: Any,
        shutdown_event: threading.Event,
        worker_factory: Optional[Callable[[Any, bool], Any]] = None,
    ):
        """
        Args:
            workers: List of (thread, worker_instance) tuples.
            shared_connection: Shared pika.BlockingConnection to reuse.
            shutdown_event: threading.Event to signal watchdog to stop.
            worker_factory: Optional callable(shared_connection, skip_prewarm) -> Worker.
                            If None, uses RabbitMqInferenceWorker (imported lazily).
        """
        self.workers = workers
        self.shared_connection = shared_connection
        self.shutdown_event = shutdown_event
        self._worker_factory = worker_factory
        self._lock = threading.Lock()
        self._check_interval = 5.0  # seconds

    def run(self):
        """Main watchdog loop. Runs until shutdown_event is set."""
        logger.info(
            f"Watchdog started, monitoring {len(self.workers)} thread(s)"
        )
        while not self.shutdown_event.is_set():
            time.sleep(self._check_interval)
            self._check_and_restart_dead_workers()
        logger.info("Watchdog loop exited")

    def _is_connection_open(self) -> bool:
        """
        Check if the shared RabbitMQ connection is open.
        Safe against None or already-closed connections.
        """
        try:
            return (
                self.shared_connection is not None
                and self.shared_connection.is_open
            )
        except Exception:
            return False

    def _reconnect_shared_connection(self) -> bool:
        """
        Attempt to reconnect the shared RabbitMQ connection.
        Uses exponential backoff (max 3 attempts).
        Tries to reuse the original connection parameters first,
        then falls back to environment variables.
        Returns True on success; updates self.shared_connection.
        """
        import pika

        logger.warning(
            "Shared RabbitMQ connection lost, attempting to reconnect..."
        )

        # Close any stale connection first
        try:
            if self.shared_connection and not self.shared_connection.is_closed:
                self.shared_connection.close()
        except Exception:
            pass  # Best-effort close

        # Try to reuse original parameters (via .params attribute on BlockingConnection)
        connection_params = None
        try:
            conn_params_obj = getattr(self.shared_connection, "params", None)
            if conn_params_obj and hasattr(conn_params_obj, "__dict__"):
                # Reconstruct from original params object
                pd = conn_params_obj.__dict__
                connection_params = pika.ConnectionParameters(
                    host=pd.get("host"),
                    port=pd.get("port"),
                    credentials=pd.get("credentials"),
                    connection_attempts=pd.get("connection_attempts", 1),
                    retry_delay=pd.get("retry_delay", 2.0),
                    socket_timeout=pd.get("socket_timeout", 5.0),
                )
                logger.info(
                    f"Reconstructing connection from original params "
                    f"(host={pd.get('host')}, port={pd.get('port')})"
                )
        except Exception as e:
            logger.debug(f"Could not reuse original params ({e}), using env fallback")

        # Fallback: read from environment
        if connection_params is None:
            rabbitmq_url = os.environ.get(
                "RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F"
            )
            connection_params = pika.URLParameters(rabbitmq_url)
            logger.info("Using RABBITMQ_URL from environment for reconnection")

        # Exponential backoff: 2s, 4s, 8s
        for attempt in range(1, 4):
            try:
                self.shared_connection = pika.BlockingConnection(connection_params)
                logger.info("Shared RabbitMQ connection re-established")
                return True
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(
                    f"Reconnection attempt {attempt}/3 failed ({e}), "
                    f"retrying in {wait}s..."
                )
                time.sleep(wait)

        logger.error(
            "Failed to reconnect shared RabbitMQ connection after 3 attempts"
        )
        return False

    def _get_worker_factory(self):
        """Return the worker factory, defaulting to RabbitMqInferenceWorker."""
        if self._worker_factory is not None:
            return self._worker_factory
        # Lazy import to avoid pulling rabbitmq_worker (and thus torch) unless needed
        from rabbitmq_worker import RabbitMqInferenceWorker

        def factory(conn, skip_prewarm):
            return RabbitMqInferenceWorker(
                shared_connection=conn, skip_prewarm=skip_prewarm
            )

        return factory

    def _check_and_restart_dead_workers(self):
        """Check each worker thread; restart any that are dead."""
        with self._lock:
            for i, (thread, worker) in enumerate(self.workers):
                if not thread.is_alive():
                    logger.warning(
                        f"Worker thread {i} ({thread.name}) is dead, restarting..."
                    )
                    try:
                        # Get factory lazily (only when actually needed)
                        worker_factory = self._get_worker_factory()

                        # Extract skip_prewarm from the dead worker
                        skip_prewarm = getattr(worker, "skip_prewarm", False)

                        # Each worker creates its own connection.
                        # Pass shared_connection=None so the worker sets owns_connection=True
                        # and is responsible for closing its own connection on restart.
                        new_worker = worker_factory(None, skip_prewarm)
                        new_thread = threading.Thread(
                            target=new_worker.start,
                            daemon=True,
                            name=f"rabbitmq-worker-{i}-restarted",
                        )
                        new_thread.start()
                        # Replace dead worker with new one
                        self.workers[i] = (new_thread, new_worker)
                        logger.info(f"Worker thread {i} restarted")
                    except Exception as e:
                        logger.error(
                            f"Failed to restart worker thread {i}: {e}"
                        )
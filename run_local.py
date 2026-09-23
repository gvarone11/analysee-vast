"""
Entry point per avviare il GPU Inference Service in locale (senza Docker).

Uso:
    cd vast/
    python run_local.py

Variabili d'ambiente opzionali:
    MODEL_CACHE_DIR         — directory con i file .pt / .pth (default: ./models_local)
    PORT                    — porta (default: 8002)
    RABBITMQ_WORKER_THREADS — numero di thread consumer interni (default: 2)

Avvia:
1. FastAPI (uvicorn) su http://localhost:PORT
   Il lifespan di main.py crea automaticamente N thread RabbitMQ consumer
   (configurati via RABBITMQ_WORKER_THREADS). Tutti i thread condividono
   la stessa connessione RabbitMQ e gli stessi pesi YOLO in VRAM.
   Ogni thread ha il proprio Predictor ultralytics indipendente:
   nessun GPU lock, inference concorrente senza conflitti.

Segnali:
- Ctrl+C ferma il processo gracefully

Carica le variabili d'ambiente dal file .env (locale, staging, production, etc.)
"""
import os
import sys
import subprocess
import signal
import time
import logging
from dotenv import load_dotenv

# Aggiungi vast/ al path in modo che gli import diretti funzionino
if __name__ == "__main__":
    vast_dir = os.path.dirname(os.path.abspath(__file__))
    if vast_dir not in sys.path:
        sys.path.insert(0, vast_dir)
    
    # Carica variabili d'ambiente dal file .env
    env_file = os.path.join(vast_dir, ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)
        print(f"✅ Loaded environment from {env_file}")
    else:
        print(f"⚠️  .env file not found at {env_file}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_local")

# Global process pid
fastapi_process = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("🛑 Ricevuto segnale di shutdown (Ctrl+C)")
    shutdown_processes()
    sys.exit(0)


def shutdown_processes():
    """Ferma FastAPI gracefully"""
    global fastapi_process
    
    if fastapi_process and fastapi_process.poll() is None:
        try:
            logger.info("Stopping FastAPI...")
            fastapi_process.terminate()
            try:
                fastapi_process.wait(timeout=5)
                logger.info("✅ FastAPI stopped")
            except subprocess.TimeoutExpired:
                logger.warning("FastAPI non ha risposto, killing...")
                fastapi_process.kill()
        except Exception as e:
            logger.error(f"Error stopping FastAPI: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    model_cache_dir = os.environ.get(
        "MODEL_CACHE_DIR",
        os.path.join(os.path.dirname(__file__), "models_local")
    )
    os.environ.setdefault("MODEL_CACHE_DIR", model_cache_dir)
    os.makedirs(model_cache_dir, exist_ok=True)
    
    # Registra signal handler per Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"")
    print(f"  🚀 GPU Inference Service (LOCAL MODE)")
    print(f"  ───────────────────────────────────────")
    print(f"  FastAPI URL:     http://localhost:{port}")
    print(f"  Health:          http://localhost:{port}/health")
    print(f"  Docs:            http://localhost:{port}/docs")
    print(f"  MODEL_CACHE_DIR: {model_cache_dir}")
    num_threads = int(os.environ.get("RABBITMQ_WORKER_THREADS", "2"))
    print(f"  🐰 RabbitMQ Threads: {num_threads} (RABBITMQ_WORKER_THREADS)")
    print(f"")
    print(f"  Press Ctrl+C to stop")
    print(f"")
    
    try:
        # Avvia FastAPI in background
        logger.info("Starting FastAPI server...")
        fastapi_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", 
             "--host", "0.0.0.0", "--port", str(port), "--log-level", "info"],
            cwd=vast_dir,
            env={**os.environ, "MODEL_CACHE_DIR": model_cache_dir}
        )
        logger.info(f"✅ FastAPI started (PID: {fastapi_process.pid})")
        logger.info(f"🐰 RabbitMQ threads managed by FastAPI lifespan ({num_threads} thread(s))")
        
        # Monitora il processo FastAPI
        while True:
            if fastapi_process.poll() is not None:
                logger.error("❌ FastAPI crashed!")
                shutdown_processes()
                sys.exit(1)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        shutdown_processes()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        shutdown_processes()
        sys.exit(1)

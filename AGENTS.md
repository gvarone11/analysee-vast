# AGENTS.md — GPU Inference Microservice for Cattle Birthing Detection

## 1. Project Overview

**Digital Future Analysee** is an **AI-powered precision livestock monitoring system** that automatically detects cattle birthing events in real-time using YOLO11 computer vision.

### Purpose
- Monitor 10-50 IP cameras in cattle barns
- Detect calving behavior patterns (calving pose, tail contraction, pre-calving signs)
- Alert farmers within 30 seconds of event detection via email/SMS
- Provide zero missed births guarantee through horizontal GPU scaling

### Business Domain
Precision livestock farming. Targets dairy and beef operations requiring:
- **High throughput**: 100+ images/day per camera without API blocking
- **Low latency**: <30s detection-to-alert time
- **High availability**: Resilient to GPU/broker outages
- **Scalability**: Add cameras and GPUs without code changes

### Key Evolution
- **v1** (Legacy): HTTP POST from Django to GPU service → blocking, lost requests on crash
- **v2** (Current): RabbitMQ RPC pattern → async, queued, horizontally scalable

---

## 2. Architecture: Message-Driven GPU Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         COMPLETE DATA FLOW                          │
└─────────────────────────────────────────────────────────────────────┘

Camera (IP Cam)
    │ JPEG stream every 30-60s
    ▼
┌──────────────────────────────┐
│  FTP Server (proftpd)        │
│  Port 21 (camera upload)     │
└──────────────┬───────────────┘
               │
               │ watchdog → S3 PUT
               ▼
        ┌────────────────┐
        │  AWS S3        │
        │ camera-{id}/   │
        │  raw/YYYYMMDD/ │
        └────────┬───────┘
                 │
                 │ (Celery periodic task)
                 ▼
┌──────────────────────────────────────┐
│  Django Backend (Port 8000)          │
│  camera/services/process.py          │
│  1. ImageFetcher (S3 GET)            │
│  2. CameraImage DB record            │
│  3. ImageProcessor instantiate       │
└──────────────┬───────────────────────┘
               │ RabbitMqRpcClient
               │ ① Publish to inference_requests
               │    (payload: image_b64, model_configs_json, crop_polygon_json)
               ▼
        ┌─────────────────┐
        │  RabbitMQ 3     │  ← Message Broker (Port 5672)
        │  AMQP 0-9-1     │
        │                 │
        │ Queue: requests │
        │ [A] [B] [C]     │
        │ [D] [E]         │
        │                 │
        │ Exchange: reply │
        │ (callbacks)     │
        └────────┬────────┘
                 │
                 │ ② Worker consumes from inference_requests
                 │    with prefetch_count=1 (fair dispatch)
                 ▼
    ┌────────────────────────────┐
    │  GPU Inference Service     │  ← Port 8002 (FastAPI)
    │  vast/rabbitmq_worker.py   │
    │                            │
    │  Thread 0: Predictor_0 ─┐  │
    │  Thread 1: Predictor_1 ─┼──► nn.Module (shared VRAM weights)
    │  Thread 2: Predictor_2 ─┤  │  1× copia modello indipendente dalla
    │  Thread N: Predictor_N ─┘  │  quantità di thread
    │                            │
    │  GPU: RTX 2060 / 3090 / A100
    │  CPU fallback: ~250ms      │ Inference time per image
    │  GPU mode: 25-100ms        │
    └───────────────┬────────────┘
                    │
                    │ ③ Publish response to callback queue
                    │    (correlation_id matches request)
                    ▼
            ┌──────────────┐
            │ Django Client│
            │ RabbitMqRpc  │ ← ④ Receive response on callback queue
            │ Client       │    (correlation_id matching via properties)
            └──────┬───────┘
                   │
                   │ Unblock process_image_streaming()
                   ▼
┌──────────────────────────────────────┐
│  Django Post-Processing              │
│  1. Decode response JSON             │
│  2. Filter detections (confidence)   │
│  3. Translate event_type to Italian  │
│  4. Create Event DB record           │
│  5. Send notifications (email/SMS)   │
│  6. Save processed_image to S3       │
└──────────────────────────────────────┘
```

### Key Architectural Principles

**Separation of Concerns**
- Django: image fetching, business logic, event creation, notifications
- GPU service: stateless YOLO inference only
- RabbitMQ: message delivery guarantee, queue persistence

**Message-Driven Async**
- Backend publishes GPU jobs without blocking
- Workers independently pull jobs at their speed
- Automatic horizontal scaling: add N workers, throughput = N × worker_speed

**Correlation ID (UUID)
- Each RPC request tagged with unique correlation_id
- Each reply goes to client-specific callback queue
- Enables request/response matching in concurrent scenarios

**Graceful Degradation**
- GPU service down? → Returns safe "No Detection" payload
- RabbitMQ timeout? → Falls back to no detection with warning log
- Pipeline continues, zero request loss

---

## 2.1 RabbitMQ Configuration & Setup (AMQP Message Broker)

### What is RabbitMQ?

RabbitMQ is an **open-source message broker** implementing AMQP 0-9-1 protocol. In this architecture:
- **Backend** publishes GPU inference jobs to the `inference_requests` queue
- **GPU Worker** consumes jobs from `inference_requests`, processes them, and publishes responses to client-specific callback queues
- **Correlation ID** (UUID) ensures request/response matching in concurrent scenarios

### RabbitMQ Components

| Component | Purpose |
|-----------|---------|
| **Queue: `inference_requests`** | Durable queue where backend publishes GPU jobs (persists on broker restart) |
| **Callback Queues** | Auto-generated per-client exclusive queues (cleanup on disconnect) |
| **Exchange (default)** | Routes messages to queues based on routing_key |
| **Binding** | Connects exchange to queue with routing key |
| **Consumer** | GPU worker subscribes to `inference_requests` with prefetch_count=1 (fair dispatch) |
| **Publisher** | Backend client publishes requests and subscribes to responses |

### RabbitMQ Environment Variables (Docker Compose)

```yaml
environment:
  RABBITMQ_DEFAULT_USER: guest           # Default username
  RABBITMQ_DEFAULT_PASS: guest           # Default password
  RABBITMQ_DEFAULT_VHOST: /              # Virtual host (/ = default)
```

These map to RabbitMQ credentials. **For production, change to strong credentials!**

### Connection URL Format

```
RABBITMQ_URL = "amqp://[username]:[password]@[host]:[port]/[vhost]"
```

**Examples**:
- Local (Docker): `amqp://guest:guest@localhost:5672/%2F` (URL-encoded `/`)
- Local (from container): `amqp://guest:guest@rabbitmq:5672/%2F`
- Vast.ai (from GPU service): `amqp://guest:guest@host.docker.internal:5672/%2F`
- Production: `amqp://producer:SecurePass123@rabbitmq.prod.internal:5672/production`

### Message Flow with Correlation ID

```
┌──────────────────────────────────────────────────────────┐
│                    REQUEST/REPLY PATTERN                 │
└──────────────────────────────────────────────────────────┘

Step 1: Backend publishes REQUEST
┌─────────────────────────────────┐
│ RabbitMqRpcClient.call(payload) │
│                                 │
│ 1. Generate UUID: correlation_id = "abc-123-def"
│ 2. Create exclusive callback_queue = "amq.gen.xyz789"
│ 3. Publish to "inference_requests" with:
│    - payload (JSON): {image_b64, model_configs, crop_polygon, request_id}
│    - reply_to: "amq.gen.xyz789"
│    - correlation_id: "abc-123-def"
│ 4. Listen on callback_queue
└────────────┬────────────────────┘
             │
             ▼
        ┌─────────────────────┐
        │  RabbitMQ Broker    │
        │  Queue: inference_  │
        │  requests [REQUEST] │
        └────────────┬────────┘
                     │
Step 2: GPU Worker consumes REQUEST
        │
        ▼
┌──────────────────────────────────┐
│ rabbitmq_worker.on_request()     │
│                                  │
│ 1. Consume from "inference_requests"
│ 2. Decode image_b64
│ 3. Run YOLO inference
│ 4. Publish RESPONSE to properties.reply_to (callback_queue)
│    - response JSON: {detections, processed_image_b64, device}
│    - correlation_id: properties.correlation_id (from request)
│ 5. ch.basic_ack() acknowledge message
└────────────┬─────────────────────┘
             │
             ▼
        ┌────────────────────┐
        │ Callback Queue     │
        │ amq.gen.xyz789     │
        │ [RESPONSE] corr_id │
        │ = "abc-123-def"    │
        └────────────┬───────┘
                     │
Step 3: Backend receives RESPONSE
        │
        ▼
┌────────────────────────────────────┐
│ RabbitMqRpcClient._on_response()   │
│                                    │
│ 1. Receive message from callback_queue
│ 2. Check properties.correlation_id == "abc-123-def" (MUST MATCH!)
│ 3. If match: set Event() → unblock call()
│ 4. Return response to caller
│ 5. If timeout: raise RpcTimeoutError after 60s
└────────────────────────────────────┘
```

### Key RabbitMQ Concepts

**Durability & Persistence**
```python
# Message persists on broker restart
BasicProperties(delivery_mode=2)  # Mode 1 = transient, 2 = persistent

# Queue survives broker restart
queue_declare(queue='inference_requests', durable=True)
```

**Fair Dispatch (Prefetch Count)**
```python
# Each worker takes only 1 message at a time
channel.basic_qos(prefetch_count=1)

# Without this: fast worker grabs all messages, slow worker starves
# Example: 10 messages, 2 workers
# ❌ No QoS: Worker1 takes all 10, Worker2 gets 0
# ✅ prefetch_count=1: Worker1 takes 1, Worker2 takes 1, then both wait
```

**Acknowledgment**
```python
# Manual ACK (not auto)
channel.basic_consume(queue='inference_requests', auto_ack=False)

# Worker must ACK after processing
channel.basic_ack(delivery_tag=method.delivery_tag)

# If worker crashes before ACK → message goes back to queue
```

### RabbitMQ Management UI

Access via browser: **http://localhost:15672**

**Default Credentials**: `guest` / `guest`

**Key Tabs**:
- **Queues**: Shows `inference_requests` queue depth, consumer count, message rate
- **Connections**: Lists active connections (backend RPC client, GPU worker)
- **Channels**: Shows channel usage per connection
- **Admin**: User management, permission configuration

**What to Monitor**:
```
inference_requests queue:
├── Ready: X messages waiting
├── Unacked: Y messages being processed
├── Consumers: 1 (should be = number of GPU workers)
└── Message Rate: Z msg/s (throughput)
```

**Example Health Check**:
```bash
# Via curl (requires guest/guest credentials)
curl -u guest:guest http://localhost:15672/api/queues/%2F/inference_requests

# Response:
{
  "name": "inference_requests",
  "vhost": "/",
  "durable": true,
  "auto_delete": false,
  "exclusive": false,
  "messages": 5,              # Ready to be consumed
  "messages_details": {...},
  "messages_ready": 5,        # NOT being processed
  "messages_unacked": 0,      # Being processed
  "consumers": 1,             # Number of active consumers
  ...
}
```

### Docker Compose Configuration

**backend/docker-compose_local.yml**:
```yaml
services:
  rabbitmq:
    image: rabbitmq:3-management-alpine
    container_name: rabbitmq_local
    ports:
      - "5672:5672"           # AMQP port
      - "15672:15672"         # Management UI port
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest
      RABBITMQ_DEFAULT_VHOST: /
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    restart: always

  web:
    depends_on:
      rabbitmq:
        condition: service_healthy
    environment:
      RABBITMQ_URL: "amqp://guest:guest@rabbitmq:5672/%2F"

volumes:
  rabbitmq_data:
    driver: local
```

### RabbitMQ Local Testing

**Phase 1: Start Broker**
```bash
cd backend
docker-compose -f docker-compose_local.yml up -d rabbitmq
sleep 5

# Verify
curl -u guest:guest http://localhost:15672/api/overview
# Response should show: {"management_version": "3.x.x", ...}
```

**Phase 2: Verify Queue**
```bash
curl -u guest:guest http://localhost:15672/api/queues/%2F/inference_requests

# If queue doesn't exist, it's created on first message publish
```

**Phase 3: Start GPU Worker**
```bash
cd ../vast
docker-compose -f docker-compose.local.yml up -d gpu-inference

# Verify worker is consuming
sleep 5
curl -u guest:guest http://localhost:15672/api/queues/%2F/inference_requests | jq '.consumers'
# Should return: 1
```

**Phase 4: Test Backend RPC Call**
```bash
cd ../backend
python test_pipeline_gpu_integration.py

# Logs should show:
# ✅ RabbitMQ broker is UP
# ✅ Worker connesso (consumers=1)
# ✅ Image inference via RabbitMQ RPC successful
```

**Phase 5: Monitor in Real-Time**
```bash
# Terminal 1: Watch queue depth
watch -n 1 'curl -s -u guest:guest http://localhost:15672/api/queues/%2F/inference_requests | jq "{ready:.messages_ready, unacked:.messages_unacked, consumers:.consumers}"'

# Terminal 2: Run concurrent test
python test_concurrent_rabbitmq.py

# You should see Ready count fluctuate as messages flow through
```

### RabbitMQ on Vast.ai (Production)

**Environment Setup**:
```bash
# On Vast.ai instance
export RABBITMQ_URL="amqp://guest:guest@rabbitmq.backend.internal:5672/%2F"
# OR if backend is on separate machine:
export RABBITMQ_URL="amqp://guest:guest@10.0.0.5:5672/%2F"
```

**Docker Compose for GPU Service** (connects back to backend RabbitMQ):
```yaml
services:
  gpu-inference:
    environment:
      RABBITMQ_URL: "amqp://guest:guest@host.docker.internal:5672/%2F"
      # host.docker.internal reaches backend's RabbitMQ from container
```

---

## 2.2 Threading & Concurrency Model

### Architettura single-container / multi-thread

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  1 Container GPU                                                                   │
│  └── uvicorn main:app (1 processo)                                                 │
│        └── lifespan: N thread daemon (RABBITMQ_WORKER_THREADS)                     │
│                                                                                    │
│   ┌──────────────────────────────────────────┐                                     │
│   │  WorkerWatchdog (thread separato)         │   ← ogni 5s verifica salute thread │
│   │  Rileva thread morto → ricrea con stessa  │                                    │
│   │  configurazione (own connection)          │                                    │
│   └─────────────────────┬────────────────────┘                                     │
│                         │ monitora                                                 │
│   ┌─────────────────────┼───────────────────────────────────────────────────┐      │
│   │  Thread 0           │  Thread 1                Thread N                 │      │
│   │  ┌──────────────┐   │  ┌──────────────┐        ┌──────────────┐         │      │
│   │  │Connection_0  │   │  │Connection_1  │  ...   │Connection_N  │         │      │
│   │  │ (propria)    │   │  │ (propria)    │        │ (propria)    │         │      │
│   │  │Channel_0     │   │  │Channel_1     │        │Channel_N     │         │      │
│   │  └──────┬───────┘   │  └──────┬───────┘        └──────┬───────┘         │      │
│   │         │ consume   │         │ consume              │ consume          │      │
│   │         ▼           │         ▼                      ▼                  │      │
│   │  inference_requests │  inference_requests   inference_requests          │      │
│   └─────────────────────┴───────────────────────────────────────────────────┘      │
│                                                                                    │
│              ╲                           ╲                                         │
│               Connessione DEDICATA per    worker                                   │
│               (non più shared_connection)                                          │
│                                                                                    │
│   Thread 0: Predictor_0 ─┐                                                         │
│   Thread 1: Predictor_1 ─┼─► nn.Module (pesi VRAM condivisi) 1×                    │
│   Thread N: Predictor_N ─┘                                                         │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Due livelli di parallelismo

| Variabile | Scope | VRAM | GPU | Caso d'uso |
|-----------|-------|------|-----|------------|
| `RABBITMQ_WORKER_THREADS` | Thread nello stesso processo | **1×** (condivisi) | CUDA serializza i kernel sullo stream default | Parallelismo I/O: un thread preleva la prossima immagine da RabbitMQ mentre un altro elabora |
| `NUM_GPU_WORKERS` | Container Docker separati | **N×** (ogni container ha la propria copia) | Contesti CUDA indipendenti → vera GPU parallelism | Throughput = N × velocità singolo worker |

### Novità: ogni worker ha la propria connessione RabbitMQ

Ogni worker thread crea una **connessione RabbitMQ dedicata** (`shared_connection=None`):

- Ogni thread ha `Connection_i` + `Channel_i` propri
- Nessuna bottleneck sulla `shared_connection`: thread indipendenti non competono per lock AMQP
- Shutdown: ogni worker chiude la propria connessione, pulito e deterministico

### Shared-weights + per-thread Predictor (model_cache.py)

```python
# _base_yolo_cache: {model_path: YOLO} — pesi caricati UNA VOLTA in VRAM
# _thread_local:    threading.local()  — un wrapper leggero per ogni thread

def load_model_thread_local(model_key, model_path) -> YOLO:
    # 1. Carica o recupera il modello base da _base_yolo_cache (lock protetto)
    # 2. Se il thread non ha ancora il wrapper, crea copy.copy(base_yolo)
    #    - copia condivide nn.Module (pesi read-only)
    #    - wrapper.predictor = None → ultralytics crea Predictor al primo uso
    # 3. Restituisce il wrapper per-thread
    # Risultato: 1× VRAM, N Predictor indipendenti per thread
```

### _gpu_lock: serializzazione del forward() (inference.py)

Il Detect head di ultralytics scrive **stato mutabile** sull'`nn.Module` condiviso durante ogni chiamata:

```python
# Dentro Detect.forward() (ultralytics):
self.anchors, self.strides = make_anchors(x, self.stride, 0.5)  # SCRIVE su self
self.shape = shape                                               # SCRIVE su self
dbox = self.decode_bboxes(self.anchors * self.strides, ...)      # LEGGE self
```

Se due thread chiamano `forward()` contemporaneamente sullo stesso `nn.Module`, il Thread B sovrascrive `self.anchors` mentre il Thread A li sta leggendo → bounding box corrotte → **No Detection non deterministico** (stessa immagine, a volte rileva, a volte no).

Per questo in `inference.py` è presente:

```python
_gpu_lock = threading.Lock()

with _gpu_lock:
    results = yolo_model(image, imgsz=input_size, verbose=False)
```

### Implicazioni pratiche

- **`_gpu_lock` serializza `model.forward()`**: elimina la race condition nel Detect head
- **Impatto throughput = zero**: CUDA null stream serializza già i kernel GPU tra thread Python; il lock non aggiunge latenza reale
- **Preprocessing sovrapposto**: decode JPEG e NMS girano in parallelo tra i thread mentre uno ha il lock GPU
- **Per vera GPU parallelism**: usa `NUM_GPU_WORKERS=N` (più container = più contesti CUDA separati)
- **Connessione dedicata per worker**: ogni thread ha `Connection_i` propria; non c'è più shared_connection
- **WorkerWatchdog**: thread separato che ogni 5s verifica `thread.is_alive()` e ricrea worker morti

---

## 2.3 WorkerWatchdog — Auto-Restart dei Thread Morti

I thread worker RabbitMQ possono uscire silenziosamente dopo aver esaurito i tentativi di riconnessione (max 10 con backoff esponenziale in `rabbitmq_worker.py`). Il WorkerWatchdog monitora e ricrea automaticamente i thread morti.

### Architettura

```
FastAPI Lifespan
  │
  ├── Thread 0: rabbitmq-worker-0 (Connection_0)
  ├── Thread 1: rabbitmq-worker-1 (Connection_1)
  ├── Thread N: rabbitmq-worker-N (Connection_N)
  │
  └── Thread: worker-watchdog (daemon)
        └── loop { sleep 5s → check each thread.is_alive()
                    → if dead: create new RabbitMqInferenceWorker(None, skip_prewarm)
                    → wrap in new Thread, start, replace in list }
```

### File: `vast/worker_watchdog.py`

```python
class WorkerWatchdog:
    # workers: List[Tuple[threading.Thread, RabbitMqInferenceWorker]]
    # shutdown_event: threading.Event — segnala l'arresto

    def run(self):
        while not self.shutdown_event.is_set():
            time.sleep(5)
            self._check_and_restart_dead_workers()

    def _check_and_restart_dead_workers(self):
        with self._lock:
            for i, (thread, worker) in enumerate(self.workers):
                if not thread.is_alive():
                    # 1. Crea nuova istanza worker (own connection)
                    new_worker = worker_factory(None, skip_prewarm)
                    # 2. Crea nuovo thread
                    new_thread = threading.Thread(target=new_worker.start)
                    new_thread.start()
                    # 3. Sostituisce in lista
                    self.workers[i] = (new_thread, new_worker)
```

### Caratteristiche chiave

| Aspetto | Dettaglio |
|---------|-----------|
| **Check interval** | 5 secondi (`WATCHDOG_INTERVAL`) |
| **Worker ricreato** | Stessa configurazione del worker morto (`skip_prewarm` preservato) |
| **Connessione** | Ogni worker ricreato ha la **propria connessione** (`shared_connection=None`) |
| **Thread-safe** | Lista workers protetta da `threading.Lock` |
| **Factory** | `_worker_factory` iniettabile per test (default: `RabbitMqInferenceWorker`) |
| **Lazy import** | `RabbitMqInferenceWorker` importato solo quando serve — evita di caricare torch all'import di `worker_watchdog.py` |
| **Shutdown pulito** | `shutdown_event.set()` → watchdog termina loop → `worker.stop()` su ogni thread |

### Integrazione con FastAPI Lifespan (main.py)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_all()

    num_workers = int(os.environ.get("RABBITMQ_WORKER_THREADS", "2"))
    worker_threads: List[Tuple[threading.Thread, Any]] = []

    for i in range(num_workers):
        worker = RabbitMqInferenceWorker(
            shared_connection=None,   # Ogni worker crea la propria connessione
            skip_prewarm=(i > 0)      # Solo il primo thread pre-warma i modelli
        )
        t = threading.Thread(target=worker.start, daemon=True)
        t.start()
        worker_threads.append((t, worker))

    # Avvia watchdog
    shutdown_event = threading.Event()
    watchdog = WorkerWatchdog(
        workers=worker_threads,
        shared_connection=None,
        shutdown_event=shutdown_event,
    )
    watchdog_thread = threading.Thread(target=watchdog.run, daemon=True)
    watchdog_thread.start()

    yield

    # Cleanup
    shutdown_event.set()
    watchdog_thread.join(timeout=10)
    for t, worker in worker_threads:
        worker.stop()
        t.join(timeout=5)
```

### Debug Endpoints (development only)

```python
# Stato worker
GET /debug/worker-status
→ { "total_workers": 2, "workers_alive": [true, true] }

# Kill worker (per test watchdog)
POST /debug/kill-worker?index=0
→ { "index": 0, "status": "signalled_stop", "was_alive": true }
```

### Test Suite: `vast/test_watchdog.py`

| Test | Cosa verifica |
|------|--------------|
| `test_watchdog_initialized_workers` | Watchdog creato con lista worker, nessun errore |
| `test_watchdog_empty_workers` | Watchdog con lista vuota, non crasha |
| `test_watchdog_restarts_dead_worker` | **Integrazione reale**: 2 MockWorker su RabbitMQ reale, 1 muore, watchdog lo riavvia |
| `test_watchdog_reconnects_and_restarts` | shared_connection chiusa → watchdog riconnette → ricrea worker |

## 3. Technology Stack

| Category | Technology | Version | Role |
|----------|-----------|---------|------|
| **Backend** | Django | 5.1 | REST API, ORM, settings management |
| **Backend** | Django REST Framework | 3.15.2 | Serialization, authentication |
| **Backend** | Celery | 5.4.0 | Async tasks (image fetch, event creation) |
| **GPU Service** | FastAPI | 0.104.1 | HTTP endpoints, async request handling |
| **Database** | PostgreSQL | 15 | Persistent: cameras, models, events, users |
| **Cache/Queue** | Redis | latest | Celery backend, result storage |
| **Message Broker** | RabbitMQ | 3 | AMQP 0-9-1, job queueing, RPC pattern |
| **AMQP Client** | pika | 1.4.0 | Python bindings for RabbitMQ (both backend + GPU worker) |
| **ML Framework** | PyTorch | 2.5.1 | Tensor computation, GPU acceleration |
| **ML Framework** | Ultralytics YOLO11 | 8.3.0 | Object detection, pre/post-processing |
| **Custom Model** | YOLO v1.2.0 | custom | Cattle birthing detection (10K training images) |
| **Container** | Docker | latest | Containerization |
| **Container** | Docker Compose | v3.9 | Orchestration (local + prod) |
| **GPU Base** | nvidia/cuda | 12.1.0 | CUDA 12.1, cuDNN 8, Ubuntu 22.04 |
| **Cloud Storage** | AWS S3 | - | Raw/processed images, model artifacts |
| **GPU Host** | Vast.ai | - | A100/RTX instance rental |
| **FTP Server** | proftpd | latest | Camera JPEG uploads |
| **Frontend** | React | 18 | Image viewer, event notifications |
| **Frontend Build** | Vite | latest | Hot reload, production bundling |
| **Frontend** | TypeScript | - | Type safety, IDE support |

### Model Details

**YOLO v1.2.0 (Custom Cattle Detector)**
- Training data: 10,000+ manually annotated cattle birthing images
- Input resolution: 1536×1536 (imgsz=1536)
- Classes:
  - `calving_pose`: Cow in specific birthing posture → **Event trigger**
  - `calving_cow_straight_tail`: Tail contraction sign → **Event trigger**
  - `pre_calving_cow`: Pre-birthing preparations
  - `normal_cow`: Background/no action
- File size: ~150 MB
- Storage: MODEL_CACHE_DIR (/tmp/yolo_models)
- **NOT committed to Git** (manual upload required per environment)

### Performance Targets (by GPU)

| GPU | Cost/h | Throughput | Latency | Use Case |
|-----|--------|-----------|---------|----------|
| RTX 2060 | $0.20 (local) | 4 req/s | 250ms/img | Local dev |
| RTX 3080 | $0.35 | 9 req/s | 110ms/img | Staging |
| RTX 3090 | $0.40 | 11 req/s | 90ms/img | **Recommended prod** |
| A100 PCIe | $1.50 | 40 req/s | 25ms/img | High-volume prod |

---

## 4. Coding Standards & Rules

### 4.1 Architecture Rules

**Rule: Stateless GPU Service**
- GPU service has NO database access, NO S3 writes, NO side effects
- All business logic stays in Django
- GPU service: pure inference function
- Rationale: Enables easy scaling, independent restart, simple testing

**Rule: Message-Driven Communication**
- Backend → GPU: Use RabbitMQ RPC (via RabbitMqRpcClient), never HTTP polling
- Enables: non-blocking, automatic queueing, horizontal worker scaling
- Never block backend thread waiting for GPU response

**Rule: Separation of Task Queues**
- Celery queue: Image fetching, event creation (belongs in backend)
- RabbitMQ queue: GPU inference jobs (belongs in GPU service)
- Rationale: Different failure domains, independent scaling

### 4.2 Model File Rules

**Rule: Model File Naming Convention**
- Format: `{model_id}_{version}_{optional_descriptor}.pt` or `{model_id}_{version}_{optional_descriptor}.pth`
- Examples: `v1.2.0.pt`, `yolov8n.pt`, `cattle_v2.3_large.pth`
- Never include absolute paths in config; use basename + MODEL_CACHE_DIR

**Rule: Model Files NOT in Git**
- .pt and .pth files are >100 MB, should not be version-controlled
- Upload manually via SCP/rsync to Vast.ai
- Document upload procedure in deployment runbook
- Store in `/tmp/yolo_models` (default MODEL_CACHE_DIR)

### 4.3 Performance Rules

**Rule: Model Caching & Concurrent Inference**
- Pesi YOLO caricati **una volta in VRAM** (`_base_yolo_cache`) condivisi tra tutti i thread
- Ogni thread RabbitMQ riceve un **wrapper YOLO leggero** (`copy.copy`) con `predictor=None`
- Il `Predictor` ultralytics viene creato per-thread al primo uso → stato pre/post-processing isolato
- **`_gpu_lock` in `inference.py`**: serializza `model.forward()` per proteggere lo stato mutabile del Detect head (`self.anchors`, `self.strides`, `self.shape`) condiviso sull'`nn.Module`. Senza lock: stessa immagine → risultati non deterministici. Impatto throughput = zero (CUDA null stream è già sequenziale)
- Cache condivisa TTL (`_model_cache`) ancora disponibile per endpoint HTTP `/infer` (thread FastAPI singolo)
- Max 10 modelli in cache, 1-hour TTL (auto-evict stale models)

**Rule: Prefetch Count = 1**
- RabbitMQ basic_qos(prefetch_count=1) on all workers
- Ensures fair distribution: each worker takes 1 job, no hoarding
- Required for horizontal scaling fairness

### 4.4 Error Handling Rules

**Rule: Graceful Degradation**
- GPU timeout/crash? → Return safe "No Detection" payload
- Do NOT fail the entire pipeline
- Log the error with correlation_id for debugging
- User receives notification: "System running in degraded mode"

**Rule: Correlation ID Tracing**
- Every RPC request tagged with UUID (correlation_id)
- Every log message includes request_id for end-to-end tracing
- Enables: request tracking through system, debugging concurrency issues

### 4.5 Testing Rules

**Rule: Test Isolation**
- Local development: use `docker-compose.local.yml` (CPU-only by default)
- GPU testing: enable `deploy.resources.devices.gpu` in compose file
- Production GPU: use `provisioning.sh` → Vast.ai automation
- Rationale: No GPU dependency for base CI/CD, tests portable across machines

**Rule: Concurrent Load Testing**
- Before production deploy, run `test_concurrent_rabbitmq.py` with N=20 threads
- Verify: all requests succeed, throughput is expected, no message loss
- Acceptable latency: last response within 2× throughput time

### 4.6 Deployment Rules

**Rule: Environment Parity**
- Code identical across local → staging → production
- Only environment variables change (RABBITMQ_URL, GPU type, MODEL_CACHE_DIR)
- Dockerfile + entrypoint.sh handle runtime setup
- Rationale: "Test what you deploy, deploy what you tested"

---

## 5. Project Structure

```
backend/
├── GMSProj/
│   ├── settings.py                      # Django settings: RABBITMQ_URL, queues, timeouts
│   ├── asgi.py
│   └── urls.py
├── camera/
│   ├── services/
│   │   ├── process.py                   # ImageProcessor.process_image_streaming()
│   │   │                                # Calls RabbitMqRpcClient for GPU inference
│   │   ├── rabbitmq_rpc.py              # RPC client implementation
│   │   │                                # AMQP publisher/consumer with correlation_id
│   │   ├── s3_handler.py                # AWS S3 upload/download
│   │   ├── image_utils.py               # JPEG encode/decode
│   │   └── notifications.py             # Email/SMS sending
│   ├── models.py                        # ORM: Camera, Event, YoloModel, etc.
│   ├── views.py                         # REST endpoints
│   └── tests/
│       ├── test_pipeline_gpu_integration.py      # End-to-end: ImageProcessor → RabbitMQ → GPU
│       └── test_concurrent_rabbitmq.py           # N threads → RabbitMQ → 1 GPU worker
├── account/
│   ├── models.py                        # User, GMSUser, UserSettings
│   └── views.py                         # Auth endpoints
├── event/
│   ├── models.py                        # Event, EventAlertRule
│   └── views.py                         # Event CRUD
├── docker-compose_local.yml             # Local dev: RabbitMQ + all services
├── docker-compose.yml                   # Production template
├── requirements.txt                     # pika>=1.3.0, Django, DRF, Celery, etc.
├── .env.sample                          # Template for environment variables
├── .env.development                     # Dev secrets (git-ignored)
├── .env.staging                         # Staging secrets (git-ignored)
├── manage.py                            # Django CLI
└── Dockerfile                           # Django container (gunicorn + static files)

vast/
├── main.py                              # FastAPI app initialization
│                                        # Lifespan: pre-warm modelli + avvia N thread worker
│                                        # + avvia WorkerWatchdog daemon thread
├── inference.py                         # YOLO inference logic (GPU/CPU auto-detection)
├── rabbitmq_worker.py                   # AMQP consumer: listens on inference_requests queue
│                                        # Decodes payload, calls run_inference(), publishes response
├── worker_watchdog.py                   # Watchdog thread: monitora e riavvia worker morti
│                                        # Ogni 5s verifica thread.is_alive()
│                                        # Ricrea worker con connessione dedicata
├── model_cache.py                       # Thread-safe TTL cache for YOLO models
│                                        # Double-checked locking pattern
├── utils.py                             # Helper: resolve_model_path(), etc.
├── entrypoint.sh                        # Dual-process container entry point
│                                        # 1 - FastAPI :8002 (background)
│                                        # 2 - RabbitMQ worker (foreground)
│                                        # Signal handlers, health checks
├── provisioning.sh                      # Vast.ai deployment script
│                                        # ENV vars setup, docker run, monitoring
├── Dockerfile                           # Base: nvidia/cuda:12.1.0
│                                        # Installs: Python, PyTorch, pika, ultralytics
│                                        # ENTRYPOINT: entrypoint.sh
├── docker-compose.local.yml             # Local GPU testing
│                                        # GPU: enabled via deploy.resources.devices
│                                        # RABBITMQ_URL: amqp://guest:guest@host.docker.internal:5672/%2F
├── docker-compose.yml                   # Production orchestration (if using Docker Swarm/K8s)
├── models_local/
│   ├── v1.2.0.pt                        # Custom cattle detector (~150 MB)
│   └── yolov8n.pt                       # Standard YOLO baseline (optional)
└── requirements.txt                     # pika>=1.3.0, torch==2.5.1, ultralytics>=8.3.0, fastapi, etc.

ftp/
├── docker-compose.dev.yml               # FTP server + watchdog
├── uploader/
│   └── app.py                           # Monitors FTP uploads, converts to S3 PUT
├── proftpd/
│   └── proftpd.conf                     # FTP server config
└── scripts/
    └── health_check.sh                  # Verify FTP connectivity

frontend/
├── src/
│   ├── components/                      # React components
│   ├── views/                           # Page components
│   ├── hooks/                           # React custom hooks
│   ├── api/                             # Axios client for backend API
│   ├── store/                           # State management (Zustand/Redux)
│   └── App.tsx                          # Root component
├── public/                              # Static assets
├── vite.config.ts                       # Vite configuration
├── tailwind.config.js                   # Tailwind CSS config
├── tsconfig.json                        # TypeScript config
├── index.html                           # Entry HTML
└── package.json                         # Node dependencies

docker-compose.yml
├── web                                  # Django backend (port 8000)
├── db                                   # PostgreSQL (port 5432)
├── redis                                # Redis Celery backend
├── rabbitmq                             # RabbitMQ broker (port 5672, 15672 mgmt)
├── celery                               # Celery worker
├── celerybeat                           # Celery beat scheduler
├── nginx                                # Nginx reverse proxy (port 80, 443)
└── volumes:
    └── postgres_data, rabbitmq_data     # Persistent data

docker-compose_local.yml
├── (same as above, but RabbitMQ + web deps on rabbitmq)
└── rabbitmq service with health checks

README.md
DEPLOYMENT.md                            # Production deployment guide
RABBITMQ_LOCAL_TESTING.md                # Local testing walkthrough
```

---

## 6. External Resources & Dependencies

### Python Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| pika | 1.4.0 | AMQP client for RabbitMQ (both backend & GPU worker) |
| django | 5.1 | Web framework, ORM, settings management |
| djangorestframework | 3.15.2 | REST API serialization, authentication |
| celery | 5.4.0 | Async task queue (separate from GPU queue) |
| fastapi | 0.104.1 | GPU service HTTP framework |
| torch | 2.5.1 | PyTorch tensor computation, CUDA support |
| ultralytics | 8.3.0 | YOLO11 object detection framework |
| pillow | latest | Image I/O, format conversion |
| opencv | latest | Image processing, bounding box drawing |
| boto3 | latest | AWS S3 client |
| psycopg2 | latest | PostgreSQL driver |
| redis | latest | Python client for Redis/Celery |
| cachetools | latest | TTL cache for YOLO models |
| requests | latest | HTTP client for health checks |

### External Services

| Service | Version | Purpose |
|---------|---------|---------|
| PostgreSQL | 15 | Primary database (cameras, events, users) |
| Redis | latest | Celery backend, session cache |
| RabbitMQ | 3 | AMQP message broker (GPU job queue) |
| AWS S3 | - | Cloud storage (raw/processed images, models) |
| Vast.ai | - | GPU rental marketplace (production A100/RTX) |
| proftpd | latest | FTP server (camera uploads) |

### Documentation & References

- [YOLO11 Docs](https://docs.ultralytics.com/) — Model architecture, inference API
- [pika Documentation](https://pika.readthedocs.io/) — RabbitMQ Python client
- [RabbitMQ Tutorials](https://www.rabbitmq.com/getstarted.html) — AMQP concepts, RPC pattern
- [FastAPI Docs](https://fastapi.tiangolo.com/) — Async HTTP framework
- [Django Docs](https://docs.djangoproject.com/) — ORM, middleware, settings
- [Vast.ai API Docs](https://www.vast.ai/docs/) — GPU instance management
- [Docker Docs](https://docs.docker.com/) — Container orchestration

---

## 7. Key Implementation Files

### Backend Integration: RabbitMQ RPC Client

**File**: `backend/camera/services/rabbitmq_rpc.py`

```python
# Establishes AMQP connection to RabbitMQ broker
# Publishes job to inference_requests queue
# Waits for response on auto-generated callback queue (correlation_id match)
# Uses process_data_events() loop to receive frames while waiting

class RabbitMqRpcClient:
    def call(payload: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Generate unique correlation_id (UUID)
        # 2. Publish to inference_requests with reply_to callback_queue
        # 3. Loop: process_data_events(timeout=1s) until response arrives
        # 4. Match correlation_id in response properties
        # 5. Return response dict or raise RpcTimeoutError
```

**Key Features**:
- Context manager protocol (`__enter__`, `__exit__`) for resource cleanup
- Thread-safe Event() for synchronization
- 3-attempt retry with 2s backoff
- Persistent message delivery (survives broker restart)
- basic_qos(prefetch_count=1) for fair dispatch

### GPU Worker: AMQP Consumer

**File**: `vast/rabbitmq_worker.py`

```python
# Connects to RabbitMQ
# Subscribes to inference_requests queue
# Processes one job at a time (prefetch_count=1)
# Decodes base64 image, calls run_inference()
# Publishes response to reply_to queue with correlation_id
# Acknowledges original message

class RabbitMqInferenceWorker:
    def on_request(ch, method, properties, body):
        # 1. Decode JSON payload
        # 2. Extract image_bytes (base64), model_configs, crop_polygon
        # 3. Call run_inference() → YOLO detections + processed_image_b64
        # 4. Publish response to properties.reply_to with same correlation_id
        # 5. ch.basic_ack() to acknowledge
```

**Key Features**:
- Automatic reconnection (max 10 attempts, exponential backoff)
- Fallback autonomo se shared connection è chiusa (owns_connection=True)
- Signal handlers (SIGTERM/SIGINT) for graceful shutdown
- Usa `load_model_thread_local()`: pesi VRAM condivisi, Predictor per-thread
- Error responses with safe "No Detection" payload

### Container Entry Point: Dual-Process Management

**File**: `vast/entrypoint.sh`

```bash
# 1. Validate environment: Python version, model cache dir
# 2. Start FastAPI background: python -m uvicorn main:app --port 8002
# 3. Health check loop: curl http://localhost:8002/health (30 attempts, 1s interval)
# 4. If FastAPI unhealthy after timeout, exit container
# 5. Start RabbitMQ worker foreground: python rabbitmq_worker.py
# 6. Monitor loop: every 5s, check both processes alive (kill -0 PID)
# 7. If either dies, exit container
# 8. On SIGTERM/SIGINT: kill both processes, wait, exit 0
```

---

## 8. Local Testing Workflow

### 8.0 Quick Start: Run Local (Dual-Process Entry Point)

**File**: `vast/run_local.py`

A convenient entry point that starts both FastAPI and RabbitMQ Worker simultaneously in local development.

**Features**:
- ✅ Automatically loads `.env` file (respects RABBITMQ_URL overrides)
- ✅ Starts FastAPI on port 8002 (configurable via PORT env var)
- ✅ Il lifespan di `main.py` avvia automaticamente N thread RabbitMQ consumer (via `RABBITMQ_WORKER_THREADS`)
- ✅ Ogni thread ha un Predictor indipendente, pesi YOLO condivisi in VRAM
- ✅ Graceful shutdown: Ctrl+C ferma FastAPI (i thread daemon terminano automaticamente)
- ✅ Unified logging output

**Basic Usage**:

```bash
cd vast/
python run_local.py
```

**Output**:
```
✅ Loaded environment from ./vast/.env

  🚀 GPU Inference Service (LOCAL MODE)
  ───────────────────────────────────────
  FastAPI URL:     http://localhost:8002
  Health:          http://localhost:8002/health
  Docs:            http://localhost:8002/docs
  MODEL_CACHE_DIR: ./models_local
  🐰 RabbitMQ Threads: 2 (RABBITMQ_WORKER_THREADS)

  Press Ctrl+C to stop

[INFO] Starting FastAPI server...
✅ FastAPI started (PID: 12345)
[INFO] 🐰 RabbitMQ threads managed by FastAPI lifespan (2 thread(s))
```

**Environment Variables**:

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | 8002 | FastAPI listening port |
| `MODEL_CACHE_DIR` | `./models_local` | Directory with .pt/.pth model files |
| `SKIP_RABBITMQ` | (empty) | Set to `1`/`true` to skip RabbitMQ worker startup |
| `RABBITMQ_URL` | (from .env) | RabbitMQ connection string; loaded from .env file |

**Usage Examples**:

```bash
# Default: FastAPI + 2 thread RabbitMQ consumer
python run_local.py

# 4 thread consumer (più parallelismo I/O)
RABBITMQ_WORKER_THREADS=4 python run_local.py

# Custom port
PORT=8003 python run_local.py

# Custom model directory
MODEL_CACHE_DIR=/path/to/models python run_local.py

# Override RabbitMQ URL from command line
RABBITMQ_URL="amqp://user:pass@localhost:5672/%2F" python run_local.py
```

**Environment File Loading**:

`run_local.py` automatically loads `.env` from the `vast/` directory with `override=True`, meaning:
- Environment variables in `.env` override any system environment variables
- Supports different `.env` files: `.env`, `.env.development`, `.env.staging`, `.env.production`
- You can symlink or copy the appropriate file: `cp .env.staging .env`

**Example .env file** (vast/.env):
```bash
RABBITMQ_URL="amqps://broydvge:xPRe6nBrj4jfzMKtgbQxWJjZ5FjRVM0v@hawk.rmq.cloudamqp.com/broydvge"
MODEL_CACHE_DIR="/tmp/yolo_models"
CUDA_VISIBLE_DEVICES=0
LOG_LEVEL="DEBUG"
```

**Graceful Shutdown**:

Premendo Ctrl+C:
1. Invia SIGINT a FastAPI (uvicorn)
2. I thread RabbitMQ sono daemon → terminano automaticamente con il processo padre
3. Attende fino a 5 secondi per la terminazione graceful
4. Uscita con code 0

**Troubleshooting**:

**Problem**: RabbitMQ Worker keeps crashing with "Connection refused"
- **Fix**: Verify RABBITMQ_URL in `.env` is correct
  ```bash
  # Local RabbitMQ (Docker)
  RABBITMQ_URL="amqp://guest:guest@localhost:5672/%2F"
  
  # Or CloudAMQP
  RABBITMQ_URL="amqps://user:pass@hawk.rmq.cloudamqp.com/user"
  ```

**Problem**: FastAPI crashes immediately
- **Fix**: Check MODEL_CACHE_DIR exists
  ```bash
  mkdir -p ./models_local
  touch ./models_local/yolov8n.pt  # or copy your model files
  ```

---

### 8.1 Setup (First Time)

```bash
# Clone repo
git clone <repo>
cd analysee

# Activate venv
.venv/Scripts/activate

# Start backend + RabbitMQ
cd backend
docker-compose -f docker-compose_local.yml up -d
sleep 10  # Wait for RabbitMQ to be ready

# Start GPU service (CPU mode for testing)
cd ../vast
docker-compose -f docker-compose.local.yml up --build -d

# Verify health
curl http://localhost:8002/health           # GPU service
curl http://localhost:15672/api/overview    # RabbitMQ (guest:guest)
```

### 8.2 End-to-End Test (Pipeline)

```bash
cd backend
python test_pipeline_gpu_integration.py
```

**Expected output**:
```
✅ RabbitMQ broker is UP (node=rabbit@rabbitmq_local)
✅ Worker connesso (consumers=1, messages_ready=0)

TEST: image_124832.jpg → POSIZIONE PARTO
 ⏱️ Inizio inferenza GPU: 12:07:50.160
 ⏱️ Fine inferenza GPU:   12:07:53.485
 ⏱️ Tempo risposta: 3.32s

✅ event_type corretto: 'Rilevato Evento Parto'
✅ Classe rilevata: 'calving pose'
✅ processed_stream valido (755854 bytes)

[OK] TUTTI I TEST PASSATI (2/2)
```

### 8.3 Concurrent Load Test

```bash
cd backend
python test_concurrent_rabbitmq.py
```

**Expected output**:
```
🚀 [12:07:50.129] Lancio 20 thread simultanei...

✅ [12:07:50.160 → 12:07:53.485]  Camera-01  3.32s → Rilevato Evento Parto
✅ [12:07:50.161 → 12:07:56.970]  Camera-02  6.81s → Rilevato Evento Parto
...

📊 RIEPILOGO: 20 camere simultanee (stress)
Richieste totali  : 20
Riuscite          : 20/20
Tempo totale muro : 4.99s
Latenza prima risp: 1.13s
Latenza media     : 3.05s
Throughput        : 4.01 req/s
```

### 8.4 GPU Testing (with RTX 2060/3080/3090)

Edit `docker-compose.local.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then restart and verify GPU usage:

```bash
docker-compose -f docker-compose.local.yml exec gpu-inference nvidia-smi
# Output: RTX 2060 GPU-Util: 45%
```

### 8.5 Monitoring in Real-Time

```bash
# RabbitMQ Management UI
open http://localhost:15672/
# Login: guest / guest
# Go to Queues tab → inference_requests
# Watch: Ready, Unacked, Consumers

# Logs from GPU service
docker-compose -f docker-compose.local.yml logs -f gpu-inference

# Logs from Django backend
docker-compose -f docker-compose_local.yml logs -f web
```

---

## 9. Production Deployment (Vast.ai)

### 9.1 Pre-Deployment Checklist

- [ ] All tests pass locally (test_pipeline_gpu_integration.py, test_concurrent_rabbitmq.py)
- [ ] .pt/.pth model files uploaded to `/tmp/yolo_models` (not in git)
- [ ] RABBITMQ_URL set in backend .env (prod RabbitMQ URL/port)
- [ ] Backend health check: `curl http://backend.prod:8000/health`
- [ ] GPU_INFERENCE_SERVICE_URL points to valid Vast.ai instance

### 9.2 Deploy GPU Service to Vast.ai

```bash
# Rent instance on Vast.ai (A100 recommended)
# SSH in
ssh -p <SSH_PORT> root@<IP>

# Upload code
scp -P <SSH_PORT> -r vast/* root@<IP>:/workspace/inference/

# Upload model files
scp -P <SSH_PORT> vast/models_local/v1.2.0.pt root@<IP>:/tmp/yolo_models/

# Run provisioning
ssh -p <SSH_PORT> root@<IP> "bash /workspace/inference/provisioning.sh"

# Health check
curl http://<IP>:8002/health
```

### 9.3 Update Backend Configuration

```bash
# backend/.env (production)
GPU_INFERENCE_SERVICE_URL=http://<VAST_IP>:8002
GPU_INFERENCE_TIMEOUT=60
RABBITMQ_HOST=rabbitmq.prod.internal  # Must match backend's RabbitMQ
```

### 9.4 Monitor Production

```bash
# RabbitMQ Management (if exposed)
http://rabbitmq.prod:15672/

# GPU Metrics
ssh -p <SSH_PORT> root@<IP> "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader -l 1"

# Service Logs
ssh -p <SSH_PORT> root@<IP> "docker logs -f <container_id>"
```

---

## 10. Troubleshooting Guide

### "No response from GPU worker after X seconds"

**Causes**:
1. RabbitMQ broker down → Check `docker-compose ps | grep rabbitmq`
2. GPU worker not consuming → Check RabbitMQ Management UI: consumers = 0
3. GPU worker crashed → Check `docker logs vast_gpu-inference_1`
4. Timeout too short → Increase INFERENCE_RPC_TIMEOUT in settings

**Fix**:
```bash
# Verify broker health
curl http://localhost:15672/api/overview -u guest:guest

# Check queue
curl http://localhost:15672/api/queues/%2F/inference_requests -u guest:guest | jq '.consumers'

# Restart worker
docker-compose -f docker-compose.local.yml restart gpu-inference
```

### "FileNotFoundError: v1.2.0.pt"

**Causes**:
1. Model file not in MODEL_CACHE_DIR
2. Wrong path in model_configs_json (absolute Windows path instead of basename)
3. Permissions issue (file not readable by app user)

**Fix**:
```bash
# Verify file exists
ls -lh /tmp/yolo_models/v1.2.0.pt

# Check permissions
chmod 644 /tmp/yolo_models/v1.2.0.pt

# Verify path resolution in settings
echo $MODEL_CACHE_DIR

# Ensure model config uses basename only
# ❌ WRONG: "model_url_s3_or_path": "C:\\models\\v1.2.0.pt"
# ✅ CORRECT: "model_url_s3_or_path": "v1.2.0.pt"
```

### "CUDA out of memory" on GPU worker

**Causes**:
1. Input image too large (imgsz mismatch)
2. Model too large for available VRAM
3. Multiple workers on same GPU (should be 1 worker per GPU)

**Fix**:
```bash
# Check available VRAM
nvidia-smi

# Scale down concurrent requests (reduce prefetch_count or worker count)
# Ensure only 1 worker per GPU

# Use smaller input size (if model supports it)
# "input_size": 1024  # instead of 1536
```

### "Connection refused" from backend to RabbitMQ

**Causes**:
1. RabbitMQ container not started
2. RABBITMQ_URL incorrect (wrong host/port)
3. Firewall blocking port 5672

**Fix**:
```bash
# Verify RabbitMQ is running
docker-compose -f docker-compose_local.yml ps | grep rabbitmq

# Test connectivity
telnet localhost 5672

# Verify URL format
echo $RABBITMQ_URL
# Expected: amqp://guest:guest@localhost:5672/%2F

# If using host.docker.internal (Windows)
RABBITMQ_URL="amqp://guest:guest@host.docker.internal:5672/%2F"
```

---

## 11. Performance Optimization Tips

### For Local Development
- Use CPU-only mode (docker-compose.local.yml default) for fast iteration
- Keep MODEL_CACHE_DIR populated to avoid re-downloads
- Reduce imgsz (e.g., 1024 instead of 1536) for faster inference during testing

### For Staging (RTX 3090)
- Use imgsz=1536 for full accuracy
- Expect ~100ms per image
- Throughput: ~10 req/s
- Concurrent load test with N=50 threads to verify stability

### For Production (A100)
- Use imgsz=1536
- Expect ~25-35ms per image
- Throughput: ~30-40 req/s
- Monitor GPU utilization (target: 80-95%)
- Add second A100 if throughput needed > 50 req/s

---

## 12. Best Practices Checklist

- [ ] Never commit .pt or .pth model files to Git
- [ ] Always test locally before production deploy
- [ ] Use correlation_id for request tracing
- [ ] Monitor RabbitMQ queue depth and consumer count
- [ ] Set GPU_INFERENCE_TIMEOUT ≥ 2× typical inference time
- [ ] Use graceful degradation (no exceptions, just "No Detection")
- [ ] Keep only 1 worker per GPU (unless GPU has 40GB+ VRAM)
- [ ] Regularly restart workers to prevent memory leaks
- [ ] Use `deploy.resources.devices.gpu` in compose (not `--gpus` flag)
- [ ] Test concurrent load before production (N=20 threads minimum)

---

## 13. Supervisor — Avvio Automatico GPU Inference Service

Il servizio GPU Inference viene eseguito senza Docker interno: `run_local.py` lancia Uvicorn come sottoprocesso e Supervisor lo mantiene attivo, riavviandolo automaticamente in caso di crash.

```
Supervisor
  └── [program:gpu-inference]
        ├── command = python run_local.py
        ├── autostart=true, autorestart=true
        ├── stdout → /var/log/gpu-inference/out.log
        └── stderr → /var/log/gpu-inference/err.log
```

### 13.1 Setup Locale (WSL/Ubuntu)

```bash
sudo apt-get update && sudo apt-get install -y supervisor
sudo mkdir -p /var/log/gpu-inference
cd vast/
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

sudo cat > /etc/supervisor/conf.d/gpu-inference.conf << 'EOF'
[program:gpu-inference]
directory=/home/<utente>/analysee/vast
command=/home/<utente>/analysee/vast/.venv/bin/python /home/<utente>/analysee/vast/run_local.py
autostart=true
autorestart=true
startsecs=10
startretries=999
stopasgroup=true
killasgroup=true
stdout_logfile=/var/log/gpu-inference/out.log
stderr_logfile=/var/log/gpu-inference/err.log
environment=PORT="8002",MODEL_CACHE_DIR="/home/<utente>/analysee/vast/models_local",RABBITMQ_WORKER_THREADS="2"
EOF

sudo systemctl enable supervisor
sudo systemctl restart supervisor
sudo supervisorctl reread && sudo supervisorctl update && sudo supervisorctl start gpu-inference
sudo supervisorctl status gpu-inference
```

### 13.2 Setup Vast.ai (onstart.sh)

```bash
cat > /root/onstart.sh << 'SCRIPT'
#!/usr/bin/env bash
set -e
APP_DIR="/workspace/inference"
cd "$APP_DIR"
echo "Starting GPU inference service setup..."
if ! command -v supervisord >/dev/null 2>&1; then
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y supervisor
fi
if [ ! -d ".venv" ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip && pip install -r requirements.txt
mkdir -p /var/log/gpu-inference
cat > /etc/supervisor/conf.d/gpu-inference.conf << 'EOF'
[program:gpu-inference]
directory=/workspace/inference
command=/workspace/inference/.venv/bin/python /workspace/inference/run_local.py
autostart=true
autorestart=true
startsecs=10
startretries=999
stopasgroup=true
killasgroup=true
stdout_logfile=/var/log/gpu-inference/out.log
stderr_logfile=/var/log/gpu-inference/err.log
environment=PORT="8002",MODEL_CACHE_DIR="/workspace/inference/models_local",RABBITMQ_WORKER_THREADS="2"
EOF
if pgrep -x supervisord >/dev/null; then
  supervisorctl reread && supervisorctl update && supervisorctl restart gpu-inference || true
else
  supervisord -c /etc/supervisor/supervisord.conf
fi
echo "GPU inference service started with supervisor"
SCRIPT
chmod +x /root/onstart.sh
```

### 13.3 Comandi Rapidi

```bash
sudo supervisorctl start gpu-inference
sudo supervisorctl stop gpu-inference
sudo supervisorctl restart gpu-inference
sudo supervisorctl status gpu-inference
sudo supervisorctl reread && sudo supervisorctl update
sudo tail -f /var/log/gpu-inference/out.log
```

### 13.4 Test di Crash

```bash
pkill -f "uvicorn main:app"
sleep 5 && sudo supervisorctl status gpu-inference
pkill -f "run_local.py"
sleep 5 && sudo supervisorctl status gpu-inference
```

### 13.5 Troubleshooting

| Errore | Soluzione |
|---|---|
| `apt-get` permessi | Usa `su -` o su Vast.ai sei già root |
| `ensurepip` non trovato | `apt-get install -y python3-venv python3-pip` |
| `no such file` da Supervisor | Verifica percorsi nel file `.conf` |
| `libGL.so.1` | `apt-get install -y libgl1 libglib2.0-0` |
| Servizio `STOPPED`/`FATAL` | Controlla `tail -50 err.log`, virtualenv, dipendenze, porta |

---

## 14. SSH e SCP — Collegarsi e Trasferire File

### 14.1 SSH su Vast.ai

```bash
ssh -p <porta> root@<ip>
```

**Alias SSH** (in `~/.ssh/config`):

```
Host vast-inference
    HostName <ip>
    Port <porta>
    User root
    IdentityFile ~/.ssh/id_rsa
```

Poi: `ssh vast-inference`

### 14.2 SCP — Trasferire File

```bash
# Caricare modello su Vast.ai
scp -P <porta> ./modello.pt root@<ip>:/workspace/inference/models_local/

# Scaricare log
scp -P <porta> root@<ip>:/var/log/gpu-inference/err.log .

# Con alias SSH
scp ./modello.pt vast-inference:/workspace/inference/models_local/
scp vast-inference:/var/log/gpu-inference/out.log .
```

### 14.3 SCP Lento — Usa tar

```bash
tar -czf models.tar.gz models_local/*.pt
scp models.tar.gz vast-inference:/workspace/inference/
ssh vast-inference "cd /workspace/inference && tar -xzf models.tar.gz"
```

---

## 15. Modelli YOLO (.pt / .pth) — Primo Avvio

| Ambiente | Percorso default |
|---|---|
| Locale | `vast/models_local/` |
| Vast.ai | `/workspace/inference/models_local/` |
| Docker | `/tmp/yolo_models` |

**Ordine di ricerca**: 1. percorso assoluto; 2. `MODEL_CACHE_DIR` + nome file; 3. auto-download Ultralytics.

### Checklist Primo Avvio

```bash
ls -la $MODEL_CACHE_DIR
grep MODEL_CACHE_DIR .env
sudo supervisorctl start gpu-inference
sudo tail -f /var/log/gpu-inference/out.log | grep "Pre-loading"
curl http://localhost:8002/cache-stats
```

### Ottenere i .pt / .pth

| Metodo | Comando |
|---|---|
| SCP | `scp -P <porta> modello.pt root@...:/workspace/inference/models_local/` |
| wget su Vast.ai | `cd $MODEL_CACHE_DIR && wget <url>` |
| Auto-download | `python -c "import ultralytics; ultralytics.YOLO('yolo11s.pt')"` |

---

**Last Updated**: 2026-06-03
**Version**: v2.0 (RabbitMQ RPC Integration)
**Maintainer**: Digital Future Development Team

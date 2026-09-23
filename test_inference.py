#!/usr/bin/env python
"""
Test diretto del GPU inference service con il modello v1.2.0.pt.

Non richiede Django, DB o Celery. Tu passi il crop_polygon come env var JSON.

Pre-requisiti:
  1. Il container GPU è in esecuzione:
       cd vast && docker-compose -f docker-compose.local.yml up --build -d
  2. v1.2.0.pt è in ./models_local/

Uso:
  cd vast
  python test_v1_inference.py

  # Con immagine reale:
  TEST_IMAGE=path/to/immagine_bovini.jpg python test_v1_inference.py

  # Con crop_polygon (JSON array):
  CROP_POLYGON_JSON='[[50,50],[1870,50],[1870,1035],[50,1035]]' TEST_IMAGE=image.jpg python test_v1_inference.py

  # Con detection classes specifiche:
  DETECTION_CLASSES="Calving Pose,Calf Appear" python test_v1_inference.py

  # Con confidence threshold diversa:
  CONFIDENCE=0.3 python test_v1_inference.py
"""

import os
import sys
import json
import base64
import time
import io
import requests

# ─── Configurazione ────────────────────────────────────────────────────────────
GPU_SERVICE_URL = os.environ.get("GPU_SERVICE_URL", "http://localhost:8002")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "yolo")  # yolo (.pt) or rfdetr (.pth)
MODEL_NAME = os.environ.get("MODEL_NAME", "v1.2.0.pt")  # can be .pt or .pth
TEST_IMAGE_PATH = os.environ.get("TEST_IMAGE", None)  # opzionale: immagine reale da disco
DETECTION_CLASSES_ENV = os.environ.get("DETECTION_CLASSES", "")  # es. "vitello,placenta"
CONFIDENCE = float(os.environ.get("CONFIDENCE", "0.3"))
CLASS_CONFIDENCE_OVERRIDES_JSON = os.environ.get("CLASS_CONFIDENCE_OVERRIDES_JSON", "{}")  # es. '{\"calf appears\":0.8,\"straight tail\":0.7}'
CROP_POLYGON_JSON_ENV = os.environ.get("CROP_POLYGON_JSON", "[[558, 28.166671752929688], [1338, 63.16667175292969], [1908, 529.1666717529297], [1910, 784.1666717529297], [5, 704.1666717529297], [19, 412.1666717529297]]")  # es. '[[50,50],[1870,50],[1870,1035],[50,1035]]'
TIMEOUT = 120  # secondi

# Auto-detect model type from file extension if MODEL_TYPE is not explicitly set
if MODEL_TYPE not in ("yolo", "rfdetr") and MODEL_NAME.endswith(".pth"):
    MODEL_TYPE = "rfdetr"
elif MODEL_TYPE not in ("yolo", "rfdetr"):
    MODEL_TYPE = "yolo"

MODELS_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "models_local")

OK = "✅"
FAIL = "❌"
WARN = "⚠️ "

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    mark = OK if condition else FAIL
    msg = f"  {mark} {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    if condition:
        passed += 1
    else:
        failed += 1
    return condition


def section(title: str):
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


def load_image() -> bytes:
    """
    Carica un'immagine di test da:
    1. TEST_IMAGE env var (immagine reale tua)
    2. image_124832.jpg nella stessa directory (se presente)
    3. Immagine sintetica 640x480 (fallback)
    """
    # 1. Immagine specificata dall'utente
    if TEST_IMAGE_PATH and os.path.exists(TEST_IMAGE_PATH):
        with open(TEST_IMAGE_PATH, "rb") as f:
            data = f.read()
        print(f"  📷 Immagine reale: {TEST_IMAGE_PATH} ({len(data) / 1024:.1f} KB)")
        return data

    # 2. Immagine di test del repo
    default_paths = [
        os.path.join(os.path.dirname(__file__), "image_091323.jpg"),
        os.path.join(os.path.dirname(__file__), "..", "backend", "image_091323.jpg"),
    ]
    for p in default_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                data = f.read()
            print(f"  📷 Immagine trovata: {p} ({len(data) / 1024:.1f} KB)")
            return data

    # 3. Immagine sintetica (PIL)
    print(f"  {WARN} Nessuna immagine reale trovata — uso immagine sintetica 640x480")
    print(f"       Per usarne una tua: TEST_IMAGE=/path/to/img.jpg python test_v1_inference.py")
    try:
        from PIL import Image as PILImage
        import numpy as np

        arr = np.zeros((480, 640, 3), dtype="uint8")
        arr[:] = [60, 80, 60]  # sfondo verde scuro (prateria)
        # Simula una sagoma animale
        arr[150:380, 100:540] = [180, 160, 120]  # corpo
        arr[80:160, 200:380] = [160, 140, 100]   # testa
        img = PILImage.fromarray(arr, "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        data = buf.getvalue()
        print(f"  📷 Immagine sintetica generata ({len(data) / 1024:.1f} KB)")
        return data
    except ImportError:
        # Minimal JPEG 1x1 come last resort
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1c"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\x0e\xff\xd9"
        )


def extract_model_classes() -> list:
    """
    Estrae tutte le classi disponibili dal modello.
    
    Per YOLO (.pt): carica il modello e legge model.names
    Per RF-DETR (.pth): restituisce l'ordine nativo noto (6 classi calving)
    
    Returns:
        Lista di nomi di classi
    """
    model_path = os.path.join(MODELS_LOCAL_DIR, MODEL_NAME)

    # RF-DETR: le classi native sono note (non estraibili staticamente dal .pth)
    if MODEL_TYPE == "rfdetr" or MODEL_NAME.endswith(".pth"):
        classes = [
            "Amniotic sac",
            "Calf appears",
            "Calving pose detected",
            "Straight tail",
            "Cow head",
            "Calf legs appear first",
        ]
        print(f"  📦 RF-DETR: 6 classi native (non estraibili staticamente)")
        return classes

    try:
        from ultralytics import YOLO
        
        if not os.path.exists(model_path):
            print(f"  {WARN} Modello non trovato per estrarre classi: {model_path}")
            return []
        
        print(f"  📦 Carico {MODEL_NAME} per estrarre le classi disponibili...")
        model = YOLO(model_path)
        
        # model.names è un dict {class_id: class_name}
        classes = list(model.names.values())
        print(f"  ✅ Classi nel modello: {classes}")
        return classes
    except Exception as e:
        print(f"  {WARN} Impossibile estrarre classi: {e}")
        return []


def build_model_config(detection_classes: list = None) -> dict:
    """
    Costruisce il model_config per v1.2.0.pt.
    
    Args:
        detection_classes: Lista di classi da filtrare. Se None o vuota, accetta tutte.
    """
    if detection_classes is None:
        # Leggi da env var se specificata
        detection_classes = (
            [c.strip() for c in DETECTION_CLASSES_ENV.split(",") if c.strip()]
            if DETECTION_CLASSES_ENV
            else []
        )

    # Auto-detect input_size: RF-DETR natural size is 560, YOLO v1.2.0 uses 1536
    default_input_size = 560 if MODEL_TYPE == "rfdetr" else 1536
    input_size = int(os.environ.get("INPUT_SIZE", str(default_input_size)))

    # Parse class_confidence_overrides from env JSON
    try:
        overrides = json.loads(CLASS_CONFIDENCE_OVERRIDES_JSON)
        if not isinstance(overrides, dict):
            overrides = {}
    except (json.JSONDecodeError, TypeError):
        overrides = {}

    # Derive model_id and model_name from MODEL_NAME
    model_base = os.path.splitext(MODEL_NAME)[0]  # e.g. "rf_v0.0.1" or "v1.2.0"
    model_id = os.environ.get("MODEL_ID", model_base)
    model_name = os.environ.get("MODEL_DISPLAY_NAME", f"Test Model ({model_base})")

    return {
        "model_id": model_id,
        "model_url_s3_or_path": MODEL_NAME,  # basename → risolto contro MODEL_CACHE_DIR
        "model_name": model_name,
        "detection_classes": detection_classes,
        "is_greyscale": False,
        "confidence_threshold": CONFIDENCE,
        "input_size": input_size,
        "class_confidence_overrides": overrides,  # keys must match detection_classes (native names!)
        "class_display_names": {},
        "model_blur": False,
    }


def build_crop_polygon() -> list:
    """
    Costruisce il crop_polygon da:
    1. CROP_POLYGON_JSON env var (es. '[[50,50],[1870,50],[1870,1035],[50,1035]]')
    2. Fallback: rettangolo full image (border 50px, 1920x1085)
    
    Formato: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] (clockwise rectangle)
    """
    # Leggi da env var se specificato
    if CROP_POLYGON_JSON_ENV:
        try:
            coords = json.loads(CROP_POLYGON_JSON_ENV)
            print(f"  ✅ Crop polygon letto da CROP_POLYGON_JSON env var: {len(coords)} vertici")
            return coords
        except Exception as e:
            print(f"  {WARN} Impossibile parsare CROP_POLYGON_JSON: {e}")
    
    # Fallback: rettangolo full image con border 50px (1920x1085)
    print(f"  💡 Crop polygon: default (full image, border 50px)")
    return [[50, 50], [1870, 50], [1870, 1035], [50, 1035]]


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 64}")
model_label = f"{MODEL_NAME} ({MODEL_TYPE})"
print(f"  GPU INFERENCE — TEST DIRETTO con {model_label}")
print(f"{'═' * 64}")
print(f"  Service URL:  {GPU_SERVICE_URL}")
print(f"  Modello:      {MODEL_NAME}")
print(f"  Model type:   {MODEL_TYPE}")
print(f"  Confidence:   {CONFIDENCE}")
detection_classes_display = DETECTION_CLASSES_ENV or "(tutte le classi)"
print(f"  Classes:      {detection_classes_display}")
print(f"  Overrides:    {CLASS_CONFIDENCE_OVERRIDES_JSON}")
polygon_source = "CROP_POLYGON_JSON env var" if CROP_POLYGON_JSON_ENV else "default (full image)"
print(f"  Crop Polygon: {polygon_source}")
print(f"  Models dir:   {MODELS_LOCAL_DIR}")


# ─── Preflight: verifica che il modello esista in models_local/ ────────────────
section("PREFLIGHT — Verifica presenza modello")

model_file = os.path.join(MODELS_LOCAL_DIR, MODEL_NAME)
model_exists = os.path.exists(model_file)

if model_exists:
    size_mb = os.path.getsize(model_file) / 1e6
    check(f"{MODEL_NAME} presente in models_local/", True, f"{size_mb:.1f} MB")
else:
    check(f"{MODEL_NAME} presente in models_local/", False,
          f"non trovato in {MODELS_LOCAL_DIR}")
    print(f"\n  {FAIL} Il modello non è nel mount directory del container.")
    print(f"       Copia {MODEL_NAME} in: {MODELS_LOCAL_DIR}")
    print(f"       Poi riavvia il container.")
    sys.exit(1)

# Verifica crop_polygon
crop_polygon_to_use = build_crop_polygon()
section(f"PREFLIGHT — Configurazione crop_polygon")
print(f"  📐 Vertici polygon: {len(crop_polygon_to_use)}")

# ─── TEST 1: Health check ──────────────────────────────────────────────────────
section("TEST 1 — GET /health")

try:
    r = requests.get(f"{GPU_SERVICE_URL}/health", timeout=10)
    check("HTTP 200", r.status_code == 200, f"status={r.status_code}")

    data = r.json()
    check("status == 'ok'", data.get("status") == "ok", data.get("status", "?"))
    device = data.get("device", "?")
    device_name = data.get("device_name", "?")
    check("device rilevato", device in ("cuda", "cpu"), device)

    print(f"\n  Device:      {device} ({device_name})")
    if device == "cpu":
        print(f"  {WARN} Stai usando CPU — l'inferenza sarà più lenta ma funziona.")
    else:
        print(f"  🎮 GPU disponibile — inferenza accelerata.")

    cache = data.get("cache", {})
    print(f"  Cache:       {cache.get('cache_size', 0)}/{cache.get('max_size', 10)} modelli in memoria")

except requests.exceptions.ConnectionError:
    check("Servizio raggiungibile", False,
          f"Connection refused su {GPU_SERVICE_URL}")
    print(f"\n  Il container non è in esecuzione o non è sulla porta 8002.")
    print(f"  Avvialo con:  cd vast && docker-compose -f docker-compose.local.yml up --build -d")
    sys.exit(1)
except Exception as e:
    check("Health check", False, str(e))
    sys.exit(1)


# ─── TEST 2: Estrai tutte le classi dal modello ─────────────────────────────────
section(f"TEST 2 — Estrai classi da {MODEL_NAME}")

model_all_classes = extract_model_classes()
if model_all_classes:
    print(f"\n  Le classi disponibili nel modello sono:")
    for i, cls in enumerate(model_all_classes, 1):
        print(f"    [{i}] {cls}")
else:
    print(f"  {WARN} Impossibile estrarre le classi dal modello.")
    print(f"       Il test continuerà senza informazioni sulle classi.")


# ─── TEST 2.5: Inferenza Senza filtro (tutte le classi) ────────────────────────
section(f"TEST 2.5 — POST /infer con {MODEL_NAME} (TUTTE LE CLASSI)")

image_bytes = load_image()
model_config = build_model_config(detection_classes=[])  # tutte le classi

print(f"\n  Config inviata:")
print(f"    model_id:          {model_config['model_id']}")
print(f"    model_name:        {model_config['model_name']}")
print(f"    confidence:        {model_config['confidence_threshold']}")
print(f"    input_size:        {model_config['input_size']}")
print(f"    detection_classes: {model_config['detection_classes'] or '(tutte)'}")
print(f"    image size:        {len(image_bytes) / 1024:.1f} KB")

try:
    t0 = time.time()
    r = requests.post(
        f"{GPU_SERVICE_URL}/infer",
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        data={
            "model_configs_json": json.dumps([model_config]),
            "crop_polygon_json": json.dumps(crop_polygon_to_use),
        },
        timeout=TIMEOUT,
    )
    elapsed = time.time() - t0

    check("HTTP 200", r.status_code == 200, f"status={r.status_code}")

    if r.status_code != 200:
        print(f"\n  Risposta errore: {r.text[:400]}")
        sys.exit(1)

    data = r.json()

    # Verifica struttura risposta
    check("event_detected presente", "event_detected" in data)
    check("detections è lista", isinstance(data.get("detections"), list))
    check("device presente", "device" in data, data.get("device"))
    check("models_run presente", "models_run" in data)
    check("nessun errore critico", data.get("error") is None, str(data.get("error")))

    # Verifica che il modello sia stato effettivamente caricato ed eseguito
    models_run = data.get("models_run", [])
    model_was_run = any(MODEL_NAME in str(m) or MODEL_NAME.replace(".pt", "").replace(".pth", "") in str(m).lower()
                        for m in models_run)
    check(f"{MODEL_NAME} eseguito", model_was_run, str(models_run))

    detections = data.get("detections", [])
    event_detected = data.get("event_detected", False)

    print(f"\n  ── Risultato Inferenza ──────────────────────────────────")
    print(f"  event_detected: {event_detected}")
    print(f"  detections:     {len(detections)}")
    print(f"  device:         {data.get('device')} ({data.get('device_name', '?')})")
    print(f"  tempo:          {elapsed:.2f}s")
    print(f"  models_run:     {models_run}")

    # Raccogli tutte le classi uniche trovate (per il report finale)
    detected_classes_unique = set()
    detected_classes_count = {}
    
    if detections:
        print(f"\n  ── Detections trovate ───────────────────────────────────")
        for i, det in enumerate(detections):
            bbox = det.get("bbox", [])
            conf = det.get("conf", 0)
            cls = det.get("class", "?")
            raw_cls = det.get("raw_class", cls)
            model_name = det.get("yolo_model_name", "?")
            print(f"  [{i+1}] class={cls!r} (raw={raw_cls!r})  conf={conf:.3f}  "
                  f"bbox={bbox}  model={model_name!r}")
            detected_classes_unique.add(cls)
            detected_classes_count[cls] = detected_classes_count.get(cls, 0) + 1
    else:
        print(f"\n  {WARN} Nessuna detection sull'immagine di test.")
        if not TEST_IMAGE_PATH:
            print(f"       Normale se hai usato un'immagine sintetica.")
            print(f"       Per testare con una tua immagine bovini:")
            print(f"       TEST_IMAGE=/path/to/img.jpg python test_v1_inference.py")
        else:
            print(f"       Il modello non ha trovato nulla nell'immagine fornita.")
            print(f"       Prova ad abbassare la confidence:  CONFIDENCE=0.1 python test_v1_inference.py")

except requests.exceptions.Timeout:
    check("Inferenza completata", False, f"Timeout dopo {TIMEOUT}s")
    sys.exit(1)
except Exception as e:
    check("Inferenza", False, str(e))
    import traceback; traceback.print_exc()
    sys.exit(1)


# ─── TEST 2.6: Class confidence overrides ─────────────────────────────────────
section(f"TEST 2.6 — POST /infer con class_confidence_overrides")
if CLASS_CONFIDENCE_OVERRIDES_JSON and CLASS_CONFIDENCE_OVERRIDES_JSON != "{}":
    try:
        overrides_test = json.loads(CLASS_CONFIDENCE_OVERRIDES_JSON)
    except (json.JSONDecodeError, TypeError):
        overrides_test = {}

    if overrides_test:
        overrides_config = build_model_config(detection_classes=[])
        # Verify overrides are present
        actual_overrides = overrides_config.get("class_confidence_overrides", {})
        check(
            "class_confidence_overrides propagate",
            actual_overrides == overrides_test,
            f"expected={overrides_test}, got={actual_overrides}",
        )

        # Run inference with overrides and verify no error
        try:
            t0 = time.time()
            r_ov = requests.post(
                f"{GPU_SERVICE_URL}/infer",
                files={"image": ("test_override.jpg", image_bytes, "image/jpeg")},
                data={
                    "model_configs_json": json.dumps([overrides_config]),
                    "crop_polygon_json": json.dumps(crop_polygon_to_use),
                },
                timeout=TIMEOUT,
            )
            elapsed_ov = time.time() - t0
            check("HTTP 200 con overrides", r_ov.status_code == 200,
                  f"status={r_ov.status_code}, tempo={elapsed_ov:.2f}s")

            data_ov = r_ov.json()
            dets_ov = data_ov.get("detections", [])
            print(f"\n  ── Risultato Inferenza con Overrides ───────────────────────")
            print(f"  detections:     {len(dets_ov)}")
            print(f"  overrides keys: {list(overrides_test.keys())}")

            for i, det in enumerate(dets_ov):
                cls = det.get("class", "?")
                conf = det.get("conf", 0)
                raw_cls = det.get("raw_class", cls)
                print(f"  [{i+1}] class={cls!r} (raw={raw_cls!r})  conf={conf:.3f}")
        except Exception as e:
            check("Inferenza con overrides", False, str(e))
    else:
        print(f"  ⏭️  Nessun override configurato (CLASS_CONFIDENCE_OVERRIDES_JSON vuoto).")
else:
    print(f"  ⏭️  Nessun override configurato. Per testare:")
    print(f"       CLASS_CONFIDENCE_OVERRIDES_JSON='{{\"calf appears\":0.8,\"straight tail\":0.7}}' python test_inference.py")


# ─── TEST 3: Immagine processata (base64) ─────────────────────────────────────
section("TEST 3 — Verifica processed_image_b64")

b64 = data.get("processed_image_b64")
check("processed_image_b64 presente", b64 is not None and b64 != "")

if b64:
    try:
        img_bytes = base64.b64decode(b64)
        is_jpeg = img_bytes[:2] == b"\xff\xd8"
        check("è un JPEG valido", is_jpeg, f"magic={img_bytes[:2].hex()}")
        check("dimensione > 1 KB", len(img_bytes) > 1024, f"{len(img_bytes) / 1024:.1f} KB")

        # Salva per ispezione visiva
        out_path = os.path.join(
            os.path.dirname(__file__), "test_output.jpg"
        )
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"\n  📷 Immagine processata salvata in:")
        print(f"     {out_path}")
        print(f"     Aprila per vedere i bounding box disegnati dal servizio.")
    except Exception as e:
        check("Decode base64", False, str(e))
else:
    print(f"  {WARN} Immagine processata vuota (atteso se nessuna detection).")


# ─── TEST 4: Cache — secondo invio deve essere più veloce ─────────────────────
section("TEST 4 — Cache modello (secondo invio deve essere più veloce)")

try:
    t0 = time.time()
    r2 = requests.post(
        f"{GPU_SERVICE_URL}/infer",
        files={"image": ("test2.jpg", image_bytes, "image/jpeg")},
        data={
            "model_configs_json": json.dumps([model_config]),
            "crop_polygon_json": json.dumps(crop_polygon_to_use),
        },
        timeout=TIMEOUT,
    )
    elapsed2 = time.time() - t0

    check("Secondo invio HTTP 200", r2.status_code == 200)
    speedup = elapsed / elapsed2 if elapsed2 > 0 else 0
    check(
        "Cache hit (secondo invio più veloce o simile)",
        elapsed2 <= elapsed * 1.5,  # tolleriamo +50%
        f"1°: {elapsed:.2f}s  2°: {elapsed2:.2f}s  speedup: {speedup:.1f}x"
    )
    print(f"  1° invio: {elapsed:.2f}s (include caricamento modello)")
    print(f"  2° invio: {elapsed2:.2f}s (da cache)")

except Exception as e:
    check("Secondo invio", False, str(e))


# ─── TEST 3: Inferenza CON tutte le classi specificate ──────────────────────
if model_all_classes:
    section(f"TEST 3 — Testa TUTTE le classi trovate: {', '.join(model_all_classes)}")
    
    # Riinvia lo stesso payload ma specificando tutte le classi
    model_config_all = build_model_config(detection_classes=model_all_classes)
    print(f"\n  Config con tutte le classi:")
    print(f"    detection_classes: {model_config_all['detection_classes']}")
    
    try:
        t0 = time.time()
        r3 = requests.post(
            f"{GPU_SERVICE_URL}/infer",
            files={"image": ("test3.jpg", image_bytes, "image/jpeg")},
            data={
                "model_configs_json": json.dumps([model_config_all]),
                "crop_polygon_json": json.dumps(crop_polygon_to_use),
            },
            timeout=TIMEOUT,
        )
        elapsed3 = time.time() - t0
        
        check("HTTP 200", r3.status_code == 200)
        data3 = r3.json()
        detections3 = data3.get("detections", [])
        
        print(f"\n  Risultato:")
        print(f"    detections:     {len(detections3)}")
        print(f"    tempo:          {elapsed3:.2f}s")
        
        if detections3:
            detected_classes_unique.update([d.get("class", "?") for d in detections3])
            detected_classes_count_all = {}
            for det in detections3:
                cls = det.get("class", "?")
                detected_classes_count_all[cls] = detected_classes_count_all.get(cls, 0) + 1
            
            print(f"\n  Classi rilevate in questo test:")
            for cls, count in sorted(detected_classes_count_all.items()):
                print(f"    {cls}: {count} rilevamenti")
    except Exception as e:
        check("Test con tutte le classi", False, str(e))


# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"\n{'═' * 64}")
print(f"  SUMMARY")
print(f"{'═' * 64}")
total = passed + failed
print(f"  Risultato test: {passed}/{total} superati")
print()

if model_all_classes:
    print(f"  Classi disponibili nel modello: {model_all_classes}")
    if detected_classes_unique:
        print(f"  Classi rilevate nelle immagini: {sorted(detected_classes_unique)}")
        print(f"  Dettagli rilevamenti: {detected_classes_count}")
    else:
        print(f"  ⚠️ Nessuna classe rilevata nelle immagini di test.")
print()

if failed == 0:
    print(f"  {OK} Tutto OK — il servizio GPU con {MODEL_NAME} funziona correttamente.")
else:
    print(f"  {FAIL} {failed} test falliti — controlla i dettagli sopra.")

print()

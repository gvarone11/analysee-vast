#!/usr/bin/env bash
set -e

APP_DIR="/workspace/inference"

echo "======================================"
echo "🚀 Vast.ai GPU Inference on-start"
echo "======================================"
echo "APP_DIR=$APP_DIR"

if [ ! -d "$APP_DIR" ]; then
  echo "❌ ERRORE: cartella $APP_DIR non trovata"
  exit 1
fi

cd "$APP_DIR"

if [ ! -f "$APP_DIR/run_local.py" ]; then
  echo "❌ ERRORE: run_local.py non trovato in $APP_DIR"
  exit 1
fi

echo "📦 Installazione dipendenze di sistema..."

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  supervisor \
  python3-venv \
  python3-pip \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  ffmpeg

echo "🐍 Preparazione virtualenv..."

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

echo "📝 Configurazione Supervisor..."

mkdir -p /var/log/gpu-inference

cat > /etc/supervisor/conf.d/gpu-inference.conf <<'EOF2'
[program:gpu-inference]
directory=/workspace/inference
command=/bin/bash -lc 'cd /workspace/inference && exec ./.venv/bin/python ./run_local.py'
autostart=true
autorestart=true
startsecs=10
startretries=999
stopasgroup=true
killasgroup=true
stdout_logfile=/var/log/gpu-inference/out.log
stderr_logfile=/var/log/gpu-inference/err.log
stdout_logfile_maxbytes=20MB
stderr_logfile_maxbytes=20MB
stdout_logfile_backups=5
stderr_logfile_backups=5
environment=PORT="8002",MODEL_CACHE_DIR="/workspace/inference/models_local",RABBITMQ_WORKER_THREADS="2"
EOF2

echo "🔁 Avvio/ricarica Supervisor..."

if pgrep -x supervisord >/dev/null; then
  echo "✅ supervisord già attivo"
else
  echo "▶️ avvio supervisord"
  supervisord -c /etc/supervisor/supervisord.conf
fi

supervisorctl reread
supervisorctl update
supervisorctl restart gpu-inference || supervisorctl start gpu-inference

echo ""
echo "📌 Stato servizio:"
supervisorctl status

echo ""
echo "🌐 Test porta 8002:"
ss -tulpn | grep 8002 || true

echo ""
echo "✅ GPU inference service started with supervisor"
echo "Log live:"
echo "tail -f /var/log/gpu-inference/out.log /var/log/gpu-inference/err.log"
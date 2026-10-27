#!/usr/bin/env bash
# One-time setup ON the Graviton box (idempotent). Run as ubuntu:
#   bash ~/vision/cloud/deploy/box_setup.sh
#
# - ~/venv-cool-api: Python 3.12 venv whose cv2 is COOL (/opt/cool, 4.14.0-pre)
#   via a .pth; FastAPI/uvicorn/boto3 only (no opencv-python, no onnxruntime).
# - Self-signed TLS cert (~/sg-tls), CN screen-guardian-l2. The Mac pins it.
# - systemd unit sg-l2: uvicorn cloud.api:app on :8443 (TLS), COOL libs.
#   The bearer token comes from Secrets Manager via the instance role.
set -euo pipefail

V=/home/ubuntu/venv-cool-api
if [ ! -x "$V/bin/python" ]; then
  python3.12 -m venv "$V"
  "$V/bin/pip" install -q --upgrade pip
  "$V/bin/pip" install -q "numpy==2.4.4" fastapi uvicorn pydantic boto3 httpx pytest
fi
echo "/opt/cool/python_3.12/site-packages/cv2/python-3.12" > "$V/lib/python3.12/site-packages/cool_cv2.pth"

mkdir -p /home/ubuntu/sg-tls
if [ ! -f /home/ubuntu/sg-tls/cert.pem ]; then
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 365 \
    -subj "/CN=screen-guardian-l2" -addext "subjectAltName=DNS:screen-guardian-l2" \
    -keyout /home/ubuntu/sg-tls/key.pem -out /home/ubuntu/sg-tls/cert.pem
  chmod 600 /home/ubuntu/sg-tls/key.pem
fi

sudo tee /etc/systemd/system/sg-l2.service >/dev/null <<'EOF'
[Unit]
Description=Screen Guardian L2 second look (COOL OpenCV on Graviton)
After=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/vision
Environment=PYTHONPATH=/home/ubuntu/vision
Environment=LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib
Environment=AWS_REGION=us-east-1
Environment=SCREEN_GUARDIAN_RUNTIME=graviton-cool-4.14
Environment=SG_PRELOAD=1
Environment=SG_DETECTOR=static
Environment=SG_STATIC_BATCH=4
Environment=SG_METRICS_SERVICE=l2-second-look
ExecStart=/home/ubuntu/venv-cool-api/bin/python -m uvicorn cloud.api:app --host 0.0.0.0 --port 8443 --ssl-keyfile /home/ubuntu/sg-tls/key.pem --ssl-certfile /home/ubuntu/sg-tls/cert.pem --no-access-log --timeout-keep-alive 75
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now sg-l2
sleep 5
systemctl is-active sg-l2
curl -sk https://127.0.0.1:8443/health | head -c 400; echo

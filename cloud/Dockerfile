# Screen Guardian evaluation API — linux/arm64 (Graviton).
#
# Stock opencv-python 5 on Ubuntu arm64. NOT the serving path: the L2 API
# runs on the COOL AMI box (<box-instance-id>) under COOL OpenCV 4.14.0-pre
# with the static-shape 320n copies (cloud/cool_model.py); see
# cloud/deploy/runbook.md. Kept only as a portable stock-OpenCV image for
# comparisons. Build (Docker daemon required), from vision/:
#   docker buildx build --platform linux/arm64 -f cloud/Dockerfile -t screen-guardian-api:stock-opencv5 .

FROM --platform=linux/arm64 python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.lock ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY vision_core ./vision_core
COPY agent ./agent
COPY policies ./policies
COPY fixtures/safe ./fixtures/safe
COPY cloud ./cloud

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "cloud.api:app", "--host", "0.0.0.0", "--port", "8080"]

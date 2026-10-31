# vloc: WeVisOne-VL native engine (transformers backend, full 11 tasks incl. mask/alpha) + sync HTTP API.
# Weights are NOT baked in: mount the checkpoint dir at /models and set WEVISONE_MODEL.
# Platforms: linux/amd64 and linux/arm64 (PyTorch cu130 index ships both manylinux wheels).
# CI (.github/workflows/docker.yml) builds both natively on v* tags and pushes ghcr.io/iintothewind/vloc.
#
# Build (context = repo root):
#   docker build -f docker/Dockerfile -t vloc:dev .
#   docker build -f docker/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t vloc:cu128 .
# Run (needs NVIDIA driver >= 580 for cu130):
#   docker run -d --gpus all -p 8000:8000 -v /path/models:/models:ro ghcr.io/iintothewind/vloc:latest
# Manual GPU smoke:
#   docker run --rm --gpus all -v /path/models:/models:ro -v /path/images:/smoke:ro \
#     ghcr.io/iintothewind/vloc:latest python /app/smoke.py
FROM python:3.12-slim-bookworm

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG WEVISONE_COMMIT=9a6bcd85404505df0d05cfb5d97512ccacf35dee

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    WEVISONE_MODEL=/models/WeVisOne-VL-2B \
    VLOC_ARTIFACT_DIR=/data/artifacts

RUN apt-get update \
    && apt-get install -y --no-install-recommends git fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision first from the CUDA index; their wheels bundle CUDA + cuDNN,
# so the host only needs the NVIDIA driver + nvidia-container-toolkit.
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url ${TORCH_INDEX}

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Engine package only; demo/eval deps (gradio, matplotlib, supervision, scipy, pycocotools) are skipped.
RUN pip install --no-deps "git+https://github.com/Tencent/WeVisOne-VL.git@${WEVISONE_COMMIT}" \
    && apt-get purge -y git && apt-get autoremove -y

COPY pyproject.toml /src/pyproject.toml
COPY vloc /src/vloc
RUN pip install /src && rm -rf /src

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data/artifacts \
    && chown -R app /data
VOLUME /data
USER app
WORKDIR /app
COPY --chown=app docker/smoke.py /app/smoke.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health',timeout=4).status==200 else 1)"
CMD ["uvicorn", "vloc.app:build_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

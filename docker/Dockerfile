# WeVisOne-VL native engine image (transformers backend, full 11 tasks incl. mask/alpha).
# Weights are NOT baked in: mount the checkpoint dir at /models and set WEVISONE_MODEL.
# Platforms: linux/amd64 and linux/arm64 (PyTorch cu130 index ships both manylinux wheels).
# CI (.github/workflows/docker.yml) builds both natively and pushes ghcr.io/iintothewind/vloc.
#
# Local build:
#   docker build -t vloc:dev docker/
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t vloc:cu128 docker/
# Smoke (needs NVIDIA driver >= 580 for cu130):
#   docker run --rm --gpus all -v /path/models:/models:ro -v /path/images:/smoke:ro \
#     -e WEVISONE_MODEL=/models/WeVisOne-VL-2B ghcr.io/iintothewind/vloc:latest
FROM python:3.12-slim-bookworm

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG WEVISONE_COMMIT=9a6bcd85404505df0d05cfb5d97512ccacf35dee

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    WEVISONE_MODEL=/models/WeVisOne-VL-2B

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision first from the CUDA index; their wheels bundle CUDA + cuDNN,
# so the host only needs the NVIDIA driver + nvidia-container-toolkit.
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url ${TORCH_INDEX}

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Engine package only; demo/eval deps (gradio, matplotlib, supervision, scipy, pycocotools) are skipped.
RUN pip install --no-deps "git+https://github.com/Tencent/WeVisOne-VL.git@${WEVISONE_COMMIT}" \
    && apt-get purge -y git && apt-get autoremove -y

RUN useradd --create-home --uid 10001 app
USER app
WORKDIR /app
COPY --chown=app smoke.py /app/smoke.py

EXPOSE 8000
CMD ["python", "/app/smoke.py"]

ARG BASE_IMAGE=533267258830.dkr.ecr.ap-south-1.amazonaws.com/slime@sha256:7a9037083c5366bad593af48a1fe106b208e0c73cafb2f2cc2c4b881e7a19f27
FROM ${BASE_IMAGE}

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        python3-venv && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-deepseek-ocr.txt /tmp/requirements-deepseek-ocr.txt
COPY requirements-vllm.txt /tmp/requirements-vllm.txt

# vLLM 0.11.2 pins PyTorch 2.9.0 and CUDA 12.8 packages. Its wheel is
# compatible with the base image's newer patch release, so retain the verified
# PyTorch 2.9.1+cu129 stack and install only the packages absent from the image.
RUN python3 -m pip install --no-cache-dir --no-deps \
        -r /tmp/requirements-vllm.txt

# DeepSeek-OCR requires an older Transformers release than the main jobs.
# Reuse the base image's CUDA-enabled PyTorch while isolating its Python deps.
RUN python3 -m venv --system-site-packages /opt/vtc-deepseek-ocr && \
    /opt/vtc-deepseek-ocr/bin/python -m pip install --no-cache-dir \
        -r /tmp/requirements-deepseek-ocr.txt

ENV VTC_DSOCR_NATIVE_PYTHON=/opt/vtc-deepseek-ocr/bin/python \
    HF_HOME=/cache/huggingface \
    VTC_DATA_DIR=/runs/data \
    VTC_RESULTS_DIR=/runs/results \
    VTC_CHECKPOINT_DIR=/runs/checkpoints \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/experiments/vtc_memory_validation
ENTRYPOINT []
CMD ["/bin/bash"]

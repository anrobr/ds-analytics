# .devcontainer/Dockerfile
ARG CUDA_TAG=13.0.1-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

# ---- System deps + Python 3.13 (deadsnakes) ----
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git curl gnupg2 software-properties-common build-essential pkg-config cmake \
    libssl-dev libffi-dev \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.13 python3.13-dev python3.13-venv \
 && /usr/bin/python3.13 -m ensurepip --upgrade \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 2

# ---- Non-root user & workspace ----
RUN useradd -ms /bin/bash vscode && adduser vscode sudo && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER vscode
WORKDIR /workspaces

# ---- uv (pkg manager) & venv ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/home/vscode/.local/bin:$PATH

# Make uv's project env (.venv) and $VIRTUAL_ENV agree
ENV VIRTUAL_ENV=.venv
ENV PATH="/workspaces/.venv/bin:$PATH"

# ---- CUDA env ----
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# ---- Copy metadata (NO hard requirement on uv.lock) ----
COPY --chown=vscode:vscode pyproject.toml .pre-commit-config.yaml /workspaces/
# If uv.lock exists in your repo, copy it; otherwise keep this commented to avoid build failures.
# COPY --chown=vscode:vscode uv.lock /workspaces/uv.lock

# ---- Resolve deps (use lock if present, otherwise generate) ----
RUN uv venv \
 && if [ -f uv.lock ] && [ -s uv.lock ]; then \
        echo 'Using existing uv.lock'; \
        uv sync --frozen; \
    else \
        echo 'No uv.lock found or it is empty; generating a fresh lock...'; \
        rm -f uv.lock; \
        uv lock && uv sync; \
    fi

# Seed/upgrade pip in the venv
RUN python -m ensurepip --upgrade && python -m pip install --upgrade pip

# ---- PyTorch (CUDA 13 nightly cu130; switch to stable cu130 when GA) ----
RUN python -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu130 \
    torch torchvision torchaudio

# ---- RAPIDS (CUDA 13, try 25.12 - early release, fallback to 25.10 if not yet available) ----
RUN set -eux; \
if ! python -m pip install --extra-index-url=https://pypi.nvidia.com \
    cudf-cu13==25.12.* \
    dask-cudf-cu13==25.12.* \
    cuml-cu13==25.12.* \
    cugraph-cu13==25.12.* \
    rmm-cu13==25.12.* \
    cupy-cuda13x; then \
    echo "RAPIDS 25.12.* not yet available on NVIDIA PyPI — falling back to 25.10.*"; \
    python -m pip install --extra-index-url=https://pypi.nvidia.com \
    cudf-cu13==25.10.* \
    dask-cudf-cu13==25.10.* \
    cuml-cu13==25.10.* \
    cugraph-cu13==25.10.* \
    rmm-cu13==25.10.* \
    cupy-cuda13x; \
fi

# ---- pre-commit ----
RUN python -m pip install pre-commit

# .devcontainer/Dockerfile
ARG CUDA_TAG=13.0.1-cudnn-runtime-ubuntu22.04
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
ENV VIRTUAL_ENV=/workspaces/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ---- CUDA env ----
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# ---- Copy metadata (NO hard requirement on uv.lock) ----
COPY --chown=vscode:vscode pyproject.toml .pre-commit-config.yaml /workspaces/
# If uv.lock exists in your repo, copy it; otherwise this line is commented out to avoid build failures.
# Uncomment the next line after you commit uv.lock:
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

# Seed pip inside the venv (Py3.13 venvs can be created without pip)
RUN python -m ensurepip --upgrade

# ---- PyTorch (CUDA 13 nightly cu130; switch to stable cu130 when GA) ----
RUN python -m pip install --upgrade pip \
 && python -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu130 \
    torch torchvision torchaudio

# ---- pre-commit (use interpreter-qualified pip) ----
RUN python -m pip install pre-commit

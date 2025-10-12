# ds-analytics

Lightweight data-science / ML environment examples and sanity checks.

This repository provides a small, reproducible workspace for data science and
machine learning experiments using a modern Python toolchain (pyproject.toml).
It includes a basic test, a GPU sanity notebook, and convenient Makefile
targets to format, lint and run tests.

## What this repo contains

- `pyproject.toml` — project metadata and dependency list used for development.
- `Makefile` — convenience targets: `fmt`, `lint`, `test`, and `gpu-check`.
- `notebooks/00_sanity_gpu.ipynb` — small notebook to validate GPU / CUDA setup.
- `tests/test_sanity.py` — a tiny unit test that verifies numpy is working.
- `.devcontainer/` — VS Code devcontainer configuration for reproducible dev env.

The project is intentionally minimal so it can be used as a starting point or
reference for onboarding, demos, and CI sanity checks.

## Quickstart — develop using the VS Code dev container (recommended)

This repository is intended to be developed inside the included VS Code Dev
Container so everyone works in a consistent environment. The devcontainer
bundles the Python version and development tools declared in `pyproject.toml`.

Prerequisites on your host machine

- VS Code (latest stable) with the Remote - Containers / Dev Containers extension
- Docker (or another container runtime compatible with VS Code Dev Containers)
- If you plan to run GPU workloads from the container: a compatible NVIDIA GPU,
	matching NVIDIA driver and CUDA toolkit installed on the host. This project
	targets CUDA 13.x (see `pyproject.toml` for library requirements); ensure
	your driver supports the CUDA version you need.
- NVIDIA Container Toolkit (nvidia-docker / nvidia-container-toolkit) if you
	want to expose GPUs to the container. Configure Docker to allow GPU access
	(for example via the `--gpus` flag or via your devcontainer.json).

Quick host checks (examples)

```bash
# Check NVIDIA driver version
nvidia-smi

# Check host CUDA toolkit (if installed)
nvcc --version

# Check nvidia-container-toolkit installation
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

If the GPU checks fail, verify your host drivers and the NVIDIA container
toolkit installation. The exact driver/CUDA pairing depends on your GPU and
the CUDA runtime used by the container images.

Tip: match host driver/toolkit to the container's CUDA version

The devcontainer image references a specific CUDA base image and GPU
settings in `.devcontainer/devcontainer.json` and the Dockerfile. Current
configuration (see `.devcontainer/devcontainer.json`) uses:

- CUDA image tag: `nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04`
- Docker runArgs: `--gpus all` (the container is configured to request all
	GPUs from the host)

Make sure your host NVIDIA driver and the nvidia-container-toolkit support
CUDA 13.x (the container installs PyTorch from the cu130 channel). If you
need a different CUDA version, edit the `CUDA_TAG` arg in
`.devcontainer/devcontainer.json` or the Dockerfile accordingly, and make
sure the host drivers/toolkit match.

How to open the project in the Dev Container (VS Code)

1. Open the repository folder in VS Code.
2. Press F1 and choose `Dev Containers: Reopen in Container` (or click the
	 green status button in the lower-left and choose the same action).
3. VS Code will build the container image (the first run can take a few
	 minutes). It will then open a new window connected to the container.

What the container provides

- A reproducible Python environment matching `pyproject.toml`.
- Common dev tools and extensions pre-installed (pre-commit, ruff, pytest,
	etc.), if configured in `.devcontainer`.
- Access to GPU/CUDA if your host has a compatible GPU and Docker is configured
	to expose it to the container (see notes below).

Running tests, linters and pre-commit inside the container

Once the container is open, use the integrated terminal (which runs inside
the container) to run the standard Make targets:

```bash
make pre-commit   # run pre-commit hooks on all files
make test         # run the test suite (runs inside the container)
make gpu-check    # quick GPU availability check (container must expose GPU)
```

GPU notes

- To run GPU workloads from inside the container, ensure Docker is configured
	to expose the GPU device(s) (for example using `--gpus` or the NVIDIA
	container toolkit). The devcontainer configuration may already include GPU
	support; if not, update your Docker/Dev Container settings to enable GPU
	passthrough.

Troubleshooting

- If the container build fails, check the VS Code Dev Containers output panel
	for the build logs. Common issues are missing permissions for Docker or a
	network/proxy preventing package downloads.
- If `pre-commit` or linters are missing, open the integrated terminal and run
	`pip install -e .[dev]` inside the container to install dev dependencies.

Why use the devcontainer?

- Eliminates "works on my machine" issues.
- Ensures consistent Python and tool versions across contributors and CI.
- Makes onboarding faster — contributors do not need to manage local
	Python installations.

## Common commands

- Run pre-commit hooks on all files:

```bash
make pre-commit
```

- Run tests (prints individual test names):

```bash
make test
```

- GPU quick-check (prints whether CUDA is available and device name):

```bash
make gpu-check
```

Or run the included notebook `notebooks/00_sanity_gpu.ipynb` inside JupyterLab.

## Development workflow

-- Create a feature branch from `main`.
-- Add tests for new behavior and run `make test`.
-- Run `make pre-commit` before opening a pull request to run formatting/linting hooks.

CI or maintainer checks should include running the test suite and linting.

## Contributing

1. Fork the repository and create a branch for your change.
2. Run the tests and pre-commit hooks locally:

```bash
make pre-commit
make test
```

3. Open a pull request with a clear description of the change.

If you'd like to propose larger changes (changing dependency versions,
adding new notebooks or examples), open an issue first so we can discuss the
scope and approach.
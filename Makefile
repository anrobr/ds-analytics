.PHONY: fmt lint test gpu-check
fmt:  ruff check --fix .
lint: ruff check .
test: pytest -q --maxfail=1 --disable-warnings --cov=src
gpu-check:
	python -c "import torch; print('CUDA:', torch.cuda.is_available()); \
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

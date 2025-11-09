.PHONY: pre-commit tests gpu-check lint

pre-commit:
	@command -v pre-commit >/dev/null 2>&1 && pre-commit run --all-files || \
	( command -v python >/dev/null 2>&1 && python -m pre_commit run --all-files ) || \
	( echo "pre-commit is not installed. Install with: pip install pre-commit" >&2; exit 1 )

tests:
	@# Use coverage if pytest supports --cov (pytest-cov installed), otherwise run without coverage
	@if pytest --help 2>/dev/null | grep -q -- --cov; then \
    	pytest -v --maxfail=1 --disable-warnings --cov=tests/; \
	else \
		echo "pytest-cov not available; running tests without coverage"; \
	pytest -v --maxfail=1 --disable-warnings tests/; \
	fi

gpu-check:
	python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

lint:
	@echo "Linting with ruff..."
	@ruff check .

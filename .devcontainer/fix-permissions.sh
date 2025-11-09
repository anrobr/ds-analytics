#!/bin/bash
# Diagnostic and permission fix script for devcontainer

echo "=== Permission Diagnostics ==="
echo "Current user: $(whoami)"
echo "User ID: $(id)"
echo ""
echo "Workspace directory ownership:"
ls -ld /workspaces/ds-and-ml
echo ""
echo "Sample files:"
ls -l /workspaces/ds-and-ml/Makefile 2>/dev/null || echo "Makefile not found"
ls -l /workspaces/ds-and-ml/pyproject.toml 2>/dev/null || echo "pyproject.toml not found"
echo ""

# Check if workspace is writable
if [ -w /workspaces/ds-and-ml ]; then
    echo "✓ Workspace is writable"
else
    echo "✗ Workspace is NOT writable - attempting fix..."

    # Try to fix with sudo
    if command -v sudo >/dev/null 2>&1; then
        echo "Running: sudo chown -R vscode:vscode /workspaces/ds-and-ml"
        sudo chown -R vscode:vscode /workspaces/ds-and-ml

        if [ $? -eq 0 ]; then
            echo "✓ Permissions fixed successfully"
        else
            echo "✗ Failed to fix permissions with sudo"
        fi
    else
        echo "✗ sudo not available"
    fi
fi

echo ""
echo "Final check:"
ls -ld /workspaces/ds-and-ml

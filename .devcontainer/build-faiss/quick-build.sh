#!/bin/bash
# Quick build helper script - run from the build-faiss directory

set -e

CONTAINER_NAME="faiss-gpu-builder:cuda13-py313"
OUTPUT_DIR="$(pwd)/output"

echo "🚀 FAISS-GPU Build Helper"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check if nvidia-docker is available
if ! docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: GPU support may not be available"
    echo "Make sure nvidia-docker2 is installed and configured"
fi

# Build the container
echo "📦 Building container image..."
docker build -t "${CONTAINER_NAME}" .

if [ $? -ne 0 ]; then
    echo "❌ Container build failed"
    exit 1
fi

echo "✅ Container built successfully"
echo ""

# Run the build
echo "🔨 Starting faiss-gpu + cuVS build..."
echo "This will take 15-30 minutes..."
echo ""

docker run --gpus all -it --rm \
    -v "${OUTPUT_DIR}:/workspace/output" \
    "${CONTAINER_NAME}" \
    bash -c "
        sudo chown -R builder:builder /workspace/output &&
        sudo chmod -R 755 /workspace/output &&
        echo '=== Building faiss-gpu ===' &&
        cd /workspace/faiss && 
        bash /workspace/build.sh && 
        echo '' && 
        echo '=== Building cuVS ===' &&
        bash /workspace/build-cuvs.sh &&
        echo '' && 
        echo '📦 Creating wheels...' && 
        cd /workspace/faiss/build/faiss/python && 
        python -m pip wheel --no-deps --wheel-dir=dist . && 
        echo 'Repairing wheel to bundle shared libraries...' &&
        python /workspace/repair-wheel.py dist/*.whl /workspace/output/install/lib &&
        mkdir -p /workspace/output/wheels && 
        cp dist/*.whl /workspace/output/wheels/ &&
        echo '' && 
        echo '🔬 Reinstalling from wheel for testing...' &&
        python -m pip uninstall -y faiss &&
        python -m pip install /workspace/output/wheels/faiss-*.whl &&
        echo '' && 
        echo '🔬 Testing integration...' &&
        python /workspace/test-integration.py &&
        echo '' &&
        echo '✅ Build complete!' &&
        echo '' &&
        echo 'Outputs:' &&
        ls -lh /workspace/output/wheels/*.whl
    "

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Success! Wheel file available at:"
    ls -lh "${OUTPUT_DIR}/wheels/"*.whl 2>/dev/null || echo "No wheel found"
else
    echo "❌ Build failed"
    exit 1
fi

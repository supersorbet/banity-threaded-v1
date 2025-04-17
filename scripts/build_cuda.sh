#!/bin/bash

# Check if NVCC is available
if ! command -v nvcc &> /dev/null; then
    echo "CUDA toolkit not found. Please install CUDA toolkit first."
    exit 1
fi

# Create build directory if it doesn't exist
mkdir -p build

# Get GPU compute capability
COMPUTE_CAP="60"  # Default to compute capability 6.0 (Pascal)
if command -v nvidia-smi &> /dev/null; then
    # Try to autodetect compute capability
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    
    # Simple mapping for common GPUs
    if [[ "$GPU_NAME" == *"RTX 4"* ]]; then
        COMPUTE_CAP="89"  # Ada Lovelace
    elif [[ "$GPU_NAME" == *"RTX 3"* ]]; then
        COMPUTE_CAP="86"  # Ampere
    elif [[ "$GPU_NAME" == *"RTX 2"* ]]; then
        COMPUTE_CAP="75"  # Turing
    elif [[ "$GPU_NAME" == *"GTX 16"* ]]; then
        COMPUTE_CAP="75"  # Turing
    elif [[ "$GPU_NAME" == *"GTX 10"* ]]; then
        COMPUTE_CAP="61"  # Pascal
    fi
    
    echo "Detected GPU: $GPU_NAME (compute_$COMPUTE_CAP)"
else
    echo "Warning: Could not detect GPU. Using default compute capability $COMPUTE_CAP."
fi

# Compile CUDA code with debug flags
nvcc src/create3_cuda.cu -o build/create3_cuda -O3 -arch=compute_$COMPUTE_CAP -code=sm_$COMPUTE_CAP \
     -lineinfo -Xcompiler -Wall,-Wextra

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    # Make the binary executable
    chmod +x build/create3_cuda
    echo "CREATE3 CUDA binary built successfully!"
else
    echo "Failed to build CREATE3 CUDA binary"
    exit 1
fi 
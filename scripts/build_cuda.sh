#!/bin/bash

# Check if NVCC is available
if ! command -v nvcc &> /dev/null; then
    echo "CUDA toolkit not found. Please install CUDA toolkit first."
    exit 1
fi

# Create build directory if it doesn't exist
mkdir -p build

# Compile CUDA code
nvcc src/vanity_cuda.cu -o build/vanity_cuda -O3 -arch=sm_35

# Make the binary executable
chmod +x build/vanity_cuda

echo "CUDA binary built successfully!" 
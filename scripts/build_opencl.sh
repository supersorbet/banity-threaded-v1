#!/bin/bash

# Check if OpenCL development files are available
if ! pkg-config --exists OpenCL; then
    echo "OpenCL development files not found. Please install OpenCL development package."
    echo "For AMD: Install ROCm or AMDGPU-PRO drivers"
    echo "For Intel: Install intel-opencl-icd and opencl-headers"
    exit 1
fi

# Create build directory if it doesn't exist
mkdir -p build

# Compile OpenCL code
g++ src/vanity_opencl.cpp -o build/vanity_opencl -O3 $(pkg-config --cflags --libs OpenCL) -I./src

# Make the binary executable
chmod +x build/vanity_opencl

echo "OpenCL binary built successfully!" 
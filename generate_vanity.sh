#!/bin/bash

# Check if OpenCL binary exists and build if needed
if [ ! -f "build/vanity_opencl" ]; then
    echo "Building OpenCL binary..."
    npm run build-opencl
fi

# Run the vanity address generator
# -opencl: Use OpenCL (GPU)
# 256: Number of threads
# 0000: Target pattern
# -x: Generate contract address
# -l: Log to file
node index.js -opencl 256 0000 -x -l

# The results will be logged to a file named VanityEth-log-{timestamp}.txt 
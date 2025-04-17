import { spawn } from 'child_process';
import { ethers } from 'ethers';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CUDA_BINARY = path.join(__dirname, '../build/create3_cuda');

// Check if CUDA binary exists and is executable
function checkCUDABinary() {
    try {
        return fs.existsSync(CUDA_BINARY) && (fs.statSync(CUDA_BINARY).mode & fs.constants.X_OK);
    } catch (error) {
        return false;
    }
}

// Find a vanity CREATE3 address using CUDA
async function findCreate3VanitySalt(
    prefix = "",
    deployer = "0x1afeB019eC12f389750f29266ed3a47567C43880", // Default CREATE69Factory
    threads = 256,
    counter = function () {}
) {
    if (!checkCUDABinary()) {
        throw new Error("CREATE3 CUDA binary not found or not executable. Please run 'npm run build-cuda' first.");
    }

    // Remove 0x prefix if present
    deployer = deployer.startsWith('0x') ? deployer.slice(2) : deployer;
    prefix = prefix.toLowerCase();

    return new Promise((resolve, reject) => {
        const args = [
            prefix,
            deployer,
            threads.toString()
        ];

        const cudaProcess = spawn(CUDA_BINARY, args);
        let output = '';
        let error = '';
        
        // Estimate addresses per iteration
        // Each iteration processes approximately threads * blockSize addresses
        const estimatedAddressesPerIteration = threads * 100;

        cudaProcess.stdout.on('data', (data) => {
            output += data.toString();
        });

        cudaProcess.stderr.on('data', (data) => {
            const text = data.toString();
            error += text;
            
            // Check for iteration updates from CUDA process
            if (text.includes('Completed') && text.includes('iterations')) {
                const match = text.match(/Completed (\d+) iterations/);
                if (match && match[1]) {
                    const iterations = parseInt(match[1]);
                    // Call the counter with estimated addresses checked
                    counter(estimatedAddressesPerIteration);
                }
            } else if (text.includes('Starting')) {
                // Initial message, call counter once to initialize
                counter(100);
            }
        });

        cudaProcess.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`CUDA process failed: ${error}`));
                return;
            }

            try {
                const result = JSON.parse(output);
                resolve(result);
            } catch (e) {
                reject(new Error(`Failed to parse CUDA output: ${e.message}`));
            }
        });

        cudaProcess.on('error', (err) => {
            reject(new Error(`Failed to start CUDA process: ${err.message}`));
        });
    });
}

export default {
    findCreate3VanitySalt,
    checkCUDABinary
}; 
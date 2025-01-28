import { spawn } from 'child_process';
import ethUtils from "ethereumjs-util";
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CUDA_BINARY = path.join(__dirname, '../build/vanity_cuda');

// Check if CUDA binary exists and is executable
function checkCUDABinary() {
    try {
        return fs.existsSync(CUDA_BINARY) && (fs.statSync(CUDA_BINARY).mode & fs.constants.X_OK);
    } catch (error) {
        return false;
    }
}

async function getVanityWalletCUDA(
    input = "",
    isChecksum = false,
    isContract = false,
    threads = 256,
    counter = function () {}
) {
    if (!checkCUDABinary()) {
        throw new Error("CUDA binary not found or not executable. Please run 'npm run build-cuda' first.");
    }

    return new Promise((resolve, reject) => {
        const args = [
            input,
            isChecksum ? '1' : '0',
            isContract ? '1' : '0',
            threads.toString()
        ];

        const cudaProcess = spawn(CUDA_BINARY, args);
        let output = '';
        let error = '';

        cudaProcess.stdout.on('data', (data) => {
            output += data.toString();
            counter(); // Increment counter for each batch processed
        });

        cudaProcess.stderr.on('data', (data) => {
            error += data.toString();
        });

        cudaProcess.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`CUDA process failed: ${error}`));
                return;
            }

            try {
                const result = JSON.parse(output);
                if (isChecksum) {
                    result.address = ethUtils.toChecksumAddress(result.address);
                }
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
    getVanityWalletCUDA,
    checkCUDABinary
}; 
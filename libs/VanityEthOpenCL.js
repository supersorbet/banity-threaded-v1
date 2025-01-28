import { spawn } from 'child_process';
import ethUtils from "ethereumjs-util";
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OPENCL_BINARY = path.join(__dirname, '../build/vanity_opencl');

// Check if OpenCL binary exists and is executable
function checkOpenCLBinary() {
    try {
        return fs.existsSync(OPENCL_BINARY) && (fs.statSync(OPENCL_BINARY).mode & fs.constants.X_OK);
    } catch (error) {
        return false;
    }
}

async function getVanityWalletOpenCL(
    input = "",
    isChecksum = false,
    isContract = false,
    threads = 256,
    counter = function () {}
) {
    if (!checkOpenCLBinary()) {
        throw new Error("OpenCL binary not found or not executable. Please run 'npm run build-opencl' first.");
    }

    return new Promise((resolve, reject) => {
        const args = [
            input,
            isChecksum ? '1' : '0',
            isContract ? '1' : '0',
            threads.toString()
        ];

        const openclProcess = spawn(OPENCL_BINARY, args);
        let output = '';
        let error = '';

        openclProcess.stdout.on('data', (data) => {
            output += data.toString();
            counter(); // Increment counter for each batch processed
        });

        openclProcess.stderr.on('data', (data) => {
            error += data.toString();
        });

        openclProcess.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`OpenCL process failed: ${error}`));
                return;
            }

            try {
                const result = JSON.parse(output);
                if (isChecksum) {
                    result.address = ethUtils.toChecksumAddress(result.address);
                }
                resolve(result);
            } catch (e) {
                reject(new Error(`Failed to parse OpenCL output: ${e.message}`));
            }
        });

        openclProcess.on('error', (err) => {
            reject(new Error(`Failed to start OpenCL process: ${err.message}`));
        });
    });
}

export default {
    getVanityWalletOpenCL,
    checkOpenCLBinary
}; 
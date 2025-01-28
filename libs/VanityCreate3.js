import { ethers } from 'ethers';
import { predictCreate3Address, generateRandomSalt } from './create3Utils.js';
import crypto from 'crypto';

function isValidHex(hex) {
    if (!hex.length) return true;
    hex = hex.toUpperCase();
    const re = /^[0-9A-F]+$/g;
    return re.test(hex);
}

async function findCreate3Salt(
    deployer,
    prefix = "",
    isChecksum = false,
    counter = function () {},
    maxAttempts = 1000000
) {
    if (!isValidHex(prefix)) throw new Error("Invalid hex prefix");
    prefix = isChecksum ? prefix : prefix.toLowerCase();

    let attempts = 0;
    while (attempts < maxAttempts) {
        counter();
        attempts++;

        // Generate random salt
        const salt = generateRandomSalt();
        
        // Predict the contract address
        let contractAddress = predictCreate3Address(salt, deployer);
        
        // Apply checksum if needed
        if (isChecksum) {
            contractAddress = ethers.getAddress(contractAddress);
        } else {
            contractAddress = contractAddress.toLowerCase();
        }

        // Check if address matches prefix
        if (contractAddress.substr(2, prefix.length) === prefix) {
            return {
                salt,
                contractAddress
            };
        }

        // Add throttling to prevent CPU overload
        if (attempts % 1000 === 0) {
            await new Promise(resolve => setTimeout(resolve, 1));
        }
    }
    throw new Error(`No matching salt found within ${maxAttempts} attempts`);
}

// GPU-accelerated version using OpenCL
async function findCreate3SaltGPU(
    deployer,
    prefix = "",
    isChecksum = false,
    threads = 256,
    counter = function () {}
) {
    // This would be implemented similar to the existing GPU vanity address generation
    // but modified for CREATE3 address prediction
    // For now, fall back to CPU version
    return findCreate3Salt(deployer, prefix, isChecksum, counter);
}

export default {
    findCreate3Salt,
    findCreate3SaltGPU,
    isValidHex
}; 
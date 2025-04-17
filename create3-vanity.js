#!/usr/bin/env node

import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import ora from 'ora';
import TimeFormat from 'hh-mm-ss';
import { ethers } from 'ethers';
import { predictCreate3Address, generateRandomSalt, verifyVanityAddress } from './libs/create3Utils.js';
import VanityCreate3CUDA from './libs/VanityCreate3CUDA.js';
import fs from 'fs';
import os from 'os';

let addps = 0;
let totalAddresses = 0;
let startTime = Date.now();
let spinner;

// Calculation of total possible combinations for a prefix
function calculateTotalCombinations(prefix) {
    return Math.pow(16, prefix.length);
}

// Format large numbers with commas
function formatNumber(num) {
    return num.toLocaleString();
}

// Calculate time remaining
function calculateETA(addressesPerSecond, remainingAddresses) {
    if (addressesPerSecond <= 0) return 'calculating...';
    const seconds = remainingAddresses / addressesPerSecond;
    return TimeFormat.fromS(seconds, "hh:mm:ss");
}

const args = yargs(hideBin(process.argv))
    .usage('Usage: $0 [options] <prefix>')
    .option('deployer', {
        alias: 'd',
        type: 'string',
        description: 'Deployer address (required)',
        default: '0x1afeB019eC12f389750f29266ed3a47567C43880', // Mainnet CREATE69Factory
        demandOption: false
    })
    .option('cpu', {
        type: 'number',
        description: 'Number of CPU threads to use',
        default: 4
    })
    .option('gpu', {
        type: 'boolean',
        description: 'Use GPU acceleration if available',
        default: false
    })
    .option('gpu-threads', {
        type: 'number',
        description: 'Number of GPU threads to use',
        default: 1024
    })
    .option('log', {
        alias: 'l',
        type: 'boolean',
        description: 'Log output to file'
    })
    .option('verify-salt', {
        type: 'string',
        description: 'Verify the predicted address for a given salt'
    })
    .example('$0 -d 0x123... 0000', 'Find salt for contract starting with 0000')
    .example('$0 --gpu 0000', 'Find salt using GPU acceleration')
    .argv;

const prefix = args._[0] || "";
if (args['verify-salt']) {
    const predictedAddress = predictCreate3Address(args['verify-salt'], args.deployer);
    console.log({
        salt: args['verify-salt'],
        contractAddress: predictedAddress,
        deployer: args.deployer
    });
    process.exit(0);
}

// Only require prefix if not verifying salt
if (!args['verify-salt'] && !prefix) {
    console.error('Error: Prefix is required');
    process.exit(1);
}

if (!ethers.isAddress(args.deployer)) {
    console.error('Error: Invalid deployer address');
    process.exit(1);
}

// Check if GPU is available
let useGPU = args.gpu;
if (useGPU && !VanityCreate3CUDA.checkCUDABinary()) {
    console.warn('Warning: GPU acceleration requested but CUDA binary not found. Falling back to CPU.');
    console.warn('Run "npm run build-cuda" to build the CUDA binary.');
    useGPU = false;
}

let logStream;
if (args.log) {
    const logFile = `Create3Vanity-log-${Date.now()}.txt`;
    logStream = fs.createWriteStream(logFile, { flags: 'a' });
}

// Initialize progress spinner
spinner = ora(`generating CREATE3 salt using ${useGPU ? 'GPU' : 'CPU'}`).start();

// Update ETA every second
setInterval(function () {
    const elapsedTime = (Date.now() - startTime) / 1000; // in seconds
    const addressesPerSecond = addps;
    const totalCombinations = calculateTotalCombinations(prefix);
    const progress = Math.min(totalAddresses / totalCombinations * 100, 99.99); // Cap at 99.99%
    const remainingAddresses = totalCombinations - totalAddresses;
    const eta = calculateETA(addressesPerSecond, remainingAddresses);
    
    spinner.text = `${useGPU ? 'GPU' : 'CPU'} mode | ` +
        `Speed: ${formatNumber(addressesPerSecond)}/s | ` +
        `Progress: ${progress.toFixed(4)}% | ` +
        `Checked: ${formatNumber(totalAddresses)} | ` +
        `ETA: ${eta}`;
    
    addps = 0;
}, 1000);

async function findSaltCPU() {
    while (true) {
        addps++;
        totalAddresses++;
        
        // Generate random salt
        const salt = generateRandomSalt();
        
        // Predict CREATE3 address
        const predictedAddress = predictCreate3Address(salt, args.deployer);
        
        // Verify prefix
        if (verifyVanityAddress(predictedAddress, prefix)) {
            return {
                salt,
                contractAddress: predictedAddress,
                deployer: args.deployer
            };
        }

        // Add throttling
        if (addps % 1000 === 0) {
            await new Promise(resolve => setTimeout(resolve, 1));
        }
    }
}

async function findSaltGPU() {
    try {
        // Counter function to increment addps
        const counter = (addressesChecked = 10000) => {
            // Each GPU call processes many addresses
            addps += addressesChecked;
            totalAddresses += addressesChecked;
        };
        
        // Call the CUDA implementation
        const result = await VanityCreate3CUDA.findCreate3VanitySalt(
            prefix,
            args.deployer,
            args['gpu-threads'],
            counter
        );
        
        return result;
    } catch (error) {
        spinner.fail(`GPU processing failed: ${error.message}`);
        console.log('Falling back to CPU processing...');
        return findSaltCPU();
    }
}

async function main() {
    // Reset counters
    addps = 0;
    totalAddresses = 0;
    startTime = Date.now();
    
    try {
        // Choose between CPU and GPU implementation
        const result = useGPU ? await findSaltGPU() : await findSaltCPU();
        
        // Calculate statistics
        const elapsedTime = (Date.now() - startTime) / 1000; // in seconds
        const addressesPerSecond = Math.round(totalAddresses / elapsedTime);
        
        spinner.succeed(JSON.stringify(result, null, 2));
        console.log(`\nStats: ${formatNumber(totalAddresses)} addresses checked in ${elapsedTime.toFixed(2)}s (${formatNumber(addressesPerSecond)}/s average)`);
        
        if (args.log && logStream) {
            logStream.write(JSON.stringify(result) + "\n");
            logStream.write(`Stats: ${formatNumber(totalAddresses)} addresses checked in ${elapsedTime.toFixed(2)}s (${formatNumber(addressesPerSecond)}/s average)\n`);
            logStream.end();
        }

    } catch (error) {
        spinner.fail(error.message);
        process.exit(1);
    }
}

main(); 
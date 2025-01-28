#!/usr/bin/env node

import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import ora from 'ora';
import TimeFormat from 'hh-mm-ss';
import { ethers } from 'ethers';
import { predictCreate3Address, generateRandomSalt, verifyVanityAddress } from './libs/create3Utils.js';
import fs from 'fs';

let addps = 0;
let spinner;

const args = yargs(hideBin(process.argv))
    .usage('Usage: $0 [options] <prefix>')
    .option('deployer', {
        alias: 'd',
        type: 'string',
        description: 'Deployer address (required)',
        default: '0xD37A8B489DbF221D614779cde7A99197dF25CE9C', // Mainnet CREATE69Factory
        demandOption: false
    })
    .option('cpu', {
        type: 'number',
        description: 'Number of CPU threads to use',
        default: 4
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

let logStream;
if (args.log) {
    const logFile = `Create3Vanity-log-${Date.now()}.txt`;
    logStream = fs.createWriteStream(logFile, { flags: 'a' });
}

// Initialize progress spinner
spinner = ora("generating CREATE3 salt").start();

// Update ETA every second
setInterval(function () {
    spinner.text = "Approximate ETA: " +
        TimeFormat.fromS(
            Math.pow(16, prefix.length) / addps,
            "hh:mm:ss"
        );
    addps = 0;
}, 1000);

async function findSalt() {
    while (true) {
        addps++;
        
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

async function main() {
    try {
        const result = await findSalt();
        
        spinner.succeed(JSON.stringify(result, null, 2));
        
        if (args.log && logStream) {
            logStream.write(JSON.stringify(result) + "\n");
            logStream.end();
        }

    } catch (error) {
        spinner.fail(error.message);
        process.exit(1);
    }
}

main(); 
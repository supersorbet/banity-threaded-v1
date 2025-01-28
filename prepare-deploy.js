#!/usr/bin/env node

import { ethers } from 'ethers';
import { formatSaltBytes32 } from './libs/create3Utils.js';

// Usage: node prepare-deploy.js <salt-from-vanity-generator> <contract-bytecode>
const args = process.argv.slice(2);
if (args.length !== 2) {
    console.error('Usage: node prepare-deploy.js <salt> <bytecode>');
    process.exit(1);
}

const [salt, bytecode] = args;

// Format the salt as bytes32
const formattedSalt = formatSaltBytes32(salt);

console.log('\nDeployment data for Remix IDE:');
console.log('==============================');
console.log('\nsalt (bytes32):');
console.log(formattedSalt);
console.log('\ncreationCode (bytes):');
console.log(bytecode.startsWith('0x') ? bytecode : '0x' + bytecode);

// Also verify the length
console.log('\nBytecode length (bytes):', Math.floor((bytecode.length - 2) / 2)); 
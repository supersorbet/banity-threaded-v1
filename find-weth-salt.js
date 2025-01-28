#!/usr/bin/env node

import VanityCreate3 from './libs/VanityCreate3.js';
import { formatSaltBytes32 } from './libs/create3Utils.js';

async function main() {
    // Get deployer address from command line
    const args = process.argv.slice(2);
    if (args.length !== 1) {
        console.error('Usage: node find-weth-salt.js <deployer-address>');
        process.exit(1);
    }
    const [deployer] = args;

    // Find salt for vanity address
    console.log('\nFinding salt for address starting with 0x133700000...');
    const result = await VanityCreate3.findCreate3Salt(
        deployer,
        '133700000',
        false,
        () => {},
        10000000 // increase max attempts
    );

    console.log('\nFound matching salt!');
    console.log('Salt:', result.salt);
    console.log('Contract address:', result.contractAddress);

    // Format deployment data for Remix
    const formattedSalt = formatSaltBytes32(result.salt);
    console.log('\nDeployment data for Remix:');
    console.log('salt (bytes32):', formattedSalt);
}

main().catch(console.error); 
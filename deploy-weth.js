#!/usr/bin/env node

import { ethers } from 'ethers';
import fs from 'fs';
import solc from 'solc';
import VanityCreate3 from './libs/VanityCreate3.js';
import { formatSaltBytes32 } from './libs/create3Utils.js';

async function compileSolidity(sourcePath) {
    const sourceCode = fs.readFileSync(sourcePath, 'utf8');
    
    const input = {
        language: 'Solidity',
        sources: {
            'WETH.sol': {
                content: sourceCode
            }
        },
        settings: {
            outputSelection: {
                '*': {
                    '*': ['*']
                }
            },
            optimizer: {
                enabled: true,
                runs: 200
            }
        }
    };

    const output = JSON.parse(solc.compile(JSON.stringify(input)));
    if (output.errors) {
        const errors = output.errors.filter(error => error.severity === 'error');
        if (errors.length > 0) {
            throw new Error(errors[0].formattedMessage);
        }
    }

    const contract = output.contracts['WETH.sol']['WETH'];
    return contract.evm.bytecode.object;
}

async function main() {
    // Get deployer address from command line
    const args = process.argv.slice(2);
    if (args.length !== 1) {
        console.error('Usage: node deploy-weth.js <deployer-address>');
        process.exit(1);
    }
    const [deployer] = args;

    // Compile WETH contract
    console.log('Compiling WETH contract...');
    const bytecode = await compileSolidity('WETH.sol');
    console.log('Contract bytecode:', bytecode);

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

    // Format deployment data
    const formattedSalt = formatSaltBytes32(result.salt);
    console.log('\nDeployment data:');
    console.log('salt (bytes32):', formattedSalt);
    console.log('creationCode (bytes):', '0x' + bytecode);
}

main().catch(console.error); 
import { ethers } from 'ethers';

// Solady's CREATE3 proxy initcode hash
const PROXY_INITCODE_HASH = '0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f';

export function predictCreate3Address(salt, deployer) {
    // First predict the proxy address using CREATE2
    const proxyAddress = ethers.getCreate2Address(
        deployer,
        salt,
        PROXY_INITCODE_HASH
    );

    // Then predict the final contract address (nonce 1)
    // This matches the exact RLP encoding in the Solidity contract
    const rlpEncoded = ethers.concat([
        '0xd6',  // 0xc0 (short RLP prefix) + 0x16 (length of: 0x94 ++ proxy ++ 0x01)
        '0x94',  // 0x80 + 0x14 (0x14 = length of an address)
        proxyAddress,
        '0x01'   // nonce
    ]);

    return ethers.getAddress('0x' + ethers.keccak256(rlpEncoded).slice(-40));
}

export function generateRandomSalt() {
    return '0x' + Array.from(ethers.randomBytes(32))
        .map(b => b.toString(16).padStart(2, '0'))
        .join('');
}

// Format salt as bytes32 for contract calls
export function formatSaltBytes32(salt) {
    return ethers.zeroPadValue(salt, 32);
}

// Verify the address matches the prefix
export function verifyVanityAddress(address, prefix) {
    return address.toLowerCase().slice(2, 2 + prefix.length) === prefix.toLowerCase();
} 
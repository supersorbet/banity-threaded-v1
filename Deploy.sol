// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

import "./CREATE3.sol";

contract Deploy {
    event Deployed(address indexed deployed);

    function deploy(bytes32 salt, bytes memory creationCode) external payable returns (address deployed) {
        // Deploy using CREATE3
        deployed = CREATE3.deployDeterministic(msg.value, creationCode, salt);
        emit Deployed(deployed);
    }

    // Helper to predict the address before deployment
    function predictAddress(bytes32 salt) external view returns (address) {
        return CREATE3.predictDeterministicAddress(salt);
    }
} 
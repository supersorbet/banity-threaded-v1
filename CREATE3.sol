// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

/// @notice Deterministic deployments agnostic to the initialization code.
/// @author Solady (https://github.com/vectorized/solady/blob/main/src/utils/CREATE3.sol)
/// @author Modified from Solmate (https://github.com/transmissions11/solmate/blob/main/src/utils/CREATE3.sol)
/// @author Modified from 0xSequence (https://github.com/0xSequence/create3/blob/master/contracts/Create3.sol)
library CREATE3 {
    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                        CUSTOM ERRORS                       */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Unable to deploy the contract.
    error DeploymentFailed();

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                      BYTECODE CONSTANTS                    */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The proxy bytecode hash.
    /// Equivalent to `keccak256(abi.encodePacked(hex"67363d3d37363d34f03d5260086018f3"))`.
    bytes32 internal constant PROXY_BYTECODE_HASH =
        0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                      CREATE3 OPERATIONS                    */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Deploys a contract using CREATE3.
    /// Returns the deployment address.
    function deploy(bytes32 salt, bytes memory creationCode) internal returns (address deployed) {
        deployed = deployDeterministic(0, creationCode, salt);
    }

    /// @dev Deploys a contract using CREATE3 and funds it with `value` wei.
    /// Returns the deployment address.
    function deployDeterministic(uint256 value, bytes memory creationCode, bytes32 salt)
        internal
        returns (address deployed)
    {
        /// @solidity memory-safe-assembly
        assembly {
            // Store the `_PROXY_BYTECODE` into scratch space.
            mstore(0x00, hex"67363d3d37363d34f03d5260086018f3")
            // Deploy the proxy using CREATE2.
            let proxy := create2(0, 0x10, 0x10, salt)
            if iszero(proxy) {
                mstore(0x00, 0x30116425) // `DeploymentFailed()`.
                revert(0x1c, 0x04)
            }
            mstore(0x14, proxy) // Store the proxy's address.
            // 0xd6 = 0xc0 (short RLP prefix) + 0x16 (length of: 0x94 ++ proxy ++ 0x01).
            // 0x94 = 0x80 + 0x14 (0x14 = the length of an address, 20 bytes, in hex).
            mstore(0x00, 0xd694)
            mstore8(0x34, 0x01)
            deployed := keccak256(0x1e, 0x17) // Compute the contract address.
            if iszero(
                mul( // The arguments of `mul` are evaluated last to first.
                    extcodesize(deployed),
                    call(gas(), proxy, value, add(creationCode, 0x20), mload(creationCode), 0x00, 0x00)
                )
            ) {
                mstore(0x00, 0x30116425) // `DeploymentFailed()`.
                revert(0x1c, 0x04)
            }
        }
    }

    /// @dev Returns the address of the contract deployed using CREATE3 with `salt`.
    function predictDeterministicAddress(bytes32 salt) internal view returns (address deployed) {
        deployed = predictDeterministicAddress(salt, address(this));
    }

    /// @dev Returns the address of the contract deployed using CREATE3 with `salt` by `deployer`.
    function predictDeterministicAddress(bytes32 salt, address deployer)
        internal
        pure
        returns (address deployed)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40) // Cache the free memory pointer.
            mstore(0x00, deployer) // Store the deployer.
            mstore8(0x0b, 0xff) // Store the prefix.
            mstore(0x20, salt) // Store the salt.
            mstore(0x40, PROXY_BYTECODE_HASH) // Store the proxy bytecode hash.

            mstore(0x14, keccak256(0x0b, 0x55)) // Store the proxy's address.
            mstore(0x40, m) // Restore the free memory pointer.
            // 0xd6 = 0xc0 (short RLP prefix) + 0x16 (length of: 0x94 ++ proxy ++ 0x01).
            // 0x94 = 0x80 + 0x14 (0x14 = the length of an address, 20 bytes, in hex).
            mstore(0x00, 0xd694)
            mstore8(0x34, 0x01)
            deployed := keccak256(0x1e, 0x17) // Compute the contract address.
        }
    }
} 
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

import "./CREATE3.sol";

/// @title CREATE3 Factory
/// @notice Factory contract for deterministic cross-chain deployments
/// @dev Uses CREATE3 pattern for consistent addresses across all chains
contract CREATE3Factory {
    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                         CUSTOM ERRORS                        */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @notice Thrown when deployment fails
    error DeploymentFailed();

    /// @notice Thrown when bytecode is empty
    error EmptyBytecode();

    /// @notice Thrown when salt has already been used
    error SaltAlreadyUsed();

    /// @notice Thrown when deployment address doesn't match expected
    error AddressMismatch(address expected, address actual);

    /// @notice Thrown when contract size is zero after deployment
    error EmptyContract();

    /// @notice Thrown when msg.value doesn't match required value
    error IncorrectValue();

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                            EVENTS                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @notice Emitted when a contract is successfully deployed
    /// @param deployer The address that initiated the deployment
    /// @param salt The salt used for deployment
    /// @param deployed The address where the contract was deployed
    /// @param value The ETH value sent with deployment
    event Deployed(
        address indexed deployer,
        bytes32 indexed salt,
        address indexed deployed,
        uint256 value
    );

    /// @notice Emitted when a deployment is verified
    /// @param deployer The address that will deploy
    /// @param salt The salt to be used
    /// @param predicted The predicted deployment address
    event DeploymentVerified(
        address indexed deployer,
        bytes32 indexed salt,
        address indexed predicted
    );

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                          FUNCTIONS                          */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @notice Deploy a contract using CREATE3
    /// @param salt The salt for deterministic address generation
    /// @param creationCode The contract creation code
    /// @param expectedAddress Optional expected deployment address (for verification)
    /// @return deployed The address where the contract was deployed
    function deploy(
        bytes32 salt,
        bytes memory creationCode,
        address expectedAddress
    ) external payable returns (address deployed) {
        // Input validation
        if (creationCode.length == 0) revert EmptyBytecode();
        
        // Predict deployment address
        address predictedAddress = CREATE3.predictDeterministicAddress(salt);
        
        // Verify expected address if provided
        if (expectedAddress != address(0)) {
            if (expectedAddress != predictedAddress) {
                revert AddressMismatch(expectedAddress, predictedAddress);
            }
            emit DeploymentVerified(msg.sender, salt, predictedAddress);
        }

        // Check if salt has been used
        if (predictedAddress.code.length > 0) {
            revert SaltAlreadyUsed();
        }

        // Deploy the contract
        deployed = CREATE3.deployDeterministic(msg.value, creationCode, salt);

        // Verify deployment
        if (deployed.code.length == 0) revert EmptyContract();
        if (deployed != predictedAddress) {
            revert AddressMismatch(predictedAddress, deployed);
        }

        // Emit deployment event
        emit Deployed(msg.sender, salt, deployed, msg.value);
    }

    /// @notice Predict deployment address for a given salt
    /// @param salt The salt to use for prediction
    /// @return predicted The predicted deployment address
    function predictDeploymentAddress(bytes32 salt) external view returns (address predicted) {
        predicted = CREATE3.predictDeterministicAddress(salt);
    }

    /// @notice Check if a salt has been used
    /// @param salt The salt to check
    /// @return true if the salt has been used
    function isSaltUsed(bytes32 salt) external view returns (bool) {
        address predictedAddress = CREATE3.predictDeterministicAddress(salt);
        return predictedAddress.code.length > 0;
    }

    /// @notice Verify deployment parameters before actual deployment
    /// @param salt The salt to use
    /// @param expectedAddress The expected deployment address
    /// @return isValid True if parameters are valid
    /// @return predictedAddress The address where contract would be deployed
    function verifyDeployment(
        bytes32 salt,
        address expectedAddress
    ) external view returns (bool isValid, address predictedAddress) {
        predictedAddress = CREATE3.predictDeterministicAddress(salt);
        
        // Check if salt is unused
        if (predictedAddress.code.length > 0) {
            return (false, predictedAddress);
        }

        // Check if address matches expected
        if (expectedAddress != address(0) && expectedAddress != predictedAddress) {
            return (false, predictedAddress);
        }

        return (true, predictedAddress);
    }

    /// @notice Receive function to accept ETH
    receive() external payable {}

    /// @notice Fallback function to accept ETH
    fallback() external payable {}
} 
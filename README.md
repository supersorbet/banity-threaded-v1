# CREATE3 Vanity Address Generator

A tool for generating deterministic vanity addresses for CREATE3 contract deployments, compatible with Solady's CREATE3 implementation.

## Features

- Generate vanity addresses for CREATE3 contract deployments
- Verify existing salt and address predictions
- Cross-chain compatible (same address on all chains)
- CPU throttling to prevent system overload
- Progress indicator with ETA
- Optional logging to file

## Installation

```bash
npm install
```

## Usage

### Generate a Vanity Address

```bash
node create3-vanity.js <prefix> [options]
```

Example (find address starting with "0000"):
```bash
node create3-vanity.js 0000 -cpu 2
```

### Verify an Existing Salt

```bash
node create3-vanity.js --verify-salt <salt>
```

Example:
```bash
node create3-vanity.js --verify-salt 0x849b3d15d82f3579ab0a2e5847624ff9625c29ea3c422306bf737b95690b9c1f
```

### Options

- `-d, --deployer`: Factory contract address (default: Mainnet CREATE69Factory)
- `--cpu`: Number of CPU threads to use (default: 4)
- `-l, --log`: Log output to file
- `--verify-salt`: Verify the predicted address for a given salt

## Technical Details

### CREATE3 Address Prediction

The tool uses the exact same address prediction logic as Solady's CREATE3 implementation:

1. First predicts the proxy address using CREATE2:
   ```solidity
   proxyAddress = CREATE2(salt, PROXY_INITCODE_HASH)
   ```

2. Then predicts the final contract address using RLP encoding:
   ```solidity
   finalAddress = keccak256(RLP([proxyAddress, 1]))[12:]
   ```

### Cross-Chain Compatibility

The generated addresses will be the same across all EVM chains when:
1. Using the same factory contract address
2. Using the same salt
3. Deploying the same contract bytecode

### Factory Contract

Default factory (Mainnet): `0xD37A8B489DbF221D614779cde7A99197dF25CE9C`

## Example Output

```json
{
  "salt": "0x849b3d15d82f3579ab0a2e5847624ff9625c29ea3c422306bf737b95690b9c1f",
  "contractAddress": "0x00003BcF3B8DAEfF0eB1c8d60ddA90ce43179968",
  "deployer": "0xD37A8B489DbF221D614779cde7A99197dF25CE9C"
}
```

## Security

- The tool uses cryptographically secure random number generation for salts
- All address predictions are verified before returning results
- CPU throttling prevents system overload

## License

MIT



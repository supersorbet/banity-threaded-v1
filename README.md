# CREATE3 Vanity Address Generator

A tool for generating deterministic vanity addresses for CREATE3 contract deployments, compatible with Solady's CREATE3 implementation.

## Features

- Generate vanity addresses for CREATE3 contract deployments
- Verify existing salt and address predictions
- Cross-chain compatible (same address on all chains)
- GPU acceleration with NVIDIA CUDA support
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

Example (using GPU acceleration):
```bash
node create3-vanity.js 0000 --gpu
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
- `--gpu`: Use GPU acceleration if available (requires CUDA)
- `--gpu-threads`: Number of GPU threads to use for CUDA (default: 1024)
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

## GPU Acceleration

This tool supports GPU acceleration using NVIDIA CUDA:

### Quick Start Guide for GPU Mining

1. **Prerequisites**
   - NVIDIA GPU with CUDA support
   - CUDA toolkit installed on your system ([NVIDIA CUDA Download](https://developer.nvidia.com/cuda-downloads))

2. **Build the CUDA binary**
   ```bash
   npm run build-cuda
   ```

3. **Run the vanity address generator with GPU support**
   ```bash
   node create3-vanity.js 0000 --gpu
   ```

4. **Monitor progress**
   The tool will display:
   - Number of addresses checked per second
   - Estimated time to completion
   - Real-time progress updates

5. **Collect your result**
   When a matching address is found, the tool will output the salt and corresponding address.

For longer prefixes (5+ characters), be prepared for longer wait times even with GPU acceleration.

### GPU Performance

For reference, here's a rough performance comparison:

| Prefix Length | CPU (4 cores) | GPU (GTX 1060) | Speedup |
|---------------|---------------|----------------|---------|
| 4 characters  | ~2-3 min      | ~10-15 sec     | ~10x    |
| 5 characters  | ~45-60 min    | ~4-5 min       | ~12x    |
| 6 characters  | ~10-12 hours  | ~1-1.5 hours   | ~10x    |

Performance will vary based on your specific GPU model and system configuration.

### Advanced GPU Options

- `--gpu-threads`: Control the number of GPU threads (default: 1024)
  ```bash
  node create3-vanity.js 0000 --gpu --gpu-threads 2048
  ```

### AMD GPU Support

Currently, the CREATE3 vanity address generation only supports NVIDIA GPUs through CUDA. 

AMD GPU support via OpenCL is planned for future releases. The codebase includes some OpenCL infrastructure for the standard vanity address generation, but this has not yet been implemented for CREATE3 addresses.

If you need to use AMD GPUs, please use the CPU mode for now:
```bash
node create3-vanity.js 0000 -cpu 8
```
any issues encountered send a message @ t.me/supersorbet . this has not been tested with anything other than older GPUs (gtx 1660)



#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "keccak_cuda.h"

// Solady's CREATE3 proxy initcode hash (hardcoded from JavaScript)
__constant__ unsigned char PROXY_INITCODE_HASH[32] = {
    0x21, 0xc3, 0x5d, 0xbe, 0x1b, 0x34, 0x4a, 0x24, 0x88, 0xcf, 0x33, 0x21, 0xd6, 0xce, 
    0x54, 0x2f, 0x8e, 0x9f, 0x30, 0x55, 0x44, 0xff, 0x09, 0xe4, 0x99, 0x3a, 0x62, 0x31, 
    0x9a, 0x49, 0x7c, 0x1f
};

__device__ void create2Address(const unsigned char* deployer, const unsigned char* salt, unsigned char* output) {
    // CREATE2 address calculation: keccak256(0xff ++ deployer ++ salt ++ keccak256(init_code))[12:]
    unsigned char buffer[1 + 20 + 32 + 32];
    buffer[0] = 0xff;
    memcpy(buffer + 1, deployer, 20);
    memcpy(buffer + 21, salt, 32);
    memcpy(buffer + 53, PROXY_INITCODE_HASH, 32);
    
    unsigned char hash[32];
    keccak256_cuda(buffer, sizeof(buffer), hash);
    
    // Copy last 20 bytes as address
    memcpy(output, hash + 12, 20);
}

__device__ void create3Address(const unsigned char* deployer, const unsigned char* salt, unsigned char* output) {
    // First predict the proxy address using CREATE2
    unsigned char proxyAddress[20];
    create2Address(deployer, salt, proxyAddress);
    
    // Then predict the final contract address (nonce 1)
    // RLP encoding: 0xd6 + 0x94 + proxyAddress + 0x01
    unsigned char rlpEncoded[23];
    rlpEncoded[0] = 0xd6;  // 0xc0 (short RLP prefix) + 0x16 (length)
    rlpEncoded[1] = 0x94;  // 0x80 + 0x14 (address length)
    memcpy(rlpEncoded + 2, proxyAddress, 20); 
    rlpEncoded[22] = 0x01; // nonce
    
    unsigned char hash[32];
    keccak256_cuda(rlpEncoded, sizeof(rlpEncoded), hash);
    
    // Copy last 20 bytes as address
    memcpy(output, hash + 12, 20);
}

__global__ void create3_vanity_kernel(unsigned char* rnd_state, const unsigned char* deployer, unsigned char* result_salt, 
                                      unsigned char* result_addr, int* found, unsigned char* prefix, size_t prefix_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Generate unique salt for this thread based on the random state
    unsigned char salt[32];
    for (int i = 0; i < 32; i++) {
        salt[i] = rnd_state[i] ^ (idx & 0xFF) ^ ((idx >> 8) & 0xFF);
    }
    
    // Compute CREATE3 address
    unsigned char addr[20];
    create3Address(deployer, salt, addr);
    
    // Convert beginning of address to hex for comparison
    char addr_hex[41];
    for (int i = 0; i < 20; i++) {
        // Convert byte to hex (lower case)
        unsigned char byte = addr[i];
        addr_hex[i*2] = (byte >> 4) <= 9 ? '0' + (byte >> 4) : 'a' + (byte >> 4) - 10;
        addr_hex[i*2+1] = (byte & 0xF) <= 9 ? '0' + (byte & 0xF) : 'a' + (byte & 0xF) - 10;
    }
    addr_hex[40] = '\0';
    
    // Check if the beginning matches our target prefix
    bool matches = true;
    for (size_t i = 0; i < prefix_len; i++) {
        if (addr_hex[i] != prefix[i]) {
            matches = false;
            break;
        }
    }
    
    if (matches && atomicExch(found, 1) == 0) {
        // Copy results (only first thread that finds match)
        memcpy(result_salt, salt, 32);
        memcpy(result_addr, addr, 20);
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <prefix> <deployer_hex> <threads>\n", argv[0]);
        return 1;
    }
    
    const char* prefix = argv[1];
    const char* deployer_hex = argv[2];
    const int num_threads = atoi(argv[3]);
    
    // Validate prefix (must be valid hex characters)
    for (size_t i = 0; i < strlen(prefix); i++) {
        char c = prefix[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
            fprintf(stderr, "Error: Prefix must contain only hex characters (0-9, a-f, A-F)\n");
            return 1;
        }
    }
    
    // Convert deployer from hex to bytes (skip 0x if present)
    unsigned char deployer[20];
    const char* deployer_ptr = deployer_hex;
    if (strncmp(deployer_hex, "0x", 2) == 0) {
        deployer_ptr += 2;
    }
    
    if (strlen(deployer_ptr) != 40) {
        fprintf(stderr, "Error: Deployer must be a 20-byte (40 hex char) Ethereum address\n");
        return 1;
    }
    
    for (int i = 0; i < 20; i++) {
        char hex[3] = {deployer_ptr[i*2], deployer_ptr[i*2+1], 0};
        deployer[i] = (unsigned char)strtol(hex, NULL, 16);
    }
    
    // Allocate device memory
    unsigned char *d_rnd_state, *d_deployer, *d_result_salt, *d_result_addr, *d_prefix;
    int *d_found;
    
    cudaMalloc(&d_rnd_state, 32);
    cudaMalloc(&d_deployer, 20);
    cudaMalloc(&d_result_salt, 32);
    cudaMalloc(&d_result_addr, 20);
    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_prefix, strlen(prefix));
    
    // Generate random initial state
    unsigned char h_rnd_state[32];
    srand(time(NULL));
    for (int i = 0; i < 32; i++) {
        h_rnd_state[i] = rand() & 0xFF;
    }
    
    // Convert prefix to lowercase
    unsigned char* prefix_lower = (unsigned char*)malloc(strlen(prefix));
    for (size_t i = 0; i < strlen(prefix); i++) {
        if (prefix[i] >= 'A' && prefix[i] <= 'F') {
            prefix_lower[i] = prefix[i] - 'A' + 'a';
        } else {
            prefix_lower[i] = prefix[i];
        }
    }
    
    // Copy data to device
    cudaMemcpy(d_rnd_state, h_rnd_state, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_deployer, deployer, 20, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix, prefix_lower, strlen(prefix), cudaMemcpyHostToDevice);
    
    // Initialize found flag
    int h_found = 0;
    cudaMemcpy(d_found, &h_found, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int block_size = 256;
    int num_blocks = (num_threads + block_size - 1) / block_size;
    
    fprintf(stderr, "Starting CREATE3 vanity address search for prefix: 0x%s\n", prefix);
    fprintf(stderr, "Using %d threads across %d blocks\n", num_threads, num_blocks);
    
    unsigned int iterations = 0;
    while (h_found == 0) {
        iterations++;
        create3_vanity_kernel<<<num_blocks, block_size>>>(d_rnd_state, d_deployer, d_result_salt, 
                                                         d_result_addr, d_found, d_prefix, strlen(prefix));
        
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
            break;
        }
        
        // Check if found
        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Update random state if not found
        if (h_found == 0) {
            for (int i = 0; i < 32; i++) {
                h_rnd_state[i] = (h_rnd_state[i] + iterations) ^ (rand() & 0xFF);
            }
            cudaMemcpy(d_rnd_state, h_rnd_state, 32, cudaMemcpyHostToDevice);
            
            // Print progress every 100 iterations
            if (iterations % 100 == 0) {
                fprintf(stderr, "Completed %d iterations...\n", iterations);
            }
        }
    }
    
    // Get results
    unsigned char h_result_salt[32];
    unsigned char h_result_addr[20];
    cudaMemcpy(h_result_salt, d_result_salt, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_addr, d_result_addr, 20, cudaMemcpyDeviceToHost);
    
    // Output JSON result
    printf("{\"salt\":\"0x");
    for (int i = 0; i < 32; i++) {
        printf("%02x", h_result_salt[i]);
    }
    printf("\",\"contractAddress\":\"0x");
    for (int i = 0; i < 20; i++) {
        printf("%02x", h_result_addr[i]);
    }
    printf("\",\"deployer\":\"0x");
    for (int i = 0; i < 20; i++) {
        printf("%02x", deployer[i]);
    }
    printf("\"}\n");
    
    // Cleanup
    free(prefix_lower);
    cudaFree(d_rnd_state);
    cudaFree(d_deployer);
    cudaFree(d_result_salt);
    cudaFree(d_result_addr);
    cudaFree(d_found);
    cudaFree(d_prefix);
    
    return 0;
} 
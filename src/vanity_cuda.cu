#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "keccak.h"

// CUDA kernel for Keccak-256 hashing
__global__ void keccak256_kernel(const unsigned char* input, size_t input_len, unsigned char* output, int* found, const char* target, size_t target_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char hash[32];
    unsigned char hex[64];
    
    // Generate unique private key for this thread
    unsigned char priv_key[32];
    for (int i = 0; i < 32; i++) {
        priv_key[i] = input[i] ^ (idx & 0xFF);
    }
    
    // Compute Keccak-256 hash
    keccak256(priv_key, 32, hash);
    
    // Convert to hex and check if matches target
    for (int i = 0; i < 32; i++) {
        sprintf((char*)&hex[i*2], "%02x", hash[i]);
    }
    
    // Check if the start matches our target
    bool matches = true;
    for (size_t i = 0; i < target_len && i < 40; i++) {
        if (hex[i] != target[i]) {
            matches = false;
            break;
        }
    }
    
    if (matches) {
        *found = idx;
        // Copy result to output
        memcpy(output, priv_key, 32);
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <pattern> <is_checksum> <is_contract> <threads>\n", argv[0]);
        return 1;
    }
    
    const char* target = argv[1];
    const int is_checksum = atoi(argv[2]);
    const int is_contract = atoi(argv[3]);
    const int num_threads = atoi(argv[4]);
    
    // Allocate device memory
    unsigned char *d_input, *d_output;
    int *d_found;
    cudaMalloc(&d_input, 32);
    cudaMalloc(&d_output, 32);
    cudaMalloc(&d_found, sizeof(int));
    
    // Generate random input
    unsigned char h_input[32];
    srand(time(NULL));
    for (int i = 0; i < 32; i++) {
        h_input[i] = rand() & 0xFF;
    }
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, 32, cudaMemcpyHostToDevice);
    
    // Initialize found flag
    int h_found = -1;
    cudaMemcpy(d_found, &h_found, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int block_size = 256;
    int num_blocks = (num_threads + block_size - 1) / block_size;
    
    while (h_found == -1) {
        keccak256_kernel<<<num_blocks, block_size>>>(d_input, 32, d_output, d_found, target, strlen(target));
        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Generate new random input if not found
        if (h_found == -1) {
            for (int i = 0; i < 32; i++) {
                h_input[i] = rand() & 0xFF;
            }
            cudaMemcpy(d_input, h_input, 32, cudaMemcpyHostToDevice);
        }
    }
    
    // Get the result
    unsigned char result[32];
    cudaMemcpy(result, d_output, 32, cudaMemcpyDeviceToHost);
    
    // Convert to JSON output
    printf("{\"privKey\":\"");
    for (int i = 0; i < 32; i++) {
        printf("%02x", result[i]);
    }
    printf("\",\"address\":\"0x");
    
    // Compute the address
    unsigned char address[20];
    keccak256(result, 32, address);
    for (int i = 0; i < 20; i++) {
        printf("%02x", address[i]);
    }
    printf("\"}\n");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_found);
    
    return 0;
} 
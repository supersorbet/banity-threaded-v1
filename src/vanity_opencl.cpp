#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "keccak.h"

// OpenCL kernel source
const char* kernelSource = R"(
    __kernel void keccak256_kernel(
        __global const uchar* input,
        __global uchar* output,
        __global int* found,
        __global const char* target,
        const uint target_len
    ) {
        int idx = get_global_id(0);
        uchar priv_key[32];
        uchar hash[32];
        char hex[64];
        
        // Generate unique private key for this work item
        for (int i = 0; i < 32; i++) {
            priv_key[i] = input[i] ^ (idx & 0xFF);
        }
        
        // Compute Keccak-256 hash
        keccak256(priv_key, 32, hash);
        
        // Convert to hex
        for (int i = 0; i < 32; i++) {
            sprintf(&hex[i*2], "%02x", hash[i]);
        }
        
        // Check if matches target
        bool matches = true;
        for (uint i = 0; i < target_len && i < 40; i++) {
            if (hex[i] != target[i]) {
                matches = false;
                break;
            }
        }
        
        if (matches) {
            atomic_xchg(found, idx);
            // Copy result
            for (int i = 0; i < 32; i++) {
                output[i] = priv_key[i];
            }
        }
    }
)";

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <pattern> <is_checksum> <is_contract> <threads>\n", argv[0]);
        return 1;
    }
    
    const char* target = argv[1];
    const int is_checksum = atoi(argv[2]);
    const int is_contract = atoi(argv[3]);
    const int num_threads = atoi(argv[4]);
    const unsigned int target_len = strlen(target);  // Store length in variable
    
    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;
    
    // Get platform and device
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    // Create context and command queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    
    // Use newer command queue creation for OpenCL 2.0+
    #ifdef CL_VERSION_2_0
        cl_queue_properties props[] = {0};
        queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    #else
        queue = clCreateCommandQueue(context, device, 0, &err);
    #endif
    
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue\n");
        return 1;
    }
    
    // Create and build program
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    if (clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "Build error: %s\n", buffer);
        return 1;
    }
    
    // Create kernel
    kernel = clCreateKernel(program, "keccak256_kernel", NULL);
    
    // Create buffers
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_WRITE, 32, NULL, NULL);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, 32, NULL, NULL);
    cl_mem d_found = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
    cl_mem d_target = clCreateBuffer(context, CL_MEM_READ_ONLY, target_len, NULL, NULL);
    
    // Generate random input
    unsigned char h_input[32];
    srand(time(NULL));
    for (int i = 0; i < 32; i++) {
        h_input[i] = rand() & 0xFF;
    }
    
    // Copy input data
    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, 32, h_input, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_target, CL_TRUE, 0, target_len, target, 0, NULL, NULL);
    
    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_found);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_target);
    clSetKernelArg(kernel, 4, sizeof(unsigned int), &target_len);
    
    // Launch kernel
    size_t global_size = num_threads;
    size_t local_size = 256;
    
    int h_found = -1;
    while (h_found == -1) {
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, d_found, CL_TRUE, 0, sizeof(int), &h_found, 0, NULL, NULL);
        
        if (h_found == -1) {
            // Generate new random input
            for (int i = 0; i < 32; i++) {
                h_input[i] = rand() & 0xFF;
            }
            clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, 32, h_input, 0, NULL, NULL);
        }
    }
    
    // Get result
    unsigned char result[32];
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, 32, result, 0, NULL, NULL);
    
    // Output JSON
    printf("{\"privKey\":\"");
    for (int i = 0; i < 32; i++) {
        printf("%02x", result[i]);
    }
    printf("\",\"address\":\"0x");
    
    // Compute address
    unsigned char address[20];
    keccak256(result, 32, address);
    for (int i = 0; i < 20; i++) {
        printf("%02x", address[i]);
    }
    printf("\"}\n");
    
    // Cleanup
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_found);
    clReleaseMemObject(d_target);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
} 
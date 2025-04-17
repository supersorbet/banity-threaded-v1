#ifndef KECCAK_CUDA_H
#define KECCAK_CUDA_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// Keccak-256 constants for CUDA
#define KECCAK_ROUNDS 24
#define KECCAK_STATE_SIZE 25
#define KECCAK_RATE 136

// CUDA device constants
__constant__ uint64_t d_keccakf_rndc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__constant__ int d_keccakf_rotc[24] = {
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

__constant__ int d_keccakf_piln[24] = {
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

// Keccak-f permutation for CUDA
__device__ void keccakf(uint64_t st[25]) {
    uint64_t t, bc[5];
    int i, j, r;

    for (r = 0; r < KECCAK_ROUNDS; r++) {
        // Theta
        for (i = 0; i < 5; i++) {
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        }

        for (i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ ((bc[(i + 1) % 5] << 1) | (bc[(i + 1) % 5] >> 63));
            for (j = 0; j < 25; j += 5) {
                st[j + i] ^= t;
            }
        }

        // Rho Pi
        t = st[1];
        for (i = 0; i < 24; i++) {
            j = d_keccakf_piln[i];
            bc[0] = st[j];
            st[j] = ((t << d_keccakf_rotc[i]) | (t >> (64 - d_keccakf_rotc[i])));
            t = bc[0];
        }

        // Chi
        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++) {
                bc[i] = st[j + i];
            }
            for (i = 0; i < 5; i++) {
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
            }
        }

        // Iota
        st[0] ^= d_keccakf_rndc[r];
    }
}

// Keccak-256 hash function for CUDA
__device__ void keccak256_cuda(const unsigned char *in, size_t inlen, unsigned char *out) {
    uint64_t st[25];
    unsigned char temp[144];
    size_t i, rsiz = 136, rsizw = 136 / 8;

    // Initialize state
    for (i = 0; i < 25; i++) {
        st[i] = 0;
    }

    // Absorb input
    while (inlen >= rsiz) {
        for (i = 0; i < rsizw; i++) {
            st[i] ^= ((uint64_t *) in)[i];
        }
        keccakf(st);
        in += rsiz;
        inlen -= rsiz;
    }

    // Last block
    memcpy(temp, in, inlen);
    temp[inlen++] = 1;
    memset(temp + inlen, 0, rsiz - inlen);
    temp[rsiz - 1] |= 0x80;

    for (i = 0; i < rsizw; i++) {
        st[i] ^= ((uint64_t *) temp)[i];
    }

    keccakf(st);

    // Output
    memcpy(out, st, 32);
}

#endif // KECCAK_CUDA_H 
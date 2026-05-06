#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define MAX 100

// Kernel
__global__ void buildString(char *S, char *RS, int n) {
    int i = threadIdx.x;

    if (i < n) {
        int start = 0;

        // Compute starting index for this thread
        for (int k = 0; k < i; k++) {
            start += (n - k);
        }

        // Copy (n - i) characters
        for (int j = 0; j < (n - i); j++) {
            RS[start + j] = S[j];
        }
    }
}

int main() {
    char h_S[MAX] = "PCAP";
    int n = strlen(h_S);

    // Total output length = n(n+1)/2
    int total_len = n * (n + 1) / 2;

    char h_RS[MAX];

    char *d_S, *d_RS;

    // Allocate device memory
    cudaMalloc((void**)&d_S, n * sizeof(char));
    cudaMalloc((void**)&d_RS, total_len * sizeof(char));

    // Copy input to device
    cudaMemcpy(d_S, h_S, n * sizeof(char), cudaMemcpyHostToDevice);

    // Launch kernel
    buildString<<<1, n>>>(d_S, d_RS, n);

    // Copy result back
    cudaMemcpy(h_RS, d_RS, total_len * sizeof(char), cudaMemcpyDeviceToHost);

    // Add null terminator for printing
    h_RS[total_len] = '\0';

    // Output
    printf("Input S: %s\n", h_S);
    printf("Output RS: %s\n", h_RS);

    // Free memory
    cudaFree(d_S);
    cudaFree(d_RS);

    return 0;
}
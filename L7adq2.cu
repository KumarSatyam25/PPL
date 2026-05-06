#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define MAX 100

__global__ void repeatString(char *Sin, char *Sout, int len, int N) {
    int i = threadIdx.x;

    if (i < len) {
        for (int j = 0; j < N; j++) {
            Sout[i + j * len] = Sin[i];
        }
    }
}

int main() {
    char h_Sin[MAX] = "Hello";
    int N = 3;

    int len = strlen(h_Sin);
    int total_len = len * N;

    char h_Sout[MAX];

    char *d_Sin, *d_Sout;

    cudaMalloc((void**)&d_Sin, len * sizeof(char));
    cudaMalloc((void**)&d_Sout, total_len * sizeof(char));

    cudaMemcpy(d_Sin, h_Sin, len * sizeof(char), cudaMemcpyHostToDevice);

    // Launch kernel
    repeatString<<<1, len>>>(d_Sin, d_Sout, len, N);

    cudaMemcpy(h_Sout, d_Sout, total_len * sizeof(char), cudaMemcpyDeviceToHost);

    h_Sout[total_len] = '\0';

    printf("Input : %s\n", h_Sin);
    printf("Output: %s\n", h_Sout);

    cudaFree(d_Sin);
    cudaFree(d_Sout);

    return 0;
}
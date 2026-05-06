#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define MAX 100

// Kernel: reverse each word
__global__ void reverseWords(char *input, char *output,
                             int *start, int *end, int numWords) {

    int i = threadIdx.x;

    if (i < numWords) {
        int s = start[i];
        int e = end[i];

        // Reverse the word
        for (int j = s; j <= e; j++) {
            output[j] = input[e - (j - s)];
        }
    }
}

int main() {
    char h_input[MAX] = "CUDA is very powerful";
    int len = strlen(h_input);

    char h_output[MAX];

    int start[MAX], end[MAX];
    int numWords = 0;

    // Find word boundaries (CPU)
    int i = 0;
    while (i < len) {
        while (i < len && h_input[i] == ' ') i++;

        if (i < len) {
            start[numWords] = i;

            while (i < len && h_input[i] != ' ') i++;

            end[numWords] = i - 1;
            numWords++;
        }
    }

    // Copy spaces as is
    for (int i = 0; i < len; i++) {
        if (h_input[i] == ' ') {
            h_output[i] = ' ';
        }
    }

    // Device memory
    char *d_input, *d_output;
    int *d_start, *d_end;

    cudaMalloc((void**)&d_input, len);
    cudaMalloc((void**)&d_output, len);
    cudaMalloc((void**)&d_start, numWords * sizeof(int));
    cudaMalloc((void**)&d_end, numWords * sizeof(int));

    cudaMemcpy(d_input, h_input, len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_start, start, numWords * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end, end, numWords * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    reverseWords<<<1, numWords>>>(d_input, d_output,
                                  d_start, d_end, numWords);

    cudaMemcpy(h_output, d_output, len, cudaMemcpyDeviceToHost);

    h_output[len] = '\0';

    printf("Input : %s\n", h_input);
    printf("Output: %s\n", h_output);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_start);
    cudaFree(d_end);

    return 0;
}
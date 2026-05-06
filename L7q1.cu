#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define MAX 100

// Kernel to count word occurrences
__global__ void countWord(char *text, char *word, int *count,
                         int text_len, int word_len) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i <= text_len - word_len) {

        int match = 1;

        for (int j = 0; j < word_len; j++) {
            if (text[i + j] != word[j]) {
                match = 0;
                break;
            }
        }

        if (match) {
            atomicAdd(count, 1);
        }
    }
}

int main() {
    char h_text[MAX] = "cuda is fast and cuda is powerful and cuda is parallel";
    char h_word[MAX] = "cuda";

    int text_len = strlen(h_text);
    int word_len = strlen(h_word);

    int h_count = 0;

    char *d_text, *d_word;
    int *d_count;

    cudaMalloc((void**)&d_text, text_len);
    cudaMalloc((void**)&d_word, word_len);
    cudaMalloc((void**)&d_count, sizeof(int));

    cudaMemcpy(d_text, h_text, text_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, h_word, word_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (text_len + blockSize - 1) / blockSize;

    countWord<<<gridSize, blockSize>>>(d_text, d_word, d_count,
                                      text_len, word_len);

    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Word '%s' occurs %d times\n", h_word, h_count);

    cudaFree(d_text);
    cudaFree(d_word);
    cudaFree(d_count);

    return 0;
}
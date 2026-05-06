#include <stdio.h>
#include <cuda.h>

#define M 4
#define N 4

__global__ void processMatrix(int *A, int *B, int rows, int cols) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {

        int val = A[i * cols + j];

        // Check border
        if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
            B[i * cols + j] = val;   // keep same
        } else {
            B[i * cols + j] = ~val;  // 1's complement
        }
    }
}

int main() {

    int h_A[M][N] = {
        {1, 2, 3, 4},
        {6, 5, 8, 3},
        {2, 4, 10, 1},
        {9, 1, 2, 5}
    };

    int h_B[M][N];

    int *d_A, *d_B;

    cudaMalloc((void**)&d_A, M*N*sizeof(int));
    cudaMalloc((void**)&d_B, M*N*sizeof(int));

    cudaMemcpy(d_A, h_A, M*N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(16,16);
    dim3 gridSize((N+15)/16, (M+15)/16);

    processMatrix<<<gridSize, blockSize>>>(d_A, d_B, M, N);

    cudaMemcpy(h_B, d_B, M*N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Output Matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_B[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
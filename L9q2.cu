#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define M 3
#define N 3

__global__ void rowPower(int *A, int *B, int rows, int cols) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {

        int val = A[i * cols + j];
        int power = i + 1;

        int result = 1;

        // Compute val^(power)
        for (int k = 0; k < power; k++) {
            result *= val;
        }

        B[i * cols + j] = result;
    }
}

int main() {
    int h_A[M][N] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    int h_B[M][N];

    int *d_A, *d_B;

    cudaMalloc((void**)&d_A, M*N*sizeof(int));
    cudaMalloc((void**)&d_B, M*N*sizeof(int));

    cudaMemcpy(d_A, h_A, M*N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(16,16);
    dim3 gridSize((N+15)/16, (M+15)/16);

    rowPower<<<gridSize, blockSize>>>(d_A, d_B, M, N);

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
#include <stdio.h>
#include <cuda.h>

#define N 3   // square matrix N x N

// =============================
// (a) One thread per row
// =============================
__global__ void mulRow(int *A, int *B, int *C, int n) {
    int i = threadIdx.x;

    if (i < n) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i*n + k] * B[k*n + j]; //i-> row, j->col, k-> kth element
            }
            C[i*n + j] = sum;
        }
    }
}

// =============================
// (b) One thread per column
// =============================
__global__ void mulCol(int *A, int *B, int *C, int n) {
    int j = threadIdx.x;

    if (j < n) {
        for (int i = 0; i < n; i++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

// =============================
// (c) One thread per element
// =============================
__global__ void mulElement(int *A, int *B, int *C, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[i*n + k] * B[k*n + j];
        }
        C[i*n + j] = sum;
    }
}

int main() {
    int h_A[N][N] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    int h_B[N][N] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };

    int h_C[N][N];

    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, N*N*sizeof(int));
    cudaMalloc((void**)&d_B, N*N*sizeof(int));
    cudaMalloc((void**)&d_C, N*N*sizeof(int));

    cudaMemcpy(d_A, h_A, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(int), cudaMemcpyHostToDevice);

    // =============================
    // (a) Row-wise
    // =============================
    mulRow<<<1, N>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nRow-wise Result:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_C[i][j]);
        }
        printf("\n");
    }

    // =============================
    // (b) Column-wise
    // =============================
    mulCol<<<1, N>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nColumn-wise Result:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_C[i][j]);
        }
        printf("\n");
    }

    // =============================
    // (c) Element-wise
    // =============================
    dim3 blockSize(16,16);
    dim3 gridSize((N+15)/16, (N+15)/16);

    mulElement<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nElement-wise Result:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_C[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
#include <stdio.h>
#include <cuda.h>

#define ROWS 3
#define COLS 3

// (a) One thread per row
__global__ void addRow(int *A, int *B, int *C, int rows, int cols) {
    int i = threadIdx.x;

    if (i < rows) {
        for (int j = 0; j < cols; j++) {
            C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
        }
    }
}

// (b) One thread per column
__global__ void addCol(int *A, int *B, int *C, int rows, int cols) {
    int j = threadIdx.x;

    if (j < cols) {
        for (int i = 0; i < rows; i++) {
            C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
        }
    }
}

// (c) One thread per element
__global__ void addElement(int *A, int *B, int *C, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
    }
}

int main() {
    int h_A[ROWS][COLS] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    int h_B[ROWS][COLS] = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };

    int h_C[ROWS][COLS];

    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, ROWS * COLS * sizeof(int));
    cudaMalloc((void**)&d_B, ROWS * COLS * sizeof(int));
    cudaMalloc((void**)&d_C, ROWS * COLS * sizeof(int));

    cudaMemcpy(d_A, h_A, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);

    // =============================
    // (a) Row-wise
    // =============================
    addRow<<<1, ROWS>>>(d_A, d_B, d_C, ROWS, COLS);
    cudaMemcpy(h_C, d_C, ROWS * COLS * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nRow-wise Result:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%d ", h_C[i][j]);
        }
        printf("\n");
    }

    // =============================
    // (b) Column-wise
    // =============================
    addCol<<<1, COLS>>>(d_A, d_B, d_C, ROWS, COLS);
    cudaMemcpy(h_C, d_C, ROWS * COLS * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nColumn-wise Result:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%d ", h_C[i][j]);
        }
        printf("\n");
    }

    // =============================
    // (c) Element-wise
    // =============================
    dim3 blockSize(16, 16);
    dim3 gridSize((COLS + 15) / 16, (ROWS + 15) / 16);

    addElement<<<gridSize, blockSize>>>(d_A, d_B, d_C, ROWS, COLS);
    cudaMemcpy(h_C, d_C, ROWS * COLS * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nElement-wise Result:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%d ", h_C[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
#include <stdio.h>
#include <cuda.h>

#define N 4

__global__ void matMul(int *A, int *B, int *C, int n) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;

        for (int k = 0; k < n; k++) {
            sum += A[row*n + k] * B[k*n + col];
        }

        C[row*n + col] = sum;
    }
}

int main() {
    int h_A[N][N], h_B[N][N], h_C[N][N];

    // Initialize matrices
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            h_A[i][j] = i + j;
            h_B[i][j] = i == j ? 1 : 0; // identity
        }

    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, N*N*sizeof(int));
    cudaMalloc((void**)&d_B, N*N*sizeof(int));
    cudaMalloc((void**)&d_C, N*N*sizeof(int));

    cudaMemcpy(d_A, h_A, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(16,16);
    dim3 gridSize((N+15)/16, (N+15)/16);

    matMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result:\n");
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++)
            printf("%d ", h_C[i][j]);
        printf("\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
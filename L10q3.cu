#include <stdio.h>
#include <cuda.h>

#define N 8
#define MASK_WIDTH 3
#define TILE_SIZE 4

__constant__ int M[MASK_WIDTH];

__global__ void tiledConv1D(int *N_arr, int *P, int width) {

    __shared__ int tile[TILE_SIZE + MASK_WIDTH - 1];

    int tx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tx;

    int halo = MASK_WIDTH / 2;

    // Load shared memory
    int shared_idx = tx + halo;

    if (i < width)
        tile[shared_idx] = N_arr[i];
    else
        tile[shared_idx] = 0;

    // Left halo
    if (tx < halo) {
        if (i - halo >= 0)
            tile[tx] = N_arr[i - halo];
        else
            tile[tx] = 0;
    }

    // Right halo
    if (tx >= TILE_SIZE - halo) {
        if (i + halo < width)
            tile[tx + MASK_WIDTH - 1] = N_arr[i + halo];
        else
            tile[tx + MASK_WIDTH - 1] = 0;
    }

    __syncthreads();

    // Convolution
    if (i < width) {
        int sum = 0;

        for (int j = 0; j < MASK_WIDTH; j++) {
            sum += tile[tx + j] * M[j];
        }

        P[i] = sum;
    }
}

int main() {

    int h_N[N] = {1,2,3,4,5,6,7,8};
    int h_M[MASK_WIDTH] = {1,0,-1};
    int h_P[N];

    int *d_N, *d_P;

    cudaMalloc((void**)&d_N, N*sizeof(int));
    cudaMalloc((void**)&d_P, N*sizeof(int));

    cudaMemcpy(d_N, h_N, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, h_M, MASK_WIDTH*sizeof(int));

    int blockSize = TILE_SIZE;
    int gridSize = (N + TILE_SIZE - 1) / TILE_SIZE;

    tiledConv1D<<<gridSize, blockSize>>>(d_N, d_P, N);

    cudaMemcpy(h_P, d_P, N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Output:\n");
    for(int i=0;i<N;i++) printf("%d ", h_P[i]);

    cudaFree(d_N); cudaFree(d_P);
    return 0;
}
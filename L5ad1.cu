#include <stdio.h>
#include <cuda.h>

__global__ void axpy(float *x, float *y, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}

int main() {
    int n = 10;
    float alpha = 2.0;

    float h_x[n], h_y[n];

    // Initialize vectors
    for (int i = 0; i < n; i++) {
        h_x[i] = i;
        h_y[i] = i;
    }

    float *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);

    axpy<<<1, n>>>(d_x, d_y, alpha, n);

    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", h_y[i]);
    }

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
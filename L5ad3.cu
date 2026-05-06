#include <stdio.h>
#include <cuda.h>

__global__ void oddEvenSort(int *arr, int n, int phase) {
    int i = threadIdx.x;

    int idx = 2 * i + phase;

    if (idx + 1 < n) {
        if (arr[idx] > arr[idx + 1]) {
            int temp = arr[idx];
            arr[idx] = arr[idx + 1];
            arr[idx + 1] = temp;
        }
    }
}

int main() {
    int n = 10;
    int h_arr[10] = {9, 7, 5, 3, 1, 2, 4, 6, 8, 0};

    int *d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < n; i++) {
        // Even phase
        oddEvenSort<<<1, n / 2>>>(d_arr, n, 0);
        cudaDeviceSynchronize();

        // Odd phase
        oddEvenSort<<<1, n / 2>>>(d_arr, n, 1);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_arr[i]);
    }

    cudaFree(d_arr);
    return 0;
}
#include <stdio.h>
#include <cuda.h>

// Kernel: one thread per row
__global__ void spmv(int num_rows, int *row_ptr, int *col_index,
                     float *data, float *x, float *y) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float sum = 0.0;

        int start = row_ptr[row];
        int end   = row_ptr[row + 1];

        for (int j = start; j < end; j++) {
            sum += data[j] * x[col_index[j]];
        }

        y[row] = sum;
    }
}

int main() {
    // Example matrix (CSR form)
    int num_rows = 4;

    // Non-zero values
    float h_data[] = {3, 1, 2, 4, 1, 1, 1};

    // Column indices
    int h_col_index[] = {0, 2, 1, 2, 3, 0, 3};

    // Row pointer
    int h_row_ptr[] = {0, 2, 4, 5, 7};

    // Input vector x
    float h_x[] = {1, 2, 3, 4};

    float h_y[4];

    int nnz = 7;

    // Device memory
    float *d_data, *d_x, *d_y;
    int *d_col_index, *d_row_ptr;

    cudaMalloc((void**)&d_data, nnz * sizeof(float));
    cudaMalloc((void**)&d_col_index, nnz * sizeof(int));
    cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_x, 4 * sizeof(float));
    cudaMalloc((void**)&d_y, 4 * sizeof(float));

    // Copy to device
    cudaMemcpy(d_data, h_data, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_index, h_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, 4 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (num_rows + blockSize - 1) / blockSize;

    spmv<<<gridSize, blockSize>>>(num_rows, d_row_ptr, d_col_index,
                                 d_data, d_x, d_y);

    // Copy result back
    cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    printf("Result vector y:\n");
    for (int i = 0; i < num_rows; i++) {
        printf("%.2f ", h_y[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(d_data);
    cudaFree(d_col_index);
    cudaFree(d_row_ptr);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
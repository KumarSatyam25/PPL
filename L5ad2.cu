#include <stdio.h>
#include <cuda.h>

#define ROWS 3
#define COLS 5

__global__ void rowSelectionSort(int *mat) {
    int row = blockIdx.x;

    for (int i = 0; i < COLS - 1; i++) {
        int min_idx = i;

        for (int j = i + 1; j < COLS; j++) {
            if (mat[row * COLS + j] < mat[row * COLS + min_idx]) {
                min_idx = j;
            }
        }

        // Swap
        int temp = mat[row * COLS + i];
        mat[row * COLS + i] = mat[row * COLS + min_idx];
        mat[row * COLS + min_idx] = temp;
    }
}

int main() {
    int h_mat[ROWS][COLS] = {
        {5, 2, 9, 1, 3},
        {8, 4, 7, 6, 0},
        {3, 1, 2, 5, 4}
    };

    int *d_mat;
    cudaMalloc(&d_mat, ROWS * COLS * sizeof(int));

    cudaMemcpy(d_mat, h_mat, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);

    rowSelectionSort<<<ROWS, 1>>>(d_mat);

    cudaMemcpy(h_mat, d_mat, ROWS * COLS * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted rows:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%d ", h_mat[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_mat);
    return 0;
}
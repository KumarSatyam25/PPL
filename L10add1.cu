#include <stdio.h>
#include <cuda.h>

#define FRIENDS 3
#define ITEMS 4

// Kernel: each thread computes one friend's total
__global__ void computeTotal(int *purchases, int *prices, int *totals) {

    int f = threadIdx.x; // friend index

    int sum = 0;

    for (int i = 0; i < ITEMS; i++) {
        sum += purchases[f * ITEMS + i] * prices[i];
    }

    totals[f] = sum;
}

int main() {

    // Prices of items
    int h_prices[ITEMS] = {10, 20, 30, 40};

    // purchases[f][i] = quantity bought by friend f of item i
    int h_purchases[FRIENDS][ITEMS] = {
        {1, 2, 0, 1},  // Friend 0
        {0, 1, 1, 2},  // Friend 1
        {2, 0, 1, 1}   // Friend 2
    };

    int h_totals[FRIENDS];
    int final_total = 0;

    int *d_prices, *d_purchases, *d_totals;

    cudaMalloc((void**)&d_prices, ITEMS*sizeof(int));
    cudaMalloc((void**)&d_purchases, FRIENDS*ITEMS*sizeof(int));
    cudaMalloc((void**)&d_totals, FRIENDS*sizeof(int));

    cudaMemcpy(d_prices, h_prices, ITEMS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_purchases, h_purchases, FRIENDS*ITEMS*sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    computeTotal<<<1, FRIENDS>>>(d_purchases, d_prices, d_totals);

    cudaMemcpy(h_totals, d_totals, FRIENDS*sizeof(int), cudaMemcpyDeviceToHost);

    // Final sum on CPU
    for (int i = 0; i < FRIENDS; i++) {
        final_total += h_totals[i];
    }

    // Display results
    printf("Friend-wise totals:\n");
    for (int i = 0; i < FRIENDS; i++) {
        printf("Friend %d: %d\n", i, h_totals[i]);
    }

    printf("Total Purchase by all friends: %d\n", final_total);

    cudaFree(d_prices);
    cudaFree(d_purchases);
    cudaFree(d_totals);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>

const int M = 3;  // Number of rows
const int N = 4;  // Number of columns
const int K = 5;  // Number of columns

#define TILE_SIZE 16

void matrixMulOpenACC(int* A, int* B, int* C, int M, int N, int K) {
    #pragma acc data copyin(A[0:M*N], B[0:N*K]) copyout(C[0:M*K])
    {
        #pragma acc parallel loop tile(TILE_SIZE, TILE_SIZE) collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                int sum = 0;
                for (int t = 0; t < (N - 1) / TILE_SIZE + 1; t++) {
                    #pragma acc loop reduction(+:sum)
                    for (int i = 0; i < TILE_SIZE; i++) {
                        int aIndex = row * N + t * TILE_SIZE + i;
                        int bIndex = (t * TILE_SIZE + i) * K + col;
                        if (t * TILE_SIZE + i < N) {
                            sum += A[aIndex] * B[bIndex];
                        }
                    }
                }
                C[row * K + col] = sum;
            }
        }
    }
}

int main() {
    int *h_A, *h_B, *h_C;
    h_A = (int*) malloc(M * N * sizeof(int));
    h_B = (int*) malloc(N * K * sizeof(int));
    h_C = (int*) malloc(M * K * sizeof(int));

    // Initialize matrices
    for (int i = 0; i < M * N; ++i) h_A[i] = i + 1;
    for (int i = 0; i < N * K; ++i) h_B[i] = i + 1;

    matrixMulOpenACC(h_A, h_B, h_C, M, N, K);

    printf("Matrix C (OpenACC):\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            printf("%d ", h_C[i * K + j]);
        }
        printf("\n");
    }

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

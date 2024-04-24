#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>

const int M = 3;  
const int N = 4; 
const int K = 5;  

void matrixMulOpenACC(int* A, int* B, int* C, int M, int N, int K) {
    #pragma acc data copyin(A[0:M*N], B[0:N*K]) copyout(C[0:M*K])
    {
        #pragma acc parallel num_gangs(M) num_workers(1)
        {
            #pragma acc loop gang
            for (int row = 0; row < M; ++row) {
                #pragma acc loop worker
                for (int col = 0; col < K; ++col) {
                    int sum = 0;
                    #pragma acc loop vector reduction(+:sum)
                    for (int i = 0; i < N; ++i) {
                        sum += A[row * N + i] * B[i * K + col];
                    }
                    C[row * K + col] = sum;
                }
            }
        }
    }
}

int main() {
    int *h_A, *h_B, *h_C;
    h_A = (int*) malloc(M * N * sizeof(int));
    h_B = (int*) malloc(N * K * sizeof(int));
    h_C = (int*) malloc(M * K * sizeof(int));

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

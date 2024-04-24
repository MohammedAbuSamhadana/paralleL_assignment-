#include <stdio.h>
#include <stdlib.h>

// Function prototypes
void matrixMultiply(int **A, int **B, int **C, int M, int N, int P);
void printMatrix(int **matrix, int rows, int cols);

int main() {
    int M = 3, N = 2, P = 3;
    int i, j;

    // Dynamically allocate memory for matrices A, B, and C
    int *A = (int *)malloc(M * sizeof(int *));
    int *B = (int *)malloc(N * sizeof(int *));
    int *C = (int *)malloc(M * sizeof(int *));

    for (i = 0; i < M; i++) {
        A[i] = (int *)malloc(N * sizeof(int));
        C[i] = (int *)malloc(P * sizeof(int));
    }
    for (i = 0; i < N; i++) {
        B[i] = (int *)malloc(P * sizeof(int));
    }

    // Assume some initialization of matrices A and B
    printf("Matrix A:\n");
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = i + j; // Simple initialization for demonstration
        }
    }
    printMatrix(A, M, N);

    printf("\nMatrix B:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < P; j++) {
            B[i][j] = i * j; // Simple initialization for demonstration
        }
    }
    printMatrix(B, N, P);

    // Perform matrix multiplication
    matrixMultiply(A, B, C, M, N, P);

    // Print the result
    printf("\nMatrix C (Result):\n");
    printMatrix(C, M, P);

    // Free memory
    for (i = 0; i < M; i++) {
        free(A[i]);
        free(C[i]);
    }
    for (i = 0; i < N; i++) {
        free(B[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}

void matrixMultiply(int **A, int **B, int **C, int M, int N, int P) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            C[i][j] = 0; // Initialize the result matrix with 0
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void printMatrix(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

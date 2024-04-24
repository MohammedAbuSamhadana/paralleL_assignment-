#include <iostream>


const int M = 3;  // Number of rows
const int N = 4;  // Number of columns
const int K = 5;  // Number of columns

_global_ void matrixMulBasic(int* A, int* B, int* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

// Function to check CUDA errors
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "Error: " << message << ", " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

int main() {
    int *h_A = new int[M * N];
    int *h_B = new int[N * K];
    int *h_C_basic = new int[M * K];

    for (int i = 0; i < M * N; ++i) h_A[i] = i + 1;
    for (int i = 0; i < N * K; ++i) h_B[i] = i + 1;

    int *d_A, *d_B, *d_C_basic;
    checkCudaError(cudaMalloc((void**)&d_A, M * N * sizeof(int)), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc((void**)&d_B, N * K * sizeof(int)), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc((void**)&d_C_basic, M * K * sizeof(int)), "cudaMalloc d_C_basic failed");

    checkCudaError(cudaMemcpy(d_A, h_A, M * N * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_A failed");
    checkCudaError(cudaMemcpy(d_B, h_B, N * K * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_B failed");

    dim3 gridSize((K + 15) / 16, (M + 15) / 16);
    dim3 blockSize(16, 16);

    matrixMulBasic<<<gridSize, blockSize>>>(d_A, d_B, d_C_basic, M, N, K);
    checkCudaError(cudaPeekAtLastError(), "Kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    checkCudaError(cudaMemcpy(h_C_basic, d_C_basic, M * K * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy d_C_basic failed");

    std::cout << "Matrix C (Basic):\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            std::cout << h_C_basic[i * K + j] << " ";
        }

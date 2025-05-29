#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>

#define TILE_SIZE 16
#define NUM_RUNS 10
#define EPSILON 1e-5

__global__ void matmul_shared_kernel(const float *A, const float *B, float *C, int m, int n, int k) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < m && t * TILE_SIZE + threadIdx.x < n) {
            tileA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < k && t * TILE_SIZE + threadIdx.y < n) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * k + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < k) {
        C[row * k + col] = sum;
    }
}

__global__ void matmul_naive_kernel(const float *A, const float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

void initialize_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

bool compare_matrices(const float *A, const float *B, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (fabs(A[i * cols + j] - B[i * cols + j]) > EPSILON) {
                printf("Mismatch at (%d, %d): %.6f vs %.6f\n", 
                       i, j, A[i * cols + j], B[i * cols + j]);
                return false;
            }
        }
    }
    return true;
}

float benchmark_kernel(void (*kernel)(const float*, const float*, float*, int, int, int),
                     const float *d_A, const float *d_B, float *d_C,
                     int m, int n, int k, dim3 grid, dim3 block) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; ++i) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / NUM_RUNS;
}

void run_comparison(int m, int n, int k) {
    std::cout << "\nRunning comparison for " << m << "x" << n << " * " << n << "x" << k << " matrices...\n";

    float *h_A = new float[m * n];
    float *h_B = new float[n * k];
    float *h_C_naive = new float[m * k];
    float *h_C_shared = new float[m * k];

    initialize_matrix(h_A, m, n);
    initialize_matrix(h_B, n, k);

    float *d_A, *d_B, *d_C_naive, *d_C_shared;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, n * k * sizeof(float));
    cudaMalloc(&d_C_naive, m * k * sizeof(float));
    cudaMalloc(&d_C_shared, m * k * sizeof(float));

    cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * k * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_naive(16, 16);
    dim3 grid_naive((k + block_naive.x - 1) / block_naive.x,
                   (m + block_naive.y - 1) / block_naive.y);
    
    dim3 block_shared(TILE_SIZE, TILE_SIZE);
    dim3 grid_shared((k + TILE_SIZE - 1) / TILE_SIZE,
                    (m + TILE_SIZE - 1) / TILE_SIZE);

    matmul_naive_kernel<<<grid_naive, block_naive>>>(d_A, d_B, d_C_naive, m, n, k);
    cudaMemcpy(h_C_naive, d_C_naive, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    matmul_shared_kernel<<<grid_shared, block_shared>>>(d_A, d_B, d_C_shared, m, n, k);
    cudaMemcpy(h_C_shared, d_C_shared, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    bool results_match = compare_matrices(h_C_naive, h_C_shared, m, k);
    std::cout << "Results verification: " << (results_match ? "PASSED" : "FAILED") << "\n";

    if (results_match) {
        float naive_time = benchmark_kernel(matmul_naive_kernel, d_A, d_B, d_C_naive, 
                                          m, n, k, grid_naive, block_naive);
        
        float shared_time = benchmark_kernel(matmul_shared_kernel, d_A, d_B, d_C_shared, 
                                           m, n, k, grid_shared, block_shared);

        double flops = 2.0 * m * n * k;
        double naive_gflops = (flops / (naive_time * 1e6)) / 1e3;
        double shared_gflops = (flops / (shared_time * 1e6)) / 1e3;

        std::cout << "\nPerformance Results:\n";
        std::cout << "Naive kernel: " << naive_time << " ms (" << naive_gflops << " GFLOPS)\n";
        std::cout << "Shared memory kernel: " << shared_time << " ms (" << shared_gflops << " GFLOPS)\n";
        std::cout << "Speedup: " << naive_time / shared_time << "x\n";
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_naive;
    delete[] h_C_shared;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_C_shared);
}

int main() {
    run_comparison(256, 256, 256);
    run_comparison(1024, 1024, 1024);
    run_comparison(2048, 2048, 2048);

    return 0;
}

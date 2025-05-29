#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define TOLERANCE 1e-5f
#define NUM_RUNS 100
#define BLOCK_SIZE 256

__global__ void softmax_kernel(float *x, int batch_size, int size) {
    int b = blockIdx.x;
    if (b < batch_size) {
        float max_val = x[b * size];
        for (int i = 1; i < size; ++i) {
            max_val = fmaxf(max_val, x[b * size + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            x[b * size + i] = expf(x[b * size + i] - max_val);
            sum += x[b * size + i];
        }

        for (int i = 0; i < size; ++i) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}

__global__ void softmax_parallel_kernel(float *x, int batch_size, int size) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    
    if (b >= batch_size) return;
    
    extern __shared__ float shared[];
    float *max_shared = shared;
    float *sum_shared = &shared[blockDim.x];
    
    float max_val = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x) {
        max_val = fmaxf(max_val, x[b * size + i]);
    }
    max_shared[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            max_shared[tid] = fmaxf(max_shared[tid], max_shared[tid + s]);
        }
        __syncthreads();
    }
    max_val = max_shared[0];
    __syncthreads();
    
    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        sum += expf(x[b * size + i] - max_val);
    }
    sum_shared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_shared[tid] += sum_shared[tid + s];
        }
        __syncthreads();
    }
    sum = sum_shared[0];
    __syncthreads();
    
    for (int i = tid; i < size; i += blockDim.x) {
        x[b * size + i] = fmaxf(expf(x[b * size + i] - max_val) / sum, 1e-7f);
    }
}

void initialize_matrix(float *matrix, int batch_size, int size) {
    srand(time(NULL));
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < size; ++i) {
            matrix[b * size + i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
}

int verify_results(float *ref, float *test, int batch_size, int size) {
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < size; ++i) {
            float diff = fabsf(ref[b * size + i] - test[b * size + i]);
            if (diff > TOLERANCE) {
                printf("Mismatch at batch %d, element %d: ref=%.6f, test=%.6f\n",
                       b, i, ref[b * size + i], test[b * size + i]);
                return 0;
            }
        }
    }
    return 1;
}

float benchmark_kernel(void (*kernel)(float*, int, int),
                     float *d_input, float *d_output,
                     int batch_size, int size,
                     int threads_per_block, size_t shared_mem,
                     cudaEvent_t start, cudaEvent_t stop) {
    kernel<<<batch_size, threads_per_block, shared_mem>>>(d_output, batch_size, size);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; ++i) {
        kernel<<<batch_size, threads_per_block, shared_mem>>>(d_output, batch_size, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    return milliseconds / NUM_RUNS;
}

int main() {
    const int batch_size = 1024;
    const int size = 4096;
    const int threads_per_block = BLOCK_SIZE;
    
    float *h_input = (float*)malloc(batch_size * size * sizeof(float));
    float *h_output_seq = (float*)malloc(batch_size * size * sizeof(float));
    float *h_output_par = (float*)malloc(batch_size * size * sizeof(float));
    
    initialize_matrix(h_input, batch_size, size);
    
    float *d_input, *d_output_seq, *d_output_par;
    cudaMalloc(&d_input, batch_size * size * sizeof(float));
    cudaMalloc(&d_output_seq, batch_size * size * sizeof(float));
    cudaMalloc(&d_output_par, batch_size * size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, batch_size * size * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMemcpy(d_output_seq, d_input, batch_size * size * sizeof(float), cudaMemcpyDeviceToDevice);
    softmax_kernel<<<batch_size, 1>>>(d_output_seq, batch_size, size);
    cudaMemcpy(h_output_seq, d_output_seq, batch_size * size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(d_output_par, d_input, batch_size * size * sizeof(float), cudaMemcpyDeviceToDevice);
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    softmax_parallel_kernel<<<batch_size, threads_per_block, shared_mem_size>>>(d_output_par, batch_size, size);
    cudaMemcpy(h_output_par, d_output_par, batch_size * size * sizeof(float), cudaMemcpyDeviceToHost);
    
    int results_match = verify_results(h_output_seq, h_output_par, batch_size, size);
    printf("Results verification: %s\n", results_match ? "PASSED" : "FAILED");
    
    if (results_match) {
        float seq_time = benchmark_kernel(softmax_kernel, d_input, d_output_seq, 
                                        batch_size, size, 1, 0, start, stop);
        
        float par_time = benchmark_kernel(softmax_parallel_kernel, d_input, d_output_par,
                                        batch_size, size, threads_per_block, shared_mem_size, start, stop);
        
        float speedup = seq_time / par_time;
        
        printf("\nPerformance Results:\n");
        printf("Sequential kernel: %.2f ms\n", seq_time);
        printf("Parallel kernel: %.2f ms\n", par_time);
        printf("Speedup: %.2fx\n", speedup);
        
        float total_ops = batch_size * size * 5;
        float seq_throughput = (total_ops / (seq_time * 1e-3)) / 1e9;
        float par_throughput = (total_ops / (par_time * 1e-3)) / 1e9;
        printf("Sequential throughput: %.2f Gops/s\n", seq_throughput);
        printf("Parallel throughput: %.2f Gops/s\n", par_throughput);
    }
    
    free(h_input);
    free(h_output_seq);
    free(h_output_par);
    cudaFree(d_input);
    cudaFree(d_output_seq);
    cudaFree(d_output_par);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

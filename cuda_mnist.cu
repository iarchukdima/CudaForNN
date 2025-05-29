#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 8192
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 32
#define EPOCHS 10
#define LEARNING_RATE 0.001

#define TILE_SIZE 16

typedef struct {
    float *weights1;
    float *weights2;
    float *bias1;
    float *bias2;
    float *grad_weights1;
    float *grad_weights2;
    float *grad_bias1;
    float *grad_bias2;
} NeuralNetwork;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

void initialize_weights(float *weights, int size) {
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

#define TILE_SIZE 16

__global__ void matmul_a_b_kernel(const float *A, const float *B, float *C, int m, int n, int k) {
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

#define TILE_SIZE 16

__global__ void matmul_a_bt_kernel(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_col = t * TILE_SIZE + threadIdx.x;
        
        if (row < m && a_col < n) {
            As[threadIdx.y][threadIdx.x] = A[row * n + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < k && b_col < n) {
            Bs[threadIdx.y][threadIdx.x] = B[col * n + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[threadIdx.x][i];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < k) {
        C[row * k + col] = sum;
    }
}

#define TILE_SIZE 16

__global__ void matmul_at_b_kernel(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (m + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_row = t * TILE_SIZE + threadIdx.y;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        if (a_row < m && row < n) {
            As[threadIdx.y][threadIdx.x] = A[a_row * n + row];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (b_row < m && col < k) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * k + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[i][threadIdx.y] * Bs[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < k) {
        C[row * k + col] = sum;
    }
}

__global__ void relu_kernel(float *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void bias_add_kernel(float *x, float *bias, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / size;
    int i = idx % size;

    if (b < batch_size && i < size) {
        x[idx] += bias[i];
    }
}

__global__ void softmax_kernel(float *x, int batch_size, int size) {
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

__global__ void clip_gradients_kernel(float *gradients, int size, float max_norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = gradients[idx];
        if (grad > max_norm) {
            gradients[idx] = max_norm;
        } else if (grad < -max_norm) {
            gradients[idx] = -max_norm;
        }
    }
}

void forward(NeuralNetwork *nn, float *d_input, float *d_hidden, float *d_output, int batch_size) {
    dim3 block_size(32, 32);
    dim3 grid_size((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);

    matmul_a_b_kernel<<<grid_size, block_size, 2 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_input, nn->weights1, d_hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    bias_add_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    relu_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    grid_size.x = (OUTPUT_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (batch_size + block_size.y - 1) / block_size.y;
    matmul_a_b_kernel<<<grid_size, block_size, 2 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_hidden, nn->weights2, d_output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    bias_add_kernel<<<(batch_size * OUTPUT_SIZE + 255) / 256, 256>>>(d_output, nn->bias2, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    softmax_kernel<<<batch_size, 256, 2 * 256 * sizeof(float) >>>(d_output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

float cross_entropy_loss(float *output, int *labels, int batch_size) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f));
    }
    return total_loss / batch_size;
}

__global__ void zero_grad_kernel(float *grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 0.0f;
    }
}

__global__ void compute_output_gradients_kernel(float *grad_output, float *output, int *labels, int batch_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        }
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
    }
}

__global__ void update_gradients_kernel(float *grad_weights, float *grad_bias, float *grad_layer, float *prev_layer, int batch_size, int prev_size, int curr_size) {
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < curr_size && j < prev_size) {
        float grad_w_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad_w_sum += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j];
        }
        atomicAdd(&grad_weights[i * prev_size + j], grad_w_sum);

        if (j == 0) {
            float grad_b_sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                grad_b_sum += grad_layer[b * curr_size + i];
            }
            atomicAdd(&grad_bias[i], grad_b_sum);
        }
    }
}

__global__ void drelu_kernel(float *x, float *d_ReLU_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_ReLU_out[idx] = x[idx] > 0.0f ? 1.0f : 0.0f;
    }
}

__global__ void multiply_gradients_kernel(float *grad1, float *grad2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad1[idx] *= grad2[idx];
    }
}

void backward(NeuralNetwork *nn, float *d_input, float *d_hidden, float *d_output, int *d_labels, int batch_size) {
    zero_grad_kernel<<<(HIDDEN_SIZE * INPUT_SIZE + 256 - 1) / 256, 256>>>(nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    zero_grad_kernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + 256 - 1) / 256, 256>>>(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    zero_grad_kernel<<<(HIDDEN_SIZE + 256 - 1) / 256, 256>>>(nn->grad_bias1, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    zero_grad_kernel<<<(OUTPUT_SIZE + 256 - 1) / 256, 256>>>(nn->grad_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    float *d_grad_output;
    CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * OUTPUT_SIZE * sizeof(float)));
    compute_output_gradients_kernel<<<(batch_size + 255) / 256, 256>>>(d_grad_output, d_output, d_labels, batch_size);
    CUDA_CHECK(cudaGetLastError());

    dim3 block_size(32, 32);
    dim3 grid_size((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (OUTPUT_SIZE + block_size.y - 1) / block_size.y);
    matmul_at_b_kernel<<<grid_size, block_size, 2 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_hidden, d_grad_output, nn->grad_weights2, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    update_gradients_kernel<<<grid_size, block_size>>>(nn->grad_weights2, nn->grad_bias2, d_grad_output, d_hidden, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    float *d_dX2;
    CUDA_CHECK(cudaMalloc(&d_dX2, batch_size * HIDDEN_SIZE * sizeof(float)));
    grid_size.x = (HIDDEN_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (batch_size + block_size.y - 1) / block_size.y;
    matmul_a_bt_kernel<<<grid_size, block_size, 2 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_grad_output, nn->weights2, d_dX2, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    float *d_grad_hidden;
    CUDA_CHECK(cudaMalloc(&d_grad_hidden, batch_size * HIDDEN_SIZE * sizeof(float)));
    drelu_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden, d_grad_hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    multiply_gradients_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_dX2, d_grad_hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    grid_size.x = (INPUT_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (HIDDEN_SIZE + block_size.y - 1) / block_size.y;
    matmul_at_b_kernel<<<grid_size, block_size, 2 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_input, d_dX2, nn->grad_weights1, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    update_gradients_kernel<<<grid_size, block_size>>>(nn->grad_weights1, nn->grad_bias1, d_dX2, d_input, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_grad_output);
    cudaFree(d_dX2);
    cudaFree(d_grad_hidden);
}

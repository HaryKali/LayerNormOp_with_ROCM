#include <hip/hip_runtime.h>

__global__ void layerNormKernel(const float* input, float* output,
                                int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float x = input[row * cols + col];
        local_sum += x;
        local_sum_sq += x * x;
    }

    sdata[threadIdx.x] = local_sum;
    sdata[threadIdx.x + blockDim.x] = local_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            sdata[threadIdx.x + blockDim.x] += sdata[threadIdx.x + blockDim.x + stride];
        }
        __syncthreads();
    }

    float mean = sdata[0] / cols;
    float var = sdata[blockDim.x] / cols - mean * mean;

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float x = input[row * cols + col];
        output[row * cols + col] = (x - mean) / sqrtf(var + eps);
    }
}

void launchLayerNorm(const float* input, float* output,
                     int rows, int cols, float eps) {
    int threadsPerBlock = 256;
    int sharedMemSize = 2 * threadsPerBlock * sizeof(float);
    layerNormKernel<<<rows, threadsPerBlock, sharedMemSize>>>(
        input, output, rows, cols, eps);
    hipDeviceSynchronize();
}

#include <cuda_runtime.h>
#include <cstdint>

namespace {
__global__ void gradient_kernel(
    const double* x, const double* first, std::int64_t rows,
    std::int64_t columns, double* gradient) {
    const std::int64_t column = blockIdx.x;
    extern __shared__ double partial[];
    double value = 0.0;
    for (std::int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
        value += x[row * columns + column] * first[row];
    }
    partial[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) gradient[column] = partial[0];
}

__global__ void hessian_kernel(
    const double* x, const double* second, std::int64_t rows,
    std::int64_t columns, double* hessian) {
    const std::int64_t flat = blockIdx.x;
    if (flat >= columns * columns) return;
    const std::int64_t left = flat / columns, right = flat % columns;
    extern __shared__ double partial[];
    double value = 0.0;
    for (std::int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
        value += x[row * columns + left] * second[row] * x[row * columns + right];
    }
    partial[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) hessian[flat] = partial[0];
}

__global__ void gradient_batch_kernel(
    const double* x, const double* first, std::int64_t batches,
    std::int64_t rows, std::int64_t columns, double* gradient) {
    const std::int64_t flat = blockIdx.x;
    if (flat >= batches * columns) return;
    const std::int64_t batch = flat / columns, column = flat % columns;
    const double* block = x + batch * rows * columns;
    extern __shared__ double partial[];
    double value = 0.0;
    for (std::int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
        value += block[row * columns + column] * first[row];
    }
    partial[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) gradient[flat] = partial[0];
}

__global__ void hessian_batch_kernel(
    const double* x, const double* second, std::int64_t batches,
    std::int64_t rows, std::int64_t columns, double* hessian) {
    const std::int64_t flat = blockIdx.x;
    if (flat >= batches * columns * columns) return;
    const std::int64_t batch = flat / (columns * columns);
    const std::int64_t within = flat % (columns * columns);
    const std::int64_t left = within / columns, right = within % columns;
    const double* block = x + batch * rows * columns;
    extern __shared__ double partial[];
    double value = 0.0;
    for (std::int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
        value += block[row * columns + left] * second[row] *
                 block[row * columns + right];
    }
    partial[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) hessian[flat] = partial[0];
}
}

extern "C" int crbstpp_cuda_moments(
    int device, const double* host_x, const double* host_first,
    const double* host_second, std::int64_t rows, std::int64_t columns,
    double* host_gradient, double* host_hessian) {
    if (cudaSetDevice(device) != cudaSuccess) return 1;
    double *x = nullptr, *first = nullptr, *second = nullptr, *gradient = nullptr, *hessian = nullptr;
    const int threads = 256;
    const std::size_t x_bytes = sizeof(double) * rows * columns;
    const std::size_t row_bytes = sizeof(double) * rows;
    const std::size_t gradient_bytes = sizeof(double) * columns;
    const std::size_t hessian_bytes = sizeof(double) * columns * columns;
    if (cudaMalloc(&x, x_bytes) != cudaSuccess || cudaMalloc(&first, row_bytes) != cudaSuccess ||
        cudaMalloc(&second, row_bytes) != cudaSuccess || cudaMalloc(&gradient, gradient_bytes) != cudaSuccess ||
        cudaMalloc(&hessian, hessian_bytes) != cudaSuccess) goto fail;
    if (cudaMemcpy(x, host_x, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(first, host_first, row_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(second, host_second, row_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    gradient_kernel<<<columns, threads, threads * sizeof(double)>>>(
        x, first, rows, columns, gradient);
    hessian_kernel<<<columns * columns, threads, threads * sizeof(double)>>>(
        x, second, rows, columns, hessian);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(host_gradient, gradient, gradient_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_hessian, hessian, hessian_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;
    cudaFree(x); cudaFree(first); cudaFree(second); cudaFree(gradient); cudaFree(hessian);
    return 0;
fail:
    if (x) cudaFree(x); if (first) cudaFree(first); if (second) cudaFree(second);
    if (gradient) cudaFree(gradient); if (hessian) cudaFree(hessian);
    return 2;
}

extern "C" int crbstpp_cuda_moments_batch(
    int device, const double* host_x, const double* host_first,
    const double* host_second, std::int64_t batches, std::int64_t rows,
    std::int64_t columns, double* host_gradient, double* host_hessian) {
    if (cudaSetDevice(device) != cudaSuccess) return 1;
    double *x = nullptr, *first = nullptr, *second = nullptr, *gradient = nullptr,
           *hessian = nullptr;
    const int threads = 256;
    const std::size_t x_bytes = sizeof(double) * batches * rows * columns;
    const std::size_t row_bytes = sizeof(double) * rows;
    const std::size_t gradient_bytes = sizeof(double) * batches * columns;
    const std::size_t hessian_bytes =
        sizeof(double) * batches * columns * columns;
    if (cudaMalloc(&x, x_bytes) != cudaSuccess ||
        cudaMalloc(&first, row_bytes) != cudaSuccess ||
        cudaMalloc(&second, row_bytes) != cudaSuccess ||
        cudaMalloc(&gradient, gradient_bytes) != cudaSuccess ||
        cudaMalloc(&hessian, hessian_bytes) != cudaSuccess) goto fail;
    if (cudaMemcpy(x, host_x, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(first, host_first, row_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(second, host_second, row_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    gradient_batch_kernel<<<batches * columns, threads, threads * sizeof(double)>>>(
        x, first, batches, rows, columns, gradient);
    hessian_batch_kernel<<<batches * columns * columns, threads, threads * sizeof(double)>>>(
        x, second, batches, rows, columns, hessian);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(host_gradient, gradient, gradient_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_hessian, hessian, hessian_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;
    cudaFree(x); cudaFree(first); cudaFree(second); cudaFree(gradient); cudaFree(hessian);
    return 0;
fail:
    if (x) cudaFree(x); if (first) cudaFree(first); if (second) cudaFree(second);
    if (gradient) cudaFree(gradient); if (hessian) cudaFree(hessian);
    return 2;
}

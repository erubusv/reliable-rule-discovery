#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <array>
#include <cstdint>
#include <mutex>

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

__global__ void weight_design_kernel(
    const double* x, const double* second, std::int64_t rows,
    std::int64_t columns, double* weighted) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t count = rows * columns;
    if (index < count) weighted[index] = x[index] * second[index / columns];
}

__global__ void weight_design_batch_kernel(
    const double* x, const double* second, std::int64_t batches,
    std::int64_t rows, std::int64_t columns, double* weighted) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t count = batches * rows * columns;
    if (index < count) {
        const std::int64_t row = (index / columns) % rows;
        weighted[index] = x[index] * second[row];
    }
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

struct Workspace {
    std::mutex mutex;
    double* x = nullptr;
    double* weighted = nullptr;
    double* first = nullptr;
    double* second = nullptr;
    double* gradient = nullptr;
    double* hessian = nullptr;
    std::size_t x_capacity = 0;
    std::size_t weighted_capacity = 0;
    std::size_t first_capacity = 0;
    std::size_t second_capacity = 0;
    std::size_t gradient_capacity = 0;
    std::size_t hessian_capacity = 0;
    cublasHandle_t handle = nullptr;
};

std::array<Workspace, 16> workspaces;

bool reserve(double** pointer, std::size_t* capacity, std::size_t bytes) {
    if (*capacity >= bytes) return true;
    if (*pointer) cudaFree(*pointer);
    *pointer = nullptr;
    *capacity = 0;
    if (cudaMalloc(reinterpret_cast<void**>(pointer), bytes) != cudaSuccess) {
        return false;
    }
    *capacity = bytes;
    return true;
}
}

extern "C" int crbstpp_cuda_moments(
    int device, const double* host_x, const double* host_first,
    const double* host_second, std::int64_t rows, std::int64_t columns,
    double* host_gradient, double* host_hessian) {
    if (device < 0 || device >= static_cast<int>(workspaces.size())) return 1;
    Workspace& workspace = workspaces[device];
    std::lock_guard<std::mutex> lock(workspace.mutex);
    if (cudaSetDevice(device) != cudaSuccess) return 1;
    const int threads = 256;
    const std::size_t x_bytes = sizeof(double) * rows * columns;
    const std::size_t row_bytes = sizeof(double) * rows;
    const std::size_t gradient_bytes = sizeof(double) * columns;
    const std::size_t hessian_bytes = sizeof(double) * columns * columns;
    if (!reserve(&workspace.x, &workspace.x_capacity, x_bytes) ||
        !reserve(&workspace.weighted, &workspace.weighted_capacity, x_bytes) ||
        !reserve(&workspace.first, &workspace.first_capacity, row_bytes) ||
        !reserve(&workspace.second, &workspace.second_capacity, row_bytes) ||
        !reserve(&workspace.gradient, &workspace.gradient_capacity, gradient_bytes) ||
        !reserve(&workspace.hessian, &workspace.hessian_capacity, hessian_bytes))
        return 2;
    if (workspace.handle == nullptr &&
        cublasCreate(&workspace.handle) != CUBLAS_STATUS_SUCCESS)
        return 2;
    if (cudaMemcpy(workspace.x, host_x, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.first, host_first, row_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.second, host_second, row_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        return 2;
    {
        const std::int64_t count = rows * columns;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        weight_design_kernel<<<blocks, threads>>>(
            workspace.x, workspace.second, rows, columns, workspace.weighted);
    }
    if (cudaGetLastError() != cudaSuccess) return 2;
    {
        const double one = 1.0, zero = 0.0;
        // Row-major n x d buffers are column-major d x n views.  Therefore
        // GEMV(N) is X^T f and GEMM(N,T) is X^T diag(w) X.
        if (cublasDgemv(
                workspace.handle, CUBLAS_OP_N, static_cast<int>(columns),
                static_cast<int>(rows), &one, workspace.x, static_cast<int>(columns),
                workspace.first, 1, &zero, workspace.gradient, 1) != CUBLAS_STATUS_SUCCESS ||
            cublasDgemm(
                workspace.handle, CUBLAS_OP_N, CUBLAS_OP_T,
                static_cast<int>(columns), static_cast<int>(columns),
                static_cast<int>(rows), &one, workspace.x, static_cast<int>(columns),
                workspace.weighted, static_cast<int>(columns), &zero,
                workspace.hessian, static_cast<int>(columns)) != CUBLAS_STATUS_SUCCESS)
            return 2;
    }
    if (cudaMemcpy(host_gradient, workspace.gradient, gradient_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_hessian, workspace.hessian, hessian_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    return 0;
}

extern "C" int crbstpp_cuda_moments_batch(
    int device, const double* host_x, const double* host_first,
    const double* host_second, std::int64_t batches, std::int64_t rows,
    std::int64_t columns, double* host_gradient, double* host_hessian) {
    if (device < 0 || device >= static_cast<int>(workspaces.size())) return 1;
    Workspace& workspace = workspaces[device];
    std::lock_guard<std::mutex> lock(workspace.mutex);
    if (cudaSetDevice(device) != cudaSuccess) return 1;
    const int threads = 256;
    const std::size_t x_bytes = sizeof(double) * batches * rows * columns;
    const std::size_t row_bytes = sizeof(double) * rows;
    const std::size_t gradient_bytes = sizeof(double) * batches * columns;
    const std::size_t hessian_bytes =
        sizeof(double) * batches * columns * columns;
    if (!reserve(&workspace.x, &workspace.x_capacity, x_bytes) ||
        !reserve(&workspace.weighted, &workspace.weighted_capacity, x_bytes) ||
        !reserve(&workspace.first, &workspace.first_capacity, row_bytes) ||
        !reserve(&workspace.second, &workspace.second_capacity, row_bytes) ||
        !reserve(&workspace.gradient, &workspace.gradient_capacity, gradient_bytes) ||
        !reserve(&workspace.hessian, &workspace.hessian_capacity, hessian_bytes))
        return 2;
    if (workspace.handle == nullptr &&
        cublasCreate(&workspace.handle) != CUBLAS_STATUS_SUCCESS)
        return 2;
    if (cudaMemcpy(workspace.x, host_x, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.first, host_first, row_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.second, host_second, row_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        return 2;
    gradient_batch_kernel<<<batches * columns, threads, threads * sizeof(double)>>>(
        workspace.x, workspace.first, batches, rows, columns, workspace.gradient);
    {
        const std::int64_t count = batches * rows * columns;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        weight_design_batch_kernel<<<blocks, threads>>>(
            workspace.x, workspace.second, batches, rows, columns,
            workspace.weighted);
    }
    if (cudaGetLastError() != cudaSuccess) return 2;
    {
        const double one = 1.0, zero = 0.0;
        const long long design_stride = rows * columns;
        const long long hessian_stride = columns * columns;
        if (cublasDgemmStridedBatched(
                workspace.handle, CUBLAS_OP_N, CUBLAS_OP_T,
                static_cast<int>(columns), static_cast<int>(columns),
                static_cast<int>(rows), &one, workspace.x,
                static_cast<int>(columns), design_stride, workspace.weighted,
                static_cast<int>(columns), design_stride, &zero,
                workspace.hessian, static_cast<int>(columns), hessian_stride,
                static_cast<int>(batches)) != CUBLAS_STATUS_SUCCESS)
            return 2;
    }
    if (
        cudaMemcpy(host_gradient, workspace.gradient, gradient_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_hessian, workspace.hessian, hessian_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    return 0;
}

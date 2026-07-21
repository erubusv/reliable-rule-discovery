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

__global__ void sparse_gradient_batch_kernel(
    const std::int64_t* rows, const std::int64_t* block_offsets, const double* values,
    const double* first, const double* second, std::int64_t candidates,
    std::int64_t blocks_per_candidate, std::int64_t knots,
    int indexed_derivatives, double* gradient, double* cross) {
    const std::int64_t flat = blockIdx.x;
    const std::int64_t dimensions = blocks_per_candidate * knots;
    if (flat >= candidates * dimensions) return;
    const std::int64_t candidate = flat / dimensions;
    const std::int64_t column = flat % dimensions;
    const std::int64_t local_block = column / knots;
    const std::int64_t knot = column % knots;
    const std::int64_t block = candidate * blocks_per_candidate + local_block;
    const std::int64_t left = block_offsets[block];
    const std::int64_t right = block_offsets[block + 1];
    extern __shared__ double partial[];
    double gradient_value = 0.0, cross_value = 0.0;
    for (std::int64_t row = left + threadIdx.x; row < right; row += blockDim.x) {
        const double value = values[row * knots + knot];
        const std::int64_t derivative_row = indexed_derivatives ? rows[row] : row;
        gradient_value += value * first[derivative_row];
        cross_value += value * second[derivative_row];
    }
    partial[threadIdx.x] = gradient_value;
    partial[blockDim.x + threadIdx.x] = cross_value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
            partial[blockDim.x + threadIdx.x] +=
                partial[blockDim.x + threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gradient[flat] = partial[0];
        cross[flat] = partial[blockDim.x];
    }
}

__device__ std::int64_t lower_bound_row(
    const std::int64_t* rows, std::int64_t left, std::int64_t right,
    std::int64_t value) {
    while (left < right) {
        const std::int64_t middle = left + (right - left) / 2;
        if (rows[middle] < value) left = middle + 1;
        else right = middle;
    }
    return left;
}

__global__ void sparse_hessian_batch_kernel(
    const std::int64_t* rows, const std::int64_t* block_offsets,
    const double* values, const double* second, std::int64_t candidates,
    std::int64_t blocks_per_candidate, std::int64_t knots,
    int indexed_derivatives, double* hessian) {
    const std::int64_t block_pairs =
        blocks_per_candidate * (blocks_per_candidate + 1) / 2;
    const std::int64_t knot_pairs = knots * knots;
    const std::int64_t flat = blockIdx.x;
    if (flat >= candidates * block_pairs) return;
    const std::int64_t candidate = flat / block_pairs;
    std::int64_t pair = flat % block_pairs;
    std::int64_t left_local = 0;
    while (pair >= blocks_per_candidate - left_local) {
        pair -= blocks_per_candidate - left_local;
        ++left_local;
    }
    const std::int64_t right_local = left_local + pair;
    const std::int64_t left_block =
        candidate * blocks_per_candidate + left_local;
    const std::int64_t right_block =
        candidate * blocks_per_candidate + right_local;
    const std::int64_t left_begin = block_offsets[left_block];
    const std::int64_t left_end = block_offsets[left_block + 1];
    const std::int64_t right_begin = block_offsets[right_block];
    const std::int64_t right_end = block_offsets[right_block + 1];
    // CRBS-TPP uses M=4.  The compiled operator supports up to M=8 while
    // selecting a launch width that bounds dynamic shared memory below 32 KiB.
    extern __shared__ double partial[];
    for (std::int64_t knot_pair = 0; knot_pair < knot_pairs; ++knot_pair) {
        partial[knot_pair * blockDim.x + threadIdx.x] = 0.0;
    }
    for (std::int64_t index = left_begin + threadIdx.x;
         index < left_end; index += blockDim.x) {
        const std::int64_t row = rows[index];
        const std::int64_t matched =
            lower_bound_row(rows, right_begin, right_end, row);
        if (matched < right_end && rows[matched] == row) {
            const double weight = second[indexed_derivatives ? row : index];
            for (std::int64_t left_knot = 0; left_knot < knots; ++left_knot) {
                const double weighted =
                    values[index * knots + left_knot] * weight;
                for (std::int64_t right_knot = 0; right_knot < knots;
                     ++right_knot) {
                    partial[(left_knot * knots + right_knot) * blockDim.x
                            + threadIdx.x] +=
                        weighted * values[matched * knots + right_knot];
                }
            }
        }
    }
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            for (std::int64_t knot_pair = 0; knot_pair < knot_pairs; ++knot_pair) {
                partial[knot_pair * blockDim.x + threadIdx.x] +=
                    partial[knot_pair * blockDim.x + threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const std::int64_t dimensions = blocks_per_candidate * knots;
        double* output = hessian + candidate * dimensions * dimensions;
        for (std::int64_t left_knot = 0; left_knot < knots; ++left_knot) {
            for (std::int64_t right_knot = 0; right_knot < knots; ++right_knot) {
                const std::int64_t knot_pair = left_knot * knots + right_knot;
                const std::int64_t left_column = left_local * knots + left_knot;
                const std::int64_t right_column = right_local * knots + right_knot;
                output[left_column * dimensions + right_column] =
                    partial[knot_pair * blockDim.x];
                if (left_local != right_local) {
                    output[right_column * dimensions + left_column] =
                        partial[knot_pair * blockDim.x];
                }
            }
        }
    }
}

struct Workspace {
    std::mutex mutex;
    double* x = nullptr;
    double* weighted = nullptr;
    double* first = nullptr;
    double* second = nullptr;
    double* gradient = nullptr;
    double* cross = nullptr;
    double* hessian = nullptr;
    std::size_t x_capacity = 0;
    std::size_t weighted_capacity = 0;
    std::size_t first_capacity = 0;
    std::size_t second_capacity = 0;
    std::size_t gradient_capacity = 0;
    std::size_t cross_capacity = 0;
    std::size_t hessian_capacity = 0;
    cublasHandle_t handle = nullptr;
};

struct SparseWorkspace {
    std::mutex mutex;
    std::int64_t* rows = nullptr;
    std::int64_t* block_offsets = nullptr;
    double* values = nullptr;
    double* first = nullptr;
    double* second = nullptr;
    double* gradient = nullptr;
    double* cross = nullptr;
    double* hessian = nullptr;
    std::size_t rows_capacity = 0;
    std::size_t block_offsets_capacity = 0;
    std::size_t values_capacity = 0;
    std::size_t first_capacity = 0;
    std::size_t second_capacity = 0;
    std::size_t gradient_capacity = 0;
    std::size_t cross_capacity = 0;
    std::size_t hessian_capacity = 0;
};

std::array<Workspace, 16> workspaces;
std::array<SparseWorkspace, 16> sparse_workspaces;

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


bool reserve_int64(
    std::int64_t** pointer, std::size_t* capacity, std::size_t bytes) {
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
    std::int64_t columns, int compute_cross, double* host_gradient,
    double* host_hessian, double* host_cross) {
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
        (compute_cross &&
         !reserve(&workspace.cross, &workspace.cross_capacity, gradient_bytes)) ||
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
    if (compute_cross) {
        gradient_batch_kernel<<<batches * columns, threads, threads * sizeof(double)>>>(
            workspace.x, workspace.second, batches, rows, columns, workspace.cross);
    }
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
        (compute_cross &&
         cudaMemcpy(host_cross, workspace.cross, gradient_bytes,
                    cudaMemcpyDeviceToHost) != cudaSuccess) ||
        cudaMemcpy(host_hessian, workspace.hessian, hessian_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    return 0;
}

int sparse_moments_batch_impl(
    int device, const std::int64_t* host_rows, const double* host_values,
    const double* host_first, const double* host_second,
    const std::int64_t* host_block_offsets, std::int64_t total_rows,
    std::int64_t derivative_rows, int indexed_derivatives,
    std::int64_t candidates, std::int64_t blocks_per_candidate,
    std::int64_t knots, double* host_gradient, double* host_hessian,
    double* host_cross) {
    if (device < 0 || device >= static_cast<int>(sparse_workspaces.size()) ||
        total_rows < 0 || derivative_rows < 0 || candidates < 1 ||
        blocks_per_candidate < 1 || knots < 1 || knots > 8)
        return 1;
    SparseWorkspace& workspace = sparse_workspaces[device];
    std::lock_guard<std::mutex> lock(workspace.mutex);
    if (cudaSetDevice(device) != cudaSuccess) return 1;
    const int threads = 256;
    const std::int64_t total_blocks = candidates * blocks_per_candidate;
    const std::int64_t dimensions = blocks_per_candidate * knots;
    const std::size_t rows_bytes = sizeof(std::int64_t) * total_rows;
    const std::size_t values_bytes = sizeof(double) * total_rows * knots;
    const std::size_t offsets_bytes = sizeof(std::int64_t) * (total_blocks + 1);
    const std::size_t derivative_bytes = sizeof(double) * derivative_rows;
    const std::size_t gradient_bytes = sizeof(double) * candidates * dimensions;
    const std::size_t hessian_bytes =
        sizeof(double) * candidates * dimensions * dimensions;
    if (!reserve_int64(&workspace.rows, &workspace.rows_capacity, rows_bytes) ||
        !reserve_int64(&workspace.block_offsets,
                       &workspace.block_offsets_capacity, offsets_bytes) ||
        !reserve(&workspace.values, &workspace.values_capacity, values_bytes) ||
        !reserve(&workspace.first, &workspace.first_capacity, derivative_bytes) ||
        !reserve(&workspace.second, &workspace.second_capacity, derivative_bytes) ||
        !reserve(&workspace.gradient, &workspace.gradient_capacity, gradient_bytes) ||
        !reserve(&workspace.cross, &workspace.cross_capacity, gradient_bytes) ||
        !reserve(&workspace.hessian, &workspace.hessian_capacity, hessian_bytes))
        return 2;
    if ((rows_bytes && cudaMemcpy(workspace.rows, host_rows, rows_bytes,
                                  cudaMemcpyHostToDevice) != cudaSuccess) ||
        (values_bytes && cudaMemcpy(workspace.values, host_values, values_bytes,
                                    cudaMemcpyHostToDevice) != cudaSuccess) ||
        (derivative_bytes &&
         cudaMemcpy(workspace.first, host_first, derivative_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess) ||
        (derivative_bytes &&
         cudaMemcpy(workspace.second, host_second, derivative_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess) ||
        cudaMemcpy(workspace.block_offsets, host_block_offsets, offsets_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return 2;
    sparse_gradient_batch_kernel<<<
        candidates * dimensions, threads, 2 * threads * sizeof(double)>>>(
        workspace.rows, workspace.block_offsets, workspace.values, workspace.first,
        workspace.second, candidates, blocks_per_candidate, knots,
        indexed_derivatives, workspace.gradient, workspace.cross);
    const std::int64_t block_pairs =
        blocks_per_candidate * (blocks_per_candidate + 1) / 2;
    int hessian_threads = 256;
    while (hessian_threads * knots * knots * static_cast<int>(sizeof(double)) >
           32 * 1024)
        hessian_threads /= 2;
    sparse_hessian_batch_kernel<<<
        candidates * block_pairs, hessian_threads,
        hessian_threads * knots * knots * sizeof(double)>>>(
        workspace.rows, workspace.block_offsets, workspace.values,
        workspace.second, candidates, blocks_per_candidate, knots,
        indexed_derivatives, workspace.hessian);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
        return 2;
    if (cudaMemcpy(host_gradient, workspace.gradient, gradient_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_cross, workspace.cross, gradient_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_hessian, workspace.hessian, hessian_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    return 0;
}

extern "C" int crbstpp_cuda_sparse_moments_batch(
    int device, const std::int64_t* host_rows, const double* host_values,
    const double* host_first, const double* host_second,
    const std::int64_t* host_block_offsets, std::int64_t total_rows,
    std::int64_t candidates, std::int64_t blocks_per_candidate,
    std::int64_t knots, double* host_gradient, double* host_hessian,
    double* host_cross) {
    return sparse_moments_batch_impl(
        device, host_rows, host_values, host_first, host_second,
        host_block_offsets, total_rows, total_rows, 0, candidates,
        blocks_per_candidate, knots, host_gradient, host_hessian, host_cross);
}

extern "C" int crbstpp_cuda_sparse_moments_indexed_batch(
    int device, const std::int64_t* host_rows, const double* host_values,
    const double* host_first, const double* host_second,
    const std::int64_t* host_block_offsets, std::int64_t total_rows,
    std::int64_t derivative_rows, std::int64_t candidates,
    std::int64_t blocks_per_candidate, std::int64_t knots,
    double* host_gradient, double* host_hessian, double* host_cross) {
    return sparse_moments_batch_impl(
        device, host_rows, host_values, host_first, host_second,
        host_block_offsets, total_rows, derivative_rows, 1, candidates,
        blocks_per_candidate, knots, host_gradient, host_hessian, host_cross);
}

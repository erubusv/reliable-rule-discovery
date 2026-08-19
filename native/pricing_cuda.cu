#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <array>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <mutex>
#include <vector>

namespace {
__global__ void weight_design_kernel(
    const double* x, const double* second, std::int64_t rows,
    std::int64_t columns, double* weighted) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t count = rows * columns;
    if (index < count) weighted[index] = x[index] * second[index / columns];
}

__global__ void gather_projected_design_kernel(
    const double* source, const std::int64_t* columns, const double* scales,
    std::int64_t rows, std::int64_t source_columns,
    std::int64_t projected_columns, double* projected) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t count = rows * projected_columns;
    if (index >= count) return;
    const std::int64_t row = index / projected_columns;
    const std::int64_t column = index % projected_columns;
    projected[index] =
        source[row * source_columns + columns[column]] * scales[column];
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

__global__ void fill_constant_kernel(
    double* values, std::int64_t count, double value) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = value;
}

__global__ void poisson_likelihood_kernel(
    const double* eta, const double* exposure, const double* event,
    std::int64_t rows, double* value, double* first, double* second) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows) return;
    const double predictor = eta[index];
    const double mean = exposure[index] * exp(predictor);
    value[index] = mean - event[index] * predictor;
    first[index] = mean - event[index];
    second[index] = mean;
}

__device__ void cloglog_event_terms_device(
    double predictor, double* value, double* first, double* second) {
    const double clipped = fmin(fmax(predictor, -745.0), 700.0);
    const double x = exp(clipped);
    if (x < 1.0e-4) {
        *value = -log(fmax(x, DBL_MIN)) + x / 2.0 - x * x / 24.0;
        *first = -1.0 + x / 2.0 - x * x / 12.0;
        *second = fmax(0.0, x / 2.0 - x * x / 6.0);
        return;
    }
    if (x > 40.0) {
        const double tail = exp(-x);
        *value = -log1p(-tail);
        *first = -x * tail / fmax(1.0 - tail, DBL_MIN);
        *second = x < 100.0 ? x * (x - 1.0) * tail : 0.0;
        *second = fmax(0.0, *second);
        return;
    }
    const double denominator = expm1(x);
    const double exponential = denominator + 1.0;
    *value = -log(-expm1(-x));
    *first = -x / denominator;
    *second = fmax(
        0.0,
        x * ((x - 1.0) * exponential + 1.0) /
            (denominator * denominator));
}

__global__ void cloglog_likelihood_kernel(
    const double* eta, const double* noevent, const double* event,
    std::int64_t rows, double* value, double* first, double* second) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows) return;
    const double predictor = eta[index];
    const double intensity = exp(fmin(fmax(predictor, -745.0), 700.0));
    double event_value = 0.0;
    double event_first = 0.0;
    double event_second = 0.0;
    cloglog_event_terms_device(
        predictor, &event_value, &event_first, &event_second);
    value[index] =
        noevent[index] * intensity + event[index] * event_value;
    first[index] =
        noevent[index] * intensity + event[index] * event_first;
    second[index] =
        noevent[index] * intensity + event[index] * event_second;
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

constexpr int kImplicitMaxSources = 3;

__device__ void add_implicit_completion_block(
    std::int64_t completion, std::int64_t observation_start,
    std::int64_t observation_end, const double* basis, int knots, int lag,
    int dimension, int block, int tile_begin, int tile_end, double* design) {
    const std::int64_t remaining = observation_end - completion;
    const int maximum = static_cast<int>(
        remaining < lag ? remaining : static_cast<std::int64_t>(lag));
    const int base = static_cast<int>(completion - observation_start);
    const int column = block * knots;
    // This routine is called once for every entity tile.  Iterating the full
    // impact horizon and discarding offsets outside the tile repeated the
    // same 30-day kernel dozens of times for long risk episodes.  Intersect
    // the strictly-future response interval with the current tile first.
    // The retained offsets and their increasing accumulation order are
    // unchanged, so this is bitwise-equivalent scheduling, not truncation.
    const int first_offset = max(1, tile_begin - base);
    const int last_offset = min(maximum, tile_end - 1 - base);
    for (int offset = first_offset; offset <= last_offset; ++offset) {
        const int global_row = base + offset;
        double* row = design +
            static_cast<std::int64_t>(global_row - tile_begin) * dimension +
            column;
        for (int knot = 0; knot < knots; ++knot)
            row[knot] += basis[knot * lag + offset - 1];
    }
}

__device__ void fill_implicit_design_block(
    const std::int64_t* source_entity_offsets, const std::int64_t* source_times,
    std::int64_t entity_count, const int* predicates, int order,
    std::int64_t window, std::int64_t entity, std::int64_t observation_start,
    std::int64_t observation_end, const double* basis, int knots, int lag,
    int dimension, int block, int tile_begin, int tile_end, double* design) {
    if (order < 1 || order > kImplicitMaxSources) return;
    std::int64_t position[kImplicitMaxSources]{};
    std::int64_t stop[kImplicitMaxSources]{};
    for (int source = 0; source < order; ++source) {
        const int predicate = predicates[source];
        if (predicate < 0) return;
        const std::int64_t base =
            static_cast<std::int64_t>(predicate) * (entity_count + 1);
        position[source] = source_entity_offsets[base + entity];
        stop[source] = source_entity_offsets[base + entity + 1];
        if (position[source] >= stop[source]) return;
    }
    if (order == 1) {
        for (std::int64_t cursor = position[0]; cursor < stop[0]; ++cursor)
            add_implicit_completion_block(
                source_times[cursor], observation_start, observation_end,
                basis, knots, lag, dimension, block, tile_begin, tile_end,
                design);
        return;
    }
    std::int64_t latest[kImplicitMaxSources];
    for (int source = 0; source < order; ++source) latest[source] = INT64_MIN;
    while (true) {
        std::int64_t next = INT64_MAX;
        for (int source = 0; source < order; ++source) {
            if (position[source] < stop[source] &&
                source_times[position[source]] < next)
                next = source_times[position[source]];
        }
        if (next == INT64_MAX) break;
        bool witnessed = true;
        std::int64_t minimum = INT64_MAX;
        std::int64_t maximum = INT64_MIN;
        for (int source = 0; source < order; ++source) {
            while (position[source] < stop[source] &&
                   source_times[position[source]] <= next)
                latest[source] = source_times[position[source]++];
            witnessed = witnessed && latest[source] != INT64_MIN;
            if (latest[source] < minimum) minimum = latest[source];
            if (latest[source] > maximum) maximum = latest[source];
        }
        if (witnessed && maximum - minimum <= window)
            add_implicit_completion_block(
                next, observation_start, observation_end, basis, knots, lag,
                dimension, block, tile_begin, tile_end, design);
    }
}

__device__ std::int64_t packed_entity_lower_bound(
    const std::int64_t* packed_entity_spans, std::int64_t begin,
    std::int64_t end, std::int64_t entity) {
    const std::int64_t target = entity << 32;
    while (begin < end) {
        const std::int64_t middle = begin + (end - begin) / 2;
        if (packed_entity_spans[middle] < target)
            begin = middle + 1;
        else
            end = middle;
    }
    return begin;
}

__device__ std::int64_t completion_time_lower_bound(
    const std::int64_t* completion_times, std::int64_t begin,
    std::int64_t end, std::int64_t target) {
    while (begin < end) {
        const std::int64_t middle = begin + (end - begin) / 2;
        if (completion_times[middle] < target)
            begin = middle + 1;
        else
            end = middle;
    }
    return begin;
}

__device__ void fill_completion_design_knot(
    const std::int64_t* completion_entity_offsets,
    const std::int64_t* completion_times,
    const std::int64_t* completion_spans,
    int completion_mode, std::int64_t entity_count, int antecedent,
    std::int64_t minimum_span, std::int64_t window,
    std::int64_t entity, std::int64_t observation_start,
    std::int64_t observation_end, const double* basis, int knots, int lag,
    int dimension, int block, int knot, int tile_begin, int tile_end,
    double* design) {
    if (antecedent < 0 || knot < 0 || knot >= knots) return;
    std::int64_t begin;
    std::int64_t end;
    if (completion_mode == 2) {
        const std::int64_t pattern_begin =
            completion_entity_offsets[static_cast<std::int64_t>(antecedent) * 2];
        const std::int64_t pattern_end = completion_entity_offsets[
            static_cast<std::int64_t>(antecedent) * 2 + 1];
        begin = packed_entity_lower_bound(
            completion_spans, pattern_begin, pattern_end, entity);
        end = packed_entity_lower_bound(
            completion_spans, begin, pattern_end, entity + 1);
    } else {
        const std::int64_t base =
            static_cast<std::int64_t>(antecedent) * (entity_count + 1);
        begin = completion_entity_offsets[base + entity];
        end = completion_entity_offsets[base + entity + 1];
    }
    // Completions are sorted by (pattern, entity, time).  A completion can
    // touch this tile only when its strictly-future lag interval intersects
    // [tile_begin, tile_end).  Narrow the entity slice before the per-knot
    // scan; without this, every tile rescanned the entity's complete history
    // even after the offset loop itself had been clipped.
    const std::int64_t first_time =
        observation_start + static_cast<std::int64_t>(tile_begin) - lag;
    const std::int64_t last_time_exclusive =
        observation_start + static_cast<std::int64_t>(tile_end) - 1;
    begin = completion_time_lower_bound(
        completion_times, begin, end, first_time);
    end = completion_time_lower_bound(
        completion_times, begin, end, last_time_exclusive);
    const int column = block * knots + knot;
    for (std::int64_t cursor = begin; cursor < end; ++cursor) {
        const std::int64_t span = completion_mode == 2
            ? (completion_spans[cursor] & 0xFFFFFFFFLL)
            : completion_spans[cursor];
        if (span <= minimum_span || span > window) continue;
        const std::int64_t completion = completion_times[cursor];
        const std::int64_t remaining = observation_end - completion;
        const int maximum = static_cast<int>(
            remaining < lag ? remaining : static_cast<std::int64_t>(lag));
        const int row_base = static_cast<int>(completion - observation_start);
        // As above, visit only the part of this completion's future kernel
        // that intersects the live tile.  Previously every tile traversed all
        // ``lag`` offsets and rejected nearly all of them inside the loop.
        const int first_offset = max(1, tile_begin - row_base);
        const int last_offset = min(maximum, tile_end - 1 - row_base);
        for (int offset = first_offset; offset <= last_offset; ++offset) {
            const int global_row = row_base + offset;
            design[
                static_cast<std::int64_t>(global_row - tile_begin) *
                    dimension +
                column] += basis[knot * lag + offset - 1];
        }
    }
}

__device__ bool implicit_candidate_entity_active(
    const std::int64_t* source_entity_offsets,
    const std::int64_t* source_times, std::int64_t entity_count,
    const int* block_predicates, const int* block_orders, int block_count,
    std::int64_t entity, std::int64_t observation_end) {
    // A candidate can contribute only if at least one of its rule/closure
    // blocks has every source witnessed before the final observed tick.  This
    // predicate is exact: a completion at observation_end has no strictly
    // future response row and all later completions are impossible.
    for (int block = 0; block < block_count; ++block) {
        const int order = block_orders[block];
        bool witnessed = order >= 1 && order <= kImplicitMaxSources;
        for (int source = 0; source < order && witnessed; ++source) {
            const int predicate =
                block_predicates[block * kImplicitMaxSources + source];
            if (predicate < 0) {
                witnessed = false;
                break;
            }
            const std::int64_t base =
                static_cast<std::int64_t>(predicate) * (entity_count + 1);
            const std::int64_t begin = source_entity_offsets[base + entity];
            const std::int64_t end = source_entity_offsets[base + entity + 1];
            witnessed = begin < end && source_times[begin] < observation_end;
        }
        if (witnessed) return true;
    }
    return false;
}

// Compute the exact strictly-future score convolution once per likelihood
// derivative state.  All rule orders, relations, W identities, and signs use
// this same sufficient statistic.  Entity boundaries prevent contributions
// from leaking across risk episodes.
__global__ void future_basis_score_kernel(
    const std::int64_t* starts, const std::int64_t* ends,
    const std::int64_t* grid_offsets, std::int64_t entity_count,
    const double* basis, int knots, int lag, const double* first,
    const unsigned char* event_by_row, const double* event_first_delta,
    int compact_derivative_mode, const int* group_by_row,
    double* future_score) {
    const std::int64_t entity = blockIdx.x;
    if (entity >= entity_count) return;
    const std::int64_t length = ends[entity] - starts[entity] + 1;
    const std::int64_t row_begin = grid_offsets[entity];
    for (std::int64_t local = threadIdx.x; local < length;
         local += blockDim.x) {
        const std::int64_t row0 = row_begin + local;
        const int maximum = static_cast<int>(min(
            static_cast<std::int64_t>(lag), length - local - 1));
        for (int knot = 0; knot < knots; ++knot) {
            double value = 0.0;
            for (int offset = 1; offset <= maximum; ++offset) {
                const std::int64_t row = row0 + offset;
                const int group = group_by_row[row];
                const double events = static_cast<double>(event_by_row[row]);
                const double row_first = compact_derivative_mode == 1
                    ? first[group] - events
                    : first[group] + events * event_first_delta[group];
                value += basis[knot * lag + offset - 1] * row_first;
            }
            future_score[row0 * knots + knot] = value;
        }
    }
}

// Exact raw gradients for provenance-aware completion blocks.  Consecutive
// candidates sharing a pattern are its frozen W identities; one block scans
// the completion stream once and sums the preconvolved scores above.  The
// full Fisher/cross/full-fit paths remain unchanged.
__global__ void completion_preconvolved_gradient_kernel(
    const std::int64_t* pattern_offsets,
    const std::int64_t* completion_times,
    const std::int64_t* packed_entity_spans,
    const std::int64_t* starts, const std::int64_t* ends,
    const std::int64_t* grid_offsets, const double* future_score, int knots,
    const int* block_predicates,
    const std::int64_t* block_minimum_spans,
    const std::int64_t* block_windows,
    int candidates, double* gradient) {
    constexpr int kMaximumWindows = 8;
    constexpr int kMaximumKnots = 8;
    const int first_candidate = blockIdx.x;
    if (first_candidate >= candidates) return;
    const int pattern = block_predicates[
        static_cast<std::int64_t>(first_candidate) * kImplicitMaxSources];
    if (first_candidate > 0 && block_predicates[
            static_cast<std::int64_t>(first_candidate - 1) *
                kImplicitMaxSources] == pattern)
        return;
    int window_count = 1;
    while (first_candidate + window_count < candidates &&
           window_count < kMaximumWindows &&
           block_predicates[
               static_cast<std::int64_t>(first_candidate + window_count) *
                   kImplicitMaxSources] == pattern)
        ++window_count;
    if (knots > kMaximumKnots || pattern < 0) return;
    double local[kMaximumWindows * kMaximumKnots];
    for (int index = 0; index < kMaximumWindows * kMaximumKnots; ++index)
        local[index] = 0.0;
    const std::int64_t begin = pattern_offsets[
        static_cast<std::int64_t>(pattern) * 2];
    const std::int64_t end = pattern_offsets[
        static_cast<std::int64_t>(pattern) * 2 + 1];
    for (std::int64_t cursor = begin + threadIdx.x; cursor < end;
         cursor += blockDim.x) {
        const std::int64_t packed = packed_entity_spans[cursor];
        const std::int64_t entity = packed >> 32;
        const std::int64_t span = packed & 0xFFFFFFFFLL;
        const std::int64_t completion = completion_times[cursor];
        if (completion >= ends[entity]) continue;
        const std::int64_t row0 =
            grid_offsets[entity] + completion - starts[entity];
        for (int knot = 0; knot < knots; ++knot) {
            const double contribution = future_score[row0 * knots + knot];
            for (int window = 0; window < window_count; ++window) {
                const int metadata = first_candidate + window;
                if (span > block_minimum_spans[metadata] &&
                    span <= block_windows[metadata])
                    local[window * kMaximumKnots + knot] += contribution;
            }
        }
    }
    extern __shared__ double reduction[];
    for (int window = 0; window < window_count; ++window) {
        for (int knot = 0; knot < knots; ++knot) {
            reduction[threadIdx.x] =
                local[window * kMaximumKnots + knot];
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride)
                    reduction[threadIdx.x] += reduction[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0)
                gradient[
                    static_cast<std::int64_t>(first_candidate + window) * knots +
                    knot] = reduction[0];
            __syncthreads();
        }
    }
}

// Direct sufficient-statistic pricing.  A block owns one candidate and a
// contiguous entity chunk, keeps the entity response in shared memory, and
// emits only gradient/upper-Fisher/current-cross partials.  This removes both
// the expanded response tile and the weighted copy consumed by GEMM.
__global__ void implicit_direct_partials_kernel(
    const std::int64_t* source_entity_offsets, const std::int64_t* source_times,
    const std::int64_t* completion_spans, int completion_mode,
    std::int64_t entity_count, const std::int64_t* starts,
    const std::int64_t* ends, const std::int64_t* grid_offsets,
    const double* basis, int knots, int lag, const int* block_predicates,
    const int* block_orders, const std::int64_t* block_minimum_spans,
    const std::int64_t* block_windows,
    const int* block_counts, int candidates, int maximum_blocks,
    const std::int64_t* candidate_entity_offsets,
    const int* candidate_entities,
    const double* first, const double* second,
    const unsigned char* event_by_row,
    const double* event_first_delta, const double* event_second_delta,
    int compact_derivative_mode,
    const int* group_by_row, const double* current_x, int current_dimension,
    const int* baseline_group_by_row, int baseline_groups,
    int baseline_dimension, const unsigned char* current_signed_state,
    double* footprint_stats,
    const int* relaxation_group_by_current_group, int relaxation_groups,
    double* group_footprint_stats,
    const int* active_current_columns, int active_current_dimension,
    const int* hessian_left, const int* hessian_right, int hessian_pairs,
    int maximum_tile_rows, int entities_per_chunk,
    std::int64_t chunks_per_candidate, double* partials) {
    const int dimension = maximum_blocks * knots;
    const int cross_count = active_current_dimension * dimension;
    const int output_dimension = dimension + hessian_pairs + cross_count;
    const int candidate = static_cast<int>(blockIdx.x / chunks_per_candidate);
    const std::int64_t chunk = blockIdx.x % chunks_per_candidate;
    if (candidate >= candidates) return;
    const std::int64_t candidate_entity_begin =
        candidate_entity_offsets[candidate];
    const std::int64_t candidate_entity_end =
        candidate_entity_offsets[candidate + 1];
    const std::int64_t active_begin =
        candidate_entity_begin + chunk * entities_per_chunk;
    const std::int64_t active_end =
        active_begin + entities_per_chunk < candidate_entity_end
            ? active_begin + entities_per_chunk
            : candidate_entity_end;
    extern __shared__ double shared[];
    double* activation = shared;
    double* moments = activation +
        static_cast<std::int64_t>(maximum_tile_rows) * dimension;
    int* active_rows = reinterpret_cast<int*>(moments + output_dimension);
    __shared__ unsigned long long active_columns;
    __shared__ unsigned long long active_current_mask;
    __shared__ int active_row_count;
    for (int output = threadIdx.x; output < output_dimension;
         output += blockDim.x)
        moments[output] = 0.0;
    __syncthreads();
    for (std::int64_t active = active_begin; active < active_end; ++active) {
        const std::int64_t entity = candidate_entities[active];
        const int length = static_cast<int>(ends[entity] - starts[entity] + 1);
        const std::int64_t candidate_metadata =
            static_cast<std::int64_t>(candidate) * maximum_blocks;
        if (!completion_mode && !implicit_candidate_entity_active(
                source_entity_offsets, source_times, entity_count,
                block_predicates +
                    candidate_metadata * kImplicitMaxSources,
                block_orders + candidate_metadata, block_counts[candidate],
                entity, ends[entity]))
            continue;
        const std::int64_t row0 = grid_offsets[entity];
        for (int tile_begin = 0; tile_begin < length;
             tile_begin += maximum_tile_rows) {
            const int tile_end =
                min(length, tile_begin + maximum_tile_rows);
            const int tile_length = tile_end - tile_begin;
            if (threadIdx.x == 0) {
                active_columns = 0ULL;
                active_current_mask = 0ULL;
                active_row_count = 0;
            }
            for (int cell = threadIdx.x; cell < tile_length * dimension;
                 cell += blockDim.x)
                activation[cell] = 0.0;
            __syncthreads();
            if (completion_mode) {
                // One thread owns one (hierarchy block, knot) column.  It
                // follows the same completion/lag order as the scalar block
                // writer, so every column has identical addition order while
                // the M independent knot columns are generated concurrently.
                for (int job = threadIdx.x;
                     job < block_counts[candidate] * knots;
                     job += blockDim.x) {
                    const int local_block = job / knots;
                    const int knot = job % knots;
                    const std::int64_t metadata =
                        static_cast<std::int64_t>(candidate) * maximum_blocks +
                        local_block;
                    fill_completion_design_knot(
                        source_entity_offsets, source_times, completion_spans,
                        completion_mode, entity_count,
                        block_predicates[
                            metadata * kImplicitMaxSources],
                        block_minimum_spans[metadata], block_windows[metadata],
                        entity, starts[entity],
                        ends[entity], basis, knots, lag, dimension, local_block,
                        knot, tile_begin, tile_end, activation);
                }
            } else {
                for (int local_block = threadIdx.x;
                     local_block < block_counts[candidate];
                     local_block += blockDim.x) {
                    const std::int64_t metadata =
                        static_cast<std::int64_t>(candidate) * maximum_blocks +
                        local_block;
                    fill_implicit_design_block(
                        source_entity_offsets, source_times, entity_count,
                        block_predicates + metadata * kImplicitMaxSources,
                        block_orders[metadata], block_windows[metadata], entity,
                        starts[entity], ends[entity], basis, knots, lag,
                        dimension, local_block, tile_begin, tile_end,
                        activation);
                }
            }
            __syncthreads();
            for (int row = threadIdx.x; row < tile_length;
                 row += blockDim.x) {
                bool active = false;
                for (int column = 0; column < dimension && !active; ++column)
                    active =
                        activation[
                            static_cast<std::int64_t>(row) * dimension +
                            column] != 0.0;
                if (active) {
                    const int destination = atomicAdd(&active_row_count, 1);
                    active_rows[destination] = row;
                }
            }
            __syncthreads();
            for (int column = threadIdx.x; column < dimension;
                 column += blockDim.x) {
                bool active = false;
                for (int position = 0;
                     position < active_row_count && !active; ++position) {
                    const int row = active_rows[position];
                    active = activation[
                        static_cast<std::int64_t>(row) * dimension + column] !=
                        0.0;
                }
                if (active) atomicOr(&active_columns, 1ULL << column);
            }
            if (active_current_dimension <= 64) {
                for (int column = threadIdx.x;
                     column < active_current_dimension;
                     column += blockDim.x) {
                    const int source_column = active_current_columns[column];
                    bool active = false;
                    for (int position = 0;
                         position < active_row_count && !active; ++position) {
                        const int row = active_rows[position];
                        const int group =
                            group_by_row[row0 + tile_begin + row];
                        active = current_x[
                            static_cast<std::int64_t>(group) *
                                current_dimension + source_column] != 0.0;
                    }
                    if (active)
                        atomicOr(&active_current_mask, 1ULL << column);
                }
            }
            __syncthreads();
            if (active_row_count == 0) continue;
            // The gradient pass has already constructed the exact union of
            // this candidate's rule/closure response rows.  Accumulate the
            // rows which are not already in the parent's *rule* footprint by
            // immutable baseline cell.  These two sufficient statistics
            // (row count, event count) define a localized saturated
            // likelihood relaxation on the host.  Importantly, baseline
            // controls do not count as parent-rule activity: only columns at
            // or beyond baseline_dimension are inspected.
            if (footprint_stats != nullptr && compact_derivative_mode != 0) {
                for (int position = threadIdx.x; position < active_row_count;
                     position += blockDim.x) {
                    const int row = active_rows[position];
                    const std::int64_t global_row = row0 + tile_begin + row;
                    const int group = group_by_row[global_row];
                    const int category = current_signed_state[group] & 3;
                    const int baseline_group =
                        baseline_group_by_row[global_row];
                    if (baseline_group >= 0 &&
                        baseline_group < baseline_groups) {
                        double* statistics = footprint_stats +
                            ((static_cast<std::int64_t>(candidate) * 4 +
                              category) * baseline_groups +
                             baseline_group) * 2;
                        atomicAdd(statistics, 1.0);
                        atomicAdd(
                            statistics + 1,
                            static_cast<double>(event_by_row[global_row]));
                    }
                }
            }
            // A second, tighter relaxation keeps every equality forced by
            // the parent's exact aggregated design.  Only mixed
            // event/no-event parent groups have positive saturated loss, so
            // the host supplies a compact map for those groups.  Candidate
            // rows themselves remain independently saturated; subtracting
            // their exact counts from each parent group therefore yields a
            // rigorous lower NLL bound without constructing M-knot rows.
            if (group_footprint_stats != nullptr &&
                relaxation_group_by_current_group != nullptr &&
                compact_derivative_mode != 0) {
                for (int position = threadIdx.x; position < active_row_count;
                     position += blockDim.x) {
                    const int row = active_rows[position];
                    const std::int64_t global_row = row0 + tile_begin + row;
                    const int group = group_by_row[global_row];
                    const int relaxation_group =
                        relaxation_group_by_current_group[group];
                    if (relaxation_group >= 0 &&
                        relaxation_group < relaxation_groups) {
                        double* statistics = group_footprint_stats +
                            (static_cast<std::int64_t>(candidate) *
                                 relaxation_groups +
                             relaxation_group) * 2;
                        atomicAdd(statistics, 1.0);
                        atomicAdd(
                            statistics + 1,
                            static_cast<double>(event_by_row[global_row]));
                    }
                }
            }
            for (int output = threadIdx.x; output < output_dimension;
                 output += blockDim.x) {
                double value = 0.0;
                if (output < dimension) {
                    if ((active_columns & (1ULL << output)) == 0ULL) continue;
                    for (int position = 0; position < active_row_count;
                         ++position) {
                        const int row = active_rows[position];
                        const std::int64_t global_row =
                            row0 + tile_begin + row;
                        const int group = group_by_row[global_row];
                        const double event_count =
                            compact_derivative_mode
                                ? static_cast<double>(event_by_row[global_row])
                                : 0.0;
                        const double row_first =
                            compact_derivative_mode == 1
                                ? first[group] - event_count
                                : (
                                    compact_derivative_mode == 2
                                        ? first[group] +
                                              event_count *
                                                  event_first_delta[group]
                                        : first[global_row]
                                );
                        value +=
                            activation[
                                static_cast<std::int64_t>(row) * dimension +
                                output] *
                            row_first;
                    }
                } else if (output < dimension + hessian_pairs) {
                    const int pair = output - dimension;
                    const int left = hessian_left[pair];
                    const int right = hessian_right[pair];
                    if ((active_columns & (1ULL << left)) == 0ULL ||
                        (active_columns & (1ULL << right)) == 0ULL)
                        continue;
                    for (int position = 0; position < active_row_count;
                         ++position) {
                        const int row = active_rows[position];
                        const std::int64_t global_row =
                            row0 + tile_begin + row;
                        const int group = group_by_row[global_row];
                        const double event_count =
                            compact_derivative_mode
                                ? static_cast<double>(event_by_row[global_row])
                                : 0.0;
                        const double row_second =
                            compact_derivative_mode == 2
                                ? second[group] +
                                      event_count * event_second_delta[group]
                                : (
                                    compact_derivative_mode == 1
                                        ? second[group]
                                        : second[global_row]
                                );
                        value +=
                            activation[
                                static_cast<std::int64_t>(row) * dimension +
                                left] *
                            row_second *
                            activation[
                                static_cast<std::int64_t>(row) * dimension +
                                right];
                    }
                } else {
                    const int cross = output - dimension - hessian_pairs;
                    const int current_column = cross / dimension;
                    const int candidate_column = cross % dimension;
                    const int source_column =
                        active_current_columns[current_column];
                    if ((active_columns & (1ULL << candidate_column)) == 0ULL)
                        continue;
                    if (active_current_dimension <= 64 &&
                        (active_current_mask &
                         (1ULL << current_column)) == 0ULL)
                        continue;
                    for (int position = 0; position < active_row_count;
                         ++position) {
                        const int row = active_rows[position];
                        const std::int64_t global_row =
                            row0 + tile_begin + row;
                        const int group = group_by_row[global_row];
                        const double event_count =
                            compact_derivative_mode
                                ? static_cast<double>(event_by_row[global_row])
                                : 0.0;
                        const double row_second =
                            compact_derivative_mode == 2
                                ? second[group] +
                                      event_count * event_second_delta[group]
                                : (
                                    compact_derivative_mode == 1
                                        ? second[group]
                                        : second[global_row]
                                );
                        value +=
                            current_x[
                                static_cast<std::int64_t>(group) *
                                    current_dimension + source_column] *
                            row_second *
                            activation[
                                static_cast<std::int64_t>(row) * dimension +
                                candidate_column];
                    }
                }
                moments[output] += value;
            }
            __syncthreads();
        }
    }
    double* destination = partials +
        (static_cast<std::int64_t>(candidate) * chunks_per_candidate + chunk) *
            output_dimension;
    for (int output = threadIdx.x; output < output_dimension;
         output += blockDim.x)
        destination[output] = moments[output];
}

// Exact parent-frozen objective changes for a batch of candidate hierarchy
// blocks.  The expensive predicate merge/completion pass is shared by one
// CUDA launch instead of being repeated by Python for every candidate.
// Candidate coefficients already include closure/rule signs.  Mode 1 is
// recurrent Poisson and mode 2 is first-event complementary log-log.
__global__ void implicit_objective_partials_kernel(
    const std::int64_t* source_entity_offsets, const std::int64_t* source_times,
    const std::int64_t* completion_spans, int completion_mode,
    std::int64_t entity_count, const std::int64_t* starts,
    const std::int64_t* ends, const std::int64_t* grid_offsets,
    const double* basis, int knots, int lag, const int* block_predicates,
    const int* block_orders, const std::int64_t* block_minimum_spans,
    const std::int64_t* block_windows,
    const int* block_counts, int candidates, int maximum_blocks,
    const std::int64_t* candidate_entity_offsets,
    const int* candidate_entities,
    const double* coefficients, const double* group_mean,
    const double* group_eta, int likelihood_mode,
    const unsigned char* event_by_row,
    const int* group_by_row, int maximum_tile_rows, int entities_per_chunk,
    std::int64_t chunks_per_candidate, double* partials) {
    const int dimension = maximum_blocks * knots;
    const int candidate = static_cast<int>(blockIdx.x / chunks_per_candidate);
    const std::int64_t chunk = blockIdx.x % chunks_per_candidate;
    if (candidate >= candidates) return;
    const std::int64_t candidate_entity_begin =
        candidate_entity_offsets[candidate];
    const std::int64_t candidate_entity_end =
        candidate_entity_offsets[candidate + 1];
    const std::int64_t active_begin =
        candidate_entity_begin + chunk * entities_per_chunk;
    const std::int64_t active_end =
        active_begin + entities_per_chunk < candidate_entity_end
            ? active_begin + entities_per_chunk
            : candidate_entity_end;
    extern __shared__ double shared[];
    double* activation = shared;
    double* reduction = activation +
        static_cast<std::int64_t>(maximum_tile_rows) * dimension;
    double value = 0.0;
    for (std::int64_t active = active_begin; active < active_end; ++active) {
        const std::int64_t entity = candidate_entities[active];
        const int length = static_cast<int>(ends[entity] - starts[entity] + 1);
        const std::int64_t candidate_metadata =
            static_cast<std::int64_t>(candidate) * maximum_blocks;
        if (!completion_mode && !implicit_candidate_entity_active(
                source_entity_offsets, source_times, entity_count,
                block_predicates +
                    candidate_metadata * kImplicitMaxSources,
                block_orders + candidate_metadata, block_counts[candidate],
                entity, ends[entity]))
            continue;
        const std::int64_t row0 = grid_offsets[entity];
        for (int tile_begin = 0; tile_begin < length;
             tile_begin += maximum_tile_rows) {
            const int tile_end = min(length, tile_begin + maximum_tile_rows);
            const int tile_length = tile_end - tile_begin;
            for (int cell = threadIdx.x; cell < tile_length * dimension;
                 cell += blockDim.x)
                activation[cell] = 0.0;
            __syncthreads();
            if (completion_mode) {
                for (int job = threadIdx.x;
                     job < block_counts[candidate] * knots;
                     job += blockDim.x) {
                    const int local_block = job / knots;
                    const int knot = job % knots;
                    const std::int64_t metadata =
                        static_cast<std::int64_t>(candidate) * maximum_blocks +
                        local_block;
                    fill_completion_design_knot(
                        source_entity_offsets, source_times, completion_spans,
                        completion_mode, entity_count,
                        block_predicates[
                            metadata * kImplicitMaxSources],
                        block_minimum_spans[metadata], block_windows[metadata],
                        entity, starts[entity],
                        ends[entity], basis, knots, lag, dimension, local_block,
                        knot, tile_begin, tile_end, activation);
                }
            } else {
                for (int local_block = threadIdx.x;
                     local_block < block_counts[candidate];
                     local_block += blockDim.x) {
                    const std::int64_t metadata =
                        static_cast<std::int64_t>(candidate) * maximum_blocks +
                        local_block;
                    fill_implicit_design_block(
                        source_entity_offsets, source_times, entity_count,
                        block_predicates + metadata * kImplicitMaxSources,
                        block_orders[metadata], block_windows[metadata], entity,
                        starts[entity], ends[entity], basis, knots, lag,
                        dimension, local_block, tile_begin, tile_end,
                        activation);
                }
            }
            __syncthreads();
            for (int row = threadIdx.x; row < tile_length; row += blockDim.x) {
                const std::int64_t global_row = row0 + tile_begin + row;
                double effect = 0.0;
                const double* design =
                    activation + static_cast<std::int64_t>(row) * dimension;
                const double* beta =
                    coefficients +
                    static_cast<std::int64_t>(candidate) * dimension;
                for (int column = 0; column < dimension; ++column)
                    effect += design[column] * beta[column];
                if (effect == 0.0) continue;
                const int group = group_by_row[global_row];
                const double old_mean = group_mean[group];
                const double event_count =
                    static_cast<double>(event_by_row[global_row]);
                // Binary observation masks are represented by one appended
                // group with zero mean/derivatives.  Such a row has exactly
                // zero likelihood opportunity, so its parent-frozen change is
                // also exactly zero.  Event rows are validated as observed
                // by the Dataset contract and deliberately do not take this
                // branch.
                if (old_mean == 0.0 && event_count == 0.0) continue;
                if (likelihood_mode == 1) {
                    const double new_mean = old_mean * exp(effect);
                    value += new_mean - old_mean - event_count * effect;
                } else {
                    const double old_eta = group_eta[group];
                    const double new_eta = old_eta + effect;
                    const double new_mean =
                        exp(fmin(fmax(new_eta, -745.0), 700.0));
                    double old_event_value = 0.0;
                    double old_event_first = 0.0;
                    double old_event_second = 0.0;
                    double new_event_value = 0.0;
                    double new_event_first = 0.0;
                    double new_event_second = 0.0;
                    cloglog_event_terms_device(
                        old_eta, &old_event_value, &old_event_first,
                        &old_event_second);
                    cloglog_event_terms_device(
                        new_eta, &new_event_value, &new_event_first,
                        &new_event_second);
                    value +=
                        (1.0 - event_count) * (new_mean - old_mean) +
                        event_count * (new_event_value - old_event_value);
                }
            }
            __syncthreads();
        }
    }
    reduction[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partials[
            static_cast<std::int64_t>(candidate) * chunks_per_candidate +
            chunk] = reduction[0];
}

__global__ void reduce_candidate_partials_kernel(
    const double* partials, int candidates,
    std::int64_t chunks_per_candidate, double* output) {
    const int candidate = blockIdx.x;
    if (candidate >= candidates) return;
    extern __shared__ double reduction[];
    double value = 0.0;
    const double* source =
        partials + static_cast<std::int64_t>(candidate) * chunks_per_candidate;
    for (std::int64_t chunk = threadIdx.x; chunk < chunks_per_candidate;
         chunk += blockDim.x)
        value += source[chunk];
    reduction[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[candidate] = reduction[0];
}

__global__ void implicit_reduce_partials_kernel(
    const double* partials, int candidates, int dimension,
    int current_dimension, const int* hessian_left,
    const int* hessian_right, int hessian_pairs,
    std::int64_t chunks_per_candidate, double* gradient, double* hessian,
    double* cross) {
    const int output_dimension =
        dimension + hessian_pairs + current_dimension * dimension;
    const std::int64_t flat = blockIdx.x;
    if (flat >= static_cast<std::int64_t>(candidates) * output_dimension) return;
    const int candidate = static_cast<int>(flat / output_dimension);
    const int output = static_cast<int>(flat % output_dimension);
    extern __shared__ double reduction[];
    double value = 0.0;
    for (std::int64_t chunk = threadIdx.x; chunk < chunks_per_candidate;
         chunk += blockDim.x) {
        value += partials[
            (static_cast<std::int64_t>(candidate) * chunks_per_candidate +
             chunk) * output_dimension + output];
    }
    reduction[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    if (output < dimension) {
        gradient[static_cast<std::int64_t>(candidate) * dimension + output] =
            reduction[0];
    } else if (output < dimension + hessian_pairs) {
        const int pair = output - dimension;
        const int left = hessian_left[pair];
        const int right = hessian_right[pair];
        double* destination = hessian +
            static_cast<std::int64_t>(candidate) * dimension * dimension;
        destination[static_cast<std::int64_t>(left) * dimension + right] =
            reduction[0];
        destination[static_cast<std::int64_t>(right) * dimension + left] =
            reduction[0];
    } else {
        const int local = output - dimension - hessian_pairs;
        cross[(static_cast<std::int64_t>(candidate) * current_dimension *
               dimension) + local] = reduction[0];
    }
}

struct Workspace {
    std::mutex mutex;
    double* x = nullptr;
    double* weighted = nullptr;
    double* projected = nullptr;
    std::int64_t* projected_columns = nullptr;
    double* projected_scales = nullptr;
    double* first = nullptr;
    double* second = nullptr;
    double* gradient = nullptr;
    double* cross = nullptr;
    double* hessian = nullptr;
    double* beta = nullptr;
    double* eta = nullptr;
    double* exposure = nullptr;
    double* event = nullptr;
    double* value = nullptr;
    double* ones = nullptr;
    std::size_t x_capacity = 0;
    std::size_t weighted_capacity = 0;
    std::size_t projected_capacity = 0;
    std::size_t projected_columns_capacity = 0;
    std::size_t projected_scales_capacity = 0;
    std::size_t first_capacity = 0;
    std::size_t second_capacity = 0;
    std::size_t gradient_capacity = 0;
    std::size_t cross_capacity = 0;
    std::size_t hessian_capacity = 0;
    std::size_t beta_capacity = 0;
    std::size_t eta_capacity = 0;
    std::size_t exposure_capacity = 0;
    std::size_t event_capacity = 0;
    std::size_t value_capacity = 0;
    std::size_t ones_capacity = 0;
    std::int64_t matrix_token = -1;
    std::int64_t matrix_rows = -1;
    std::int64_t matrix_columns = -1;
    std::int64_t likelihood_token = -1;
    int likelihood_mode = 0;
    std::int64_t projection_token = -1;
    std::int64_t projection_columns_count = -1;
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
    // Indexed sparse batches repeatedly use one immutable likelihood
    // derivative grid.  Keep it resident until the caller changes the token;
    // response rows and values remain candidate-specific and are still copied
    // for every call.
    std::int64_t derivative_token = -1;
    std::int64_t derivative_rows = -1;
    std::int64_t geometry_token = -1;
    std::int64_t geometry_total_rows = -1;
    std::int64_t geometry_candidates = -1;
    std::int64_t geometry_blocks = -1;
    std::int64_t geometry_knots = -1;
};

struct ImplicitWorkspace {
    std::mutex mutex;
    std::int64_t* source_offsets = nullptr;
    std::int64_t* source_times = nullptr;
    std::int64_t* source_spans = nullptr;
    std::int64_t* starts = nullptr;
    std::int64_t* ends = nullptr;
    std::int64_t* grid_offsets = nullptr;
    double* basis = nullptr;
    int* block_predicates = nullptr;
    int* block_orders = nullptr;
    std::int64_t* block_minimum_spans = nullptr;
    std::int64_t* block_windows = nullptr;
    int* block_counts = nullptr;
    std::int64_t* candidate_entity_offsets = nullptr;
    int* candidate_entities = nullptr;
    double* first = nullptr;
    double* second = nullptr;
    unsigned char* event_by_row = nullptr;
    double* event_first_delta = nullptr;
    double* event_second_delta = nullptr;
    double* group_eta = nullptr;
    int* group_by_row = nullptr;
    int* baseline_group_by_row = nullptr;
    unsigned char* current_signed_state = nullptr;
    int* relaxation_group_by_current_group = nullptr;
    int* hessian_left = nullptr;
    int* hessian_right = nullptr;
    int* active_current_columns = nullptr;
    double* current_x = nullptr;
    double* future_score = nullptr;
    double* gradient = nullptr;
    double* hessian = nullptr;
    double* cross = nullptr;
    double* partials = nullptr;
    double* footprint_stats = nullptr;
    double* group_footprint_stats = nullptr;
    std::size_t source_offsets_capacity = 0;
    std::size_t source_times_capacity = 0;
    std::size_t source_spans_capacity = 0;
    std::size_t starts_capacity = 0;
    std::size_t ends_capacity = 0;
    std::size_t grid_offsets_capacity = 0;
    std::size_t basis_capacity = 0;
    std::size_t block_predicates_capacity = 0;
    std::size_t block_orders_capacity = 0;
    std::size_t block_minimum_spans_capacity = 0;
    std::size_t block_windows_capacity = 0;
    std::size_t block_counts_capacity = 0;
    std::size_t candidate_entity_offsets_capacity = 0;
    std::size_t candidate_entities_capacity = 0;
    std::size_t first_capacity = 0;
    std::size_t second_capacity = 0;
    std::size_t event_by_row_capacity = 0;
    std::size_t event_first_delta_capacity = 0;
    std::size_t event_second_delta_capacity = 0;
    std::size_t group_eta_capacity = 0;
    std::size_t group_by_row_capacity = 0;
    std::size_t baseline_group_by_row_capacity = 0;
    std::size_t current_signed_state_capacity = 0;
    std::size_t relaxation_group_by_current_group_capacity = 0;
    std::size_t hessian_left_capacity = 0;
    std::size_t hessian_right_capacity = 0;
    std::size_t active_current_columns_capacity = 0;
    std::size_t current_x_capacity = 0;
    std::size_t future_score_capacity = 0;
    std::size_t gradient_capacity = 0;
    std::size_t hessian_capacity = 0;
    std::size_t cross_capacity = 0;
    std::size_t partials_capacity = 0;
    std::size_t footprint_stats_capacity = 0;
    std::size_t group_footprint_stats_capacity = 0;
    std::int64_t source_token = -1;
    std::int64_t source_entities = -1;
    std::int64_t source_predicates = -1;
    std::int64_t source_events = -1;
    int source_completion_mode = 0;
    int maximum_entity_rows = -1;
    std::int64_t derivative_token = -1;
    std::int64_t derivative_rows = -1;
    int derivative_compact_mode = 0;
    std::int64_t objective_token = -1;
    int objective_mode = 0;
    std::int64_t current_token = -1;
    std::int64_t current_rows = -1;
    std::int64_t current_groups = -1;
    int current_dimension = -1;
    std::int64_t future_score_token = -1;
    std::int64_t future_score_rows = -1;
    int future_score_knots = -1;
    int future_score_lag = -1;
    std::int64_t baseline_token = -1;
    std::int64_t baseline_rows = -1;
    int baseline_groups = -1;
    int baseline_dimension = -1;
    int hessian_dimension = -1;
};

std::array<Workspace, 16> workspaces;
std::array<SparseWorkspace, 16> sparse_workspaces;
std::array<ImplicitWorkspace, 16> implicit_workspaces;

template <typename T>
void release_buffer(T** pointer, std::size_t* capacity) {
    if (*pointer != nullptr) cudaFree(*pointer);
    *pointer = nullptr;
    *capacity = 0;
}

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

bool reserve_int(
    int** pointer, std::size_t* capacity, std::size_t bytes) {
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

bool reserve_byte(
    unsigned char** pointer, std::size_t* capacity, std::size_t bytes) {
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
    workspace.matrix_token = -1;
    workspace.matrix_rows = -1;
    workspace.matrix_columns = -1;
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

extern "C" int crbstpp_cuda_moments_resident(
    int device, std::int64_t matrix_token, const double* host_x,
    const double* host_first, const double* host_second, std::int64_t rows,
    std::int64_t columns, double* host_gradient, double* host_hessian) {
    if (device < 0 || device >= static_cast<int>(workspaces.size())) return 1;
    Workspace& workspace = workspaces[device];
    std::lock_guard<std::mutex> lock(workspace.mutex);
    if (cudaSetDevice(device) != cudaSuccess) return 1;
    const int threads = 256;
    const std::size_t x_bytes = sizeof(double) * rows * columns;
    const std::size_t row_bytes = sizeof(double) * rows;
    const std::size_t gradient_bytes = sizeof(double) * columns;
    const std::size_t hessian_bytes = sizeof(double) * columns * columns;
    const bool same_matrix =
        workspace.matrix_token == matrix_token &&
        workspace.matrix_rows == rows &&
        workspace.matrix_columns == columns &&
        workspace.x_capacity >= x_bytes;
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
    if ((!same_matrix &&
         cudaMemcpy(workspace.x, host_x, x_bytes, cudaMemcpyHostToDevice) !=
             cudaSuccess) ||
        cudaMemcpy(workspace.first, host_first, row_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.second, host_second, row_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return 2;
    workspace.matrix_token = matrix_token;
    workspace.matrix_rows = rows;
    workspace.matrix_columns = columns;
    {
        const std::int64_t count = rows * columns;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        weight_design_kernel<<<blocks, threads>>>(
            workspace.x, workspace.second, rows, columns, workspace.weighted);
    }
    if (cudaGetLastError() != cudaSuccess) return 2;
    const double alpha = 1.0, beta = 0.0;
    if (cublasDgemv(
            workspace.handle, CUBLAS_OP_N, static_cast<int>(columns),
            static_cast<int>(rows), &alpha, workspace.x,
            static_cast<int>(columns), workspace.first, 1, &beta,
            workspace.gradient, 1) != CUBLAS_STATUS_SUCCESS ||
        cublasDgemm(
            workspace.handle, CUBLAS_OP_N, CUBLAS_OP_T,
            static_cast<int>(columns), static_cast<int>(columns),
            static_cast<int>(rows), &alpha, workspace.x,
            static_cast<int>(columns), workspace.weighted,
            static_cast<int>(columns), &beta, workspace.hessian,
            static_cast<int>(columns)) != CUBLAS_STATUS_SUCCESS)
        return 2;
    if (cudaMemcpy(host_gradient, workspace.gradient, gradient_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_hessian, workspace.hessian, hessian_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    return 0;
}

extern "C" int crbstpp_cuda_eta_resident(
    int device, std::int64_t matrix_token, const double* host_x,
    const double* host_beta, std::int64_t rows, std::int64_t columns,
    double* host_eta) {
    if (device < 0 || device >= static_cast<int>(workspaces.size())) return 1;
    Workspace& workspace = workspaces[device];
    std::lock_guard<std::mutex> lock(workspace.mutex);
    if (cudaSetDevice(device) != cudaSuccess) return 1;
    const std::size_t x_bytes = sizeof(double) * rows * columns;
    const std::size_t beta_bytes = sizeof(double) * columns;
    const std::size_t eta_bytes = sizeof(double) * rows;
    const bool same_matrix =
        workspace.matrix_token == matrix_token &&
        workspace.matrix_rows == rows &&
        workspace.matrix_columns == columns &&
        workspace.x_capacity >= x_bytes;
    if (!reserve(&workspace.x, &workspace.x_capacity, x_bytes) ||
        !reserve(&workspace.beta, &workspace.beta_capacity, beta_bytes) ||
        !reserve(&workspace.eta, &workspace.eta_capacity, eta_bytes))
        return 2;
    if (workspace.handle == nullptr &&
        cublasCreate(&workspace.handle) != CUBLAS_STATUS_SUCCESS)
        return 2;
    if ((!same_matrix &&
         cudaMemcpy(workspace.x, host_x, x_bytes, cudaMemcpyHostToDevice) !=
             cudaSuccess) ||
        cudaMemcpy(workspace.beta, host_beta, beta_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return 2;
    workspace.matrix_token = matrix_token;
    workspace.matrix_rows = rows;
    workspace.matrix_columns = columns;
    const double alpha = 1.0, beta = 0.0;
    // Row-major X(rows, columns) is column-major X.T(columns, rows).
    if (cublasDgemv(
            workspace.handle, CUBLAS_OP_T, static_cast<int>(columns),
            static_cast<int>(rows), &alpha, workspace.x,
            static_cast<int>(columns), workspace.beta, 1, &beta,
            workspace.eta, 1) != CUBLAS_STATUS_SUCCESS ||
        cudaMemcpy(host_eta, workspace.eta, eta_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    return 0;
}

extern "C" int crbstpp_cuda_projected_objective_resident(
    int device, std::int64_t matrix_token, std::int64_t projection_token,
    const double* host_x, const double* host_beta,
    const std::int64_t* host_columns, const double* host_scales,
    const double* host_primary, const double* host_event,
    std::int64_t rows, std::int64_t source_columns,
    std::int64_t projected_columns, int likelihood_mode,
    int compute_moments, double* host_nll, double* host_gradient,
    double* host_hessian) {
    if (device < 0 || device >= static_cast<int>(workspaces.size()) ||
        rows < 1 || rows > INT32_MAX || source_columns < 1 ||
        source_columns > INT32_MAX || projected_columns < 1 ||
        projected_columns > source_columns || host_nll == nullptr ||
        host_columns == nullptr || host_scales == nullptr ||
        (likelihood_mode != 1 && likelihood_mode != 2))
        return 1;
    Workspace& workspace = workspaces[device];
    std::lock_guard<std::mutex> lock(workspace.mutex);
    if (cudaSetDevice(device) != cudaSuccess) return 1;
    const int threads = 256;
    const std::size_t x_bytes = sizeof(double) * rows * source_columns;
    const std::size_t projected_bytes =
        sizeof(double) * rows * projected_columns;
    const std::size_t row_bytes = sizeof(double) * rows;
    const std::size_t beta_bytes = sizeof(double) * projected_columns;
    const std::size_t column_bytes =
        sizeof(std::int64_t) * projected_columns;
    const std::size_t scale_bytes = sizeof(double) * projected_columns;
    const std::size_t gradient_bytes = sizeof(double) * projected_columns;
    const std::size_t hessian_bytes =
        sizeof(double) * projected_columns * projected_columns;
    const bool same_matrix =
        workspace.matrix_token == matrix_token &&
        workspace.matrix_rows == rows &&
        workspace.matrix_columns == source_columns &&
        workspace.x_capacity >= x_bytes;
    const bool same_likelihood =
        same_matrix && workspace.likelihood_token == matrix_token &&
        workspace.likelihood_mode == likelihood_mode &&
        workspace.exposure_capacity >= row_bytes &&
        workspace.event_capacity >= row_bytes;
    const bool same_projection =
        same_matrix && workspace.projection_token == projection_token &&
        workspace.projection_columns_count == projected_columns &&
        workspace.projected_capacity >= projected_bytes;
    if (!reserve(&workspace.x, &workspace.x_capacity, x_bytes) ||
        !reserve(&workspace.projected, &workspace.projected_capacity,
                 projected_bytes) ||
        !reserve_int64(&workspace.projected_columns,
                       &workspace.projected_columns_capacity, column_bytes) ||
        !reserve(&workspace.projected_scales,
                 &workspace.projected_scales_capacity, scale_bytes) ||
        !reserve(&workspace.weighted, &workspace.weighted_capacity,
                 projected_bytes) ||
        !reserve(&workspace.beta, &workspace.beta_capacity, beta_bytes) ||
        !reserve(&workspace.eta, &workspace.eta_capacity, row_bytes) ||
        !reserve(&workspace.exposure, &workspace.exposure_capacity, row_bytes) ||
        !reserve(&workspace.event, &workspace.event_capacity, row_bytes) ||
        !reserve(&workspace.value, &workspace.value_capacity, row_bytes) ||
        !reserve(&workspace.ones, &workspace.ones_capacity, row_bytes) ||
        !reserve(&workspace.first, &workspace.first_capacity, row_bytes) ||
        !reserve(&workspace.second, &workspace.second_capacity, row_bytes) ||
        (compute_moments &&
         !reserve(&workspace.gradient, &workspace.gradient_capacity,
                  gradient_bytes)) ||
        (compute_moments == 1 &&
         !reserve(&workspace.hessian, &workspace.hessian_capacity,
                  hessian_bytes)))
        return 2;
    if (workspace.handle == nullptr &&
        cublasCreate(&workspace.handle) != CUBLAS_STATUS_SUCCESS)
        return 2;
    if ((!same_matrix &&
         cudaMemcpy(workspace.x, host_x, x_bytes, cudaMemcpyHostToDevice) !=
             cudaSuccess) ||
        (!same_likelihood &&
         (cudaMemcpy(workspace.exposure, host_primary, row_bytes,
                     cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(workspace.event, host_event, row_bytes,
                     cudaMemcpyHostToDevice) != cudaSuccess)) ||
        cudaMemcpy(workspace.beta, host_beta, beta_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return 2;
    if (!same_projection) {
        if (cudaMemcpy(workspace.projected_columns, host_columns, column_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(workspace.projected_scales, host_scales, scale_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return 2;
        const std::int64_t count = rows * projected_columns;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        gather_projected_design_kernel<<<blocks, threads>>>(
            workspace.x, workspace.projected_columns,
            workspace.projected_scales, rows, source_columns,
            projected_columns, workspace.projected);
        if (cudaGetLastError() != cudaSuccess) return 2;
    }
    workspace.matrix_token = matrix_token;
    workspace.matrix_rows = rows;
    workspace.matrix_columns = source_columns;
    workspace.likelihood_token = matrix_token;
    workspace.likelihood_mode = likelihood_mode;
    workspace.projection_token = projection_token;
    workspace.projection_columns_count = projected_columns;

    const double alpha = 1.0, zero = 0.0;
    if (cublasDgemv(
            workspace.handle, CUBLAS_OP_T,
            static_cast<int>(projected_columns), static_cast<int>(rows),
            &alpha, workspace.projected,
            static_cast<int>(projected_columns), workspace.beta, 1,
            &zero, workspace.eta, 1) != CUBLAS_STATUS_SUCCESS)
        return 2;
    const int row_blocks = static_cast<int>((rows + threads - 1) / threads);
    if (likelihood_mode == 1) {
        poisson_likelihood_kernel<<<row_blocks, threads>>>(
            workspace.eta, workspace.exposure, workspace.event, rows,
            workspace.value, workspace.first, workspace.second);
    } else {
        cloglog_likelihood_kernel<<<row_blocks, threads>>>(
            workspace.eta, workspace.exposure, workspace.event, rows,
            workspace.value, workspace.first, workspace.second);
    }
    fill_constant_kernel<<<row_blocks, threads>>>(workspace.ones, rows, 1.0);
    if (cudaGetLastError() != cudaSuccess) return 2;
    double nll = 0.0;
    if (cublasDdot(workspace.handle, static_cast<int>(rows), workspace.ones, 1,
                   workspace.value, 1, &nll) != CUBLAS_STATUS_SUCCESS)
        return 2;
    if (compute_moments) {
        if (cublasDgemv(
                workspace.handle, CUBLAS_OP_N,
                static_cast<int>(projected_columns), static_cast<int>(rows),
                &alpha, workspace.projected,
                static_cast<int>(projected_columns), workspace.first, 1,
                &zero, workspace.gradient, 1) != CUBLAS_STATUS_SUCCESS)
            return 2;
    }
    if (compute_moments == 1) {
        const std::int64_t count = rows * projected_columns;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        weight_design_kernel<<<blocks, threads>>>(
            workspace.projected, workspace.second, rows, projected_columns,
            workspace.weighted);
        if (cudaGetLastError() != cudaSuccess ||
            cublasDgemm(
                workspace.handle, CUBLAS_OP_N, CUBLAS_OP_T,
                static_cast<int>(projected_columns),
                static_cast<int>(projected_columns), static_cast<int>(rows),
                &alpha, workspace.projected,
                static_cast<int>(projected_columns), workspace.weighted,
                static_cast<int>(projected_columns), &zero,
                workspace.hessian,
                static_cast<int>(projected_columns)) != CUBLAS_STATUS_SUCCESS)
            return 2;
    }
    *host_nll = nll;
    if (compute_moments &&
        cudaMemcpy(host_gradient, workspace.gradient, gradient_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    if (compute_moments == 1 &&
        cudaMemcpy(host_hessian, workspace.hessian, hessian_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    return 0;
}

extern "C" int crbstpp_cuda_poisson_objective_resident(
    int device, std::int64_t matrix_token, const double* host_x,
    const double* host_beta, const double* host_exposure,
    const double* host_event, std::int64_t rows, std::int64_t columns,
    int compute_moments, double* host_nll, double* host_eta,
    double* host_gradient, double* host_hessian) {
    if (device < 0 || device >= static_cast<int>(workspaces.size()) ||
        rows < 1 || rows > INT32_MAX || columns < 1 || columns > INT32_MAX ||
        host_nll == nullptr)
        return 1;
    Workspace& workspace = workspaces[device];
    std::lock_guard<std::mutex> lock(workspace.mutex);
    if (cudaSetDevice(device) != cudaSuccess) return 1;
    const int threads = 256;
    const std::size_t x_bytes = sizeof(double) * rows * columns;
    const std::size_t row_bytes = sizeof(double) * rows;
    const std::size_t beta_bytes = sizeof(double) * columns;
    const std::size_t gradient_bytes = sizeof(double) * columns;
    const std::size_t hessian_bytes = sizeof(double) * columns * columns;
    const bool same_matrix =
        workspace.matrix_token == matrix_token &&
        workspace.matrix_rows == rows &&
        workspace.matrix_columns == columns &&
        workspace.x_capacity >= x_bytes;
    const bool same_likelihood =
        same_matrix && workspace.likelihood_token == matrix_token &&
        workspace.likelihood_mode == 1 &&
        workspace.exposure_capacity >= row_bytes &&
        workspace.event_capacity >= row_bytes;
    if (!reserve(&workspace.x, &workspace.x_capacity, x_bytes) ||
        !reserve(&workspace.beta, &workspace.beta_capacity, beta_bytes) ||
        !reserve(&workspace.eta, &workspace.eta_capacity, row_bytes) ||
        !reserve(&workspace.exposure, &workspace.exposure_capacity, row_bytes) ||
        !reserve(&workspace.event, &workspace.event_capacity, row_bytes) ||
        !reserve(&workspace.value, &workspace.value_capacity, row_bytes) ||
        !reserve(&workspace.ones, &workspace.ones_capacity, row_bytes) ||
        !reserve(&workspace.first, &workspace.first_capacity, row_bytes) ||
        !reserve(&workspace.second, &workspace.second_capacity, row_bytes) ||
        (compute_moments &&
         !reserve(&workspace.gradient, &workspace.gradient_capacity,
                  gradient_bytes)) ||
        (compute_moments == 1 &&
         (!reserve(&workspace.weighted, &workspace.weighted_capacity, x_bytes) ||
          !reserve(&workspace.hessian, &workspace.hessian_capacity,
                   hessian_bytes))))
        return 2;
    if (workspace.handle == nullptr &&
        cublasCreate(&workspace.handle) != CUBLAS_STATUS_SUCCESS)
        return 2;
    if ((!same_matrix &&
         cudaMemcpy(workspace.x, host_x, x_bytes, cudaMemcpyHostToDevice) !=
             cudaSuccess) ||
        (!same_likelihood &&
         (cudaMemcpy(workspace.exposure, host_exposure, row_bytes,
                     cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(workspace.event, host_event, row_bytes,
                     cudaMemcpyHostToDevice) != cudaSuccess)) ||
        cudaMemcpy(workspace.beta, host_beta, beta_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return 2;
    workspace.matrix_token = matrix_token;
    workspace.matrix_rows = rows;
    workspace.matrix_columns = columns;
    workspace.likelihood_token = matrix_token;
    workspace.likelihood_mode = 1;
    const double alpha = 1.0, zero = 0.0;
    if (cublasDgemv(
            workspace.handle, CUBLAS_OP_T, static_cast<int>(columns),
            static_cast<int>(rows), &alpha, workspace.x,
            static_cast<int>(columns), workspace.beta, 1, &zero,
            workspace.eta, 1) != CUBLAS_STATUS_SUCCESS)
        return 2;
    const int row_blocks = static_cast<int>((rows + threads - 1) / threads);
    poisson_likelihood_kernel<<<row_blocks, threads>>>(
        workspace.eta, workspace.exposure, workspace.event, rows,
        workspace.value, workspace.first, workspace.second);
    fill_constant_kernel<<<row_blocks, threads>>>(workspace.ones, rows, 1.0);
    if (cudaGetLastError() != cudaSuccess) return 2;
    double nll = 0.0;
    if (cublasDdot(
            workspace.handle, static_cast<int>(rows), workspace.ones, 1,
            workspace.value, 1, &nll) != CUBLAS_STATUS_SUCCESS)
        return 2;
    if (compute_moments) {
        if (cublasDgemv(
                workspace.handle, CUBLAS_OP_N, static_cast<int>(columns),
                static_cast<int>(rows), &alpha, workspace.x,
                static_cast<int>(columns), workspace.first, 1, &zero,
                workspace.gradient, 1) != CUBLAS_STATUS_SUCCESS)
            return 2;
    }
    if (compute_moments == 1) {
        const std::int64_t count = rows * columns;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        weight_design_kernel<<<blocks, threads>>>(
            workspace.x, workspace.second, rows, columns,
            workspace.weighted);
        if (cudaGetLastError() != cudaSuccess ||
            cublasDgemm(
                workspace.handle, CUBLAS_OP_N, CUBLAS_OP_T,
                static_cast<int>(columns), static_cast<int>(columns),
                static_cast<int>(rows), &alpha, workspace.x,
                static_cast<int>(columns), workspace.weighted,
                static_cast<int>(columns), &zero, workspace.hessian,
                static_cast<int>(columns)) != CUBLAS_STATUS_SUCCESS)
            return 2;
    }
    *host_nll = nll;
    if (host_eta != nullptr &&
        cudaMemcpy(host_eta, workspace.eta, row_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    if (compute_moments &&
        cudaMemcpy(host_gradient, workspace.gradient, gradient_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    if (compute_moments == 1 &&
        cudaMemcpy(host_hessian, workspace.hessian, hessian_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    return 0;
}

extern "C" int crbstpp_cuda_cloglog_objective_resident(
    int device, std::int64_t matrix_token, const double* host_x,
    const double* host_beta, const double* host_noevent,
    const double* host_event, std::int64_t rows, std::int64_t columns,
    int compute_moments, double* host_nll, double* host_eta,
    double* host_gradient, double* host_hessian) {
    if (device < 0 || device >= static_cast<int>(workspaces.size()) ||
        rows < 1 || rows > INT32_MAX || columns < 1 || columns > INT32_MAX ||
        host_nll == nullptr)
        return 1;
    Workspace& workspace = workspaces[device];
    std::lock_guard<std::mutex> lock(workspace.mutex);
    if (cudaSetDevice(device) != cudaSuccess) return 1;
    const int threads = 256;
    const std::size_t x_bytes = sizeof(double) * rows * columns;
    const std::size_t row_bytes = sizeof(double) * rows;
    const std::size_t beta_bytes = sizeof(double) * columns;
    const std::size_t gradient_bytes = sizeof(double) * columns;
    const std::size_t hessian_bytes = sizeof(double) * columns * columns;
    const bool same_matrix =
        workspace.matrix_token == matrix_token &&
        workspace.matrix_rows == rows &&
        workspace.matrix_columns == columns &&
        workspace.x_capacity >= x_bytes;
    const bool same_likelihood =
        same_matrix && workspace.likelihood_token == matrix_token &&
        workspace.likelihood_mode == 2 &&
        workspace.exposure_capacity >= row_bytes &&
        workspace.event_capacity >= row_bytes;
    if (!reserve(&workspace.x, &workspace.x_capacity, x_bytes) ||
        !reserve(&workspace.beta, &workspace.beta_capacity, beta_bytes) ||
        !reserve(&workspace.eta, &workspace.eta_capacity, row_bytes) ||
        !reserve(&workspace.exposure, &workspace.exposure_capacity, row_bytes) ||
        !reserve(&workspace.event, &workspace.event_capacity, row_bytes) ||
        !reserve(&workspace.value, &workspace.value_capacity, row_bytes) ||
        !reserve(&workspace.ones, &workspace.ones_capacity, row_bytes) ||
        !reserve(&workspace.first, &workspace.first_capacity, row_bytes) ||
        !reserve(&workspace.second, &workspace.second_capacity, row_bytes) ||
        (compute_moments &&
         !reserve(&workspace.gradient, &workspace.gradient_capacity,
                  gradient_bytes)) ||
        (compute_moments == 1 &&
         (!reserve(&workspace.weighted, &workspace.weighted_capacity, x_bytes) ||
          !reserve(&workspace.hessian, &workspace.hessian_capacity,
                   hessian_bytes))))
        return 2;
    if (workspace.handle == nullptr &&
        cublasCreate(&workspace.handle) != CUBLAS_STATUS_SUCCESS)
        return 2;
    if ((!same_matrix &&
         cudaMemcpy(workspace.x, host_x, x_bytes, cudaMemcpyHostToDevice) !=
             cudaSuccess) ||
        (!same_likelihood &&
         (cudaMemcpy(workspace.exposure, host_noevent, row_bytes,
                     cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(workspace.event, host_event, row_bytes,
                     cudaMemcpyHostToDevice) != cudaSuccess)) ||
        cudaMemcpy(workspace.beta, host_beta, beta_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return 2;
    workspace.matrix_token = matrix_token;
    workspace.matrix_rows = rows;
    workspace.matrix_columns = columns;
    workspace.likelihood_token = matrix_token;
    workspace.likelihood_mode = 2;
    const double alpha = 1.0, zero = 0.0;
    if (cublasDgemv(
            workspace.handle, CUBLAS_OP_T, static_cast<int>(columns),
            static_cast<int>(rows), &alpha, workspace.x,
            static_cast<int>(columns), workspace.beta, 1, &zero,
            workspace.eta, 1) != CUBLAS_STATUS_SUCCESS)
        return 2;
    const int row_blocks = static_cast<int>((rows + threads - 1) / threads);
    cloglog_likelihood_kernel<<<row_blocks, threads>>>(
        workspace.eta, workspace.exposure, workspace.event, rows,
        workspace.value, workspace.first, workspace.second);
    fill_constant_kernel<<<row_blocks, threads>>>(workspace.ones, rows, 1.0);
    if (cudaGetLastError() != cudaSuccess) return 2;
    double nll = 0.0;
    if (cublasDdot(
            workspace.handle, static_cast<int>(rows), workspace.ones, 1,
            workspace.value, 1, &nll) != CUBLAS_STATUS_SUCCESS)
        return 2;
    if (compute_moments) {
        if (cublasDgemv(
                workspace.handle, CUBLAS_OP_N, static_cast<int>(columns),
                static_cast<int>(rows), &alpha, workspace.x,
                static_cast<int>(columns), workspace.first, 1, &zero,
                workspace.gradient, 1) != CUBLAS_STATUS_SUCCESS)
            return 2;
    }
    if (compute_moments == 1) {
        const std::int64_t count = rows * columns;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        weight_design_kernel<<<blocks, threads>>>(
            workspace.x, workspace.second, rows, columns,
            workspace.weighted);
        if (cudaGetLastError() != cudaSuccess ||
            cublasDgemm(
                workspace.handle, CUBLAS_OP_N, CUBLAS_OP_T,
                static_cast<int>(columns), static_cast<int>(columns),
                static_cast<int>(rows), &alpha, workspace.x,
                static_cast<int>(columns), workspace.weighted,
                static_cast<int>(columns), &zero, workspace.hessian,
                static_cast<int>(columns)) != CUBLAS_STATUS_SUCCESS)
            return 2;
    }
    *host_nll = nll;
    if (host_eta != nullptr &&
        cudaMemcpy(host_eta, workspace.eta, row_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    if (compute_moments &&
        cudaMemcpy(host_gradient, workspace.gradient, gradient_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    if (compute_moments == 1 &&
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
    workspace.matrix_token = -1;
    workspace.matrix_rows = -1;
    workspace.matrix_columns = -1;
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

extern "C" int crbstpp_cuda_implicit_moments_batch(
    int device,
    const std::int64_t* host_source_offsets,
    const std::int64_t* host_source_times,
    const std::int64_t* host_source_spans,
    int completion_mode,
    const std::int64_t* host_starts,
    const std::int64_t* host_ends,
    const std::int64_t* host_grid_offsets,
    const double* host_basis,
    std::int64_t source_token,
    std::int64_t predicate_count,
    std::int64_t entity_count,
    std::int64_t source_event_count,
    int knots,
    int lag,
    const int* host_block_predicates,
    const int* host_block_orders,
    const std::int64_t* host_block_minimum_spans,
    const std::int64_t* host_block_windows,
    const int* host_block_counts,
    const std::int64_t* host_candidate_entity_offsets,
    const int* host_candidate_entities,
    int candidates,
    int maximum_blocks,
    const double* host_first,
    const double* host_second,
    const unsigned char* host_event_by_row,
    const double* host_event_first_delta,
    const double* host_event_second_delta,
    int compact_derivative_mode,
    std::int64_t derivative_token,
    std::int64_t rows,
    const int* host_group_by_row,
    const double* host_current_x,
    std::int64_t current_groups,
    int current_dimension,
    const int* host_baseline_group_by_row,
    int baseline_groups,
    int baseline_dimension,
    const unsigned char* host_current_signed_state,
    const int* host_relaxation_group_by_current_group,
    int relaxation_groups,
    int collect_group_footprint_stats,
    const int* host_active_current_columns,
    int active_current_dimension,
    int gradient_only,
    int collect_footprint_stats,
    double* host_gradient,
    double* host_hessian,
    double* host_cross,
    double* host_footprint_stats,
    double* host_group_footprint_stats) {
    if (device < 0 || device >= static_cast<int>(implicit_workspaces.size()) ||
        predicate_count < 1 || entity_count < 1 || source_event_count < 0 ||
        knots < 1 || knots > 8 || lag < 1 || candidates < 1 ||
        maximum_blocks < 1 || maximum_blocks > 8 || rows < 1 ||
        current_groups < 1 || current_dimension < 1 ||
        baseline_groups < 1 || baseline_dimension < 1 ||
        baseline_dimension > current_dimension ||
        active_current_dimension < (gradient_only ? 0 : 1) ||
        active_current_dimension > current_dimension ||
        (completion_mode && host_source_spans == nullptr) ||
        completion_mode < 0 || completion_mode > 2 ||
        (compact_derivative_mode &&
         host_event_by_row == nullptr) ||
        (compact_derivative_mode == 2 &&
         (host_event_first_delta == nullptr ||
          host_event_second_delta == nullptr)) ||
        compact_derivative_mode < 0 || compact_derivative_mode > 2 ||
        gradient_only < 0 || gradient_only > 1 ||
        collect_footprint_stats < 0 || collect_footprint_stats > 1 ||
        collect_group_footprint_stats < 0 ||
        collect_group_footprint_stats > 1 ||
        (collect_footprint_stats &&
         (compact_derivative_mode == 0 ||
          host_baseline_group_by_row == nullptr ||
          host_current_signed_state == nullptr ||
          host_footprint_stats == nullptr)) ||
        (collect_group_footprint_stats &&
         (compact_derivative_mode == 0 || relaxation_groups < 1 ||
          host_relaxation_group_by_current_group == nullptr ||
          host_group_footprint_stats == nullptr)) ||
        host_grid_offsets[entity_count] != rows)
        return 1;
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(candidates) * maximum_blocks;
         ++index) {
        if (host_block_minimum_spans[index] >= host_block_windows[index])
            return 1;
    }
    for (int index = 0; index < active_current_dimension; ++index) {
        if (host_active_current_columns[index] < 0 ||
            host_active_current_columns[index] >= current_dimension)
            return 1;
    }
    if (host_candidate_entity_offsets[0] != 0)
        return 1;
    const std::int64_t candidate_entity_count =
        host_candidate_entity_offsets[candidates];
    if (candidate_entity_count < 0)
        return 1;
    for (int candidate = 0; candidate < candidates; ++candidate) {
        if (host_block_counts[candidate] < 1 ||
            host_block_counts[candidate] > maximum_blocks ||
            host_candidate_entity_offsets[candidate] >
                host_candidate_entity_offsets[candidate + 1])
            return 1;
    }
    for (std::int64_t index = 0; index < candidate_entity_count; ++index) {
        if (host_candidate_entities[index] < 0 ||
            host_candidate_entities[index] >= entity_count)
            return 1;
    }
    ImplicitWorkspace& workspace = implicit_workspaces[device];
    std::lock_guard<std::mutex> lock(workspace.mutex);
    if (cudaSetDevice(device) != cudaSuccess) return 1;

    const int dimension = maximum_blocks * knots;
    const std::size_t source_offset_width =
        completion_mode == 2 ? 2 : static_cast<std::size_t>(entity_count + 1);
    const std::size_t source_offsets_bytes = sizeof(std::int64_t) *
        predicate_count * source_offset_width;
    const std::size_t source_times_bytes =
        sizeof(std::int64_t) * source_event_count;
    const std::size_t source_spans_bytes =
        completion_mode
            ? sizeof(std::int64_t) * source_event_count
            : 0;
    const std::size_t entity_bytes = sizeof(std::int64_t) * entity_count;
    const std::size_t grid_offsets_bytes =
        sizeof(std::int64_t) * (entity_count + 1);
    const std::size_t basis_bytes = sizeof(double) * knots * lag;
    const std::size_t metadata_count =
        static_cast<std::size_t>(candidates) * maximum_blocks;
    const std::size_t predicate_bytes =
        sizeof(int) * metadata_count * kImplicitMaxSources;
    const std::size_t order_bytes = sizeof(int) * metadata_count;
    const std::size_t window_bytes = sizeof(std::int64_t) * metadata_count;
    const std::size_t count_bytes = sizeof(int) * candidates;
    const std::size_t candidate_entity_offset_bytes =
        sizeof(std::int64_t) * (static_cast<std::size_t>(candidates) + 1);
    const std::size_t candidate_entity_bytes =
        sizeof(int) * static_cast<std::size_t>(candidate_entity_count);
    const std::int64_t derivative_rows =
        compact_derivative_mode ? current_groups : rows;
    const std::size_t derivative_bytes =
        sizeof(double) * derivative_rows;
    const std::size_t event_bytes =
        compact_derivative_mode ? sizeof(unsigned char) * rows : 0;
    const std::size_t event_delta_bytes =
        compact_derivative_mode == 2
            ? sizeof(double) * current_groups
            : 0;
    const std::size_t groups_bytes = sizeof(int) * rows;
    const std::size_t footprint_bytes = collect_footprint_stats
        ? sizeof(double) * candidates * 4 * baseline_groups * 2
        : 0;
    const std::size_t group_footprint_bytes = collect_group_footprint_stats
        ? sizeof(double) * candidates * relaxation_groups * 2
        : 0;
    const std::size_t current_x_bytes =
        sizeof(double) * current_groups * current_dimension;
    const std::size_t current_signed_state_bytes = collect_footprint_stats
        ? sizeof(unsigned char) * current_groups
        : 0;
    const std::size_t relaxation_group_bytes = collect_group_footprint_stats
        ? sizeof(int) * current_groups
        : 0;
    const std::size_t active_current_columns_bytes =
        sizeof(int) * active_current_dimension;
    const std::size_t gradient_bytes =
        sizeof(double) * candidates * dimension;
    const bool direct_completion_gradient =
        gradient_only && completion_mode == 2 && maximum_blocks == 1 &&
        compact_derivative_mode != 0 && !collect_footprint_stats &&
        !collect_group_footprint_stats;
    const std::size_t future_score_bytes = direct_completion_gradient
        ? sizeof(double) * static_cast<std::size_t>(rows) * knots
        : 0;
    const std::size_t hessian_bytes = gradient_only
        ? 0
        : sizeof(double) * candidates * dimension * dimension;
    const std::size_t cross_bytes = gradient_only
        ? 0
        : sizeof(double) * candidates * active_current_dimension * dimension;

    const bool source_reallocation =
        workspace.source_offsets_capacity < source_offsets_bytes ||
        workspace.source_times_capacity < source_times_bytes ||
        (completion_mode &&
         workspace.source_spans_capacity < source_spans_bytes) ||
        workspace.starts_capacity < entity_bytes ||
        workspace.ends_capacity < entity_bytes ||
        workspace.grid_offsets_capacity < grid_offsets_bytes ||
        workspace.basis_capacity < basis_bytes;
    const bool derivative_reallocation =
        workspace.first_capacity < derivative_bytes ||
        workspace.second_capacity < derivative_bytes ||
        (compact_derivative_mode &&
         workspace.event_by_row_capacity < event_bytes) ||
        (compact_derivative_mode == 2 &&
         (workspace.event_first_delta_capacity < event_delta_bytes ||
          workspace.event_second_delta_capacity < event_delta_bytes));
    const bool current_reallocation =
        workspace.group_by_row_capacity < groups_bytes ||
        workspace.current_x_capacity < current_x_bytes;
    const bool future_score_reallocation =
        direct_completion_gradient &&
        workspace.future_score_capacity < future_score_bytes;
    const bool baseline_reallocation = collect_footprint_stats &&
        workspace.baseline_group_by_row_capacity < groups_bytes;
    if (source_reallocation) workspace.source_token = -1;
    if (derivative_reallocation) workspace.derivative_token = -1;
    if (current_reallocation) workspace.current_token = -1;
    if (future_score_reallocation) workspace.future_score_token = -1;
    if (baseline_reallocation) workspace.baseline_token = -1;
    if (!reserve_int64(&workspace.source_offsets,
                       &workspace.source_offsets_capacity,
                       source_offsets_bytes) ||
        !reserve_int64(&workspace.source_times,
                       &workspace.source_times_capacity,
                       source_times_bytes) ||
        (completion_mode &&
         !reserve_int64(&workspace.source_spans,
                        &workspace.source_spans_capacity,
                        source_spans_bytes)) ||
        !reserve_int64(&workspace.starts, &workspace.starts_capacity,
                       entity_bytes) ||
        !reserve_int64(&workspace.ends, &workspace.ends_capacity,
                       entity_bytes) ||
        !reserve_int64(&workspace.grid_offsets,
                       &workspace.grid_offsets_capacity,
                       grid_offsets_bytes) ||
        !reserve(&workspace.basis, &workspace.basis_capacity, basis_bytes) ||
        !reserve_int(&workspace.block_predicates,
                     &workspace.block_predicates_capacity, predicate_bytes) ||
        !reserve_int(&workspace.block_orders,
                     &workspace.block_orders_capacity, order_bytes) ||
        !reserve_int64(&workspace.block_minimum_spans,
                       &workspace.block_minimum_spans_capacity, window_bytes) ||
        !reserve_int64(&workspace.block_windows,
                       &workspace.block_windows_capacity, window_bytes) ||
        !reserve_int(&workspace.block_counts,
                     &workspace.block_counts_capacity, count_bytes) ||
        !reserve_int64(&workspace.candidate_entity_offsets,
                       &workspace.candidate_entity_offsets_capacity,
                       candidate_entity_offset_bytes) ||
        (candidate_entity_bytes &&
         !reserve_int(&workspace.candidate_entities,
                      &workspace.candidate_entities_capacity,
                      candidate_entity_bytes)) ||
        !reserve(&workspace.first, &workspace.first_capacity, derivative_bytes) ||
        !reserve(&workspace.second, &workspace.second_capacity, derivative_bytes) ||
        (compact_derivative_mode &&
         !reserve_byte(&workspace.event_by_row,
                       &workspace.event_by_row_capacity, event_bytes)) ||
        (compact_derivative_mode == 2 &&
         (!reserve(&workspace.event_first_delta,
                   &workspace.event_first_delta_capacity, event_delta_bytes) ||
          !reserve(&workspace.event_second_delta,
                   &workspace.event_second_delta_capacity, event_delta_bytes))) ||
        !reserve_int(&workspace.group_by_row,
                     &workspace.group_by_row_capacity, groups_bytes) ||
        (active_current_columns_bytes &&
         !reserve_int(&workspace.active_current_columns,
                      &workspace.active_current_columns_capacity,
                      active_current_columns_bytes)) ||
        !reserve(&workspace.current_x, &workspace.current_x_capacity,
                 current_x_bytes) ||
        (direct_completion_gradient &&
         !reserve(&workspace.future_score,
                  &workspace.future_score_capacity, future_score_bytes)) ||
        !reserve(&workspace.gradient, &workspace.gradient_capacity,
                 gradient_bytes) ||
        (!gradient_only &&
         !reserve(&workspace.hessian, &workspace.hessian_capacity,
                  hessian_bytes)) ||
        (!gradient_only &&
         !reserve(&workspace.cross, &workspace.cross_capacity, cross_bytes)))
        return 2;
    if (collect_footprint_stats &&
        !reserve_int(&workspace.baseline_group_by_row,
                     &workspace.baseline_group_by_row_capacity, groups_bytes))
        return 2;
    if (collect_footprint_stats &&
        !reserve_byte(&workspace.current_signed_state,
                      &workspace.current_signed_state_capacity,
                      current_signed_state_bytes))
        return 2;
    if (collect_footprint_stats &&
        !reserve(&workspace.footprint_stats,
                 &workspace.footprint_stats_capacity, footprint_bytes))
        return 2;
    if (collect_group_footprint_stats &&
        (!reserve_int(&workspace.relaxation_group_by_current_group,
                      &workspace.relaxation_group_by_current_group_capacity,
                      relaxation_group_bytes) ||
         !reserve(&workspace.group_footprint_stats,
                  &workspace.group_footprint_stats_capacity,
                  group_footprint_bytes)))
        return 2;
    // ``source_token`` identifies the immutable time/span values and shared
    // context, not the wave-local predicate offset table.  A relation-aware
    // dictionary may select disjoint slices of one global completion store in
    // each wave.  Retain the multi-GiB values once per GPU while copying only
    // the current offset rows below.  Predicate cardinality is intentionally
    // absent from this cache key.
    const bool reuse_source =
        source_token >= 0 && !source_reallocation &&
        workspace.source_token == source_token &&
        workspace.source_entities == entity_count &&
        workspace.source_events == source_event_count &&
        workspace.source_completion_mode == completion_mode;
    if (cudaMemcpy(workspace.source_offsets, host_source_offsets,
                   source_offsets_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        return 2;
    if (!reuse_source) {
        if ((source_times_bytes &&
             cudaMemcpy(workspace.source_times, host_source_times,
                        source_times_bytes, cudaMemcpyHostToDevice) != cudaSuccess) ||
            (source_spans_bytes &&
             cudaMemcpy(workspace.source_spans, host_source_spans,
                        source_spans_bytes, cudaMemcpyHostToDevice) != cudaSuccess) ||
            cudaMemcpy(workspace.starts, host_starts, entity_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(workspace.ends, host_ends, entity_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(workspace.grid_offsets, host_grid_offsets,
                       grid_offsets_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(workspace.basis, host_basis, basis_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return 2;
        workspace.source_token = source_token;
        workspace.source_entities = entity_count;
        workspace.source_predicates = predicate_count;
        workspace.source_events = source_event_count;
        workspace.source_completion_mode = completion_mode;
    }
    // The resident offset table now describes this wave even when its value
    // vectors were reused.  Objective evaluation immediately following the
    // moments call therefore sees the matching predicate count.
    workspace.source_predicates = predicate_count;
    const bool reuse_derivatives =
        derivative_token >= 0 && !derivative_reallocation &&
        workspace.derivative_token == derivative_token &&
        workspace.derivative_rows == derivative_rows &&
        workspace.derivative_compact_mode == compact_derivative_mode;
    if (!reuse_derivatives) {
        if (cudaMemcpy(workspace.first, host_first, derivative_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(workspace.second, host_second, derivative_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            (compact_derivative_mode &&
             cudaMemcpy(workspace.event_by_row, host_event_by_row,
                        event_bytes, cudaMemcpyHostToDevice) != cudaSuccess) ||
            (compact_derivative_mode == 2 &&
             (cudaMemcpy(workspace.event_first_delta, host_event_first_delta,
                         event_delta_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
              cudaMemcpy(workspace.event_second_delta, host_event_second_delta,
                         event_delta_bytes, cudaMemcpyHostToDevice) != cudaSuccess)))
            return 2;
        workspace.derivative_token = derivative_token;
        workspace.derivative_rows = derivative_rows;
        workspace.derivative_compact_mode = compact_derivative_mode;
    }
    const bool reuse_current =
        derivative_token >= 0 && !current_reallocation &&
        workspace.current_token == derivative_token &&
        workspace.current_rows == rows &&
        workspace.current_groups == current_groups &&
        workspace.current_dimension == current_dimension;
    if (!reuse_current) {
        if (cudaMemcpy(workspace.group_by_row, host_group_by_row, groups_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(workspace.current_x, host_current_x, current_x_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return 2;
        workspace.current_token = derivative_token;
        workspace.current_rows = rows;
        workspace.current_groups = current_groups;
        workspace.current_dimension = current_dimension;
    }
    const bool reuse_baseline = collect_footprint_stats &&
        derivative_token >= 0 && !baseline_reallocation &&
        workspace.baseline_token == derivative_token &&
        workspace.baseline_rows == rows &&
        workspace.baseline_groups == baseline_groups &&
        workspace.baseline_dimension == baseline_dimension;
    if (collect_footprint_stats && !reuse_baseline) {
        if (cudaMemcpy(workspace.baseline_group_by_row,
                       host_baseline_group_by_row, groups_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(workspace.current_signed_state,
                       host_current_signed_state, current_signed_state_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return 2;
        workspace.baseline_token = derivative_token;
        workspace.baseline_rows = rows;
        workspace.baseline_groups = baseline_groups;
        workspace.baseline_dimension = baseline_dimension;
    }
    if (collect_group_footprint_stats &&
        cudaMemcpy(workspace.relaxation_group_by_current_group,
                   host_relaxation_group_by_current_group,
                   relaxation_group_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return 2;
    if (cudaMemcpy(workspace.block_predicates, host_block_predicates,
                   predicate_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.block_orders, host_block_orders, order_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.block_minimum_spans, host_block_minimum_spans,
                   window_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.block_windows, host_block_windows, window_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.block_counts, host_block_counts, count_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.candidate_entity_offsets,
                   host_candidate_entity_offsets,
                   candidate_entity_offset_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        (candidate_entity_bytes &&
         cudaMemcpy(workspace.candidate_entities, host_candidate_entities,
                    candidate_entity_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess) ||
        (active_current_columns_bytes &&
         cudaMemcpy(workspace.active_current_columns,
                    host_active_current_columns,
                    active_current_columns_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess) ||
        cudaMemset(workspace.gradient, 0, gradient_bytes) != cudaSuccess ||
        (!gradient_only &&
         cudaMemset(workspace.hessian, 0, hessian_bytes) != cudaSuccess) ||
        (!gradient_only &&
         cudaMemset(workspace.cross, 0, cross_bytes) != cudaSuccess))
        return 2;
    if (collect_footprint_stats &&
        cudaMemset(workspace.footprint_stats, 0, footprint_bytes) != cudaSuccess)
        return 2;
    if (collect_group_footprint_stats &&
        cudaMemset(workspace.group_footprint_stats, 0,
                   group_footprint_bytes) != cudaSuccess)
        return 2;

    if (direct_completion_gradient) {
        const int threads = 256;
        const bool reuse_future_score =
            derivative_token >= 0 &&
            workspace.future_score_token == derivative_token &&
            workspace.future_score_rows == rows &&
            workspace.future_score_knots == knots &&
            workspace.future_score_lag == lag;
        if (!reuse_future_score) {
            future_basis_score_kernel<<<
                static_cast<int>(entity_count), threads>>>(
                workspace.starts, workspace.ends, workspace.grid_offsets,
                entity_count, workspace.basis, knots, lag, workspace.first,
                workspace.event_by_row, workspace.event_first_delta,
                compact_derivative_mode, workspace.group_by_row,
                workspace.future_score);
            if (cudaGetLastError() != cudaSuccess) return 2;
            workspace.future_score_token = derivative_token;
            workspace.future_score_rows = rows;
            workspace.future_score_knots = knots;
            workspace.future_score_lag = lag;
        }
        completion_preconvolved_gradient_kernel<<<
            candidates, threads, threads * sizeof(double)>>>(
            workspace.source_offsets, workspace.source_times,
            workspace.source_spans, workspace.starts, workspace.ends,
            workspace.grid_offsets, workspace.future_score, knots,
            workspace.block_predicates, workspace.block_minimum_spans,
            workspace.block_windows, candidates,
            workspace.gradient);
        if (cudaGetLastError() != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(host_gradient, workspace.gradient, gradient_bytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            return 2;
        return 0;
    }

    const int hessian_pairs = gradient_only
        ? 0
        : dimension * (dimension + 1) / 2;
    const std::size_t pair_bytes = sizeof(int) * hessian_pairs;
    if (!gradient_only &&
        (!reserve_int(&workspace.hessian_left,
                      &workspace.hessian_left_capacity, pair_bytes) ||
         !reserve_int(&workspace.hessian_right,
                      &workspace.hessian_right_capacity, pair_bytes)))
        return 2;
    if (!gradient_only && workspace.hessian_dimension != dimension) {
        std::vector<int> host_hessian_left(hessian_pairs);
        std::vector<int> host_hessian_right(hessian_pairs);
        int pair = 0;
        for (int left = 0; left < dimension; ++left) {
            for (int right = left; right < dimension; ++right) {
                host_hessian_left[pair] = left;
                host_hessian_right[pair] = right;
                ++pair;
            }
        }
        if (cudaMemcpy(workspace.hessian_left, host_hessian_left.data(), pair_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(workspace.hessian_right, host_hessian_right.data(), pair_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return 2;
        workspace.hessian_dimension = dimension;
    }
    int maximum_entity_rows = workspace.maximum_entity_rows;
    if (!reuse_source || maximum_entity_rows < 1) {
        maximum_entity_rows = 0;
        for (std::int64_t entity = 0; entity < entity_count; ++entity) {
            const std::int64_t length =
                host_ends[entity] - host_starts[entity] + 1;
            if (length < 1 || length > INT32_MAX) return 1;
            maximum_entity_rows = std::max(
                maximum_entity_rows, static_cast<int>(length));
        }
        workspace.maximum_entity_rows = maximum_entity_rows;
    }
    const int effective_current_dimension = gradient_only
        ? 0
        : active_current_dimension;
    const int output_dimension = dimension + hessian_pairs +
        effective_current_dimension * dimension;
    std::int64_t maximum_active_entities = 0;
    for (int candidate = 0; candidate < candidates; ++candidate)
        maximum_active_entities = std::max(
            maximum_active_entities,
            host_candidate_entity_offsets[candidate + 1] -
                host_candidate_entity_offsets[candidate]);
    int entities_per_chunk = 256;
    std::int64_t chunks_per_candidate =
        std::max<std::int64_t>(
            1,
            (maximum_active_entities + entities_per_chunk - 1) /
                entities_per_chunk);
    std::size_t partial_bytes = sizeof(double) * candidates *
        static_cast<std::size_t>(chunks_per_candidate) * output_dimension;
    constexpr std::size_t kPartialBudget = 768ULL * 1024 * 1024;
    while (partial_bytes > kPartialBudget &&
           entities_per_chunk < std::max<std::int64_t>(
               1, maximum_active_entities)) {
        entities_per_chunk *= 2;
        chunks_per_candidate =
            std::max<std::int64_t>(
                1,
                (maximum_active_entities + entities_per_chunk - 1) /
                    entities_per_chunk);
        partial_bytes = sizeof(double) * candidates *
            static_cast<std::size_t>(chunks_per_candidate) * output_dimension;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        return 2;
    const std::size_t shared_limit = std::max(
        static_cast<std::size_t>(properties.sharedMemPerBlock),
        static_cast<std::size_t>(properties.sharedMemPerBlockOptin));
    const std::size_t fixed_shared =
        sizeof(double) * static_cast<std::size_t>(output_dimension);
    const std::size_t per_row_shared =
        sizeof(double) * static_cast<std::size_t>(dimension) + sizeof(int);
    if (shared_limit <= fixed_shared + per_row_shared)
        return 2;
    const int maximum_tile_rows = std::max(
        1,
        std::min(
            maximum_entity_rows,
            static_cast<int>(
                (shared_limit - fixed_shared) / per_row_shared)));
    const std::size_t shared_bytes =
        fixed_shared +
        static_cast<std::size_t>(maximum_tile_rows) * per_row_shared;
    if (shared_bytes > static_cast<std::size_t>(properties.sharedMemPerBlock) &&
        cudaFuncSetAttribute(
            implicit_direct_partials_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)) != cudaSuccess)
        return 2;
    if (!reserve(&workspace.partials, &workspace.partials_capacity,
                 partial_bytes))
        return 2;
    const int threads = 256;
    implicit_direct_partials_kernel<<<
        static_cast<std::int64_t>(candidates) * chunks_per_candidate, threads,
        shared_bytes>>>(
        workspace.source_offsets, workspace.source_times,
        workspace.source_spans, completion_mode, entity_count,
        workspace.starts, workspace.ends, workspace.grid_offsets,
        workspace.basis, knots, lag, workspace.block_predicates,
        workspace.block_orders, workspace.block_minimum_spans,
        workspace.block_windows,
        workspace.block_counts, candidates, maximum_blocks,
        workspace.candidate_entity_offsets, workspace.candidate_entities,
        workspace.first,
        workspace.second, workspace.event_by_row,
        workspace.event_first_delta, workspace.event_second_delta,
        compact_derivative_mode,
        workspace.group_by_row, workspace.current_x, current_dimension,
        workspace.baseline_group_by_row, baseline_groups, baseline_dimension,
        workspace.current_signed_state,
        collect_footprint_stats ? workspace.footprint_stats : nullptr,
        collect_group_footprint_stats
            ? workspace.relaxation_group_by_current_group
            : nullptr,
        relaxation_groups,
        collect_group_footprint_stats
            ? workspace.group_footprint_stats
            : nullptr,
        workspace.active_current_columns, effective_current_dimension,
        workspace.hessian_left, workspace.hessian_right, hessian_pairs,
        maximum_tile_rows, entities_per_chunk, chunks_per_candidate,
        workspace.partials);
    implicit_reduce_partials_kernel<<<
        static_cast<std::int64_t>(candidates) * output_dimension, threads,
        threads * sizeof(double)>>>(
        workspace.partials, candidates, dimension, effective_current_dimension,
        workspace.hessian_left, workspace.hessian_right, hessian_pairs,
        chunks_per_candidate, workspace.gradient, workspace.hessian,
        workspace.cross);
    if (cudaGetLastError() != cudaSuccess) return 2;
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(host_gradient, workspace.gradient, gradient_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        (!gradient_only &&
         cudaMemcpy(host_hessian, workspace.hessian, hessian_bytes,
                    cudaMemcpyDeviceToHost) != cudaSuccess) ||
        (!gradient_only &&
         cudaMemcpy(host_cross, workspace.cross, cross_bytes,
                    cudaMemcpyDeviceToHost) != cudaSuccess) ||
        (collect_footprint_stats &&
         cudaMemcpy(host_footprint_stats, workspace.footprint_stats,
                    footprint_bytes, cudaMemcpyDeviceToHost) != cudaSuccess))
        return 2;
    if (collect_group_footprint_stats &&
        cudaMemcpy(host_group_footprint_stats,
                   workspace.group_footprint_stats,
                   group_footprint_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    return 0;
}

extern "C" int crbstpp_cuda_implicit_objective_batch(
    int device, std::int64_t source_token, std::int64_t derivative_token,
    std::int64_t entity_count, int knots, int lag,
    const int* host_block_predicates, const int* host_block_orders,
    const std::int64_t* host_block_minimum_spans,
    const std::int64_t* host_block_windows, const int* host_block_counts,
    const std::int64_t* host_candidate_entity_offsets,
    const int* host_candidate_entities,
    int candidates, int maximum_blocks, const double* host_coefficients,
    const double* host_group_eta, std::int64_t current_groups,
    int likelihood_mode,
    int maximum_entity_rows, double* host_objective_delta) {
    if (device < 0 || device >= static_cast<int>(implicit_workspaces.size()) ||
        entity_count < 1 || knots < 1 || knots > 8 || lag < 1 ||
        candidates < 1 || maximum_blocks < 1 || maximum_blocks > 8 ||
        likelihood_mode < 1 || likelihood_mode > 2 ||
        (likelihood_mode == 2 && host_group_eta == nullptr) ||
        maximum_entity_rows < 1)
        return 1;
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(candidates) * maximum_blocks;
         ++index) {
        if (host_block_minimum_spans[index] >= host_block_windows[index])
            return 1;
    }
    if (host_candidate_entity_offsets[0] != 0)
        return 1;
    const std::int64_t candidate_entity_count =
        host_candidate_entity_offsets[candidates];
    if (candidate_entity_count < 0)
        return 1;
    std::int64_t maximum_candidate_entities = 0;
    for (int candidate = 0; candidate < candidates; ++candidate) {
        const std::int64_t begin =
            host_candidate_entity_offsets[candidate];
        const std::int64_t end =
            host_candidate_entity_offsets[candidate + 1];
        if (begin > end)
            return 1;
        maximum_candidate_entities =
            std::max(maximum_candidate_entities, end - begin);
    }
    for (std::int64_t index = 0; index < candidate_entity_count; ++index) {
        if (host_candidate_entities[index] < 0 ||
            host_candidate_entities[index] >= entity_count)
            return 1;
    }
    ImplicitWorkspace& workspace = implicit_workspaces[device];
    std::lock_guard<std::mutex> lock(workspace.mutex);
    if (cudaSetDevice(device) != cudaSuccess ||
        workspace.source_token != source_token ||
        workspace.source_entities != entity_count ||
        workspace.derivative_token != derivative_token ||
        workspace.derivative_compact_mode != likelihood_mode ||
        workspace.current_token != derivative_token ||
        workspace.current_groups != current_groups ||
        workspace.event_by_row == nullptr || workspace.group_by_row == nullptr)
        return 1;
    const int dimension = maximum_blocks * knots;
    const std::size_t metadata_count =
        static_cast<std::size_t>(candidates) * maximum_blocks;
    const std::size_t predicate_bytes =
        sizeof(int) * metadata_count * kImplicitMaxSources;
    const std::size_t order_bytes = sizeof(int) * metadata_count;
    const std::size_t window_bytes = sizeof(std::int64_t) * metadata_count;
    const std::size_t count_bytes = sizeof(int) * candidates;
    const std::size_t candidate_entity_offset_bytes =
        sizeof(std::int64_t) * (static_cast<std::size_t>(candidates) + 1);
    const std::size_t candidate_entity_bytes =
        sizeof(int) * static_cast<std::size_t>(candidate_entity_count);
    const std::size_t coefficient_bytes =
        sizeof(double) * candidates * dimension;
    const std::size_t group_eta_bytes =
        likelihood_mode == 2 ? sizeof(double) * current_groups : 0;
    const std::size_t output_bytes = sizeof(double) * candidates;
    if (!reserve_int(&workspace.block_predicates,
                     &workspace.block_predicates_capacity, predicate_bytes) ||
        !reserve_int(&workspace.block_orders,
                     &workspace.block_orders_capacity, order_bytes) ||
        !reserve_int64(&workspace.block_minimum_spans,
                       &workspace.block_minimum_spans_capacity, window_bytes) ||
        !reserve_int64(&workspace.block_windows,
                       &workspace.block_windows_capacity, window_bytes) ||
        !reserve_int(&workspace.block_counts,
                     &workspace.block_counts_capacity, count_bytes) ||
        !reserve_int64(&workspace.candidate_entity_offsets,
                       &workspace.candidate_entity_offsets_capacity,
                       candidate_entity_offset_bytes) ||
        (candidate_entity_bytes &&
         !reserve_int(&workspace.candidate_entities,
                      &workspace.candidate_entities_capacity,
                      candidate_entity_bytes)) ||
        !reserve(&workspace.hessian, &workspace.hessian_capacity,
                 coefficient_bytes) ||
        (likelihood_mode == 2 &&
         !reserve(&workspace.group_eta, &workspace.group_eta_capacity,
                  group_eta_bytes)) ||
        !reserve(&workspace.gradient, &workspace.gradient_capacity, output_bytes))
        return 2;
    if (cudaMemcpy(workspace.block_predicates, host_block_predicates,
                   predicate_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.block_orders, host_block_orders, order_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.block_minimum_spans, host_block_minimum_spans,
                   window_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.block_windows, host_block_windows, window_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.block_counts, host_block_counts, count_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace.candidate_entity_offsets,
                   host_candidate_entity_offsets,
                   candidate_entity_offset_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        (candidate_entity_bytes &&
         cudaMemcpy(workspace.candidate_entities, host_candidate_entities,
                    candidate_entity_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess) ||
        (likelihood_mode == 2 &&
         (workspace.objective_token != derivative_token ||
          workspace.objective_mode != likelihood_mode) &&
         cudaMemcpy(workspace.group_eta, host_group_eta, group_eta_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess) ||
        cudaMemcpy(workspace.hessian, host_coefficients, coefficient_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return 2;
    workspace.objective_token = derivative_token;
    workspace.objective_mode = likelihood_mode;
    const int threads = 256;
    const std::int64_t chunks_per_candidate =
        std::max<std::int64_t>(1, (maximum_candidate_entities + 255) / 256);
    const std::size_t partial_bytes =
        sizeof(double) * candidates * chunks_per_candidate;
    if (!reserve(&workspace.partials, &workspace.partials_capacity,
                 partial_bytes))
        return 2;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        return 2;
    const std::size_t shared_limit = std::max(
        static_cast<std::size_t>(properties.sharedMemPerBlock),
        static_cast<std::size_t>(properties.sharedMemPerBlockOptin));
    const std::size_t reduction_bytes = sizeof(double) * threads;
    const std::size_t per_row_shared =
        sizeof(double) * static_cast<std::size_t>(dimension);
    if (shared_limit <= reduction_bytes + per_row_shared)
        return 2;
    const int maximum_tile_rows = std::max(
        1,
        std::min(
            maximum_entity_rows,
            static_cast<int>(
                (shared_limit - reduction_bytes) / per_row_shared)));
    const std::size_t shared_bytes =
        reduction_bytes +
        static_cast<std::size_t>(maximum_tile_rows) * per_row_shared;
    if (shared_bytes > static_cast<std::size_t>(properties.sharedMemPerBlock) &&
        cudaFuncSetAttribute(
            implicit_objective_partials_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)) != cudaSuccess)
        return 2;
    implicit_objective_partials_kernel<<<
        static_cast<std::int64_t>(candidates) * chunks_per_candidate, threads,
        shared_bytes>>>(
        workspace.source_offsets, workspace.source_times,
        workspace.source_spans, workspace.source_completion_mode,
        entity_count,
        workspace.starts, workspace.ends, workspace.grid_offsets,
        workspace.basis, knots, lag, workspace.block_predicates,
        workspace.block_orders, workspace.block_minimum_spans,
        workspace.block_windows,
        workspace.block_counts, candidates, maximum_blocks,
        workspace.candidate_entity_offsets, workspace.candidate_entities,
        workspace.hessian, workspace.first, workspace.group_eta,
        likelihood_mode, workspace.event_by_row, workspace.group_by_row,
        maximum_tile_rows,
        256, chunks_per_candidate, workspace.partials);
    reduce_candidate_partials_kernel<<<
        candidates, threads, threads * sizeof(double)>>>(
        workspace.partials, candidates, chunks_per_candidate,
        workspace.gradient);
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(host_objective_delta, workspace.gradient, output_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    return 0;
}

int sparse_moments_batch_impl(
    int device, const std::int64_t* host_rows, const double* host_values,
    const double* host_first, const double* host_second,
    const std::int64_t* host_block_offsets, std::int64_t total_rows,
    std::int64_t derivative_rows, int indexed_derivatives,
    std::int64_t derivative_token, std::int64_t geometry_token,
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
    const bool geometry_reallocation =
        workspace.rows_capacity < rows_bytes ||
        workspace.values_capacity < values_bytes ||
        workspace.block_offsets_capacity < offsets_bytes;
    if (geometry_reallocation) workspace.geometry_token = -1;
    const bool derivative_reallocation =
        workspace.first_capacity < derivative_bytes ||
        workspace.second_capacity < derivative_bytes;
    if (derivative_reallocation) {
        workspace.derivative_token = -1;
        workspace.derivative_rows = -1;
    }
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
    const bool reuse_derivatives =
        indexed_derivatives && derivative_token >= 0 &&
        !derivative_reallocation &&
        workspace.derivative_token == derivative_token &&
        workspace.derivative_rows == derivative_rows;
    const bool reuse_geometry =
        indexed_derivatives && geometry_token >= 0 &&
        !geometry_reallocation &&
        workspace.geometry_token == geometry_token &&
        workspace.geometry_total_rows == total_rows &&
        workspace.geometry_candidates == candidates &&
        workspace.geometry_blocks == blocks_per_candidate &&
        workspace.geometry_knots == knots;
    if ((!reuse_geometry && rows_bytes &&
         cudaMemcpy(workspace.rows, host_rows, rows_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess) ||
        (!reuse_geometry && values_bytes &&
         cudaMemcpy(workspace.values, host_values, values_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess) ||
        (!reuse_derivatives && derivative_bytes &&
         cudaMemcpy(workspace.first, host_first, derivative_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess) ||
        (!reuse_derivatives && derivative_bytes &&
         cudaMemcpy(workspace.second, host_second, derivative_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess) ||
        (!reuse_geometry &&
         cudaMemcpy(workspace.block_offsets, host_block_offsets, offsets_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess))
        return 2;
    if (!reuse_geometry && indexed_derivatives && geometry_token >= 0) {
        workspace.geometry_token = geometry_token;
        workspace.geometry_total_rows = total_rows;
        workspace.geometry_candidates = candidates;
        workspace.geometry_blocks = blocks_per_candidate;
        workspace.geometry_knots = knots;
    } else if (!indexed_derivatives || geometry_token < 0) {
        workspace.geometry_token = -1;
    }
    if (!reuse_derivatives && indexed_derivatives && derivative_token >= 0) {
        workspace.derivative_token = derivative_token;
        workspace.derivative_rows = derivative_rows;
    } else if (!indexed_derivatives || derivative_token < 0) {
        workspace.derivative_token = -1;
        workspace.derivative_rows = -1;
    }
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
        host_block_offsets, total_rows, total_rows, 0, -1, -1, candidates,
        blocks_per_candidate, knots, host_gradient, host_hessian, host_cross);
}

extern "C" int crbstpp_cuda_sparse_moments_indexed_batch(
    int device, const std::int64_t* host_rows, const double* host_values,
    const double* host_first, const double* host_second,
    const std::int64_t* host_block_offsets, std::int64_t total_rows,
    std::int64_t derivative_rows, std::int64_t derivative_token,
    std::int64_t candidates,
    std::int64_t blocks_per_candidate, std::int64_t knots,
    double* host_gradient, double* host_hessian, double* host_cross) {
    return sparse_moments_batch_impl(
        device, host_rows, host_values, host_first, host_second,
        host_block_offsets, total_rows, derivative_rows, 1, derivative_token, -1,
        candidates,
        blocks_per_candidate, knots, host_gradient, host_hessian, host_cross);
}

extern "C" int crbstpp_cuda_sparse_moments_indexed_resident(
    int device, const std::int64_t* host_rows, const double* host_values,
    const double* host_first, const double* host_second,
    const std::int64_t* host_block_offsets, std::int64_t total_rows,
    std::int64_t derivative_rows, std::int64_t derivative_token,
    std::int64_t geometry_token, std::int64_t candidates,
    std::int64_t blocks_per_candidate, std::int64_t knots,
    double* host_gradient, double* host_hessian, double* host_cross) {
    return sparse_moments_batch_impl(
        device, host_rows, host_values, host_first, host_second,
        host_block_offsets, total_rows, derivative_rows, 1, derivative_token,
        geometry_token, candidates, blocks_per_candidate, knots,
        host_gradient, host_hessian, host_cross);
}

extern "C" int crbstpp_cuda_release_workspace(int device) {
    if (device < 0 || device >= static_cast<int>(workspaces.size())) return 1;
    Workspace& dense = workspaces[device];
    SparseWorkspace& sparse = sparse_workspaces[device];
    ImplicitWorkspace& implicit = implicit_workspaces[device];
    std::unique_lock<std::mutex> dense_lock(dense.mutex, std::defer_lock);
    std::unique_lock<std::mutex> sparse_lock(sparse.mutex, std::defer_lock);
    std::unique_lock<std::mutex> implicit_lock(implicit.mutex, std::defer_lock);
    std::lock(dense_lock, sparse_lock, implicit_lock);
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
        return 2;

    release_buffer(&dense.x, &dense.x_capacity);
    release_buffer(&dense.weighted, &dense.weighted_capacity);
    release_buffer(&dense.projected, &dense.projected_capacity);
    release_buffer(
        &dense.projected_columns, &dense.projected_columns_capacity);
    release_buffer(&dense.projected_scales, &dense.projected_scales_capacity);
    release_buffer(&dense.first, &dense.first_capacity);
    release_buffer(&dense.second, &dense.second_capacity);
    release_buffer(&dense.gradient, &dense.gradient_capacity);
    release_buffer(&dense.cross, &dense.cross_capacity);
    release_buffer(&dense.hessian, &dense.hessian_capacity);
    release_buffer(&dense.beta, &dense.beta_capacity);
    release_buffer(&dense.eta, &dense.eta_capacity);
    release_buffer(&dense.exposure, &dense.exposure_capacity);
    release_buffer(&dense.event, &dense.event_capacity);
    release_buffer(&dense.value, &dense.value_capacity);
    release_buffer(&dense.ones, &dense.ones_capacity);
    dense.matrix_token = -1;
    dense.matrix_rows = -1;
    dense.matrix_columns = -1;
    dense.likelihood_token = -1;
    dense.likelihood_mode = 0;
    dense.projection_token = -1;
    dense.projection_columns_count = -1;
    if (dense.handle != nullptr) {
        cublasDestroy(dense.handle);
        dense.handle = nullptr;
    }

    release_buffer(&sparse.rows, &sparse.rows_capacity);
    release_buffer(&sparse.block_offsets, &sparse.block_offsets_capacity);
    release_buffer(&sparse.values, &sparse.values_capacity);
    release_buffer(&sparse.first, &sparse.first_capacity);
    release_buffer(&sparse.second, &sparse.second_capacity);
    release_buffer(&sparse.gradient, &sparse.gradient_capacity);
    release_buffer(&sparse.cross, &sparse.cross_capacity);
    release_buffer(&sparse.hessian, &sparse.hessian_capacity);
    sparse.derivative_token = -1;
    sparse.derivative_rows = -1;
    sparse.geometry_token = -1;
    sparse.geometry_total_rows = -1;
    sparse.geometry_candidates = -1;
    sparse.geometry_blocks = -1;
    sparse.geometry_knots = -1;

    release_buffer(&implicit.source_offsets, &implicit.source_offsets_capacity);
    release_buffer(&implicit.source_times, &implicit.source_times_capacity);
    release_buffer(&implicit.source_spans, &implicit.source_spans_capacity);
    release_buffer(&implicit.starts, &implicit.starts_capacity);
    release_buffer(&implicit.ends, &implicit.ends_capacity);
    release_buffer(&implicit.grid_offsets, &implicit.grid_offsets_capacity);
    release_buffer(&implicit.basis, &implicit.basis_capacity);
    release_buffer(&implicit.block_predicates, &implicit.block_predicates_capacity);
    release_buffer(&implicit.block_orders, &implicit.block_orders_capacity);
    release_buffer(&implicit.block_minimum_spans,
                   &implicit.block_minimum_spans_capacity);
    release_buffer(&implicit.block_windows, &implicit.block_windows_capacity);
    release_buffer(&implicit.block_counts, &implicit.block_counts_capacity);
    release_buffer(&implicit.candidate_entity_offsets,
                   &implicit.candidate_entity_offsets_capacity);
    release_buffer(&implicit.candidate_entities,
                   &implicit.candidate_entities_capacity);
    release_buffer(&implicit.first, &implicit.first_capacity);
    release_buffer(&implicit.second, &implicit.second_capacity);
    release_buffer(&implicit.event_by_row, &implicit.event_by_row_capacity);
    release_buffer(
        &implicit.event_first_delta, &implicit.event_first_delta_capacity);
    release_buffer(
        &implicit.event_second_delta, &implicit.event_second_delta_capacity);
    release_buffer(&implicit.group_eta, &implicit.group_eta_capacity);
    release_buffer(&implicit.group_by_row, &implicit.group_by_row_capacity);
    release_buffer(&implicit.baseline_group_by_row,
                   &implicit.baseline_group_by_row_capacity);
    release_buffer(&implicit.current_signed_state,
                   &implicit.current_signed_state_capacity);
    release_buffer(&implicit.relaxation_group_by_current_group,
                   &implicit.relaxation_group_by_current_group_capacity);
    release_buffer(&implicit.hessian_left, &implicit.hessian_left_capacity);
    release_buffer(&implicit.hessian_right, &implicit.hessian_right_capacity);
    release_buffer(&implicit.active_current_columns,
                   &implicit.active_current_columns_capacity);
    release_buffer(&implicit.current_x, &implicit.current_x_capacity);
    release_buffer(&implicit.future_score, &implicit.future_score_capacity);
    release_buffer(&implicit.gradient, &implicit.gradient_capacity);
    release_buffer(&implicit.hessian, &implicit.hessian_capacity);
    release_buffer(&implicit.cross, &implicit.cross_capacity);
    release_buffer(&implicit.partials, &implicit.partials_capacity);
    release_buffer(&implicit.footprint_stats,
                   &implicit.footprint_stats_capacity);
    release_buffer(&implicit.group_footprint_stats,
                   &implicit.group_footprint_stats_capacity);
    implicit.source_token = -1;
    implicit.source_entities = -1;
    implicit.source_predicates = -1;
    implicit.source_events = -1;
    implicit.source_completion_mode = 0;
    implicit.maximum_entity_rows = -1;
    implicit.derivative_token = -1;
    implicit.derivative_rows = -1;
    implicit.derivative_compact_mode = 0;
    implicit.objective_token = -1;
    implicit.objective_mode = 0;
    implicit.current_token = -1;
    implicit.current_rows = -1;
    implicit.current_groups = -1;
    implicit.current_dimension = -1;
    implicit.future_score_token = -1;
    implicit.future_score_rows = -1;
    implicit.future_score_knots = -1;
    implicit.future_score_lag = -1;
    implicit.baseline_token = -1;
    implicit.baseline_rows = -1;
    implicit.baseline_groups = -1;
    implicit.baseline_dimension = -1;
    implicit.hessian_dimension = -1;
    return 0;
}

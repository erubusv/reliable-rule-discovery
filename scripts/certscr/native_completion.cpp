#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <queue>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <numeric>
#include <vector>

namespace {
std::uint64_t mix_word(std::uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value;
}

std::uint64_t hash_row(const float* values, const std::int64_t width) {
  // Hash two float32 columns per word with one multiply per 64-bit word.
  // The previous nested avalanche used three multiplies plus several rotates
  // for every pair and dominated wide triplet grouping. Equality is still
  // decided by the collision-chain memcmp below, so a hash change cannot merge
  // unequal rows or alter a sufficient statistic.
  std::uint64_t hash = 1469598103934665603ULL ^
                       static_cast<std::uint64_t>(width);
  constexpr std::uint64_t prime = 1099511628211ULL;
  std::int64_t column = 0;
  for (; column + 1 < width; column += 2) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, values + column, sizeof(bits));
    hash ^= bits;
    hash *= prime;
  }
  if (column < width) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, values + column, sizeof(bits));
    hash ^= static_cast<std::uint64_t>(bits);
    hash *= prime;
  }
  return mix_word(hash);
}

class FlatRowIndex {
 public:
  explicit FlatRowIndex(const std::int64_t expected_rows,
                        const std::int64_t initial_expected_rows = -1) {
    max_rows_ = expected_rows;
    const std::int64_t initial =
        initial_expected_rows < 0
            ? expected_rows
            : std::min(expected_rows, initial_expected_rows);
    std::size_t required = static_cast<std::size_t>(
        std::max<std::int64_t>(2, initial + initial / 2 + 1));
    std::size_t capacity = 2;
    while (capacity < required) {
      capacity <<= 1;
    }
    mask_ = capacity - 1;
    groups_.assign(capacity, -1);
    hashes_.resize(capacity);
  }

  std::int64_t find(const std::uint64_t hash,
                    const float* row,
                    const float* grouped_rows,
                    const std::int64_t columns,
                    std::size_t* empty_slot) {
    if ((size_ + 1) * 10 >= groups_.size() * 7 &&
        size_ < static_cast<std::size_t>(max_rows_)) {
      grow();
    }
    const std::size_t row_bytes =
        static_cast<std::size_t>(columns) * sizeof(float);
    std::size_t slot = static_cast<std::size_t>(hash) & mask_;
    while (groups_[slot] >= 0) {
      const std::int32_t group = groups_[slot];
      if (hashes_[slot] == hash &&
          std::memcmp(row,
                      grouped_rows + static_cast<std::int64_t>(group) * columns,
                      row_bytes) == 0) {
        return static_cast<std::int64_t>(group);
      }
      slot = (slot + 1) & mask_;
    }
    *empty_slot = slot;
    return -1;
  }

  void insert(const std::size_t slot,
              const std::uint64_t hash,
              const std::int64_t group) {
    groups_[slot] = static_cast<std::int32_t>(group);
    hashes_[slot] = hash;
    ++size_;
  }

 private:
  void grow() {
    const std::size_t capacity = groups_.size() * 2;
    std::vector<std::int32_t> next_groups(capacity, -1);
    std::vector<std::uint64_t> next_hashes(capacity);
    const std::size_t next_mask = capacity - 1;
    for (std::size_t old = 0; old < groups_.size(); ++old) {
      if (groups_[old] < 0) {
        continue;
      }
      std::size_t slot = static_cast<std::size_t>(hashes_[old]) & next_mask;
      while (next_groups[slot] >= 0) {
        slot = (slot + 1) & next_mask;
      }
      next_groups[slot] = groups_[old];
      next_hashes[slot] = hashes_[old];
    }
    groups_.swap(next_groups);
    hashes_.swap(next_hashes);
    mask_ = next_mask;
  }

  std::int64_t max_rows_ = 0;
  std::size_t mask_ = 1;
  std::size_t size_ = 0;
  std::vector<std::int32_t> groups_;
  std::vector<std::uint64_t> hashes_;
};
}  // namespace

extern "C" std::int64_t certscr_linear_completions(
    const std::int32_t q,
    const std::int32_t* const* sequences,
    const std::int32_t* const* times,
    const std::int64_t* lengths,
    const std::int32_t* sequence_lookup,
    const std::int64_t n_sequences,
    const std::int32_t max_span,
    std::int32_t* output_sequences,
    std::int32_t* output_times,
    std::int32_t* output_spans,
    const std::int64_t capacity) {
  if (q < 1 || q > 3 || sequences == nullptr || times == nullptr ||
      lengths == nullptr || sequence_lookup == nullptr || capacity < 0) {
    return -1;
  }
  std::int64_t position[3] = {0, 0, 0};
  std::int64_t written = 0;
  constexpr std::int32_t missing = std::numeric_limits<std::int32_t>::min();

  while (true) {
    std::int32_t sequence = std::numeric_limits<std::int32_t>::max();
    bool available = false;
    for (std::int32_t source = 0; source < q; ++source) {
      if (position[source] < lengths[source]) {
        sequence = std::min(sequence, sequences[source][position[source]]);
        available = true;
      }
    }
    if (!available) {
      break;
    }
    if (sequence < 0 || static_cast<std::int64_t>(sequence) >= n_sequences) {
      return -2;
    }
    if (sequence_lookup[sequence] < 0) {
      for (std::int32_t source = 0; source < q; ++source) {
        while (position[source] < lengths[source] &&
               sequences[source][position[source]] == sequence) {
          ++position[source];
        }
      }
      continue;
    }

    std::int32_t latest[3] = {missing, missing, missing};
    while (true) {
      std::int32_t now = std::numeric_limits<std::int32_t>::max();
      bool sequence_available = false;
      for (std::int32_t source = 0; source < q; ++source) {
        if (position[source] < lengths[source] &&
            sequences[source][position[source]] == sequence) {
          now = std::min(now, times[source][position[source]]);
          sequence_available = true;
        }
      }
      if (!sequence_available) {
        break;
      }
      for (std::int32_t source = 0; source < q; ++source) {
        while (position[source] < lengths[source] &&
               sequences[source][position[source]] == sequence &&
               times[source][position[source]] == now) {
          latest[source] = now;
          ++position[source];
        }
      }
      bool complete = true;
      std::int32_t earliest = latest[0];
      std::int32_t latest_time = latest[0];
      for (std::int32_t source = 0; source < q; ++source) {
        if (latest[source] == missing) {
          complete = false;
          break;
        }
        earliest = std::min(earliest, latest[source]);
        latest_time = std::max(latest_time, latest[source]);
      }
      if (!complete) {
        continue;
      }
      const std::int32_t span = latest_time - earliest;
      if (max_span >= 0 && span > max_span) {
        continue;
      }
      if (written >= capacity) {
        return -3;
      }
      output_sequences[written] = sequence;
      output_times[written] = now;
      output_spans[written] = span;
      ++written;
    }
  }
  return written;
}

extern "C" std::int64_t certscr_batched_sparse_rule_moments(
    const float* grid_values,
    const double* grid_weights_by_state,
    const std::int64_t* grid_weight_positions,
    const std::int64_t grid_weight_rows,
    const std::int64_t grid_rows,
    const std::int32_t states,
    const std::int32_t width,
    const float* event_values,
    const double* event_first_by_state,
    const double* event_second_by_state,
    const std::int64_t event_rows,
    const std::int32_t include_event_second,
    const std::int32_t worker_count,
    double* output_gradient,
    double* output_information) {
  if (grid_rows < 0 || grid_weight_rows < grid_rows || event_rows < 0 ||
      states < 1 || width < 1 ||
      worker_count < 1 || grid_values == nullptr ||
      grid_weights_by_state == nullptr ||
      output_gradient == nullptr || output_information == nullptr ||
      (event_rows > 0 &&
       (event_values == nullptr || event_first_by_state == nullptr ||
        event_second_by_state == nullptr))) {
    return -1;
  }
  const std::size_t state_count = static_cast<std::size_t>(states);
  const std::size_t column_count = static_cast<std::size_t>(width);
  const std::size_t gradient_size = column_count * state_count;
  const std::size_t information_size =
      column_count * column_count * state_count;
  std::fill(output_gradient, output_gradient + gradient_size, 0.0);
  std::fill(output_information,
            output_information + information_size,
            0.0);

  // States are independent. Each state preserves chronological row order, so
  // parallel execution changes no floating-point accumulation within a score.
  // State-major weights also avoid the strided reads of a row-major residual
  // matrix. Small moment accumulators remain hot in each worker's L1 cache.
  auto compute_state = [&](const std::int32_t state) {
    std::vector<double> gradient(column_count, 0.0);
    std::vector<double> information(column_count * column_count, 0.0);
    const double* grid_weights =
        grid_weights_by_state +
        static_cast<std::int64_t>(state) * grid_weight_rows;
    for (std::int64_t row = 0; row < grid_rows; ++row) {
      const float* values = grid_values + row * width;
      const std::int64_t weight_row =
          grid_weight_positions == nullptr ? row : grid_weight_positions[row];
      if (weight_row < 0 || weight_row >= grid_weight_rows) {
        return;
      }
      const double weight = grid_weights[weight_row];
      for (std::int32_t left = 0; left < width; ++left) {
        const double left_value = static_cast<double>(values[left]);
        gradient[static_cast<std::size_t>(left)] += left_value * weight;
        for (std::int32_t right = 0; right <= left; ++right) {
          information[static_cast<std::size_t>(left) * width + right] +=
              left_value * static_cast<double>(values[right]) * weight;
        }
      }
    }
    const double* event_first =
        event_first_by_state + static_cast<std::int64_t>(state) * event_rows;
    const double* event_second =
        event_second_by_state + static_cast<std::int64_t>(state) * event_rows;
    for (std::int64_t row = 0; row < event_rows; ++row) {
      const float* values = event_values + row * width;
      for (std::int32_t left = 0; left < width; ++left) {
        const double left_value = static_cast<double>(values[left]);
        gradient[static_cast<std::size_t>(left)] +=
            left_value * event_first[row];
        if (include_event_second != 0) {
          for (std::int32_t right = 0; right <= left; ++right) {
            information[static_cast<std::size_t>(left) * width + right] +=
                left_value * static_cast<double>(values[right]) *
                event_second[row];
          }
        }
      }
    }
    for (std::int32_t left = 0; left < width; ++left) {
      output_gradient[static_cast<std::size_t>(left) * states + state] =
          gradient[static_cast<std::size_t>(left)];
      for (std::int32_t right = 0; right <= left; ++right) {
        output_information[
            (static_cast<std::size_t>(left) * width + right) * states + state] =
            information[static_cast<std::size_t>(left) * width + right];
      }
    }
  };
  const std::int32_t workers = std::min(states, worker_count);
  if (workers <= 1 || grid_rows + event_rows < 16384) {
    for (std::int32_t state = 0; state < states; ++state) {
      compute_state(state);
    }
  } else {
    std::atomic<std::int32_t> next_state{0};
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers));
    for (std::int32_t worker = 0; worker < workers; ++worker) {
      threads.emplace_back([&]() {
        while (true) {
          const std::int32_t state = next_state.fetch_add(1);
          if (state >= states) {
            break;
          }
          compute_state(state);
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  for (std::int32_t left = 0; left < width; ++left) {
    for (std::int32_t right = 0; right < left; ++right) {
      for (std::int32_t state = 0; state < states; ++state) {
        output_information[
            (static_cast<std::size_t>(right) * width + left) * states +
            state] = output_information[
            (static_cast<std::size_t>(left) * width + right) * states +
            state];
      }
    }
  }
  return grid_rows + event_rows;
}

extern "C" std::int64_t certscr_sorted_unique_int64_union(
    const std::int64_t* const* arrays,
    const std::int64_t* lengths,
    const std::int32_t array_count,
    std::int64_t* output,
    const std::int64_t capacity) {
  if (arrays == nullptr || lengths == nullptr || output == nullptr ||
      array_count < 0 || capacity < 0) {
    return -1;
  }
  std::vector<std::int64_t> positions(static_cast<std::size_t>(array_count), 0);
  bool has_values = false;
  std::int64_t minimum_value = std::numeric_limits<std::int64_t>::max();
  std::int64_t maximum_value = std::numeric_limits<std::int64_t>::min();
  for (std::int32_t index = 0; index < array_count; ++index) {
    if (lengths[index] < 0 || (lengths[index] > 0 && arrays[index] == nullptr)) {
      return -2;
    }
    for (std::int64_t position = 1; position < lengths[index]; ++position) {
      if (arrays[index][position] <= arrays[index][position - 1]) {
        return -4;
      }
    }
    if (lengths[index] > 0) {
      has_values = true;
      minimum_value = std::min(minimum_value, arrays[index][0]);
      maximum_value = std::max(
          maximum_value, arrays[index][lengths[index] - 1]);
    }
  }
  if (!has_values) {
    return 0;
  }
  const __int128 span128 =
      static_cast<__int128>(maximum_value) -
      static_cast<__int128>(minimum_value) + 1;
  constexpr std::uint64_t max_bitmap_bytes = 512ULL * 1024ULL * 1024ULL;
  if (array_count > 3 && span128 > 0 &&
      span128 <= static_cast<__int128>(
                     std::numeric_limits<std::uint64_t>::max() - 63)) {
    const std::uint64_t span = static_cast<std::uint64_t>(span128);
    const std::uint64_t word_count = (span + 63) / 64;
    if (word_count <= max_bitmap_bytes / sizeof(std::uint64_t) &&
        word_count <= static_cast<std::uint64_t>(capacity)) {
      std::vector<std::uint64_t> bitmap(
          static_cast<std::size_t>(word_count), 0);
      for (std::int32_t index = 0; index < array_count; ++index) {
        for (std::int64_t position = 0; position < lengths[index]; ++position) {
          const std::uint64_t offset = static_cast<std::uint64_t>(
              static_cast<__int128>(arrays[index][position]) -
              static_cast<__int128>(minimum_value));
          bitmap[static_cast<std::size_t>(offset >> 6)] |=
              std::uint64_t{1} << (offset & 63);
        }
      }
      std::int64_t written = 0;
      for (std::uint64_t word_index = 0; word_index < word_count;
           ++word_index) {
        std::uint64_t word = bitmap[static_cast<std::size_t>(word_index)];
        while (word != 0) {
          const std::uint64_t bit =
              static_cast<std::uint64_t>(__builtin_ctzll(word));
          if (written >= capacity) {
            return -3;
          }
          output[written++] = static_cast<std::int64_t>(
              static_cast<__int128>(minimum_value) +
              static_cast<__int128>((word_index << 6) + bit));
          word &= word - 1;
        }
      }
      return written;
    }
  }
  using HeapItem = std::pair<std::int64_t, std::int32_t>;
  std::priority_queue<
      HeapItem,
      std::vector<HeapItem>,
      std::greater<HeapItem>> heap;
  for (std::int32_t index = 0; index < array_count; ++index) {
    if (lengths[index] > 0) {
      heap.emplace(arrays[index][0], index);
    }
  }
  std::int64_t written = 0;
  while (!heap.empty()) {
    const std::int64_t next_value = heap.top().first;
    if (written >= capacity) {
      return -3;
    }
    output[written++] = next_value;
    // Every input is strictly increasing, so there is at most one heap entry
    // per array. Consume every array sharing this value before emitting the
    // next unique row. The resulting bytes are identical to the former head
    // scan while changing O(unique_rows * array_count) to O(total_rows log K).
    while (!heap.empty() && heap.top().first == next_value) {
      const std::int32_t index = heap.top().second;
      heap.pop();
      auto& position = positions[static_cast<std::size_t>(index)];
      ++position;
      if (position < lengths[index]) {
        heap.emplace(arrays[index][position], index);
      }
    }
  }
  return written;
}

extern "C" std::int64_t certscr_sorted_grid_sequences(
    const std::int64_t* offsets,
    const std::int64_t sequence_count,
    const std::int64_t* rows,
    const std::int64_t row_count,
    std::int32_t* output) {
  if (offsets == nullptr || rows == nullptr || output == nullptr ||
      sequence_count < 1 || row_count < 0 || offsets[0] != 0) {
    return -1;
  }
  for (std::int64_t sequence = 0; sequence < sequence_count; ++sequence) {
    if (offsets[sequence + 1] < offsets[sequence]) {
      return -2;
    }
  }
  std::int64_t sequence = 0;
  for (std::int64_t index = 0; index < row_count; ++index) {
    if (rows[index] < 0 || rows[index] >= offsets[sequence_count] ||
        (index > 0 && rows[index] < rows[index - 1])) {
      return -3;
    }
    while (sequence + 1 < sequence_count &&
           rows[index] >= offsets[sequence + 1]) {
      ++sequence;
    }
    output[index] = static_cast<std::int32_t>(sequence);
  }
  return row_count;
}

extern "C" std::int64_t certscr_sorted_unique_int64_union_with_positions(
    const std::int64_t* const* arrays,
    const std::int64_t* lengths,
    const std::int32_t array_count,
    std::int64_t* output,
    std::int32_t* const* output_positions,
    const std::int64_t capacity,
    const std::int32_t validate_sorted) {
  if (arrays == nullptr || lengths == nullptr || output == nullptr ||
      output_positions == nullptr || array_count < 0 || capacity < 0 ||
      capacity > std::numeric_limits<std::int32_t>::max()) {
    return -1;
  }
  std::vector<std::int64_t> positions(static_cast<std::size_t>(array_count), 0);
  for (std::int32_t index = 0; index < array_count; ++index) {
    if (lengths[index] < 0 ||
        (lengths[index] > 0 &&
         (arrays[index] == nullptr || output_positions[index] == nullptr))) {
      return -2;
    }
    if (validate_sorted != 0) {
      for (std::int64_t position = 1; position < lengths[index]; ++position) {
        if (arrays[index][position] <= arrays[index][position - 1]) {
          return -4;
        }
      }
    }
  }
  std::int64_t written = 0;
  while (true) {
    std::int64_t next_value = std::numeric_limits<std::int64_t>::max();
    bool found = false;
    for (std::int32_t index = 0; index < array_count; ++index) {
      const std::int64_t position = positions[static_cast<std::size_t>(index)];
      if (position < lengths[index]) {
        next_value = std::min(next_value, arrays[index][position]);
        found = true;
      }
    }
    if (!found) {
      break;
    }
    if (written >= capacity) {
      return -3;
    }
    output[written] = next_value;
    for (std::int32_t index = 0; index < array_count; ++index) {
      auto& position = positions[static_cast<std::size_t>(index)];
      if (position < lengths[index] && arrays[index][position] == next_value) {
        output_positions[index][position] = static_cast<std::int32_t>(written);
        ++position;
      }
    }
    ++written;
  }
  return written;
}

extern "C" std::int64_t certscr_sparse_component_integral(
    const std::int64_t* summary_indices,
    const double* summary_eta,
    const std::int64_t summary_count,
    const double inactive_eta,
    const std::int64_t* block_indices,
    const float* block_values,
    const std::int64_t block_count,
    const std::int32_t width,
    const double* coefficients,
    const double* row_weights,
    const std::int64_t* sequence_offsets,
    const std::int64_t sequence_count,
    const std::int32_t validate_sorted,
    double* output) {
  if (summary_indices == nullptr || summary_eta == nullptr ||
      block_indices == nullptr || block_values == nullptr ||
      coefficients == nullptr || sequence_offsets == nullptr ||
      output == nullptr || summary_count < 0 ||
      block_count < 0 || width < 1 || sequence_count < 1 ||
      sequence_offsets[0] != 0) {
    return -1;
  }
  if (validate_sorted != 0) {
    for (std::int64_t index = 1; index < summary_count; ++index) {
      if (summary_indices[index] <= summary_indices[index - 1]) {
        return -2;
      }
    }
    for (std::int64_t index = 1; index < block_count; ++index) {
      if (block_indices[index] <= block_indices[index - 1]) {
        return -3;
      }
    }
  }
  for (std::int64_t sequence = 0; sequence < sequence_count; ++sequence) {
    if (sequence_offsets[sequence + 1] < sequence_offsets[sequence]) {
      return -4;
    }
  }
  std::int64_t summary_position = 0;
  std::int64_t sequence = 0;
  for (std::int64_t row = 0; row < block_count; ++row) {
    const std::int64_t grid_row = block_indices[row];
    if (grid_row < 0 || grid_row >= sequence_offsets[sequence_count]) {
      return -5;
    }
    while (summary_position < summary_count &&
           summary_indices[summary_position] < grid_row) {
      ++summary_position;
    }
    while (sequence + 1 < sequence_count &&
           grid_row >= sequence_offsets[sequence + 1]) {
      ++sequence;
    }
    const double eta =
        summary_position < summary_count &&
                summary_indices[summary_position] == grid_row
            ? summary_eta[summary_position]
            : inactive_eta;
    double activation = 0.0;
    const float* values = block_values + row * static_cast<std::int64_t>(width);
    for (std::int32_t column = 0; column < width; ++column) {
      activation += static_cast<double>(values[column]) * coefficients[column];
    }
    const double quadrature = row_weights == nullptr ? 1.0 : row_weights[row];
    output[sequence] += quadrature * std::exp(eta) * activation;
  }
  return block_count;
}

extern "C" std::int64_t certscr_add_sparse_linear_predictor(
    const std::int32_t* positions,
    const float* values,
    const std::int64_t row_count,
    const std::int32_t width,
    const double* coefficients,
    const double scale,
    double* output,
    const std::int64_t output_count) {
  if (positions == nullptr || values == nullptr || coefficients == nullptr ||
      output == nullptr || row_count < 0 || width < 1 || output_count < 0) {
    return -1;
  }
  for (std::int64_t row = 0; row < row_count; ++row) {
    const std::int32_t destination = positions[row];
    if (destination < 0 || destination >= output_count) {
      return -2;
    }
    const float* source = values + row * static_cast<std::int64_t>(width);
    double value = 0.0;
    for (std::int32_t column = 0; column < width; ++column) {
      value += static_cast<double>(source[column]) * coefficients[column];
    }
    output[destination] += scale * value;
  }
  return row_count;
}

extern "C" std::int64_t certscr_group_float32_rows(
    const float* matrix,
    const double* weights,
    const std::int64_t rows,
    const std::int64_t columns,
    float* output_matrix,
    double* output_weights) {
  if (matrix == nullptr || weights == nullptr || output_matrix == nullptr ||
      output_weights == nullptr || rows < 0 || columns < 1) {
    return -1;
  }
  FlatRowIndex index(rows);
  std::int64_t groups = 0;
  const std::size_t row_bytes = static_cast<std::size_t>(columns) * sizeof(float);
  for (std::int64_t row = 0; row < rows; ++row) {
    const float* input = matrix + row * columns;
    const std::uint64_t hash = hash_row(input, columns);
    std::size_t empty_slot = 0;
    const std::int64_t group = index.find(
        hash, input, output_matrix, columns, &empty_slot);
    if (group >= 0) {
      output_weights[group] += weights[row];
      continue;
    }
    // ``matrix`` and ``output_matrix`` may alias in the solver-only in-place
    // compaction path.  Stable grouping writes only at/before the current
    // input row, so memmove is safe and cannot overwrite a future row.
    std::memmove(output_matrix + groups * columns, input, row_bytes);
    output_weights[groups] = weights[row];
    index.insert(empty_slot, hash, groups);
    ++groups;
  }
  std::int64_t kept = 0;
  for (std::int64_t group = 0; group < groups; ++group) {
    if (output_weights[group] <= 0.0) {
      continue;
    }
    if (kept != group) {
      std::memmove(
          output_matrix + kept * columns,
          output_matrix + group * columns,
          row_bytes);
      output_weights[kept] = output_weights[group];
    }
    ++kept;
  }
  return kept;
}

extern "C" std::int64_t certscr_group_float32_rows_partitioned(
    const float* matrix,
    const double* event_weights,
    const double* grid_weights,
    const std::int64_t rows,
    const std::int64_t event_rows,
    const std::int64_t columns,
    float* output_matrix,
    double* output_weights,
    std::int64_t* output_event_rows) {
  if (matrix == nullptr || event_weights == nullptr || grid_weights == nullptr ||
      output_matrix == nullptr || output_weights == nullptr ||
      output_event_rows == nullptr || rows < 0 || event_rows < 0 ||
      event_rows > rows || columns < 1) {
    return -1;
  }
  // Event and quadrature rows must remain separate sufficient-statistic
  // partitions even when their covariates are bit-identical.  Group each
  // partition into adjacent regions of one output allocation.  This produces
  // exactly the same stable first-occurrence ordering as two independent
  // calls followed by concatenate, without retaining those two temporary
  // matrices and a third concatenated copy at peak memory.
  const std::int64_t grouped_events = certscr_group_float32_rows(
      matrix,
      event_weights,
      event_rows,
      columns,
      output_matrix,
      output_weights);
  if (grouped_events < 0) {
    return grouped_events;
  }
  const std::int64_t grouped_grid = certscr_group_float32_rows(
      matrix + event_rows * columns,
      grid_weights,
      rows - event_rows,
      columns,
      output_matrix + grouped_events * columns,
      output_weights + grouped_events);
  if (grouped_grid < 0) {
    return grouped_grid;
  }
  *output_event_rows = grouped_events;
  return grouped_events + grouped_grid;
}

extern "C" std::int64_t certscr_group_sparse_design_partitioned(
    const std::int64_t* active_rows,
    const double* active_weights,
    const std::int64_t active_count,
    const std::int64_t* const* block_rows,
    const float* const* block_grid_values,
    const float* const* block_event_values,
    const std::int64_t* block_lengths,
    const std::int32_t* block_widths,
    const float* block_signs,
    const std::int32_t block_count,
    const std::int64_t event_rows,
    const double* event_weights,
    const double inactive_weight,
    const std::int32_t columns,
    float* output_matrix,
    double* output_weights,
    std::int64_t* output_representatives,
    const std::int64_t capacity,
    std::int64_t* output_active_groups,
    std::int64_t* output_event_rows) {
  if (active_rows == nullptr || active_weights == nullptr ||
      block_rows == nullptr || block_grid_values == nullptr ||
      block_event_values == nullptr || block_lengths == nullptr ||
      block_widths == nullptr || block_signs == nullptr ||
      event_weights == nullptr || output_matrix == nullptr ||
      output_weights == nullptr || output_representatives == nullptr ||
      output_event_rows == nullptr ||
      active_count < 0 || block_count < 1 || event_rows < 0 ||
      columns < 1 || capacity < event_rows + 1 || inactive_weight < 0.0) {
    return -1;
  }
  std::int32_t expected_columns = 1;
  for (std::int32_t block = 0; block < block_count; ++block) {
    if (block_lengths[block] < 0 || block_widths[block] < 1 ||
        block_rows[block] == nullptr || block_grid_values[block] == nullptr ||
        block_event_values[block] == nullptr ||
        !std::isfinite(static_cast<double>(block_signs[block]))) {
      return -2;
    }
    expected_columns += block_widths[block];
    for (std::int64_t position = 1; position < block_lengths[block]; ++position) {
      if (block_rows[block][position] <= block_rows[block][position - 1]) {
        return -3;
      }
    }
  }
  if (expected_columns != columns) {
    return -4;
  }
  for (std::int64_t row = 1; row < active_count; ++row) {
    if (active_rows[row] <= active_rows[row - 1]) {
      return -5;
    }
  }

  std::vector<float> row(static_cast<std::size_t>(columns), 0.0F);
  const std::size_t row_bytes = static_cast<std::size_t>(columns) * sizeof(float);

  auto group_partition = [&](const bool events,
                             const std::int64_t input_count,
                             const std::int64_t output_offset,
                             const double extra_zero_weight,
                             std::vector<std::int64_t>* grid_positions)
      -> std::int64_t {
    FlatRowIndex index(input_count + static_cast<std::int64_t>(extra_zero_weight > 0.0));
    std::int64_t groups = 0;

    auto insert = [&](const double weight,
                      const std::int64_t representative,
                      std::int64_t* inserted_group) -> bool {
      if (!(weight > 0.0) || !std::isfinite(weight)) {
        return weight == 0.0;
      }
      const std::uint64_t hash = hash_row(row.data(), columns);
      std::size_t empty_slot = 0;
      const std::int64_t group = index.find(
          hash,
          row.data(),
          output_matrix + output_offset * columns,
          columns,
          &empty_slot);
      if (group >= 0) {
        output_weights[output_offset + group] += weight;
        if (inserted_group != nullptr) {
          *inserted_group = group;
        }
        return std::isfinite(output_weights[output_offset + group]);
      }
      if (output_offset + groups >= capacity) {
        return false;
      }
      std::memcpy(
          output_matrix + (output_offset + groups) * columns,
          row.data(),
          row_bytes);
      output_weights[output_offset + groups] = weight;
      output_representatives[output_offset + groups] = representative;
      index.insert(empty_slot, hash, groups);
      if (inserted_group != nullptr) {
        *inserted_group = groups;
      }
      ++groups;
      return true;
    };

    for (std::int64_t input = 0; input < input_count; ++input) {
      std::fill(row.begin(), row.end(), 0.0F);
      row[0] = 1.0F;
      std::int32_t column = 1;
      for (std::int32_t block = 0; block < block_count; ++block) {
        const std::int32_t width = block_widths[block];
        const float sign = block_signs[block];
        const float* values = nullptr;
        if (events) {
          values = block_event_values[block] + input * static_cast<std::int64_t>(width);
        } else {
          auto& position = (*grid_positions)[static_cast<std::size_t>(block)];
          while (position < block_lengths[block] &&
                 block_rows[block][position] < active_rows[input]) {
            ++position;
          }
          if (position < block_lengths[block] &&
              block_rows[block][position] == active_rows[input]) {
            values = block_grid_values[block] +
                     position * static_cast<std::int64_t>(width);
          }
        }
        if (values != nullptr) {
          for (std::int32_t inner = 0; inner < width; ++inner) {
            row[column + inner] = sign * values[inner];
          }
        }
        column += width;
      }
      const double weight = events ? event_weights[input] : active_weights[input];
      const std::int64_t representative =
          events ? -(input + 1) : active_rows[input];
      std::int64_t* assigned =
          (!events && output_active_groups != nullptr)
              ? output_active_groups + input
              : nullptr;
      if (!insert(weight, representative, assigned)) {
        return -1;
      }
    }
    if (!events && extra_zero_weight > 0.0) {
      std::fill(row.begin(), row.end(), 0.0F);
      row[0] = 1.0F;
      if (!insert(
              extra_zero_weight,
              std::numeric_limits<std::int64_t>::min(),
              nullptr)) {
        return -1;
      }
    }
    return groups;
  };

  std::vector<std::int64_t> unused_positions;
  const std::int64_t grouped_events = group_partition(
      true, event_rows, 0, 0.0, &unused_positions);
  if (grouped_events < 0) {
    return -6;
  }
  std::vector<std::int64_t> grid_positions(
      static_cast<std::size_t>(block_count), 0);
  const std::int64_t grouped_grid = group_partition(
      false,
      active_count,
      grouped_events,
      inactive_weight,
      &grid_positions);
  if (grouped_grid < 0) {
    return -7;
  }
  *output_event_rows = grouped_events;
  return grouped_events + grouped_grid;
}

extern "C" std::int64_t certscr_refine_sparse_base_partitioned(
    const std::int64_t* active_rows,
    const double* active_weights,
    const std::int64_t active_count,
    const std::int64_t* base_source_rows,
    const std::int64_t* base_source_groups,
    const std::int64_t base_source_count,
    const float* base_grid_design,
    const double* base_grid_weights,
    const std::int64_t base_group_count,
    const std::int64_t zero_base_group,
    const std::int32_t base_columns,
    const float* base_event_design,
    const std::int64_t* const* block_rows,
    const float* const* block_grid_values,
    const float* const* block_event_values,
    const std::int64_t* block_lengths,
    const std::int32_t* block_widths,
    const float* block_signs,
    const std::int32_t block_count,
    const std::int64_t event_rows,
    const double* event_weights,
    const std::int32_t columns,
    float* output_matrix,
    double* output_weights,
    std::int64_t* output_representatives,
    const std::int64_t capacity,
    std::int32_t* output_active_groups,
    std::int32_t* output_background_groups,
    std::int32_t* output_event_groups,
    std::int64_t* output_event_rows) {
  if (active_rows == nullptr || active_weights == nullptr ||
      base_source_rows == nullptr || base_source_groups == nullptr ||
      base_grid_design == nullptr || base_grid_weights == nullptr ||
      base_event_design == nullptr || block_rows == nullptr ||
      block_grid_values == nullptr || block_event_values == nullptr ||
      block_lengths == nullptr || block_widths == nullptr ||
      block_signs == nullptr || event_weights == nullptr ||
      output_matrix == nullptr || output_weights == nullptr ||
      output_representatives == nullptr || output_event_rows == nullptr ||
      active_count < 0 || base_source_count < 0 || base_group_count < 1 ||
      base_columns < 1 || block_count < 1 || event_rows < 0 ||
      capacity < event_rows + base_group_count || columns <= base_columns) {
    return -1;
  }
  std::int32_t expected_columns = base_columns;
  for (std::int32_t block = 0; block < block_count; ++block) {
    if (block_lengths[block] < 0 || block_widths[block] < 1 ||
        block_rows[block] == nullptr || block_grid_values[block] == nullptr ||
        block_event_values[block] == nullptr ||
        !std::isfinite(static_cast<double>(block_signs[block]))) {
      return -2;
    }
    expected_columns += block_widths[block];
    for (std::int64_t position = 1; position < block_lengths[block]; ++position) {
      if (block_rows[block][position] <= block_rows[block][position - 1]) {
        return -3;
      }
    }
  }
  if (expected_columns != columns) {
    return -4;
  }
  for (std::int64_t row = 1; row < active_count; ++row) {
    if (active_rows[row] <= active_rows[row - 1]) {
      return -5;
    }
  }
  for (std::int64_t row = 0; row < base_source_count; ++row) {
    if ((row > 0 && base_source_rows[row] <= base_source_rows[row - 1]) ||
        base_source_groups[row] < 0 ||
        base_source_groups[row] >= base_group_count) {
      return -6;
    }
  }

  std::vector<float> row(static_cast<std::size_t>(columns), 0.0F);
  const std::size_t row_bytes = static_cast<std::size_t>(columns) * sizeof(float);
  auto group_partition = [&](const bool events,
                             const std::int64_t input_count,
                             const std::int64_t output_offset,
                             std::vector<double>* remaining,
                             std::vector<std::int64_t>* block_positions)
      -> std::int64_t {
    const std::int64_t expected =
        events ? input_count : input_count + base_group_count;
    const std::int64_t initial_expected = events
        ? expected
        : std::min(
              expected,
              base_group_count + std::max<std::int64_t>(1, input_count / 4));
    FlatRowIndex index(expected, initial_expected);
    std::int64_t groups = 0;
    std::int64_t base_source_position = 0;

    auto insert = [&](const double weight,
                      const std::int64_t representative,
                      std::int32_t* assigned_group) -> bool {
      if (!(weight > 0.0) || !std::isfinite(weight)) {
        if (assigned_group != nullptr) {
          *assigned_group = -1;
        }
        return weight == 0.0;
      }
      const std::uint64_t hash = hash_row(row.data(), columns);
      std::size_t empty_slot = 0;
      const std::int64_t group = index.find(
          hash,
          row.data(),
          output_matrix + output_offset * columns,
          columns,
          &empty_slot);
      if (group >= 0) {
        output_weights[output_offset + group] += weight;
        if (assigned_group != nullptr) {
          *assigned_group = static_cast<std::int32_t>(group);
        }
        return std::isfinite(output_weights[output_offset + group]);
      }
      if (output_offset + groups >= capacity) {
        return false;
      }
      std::memcpy(output_matrix + (output_offset + groups) * columns,
                  row.data(), row_bytes);
      output_weights[output_offset + groups] = weight;
      output_representatives[output_offset + groups] = representative;
      index.insert(empty_slot, hash, groups);
      if (assigned_group != nullptr) {
        *assigned_group = static_cast<std::int32_t>(groups);
      }
      ++groups;
      return true;
    };

    for (std::int64_t input = 0; input < input_count; ++input) {
      std::fill(row.begin(), row.end(), 0.0F);
      std::int64_t base_group = 0;
      if (events) {
        std::memcpy(row.data(),
                    base_event_design + input * static_cast<std::int64_t>(base_columns),
                    static_cast<std::size_t>(base_columns) * sizeof(float));
      } else {
        // Both streams are strictly increasing.  A monotone merge is exact
        // and replaces one log2(N) binary search per active loan-month with a
        // single sequential pass over the fixed source map.
        while (base_source_position < base_source_count &&
               base_source_rows[base_source_position] < active_rows[input]) {
          ++base_source_position;
        }
        if (base_source_position < base_source_count &&
            base_source_rows[base_source_position] == active_rows[input]) {
          base_group = base_source_groups[base_source_position];
        } else {
          if (zero_base_group < 0 || zero_base_group >= base_group_count) {
            return -1;
          }
          base_group = zero_base_group;
        }
        std::memcpy(row.data(),
                    base_grid_design + base_group * static_cast<std::int64_t>(base_columns),
                    static_cast<std::size_t>(base_columns) * sizeof(float));
        const std::size_t base_group_index =
            static_cast<std::size_t>(base_group);
        (*remaining)[base_group_index] -= active_weights[input];
      }
      std::int32_t column = base_columns;
      for (std::int32_t block = 0; block < block_count; ++block) {
        const std::int32_t width = block_widths[block];
        const float sign = block_signs[block];
        const float* values = nullptr;
        if (events) {
          values = block_event_values[block] +
                   input * static_cast<std::int64_t>(width);
        } else {
          auto& position = (*block_positions)[static_cast<std::size_t>(block)];
          while (position < block_lengths[block] &&
                 block_rows[block][position] < active_rows[input]) {
            ++position;
          }
          if (position < block_lengths[block] &&
              block_rows[block][position] == active_rows[input]) {
            values = block_grid_values[block] +
                     position * static_cast<std::int64_t>(width);
          }
        }
        if (values != nullptr) {
          for (std::int32_t inner = 0; inner < width; ++inner) {
            row[column + inner] = sign * values[inner];
          }
        }
        column += width;
      }
      const double weight = events ? event_weights[input] : active_weights[input];
      const std::int64_t representative =
          events ? -(input + 1) : active_rows[input];
      std::int32_t* assigned = events
          ? (output_event_groups == nullptr ? nullptr : output_event_groups + input)
          : (output_active_groups == nullptr ? nullptr : output_active_groups + input);
      if (!insert(weight, representative, assigned)) {
        return -2;
      }
    }

    if (!events) {
      for (std::int64_t base_group = 0; base_group < base_group_count;
           ++base_group) {
        const std::size_t base_group_index =
            static_cast<std::size_t>(base_group);
        double weight = (*remaining)[base_group_index];
        const double epsilon = std::numeric_limits<double>::epsilon();
        const double scale = std::max(
            1.0, std::abs(base_grid_weights[base_group]));
        double tolerance = 128.0 * epsilon * scale;
        if (weight < -tolerance) {
          // The common unit-weight path is exact and never enters here.  For
          // a rare IPW cancellation, recover the group-specific operation
          // count with one cold rescan instead of allocating/zeroing a
          // base_group_count-sized counter vector on every successful W.
          std::int64_t removal_count = 0;
          std::int64_t source_position = 0;
          for (std::int64_t input = 0; input < active_count; ++input) {
            while (source_position < base_source_count &&
                   base_source_rows[source_position] < active_rows[input]) {
              ++source_position;
            }
            const std::int64_t assigned_group =
                source_position < base_source_count &&
                        base_source_rows[source_position] == active_rows[input]
                    ? base_source_groups[source_position]
                    : zero_base_group;
            removal_count += assigned_group == base_group;
          }
          const double operation_error =
              static_cast<double>(removal_count + 1) * epsilon;
          const double gamma = operation_error < 0.5
              ? operation_error / (1.0 - operation_error)
              : 1.0;
          tolerance = (128.0 * epsilon + gamma) * scale;
        }
        if (weight < 0.0 && weight >= -tolerance) {
          weight = 0.0;
        }
        if (weight < 0.0 || !std::isfinite(weight)) {
          return -3;
        }
        std::fill(row.begin(), row.end(), 0.0F);
        std::memcpy(row.data(),
                    base_grid_design + base_group * static_cast<std::int64_t>(base_columns),
                    static_cast<std::size_t>(base_columns) * sizeof(float));
        std::int32_t* assigned = output_background_groups == nullptr
            ? nullptr
            : output_background_groups + base_group;
        if (!insert(
                weight,
                std::numeric_limits<std::int64_t>::min(),
                assigned)) {
          return -4;
        }
      }
    }
    return groups;
  };

  std::vector<double> remaining(
      base_grid_weights, base_grid_weights + base_group_count);
  std::vector<std::int64_t> unused_positions;
  const std::int64_t grouped_events = group_partition(
      true, event_rows, 0, &remaining, &unused_positions);
  if (grouped_events < 0) {
    return -7;
  }
  std::vector<std::int64_t> block_positions(
      static_cast<std::size_t>(block_count), 0);
  const std::int64_t grouped_grid = group_partition(
      false, active_count, grouped_events, &remaining, &block_positions);
  if (grouped_grid < 0) {
    return -8;
  }
  *output_event_rows = grouped_events;
  return grouped_events + grouped_grid;
}

extern "C" std::int64_t certscr_update_sparse_design_partitioned(
    const std::int64_t* active_rows,
    const double* active_weights,
    const std::int64_t active_count,
    const std::int32_t* old_grid_group_map,
    const std::int64_t grid_count,
    const float* old_grid_design,
    const double* old_grid_weights,
    const std::int64_t old_grid_groups,
    const std::int32_t* old_event_group_map,
    const std::int64_t event_rows,
    const float* old_event_design,
    const std::int64_t old_event_groups,
    const double* event_weights,
    const std::int64_t* const* block_rows,
    const float* const* block_grid_values,
    const float* const* block_event_values,
    const std::int64_t* block_lengths,
    const std::int32_t* block_widths,
    const std::int32_t* block_offsets,
    const float* block_signs,
    const std::int32_t block_count,
    const std::int32_t old_columns,
    const std::int32_t columns,
    float* output_matrix,
    double* output_weights,
    std::int64_t* output_representatives,
    const std::int64_t capacity,
    std::int32_t* output_active_groups,
    std::int32_t* output_background_groups,
    std::int32_t* output_event_groups,
    std::int64_t* output_event_rows) {
  if (active_rows == nullptr || active_weights == nullptr ||
      old_grid_group_map == nullptr || old_grid_design == nullptr ||
      old_grid_weights == nullptr || old_event_group_map == nullptr ||
      old_event_design == nullptr || event_weights == nullptr ||
      block_rows == nullptr || block_grid_values == nullptr ||
      block_event_values == nullptr || block_lengths == nullptr ||
      block_widths == nullptr || block_offsets == nullptr ||
      block_signs == nullptr || output_matrix == nullptr ||
      output_weights == nullptr || output_representatives == nullptr ||
      output_active_groups == nullptr || output_background_groups == nullptr ||
      output_event_groups == nullptr || output_event_rows == nullptr ||
      active_count < 0 || grid_count < 0 || old_grid_groups < 1 ||
      event_rows < 0 || old_event_groups < 0 || block_count < 1 ||
      old_columns < 1 || columns < old_columns ||
      capacity < event_rows + old_grid_groups) {
    return -1;
  }
  for (std::int32_t block = 0; block < block_count; ++block) {
    if (block_lengths[block] < 0 || block_widths[block] < 1 ||
        block_offsets[block] < 0 ||
        block_offsets[block] + block_widths[block] > columns ||
        block_rows[block] == nullptr || block_grid_values[block] == nullptr ||
        block_event_values[block] == nullptr ||
        !std::isfinite(static_cast<double>(block_signs[block]))) {
      return -2;
    }
    for (std::int64_t position = 1; position < block_lengths[block]; ++position) {
      if (block_rows[block][position] <= block_rows[block][position - 1]) {
        return -3;
      }
    }
  }
  for (std::int64_t row_index = 0; row_index < active_count; ++row_index) {
    if (active_rows[row_index] < 0 || active_rows[row_index] >= grid_count ||
        (row_index > 0 && active_rows[row_index] <= active_rows[row_index - 1])) {
      return -4;
    }
  }

  std::vector<float> row(static_cast<std::size_t>(columns), 0.0F);
  const std::size_t row_bytes =
      static_cast<std::size_t>(columns) * sizeof(float);
  const std::size_t old_row_bytes =
      static_cast<std::size_t>(old_columns) * sizeof(float);
  // ``old_grid_weights`` was formed by summing many positive sequence/IPW
  // masses.  Incremental refinement removes the active rows one at a time.
  // When an old group is exhausted, the two legal summation orders can leave
  // a tiny negative residual.  The rare slow path below recovers the number
  // of removals so it can use the standard gamma_k floating-point error bound,
  // rather than paying for a large counter array on every successful update.
  auto group_partition = [&](const bool events,
                             const std::int64_t input_count,
                             const std::int64_t output_offset,
                             std::vector<double>* remaining,
                             std::vector<std::int64_t>* block_positions)
      -> std::int64_t {
    const std::int64_t expected =
        events ? input_count : input_count + old_grid_groups;
    const std::int64_t initial_expected = events
        ? expected
        : std::min(
              expected,
              old_grid_groups + std::max<std::int64_t>(1, input_count / 3));
    FlatRowIndex index(expected, initial_expected);
    std::int64_t groups = 0;

    auto insert = [&](const double weight,
                      const std::int64_t representative,
                      std::int32_t* assigned) -> bool {
      if (!(weight > 0.0) || !std::isfinite(weight)) {
        *assigned = -1;
        return weight == 0.0;
      }
      const std::uint64_t hash = hash_row(row.data(), columns);
      std::size_t empty_slot = 0;
      const std::int64_t group = index.find(
          hash,
          row.data(),
          output_matrix + output_offset * columns,
          columns,
          &empty_slot);
      if (group >= 0) {
        output_weights[output_offset + group] += weight;
        *assigned = static_cast<std::int32_t>(group);
        return std::isfinite(output_weights[output_offset + group]);
      }
      if (output_offset + groups >= capacity) {
        return false;
      }
      std::memcpy(output_matrix + (output_offset + groups) * columns,
                  row.data(), row_bytes);
      output_weights[output_offset + groups] = weight;
      output_representatives[output_offset + groups] = representative;
      index.insert(empty_slot, hash, groups);
      *assigned = static_cast<std::int32_t>(groups);
      ++groups;
      return true;
    };

    for (std::int64_t input = 0; input < input_count; ++input) {
      std::int32_t old_group = 0;
      if (events) {
        old_group = old_event_group_map[input];
        if (old_group < 0 || old_group >= old_event_groups) {
          return -1;
        }
        std::fill(row.begin(), row.end(), 0.0F);
        std::memcpy(row.data(),
                    old_event_design +
                        static_cast<std::int64_t>(old_group) * old_columns,
                    old_row_bytes);
      } else {
        old_group = old_grid_group_map[active_rows[input]];
        if (old_group < 0 || old_group >= old_grid_groups) {
          return -2;
        }
        std::fill(row.begin(), row.end(), 0.0F);
        std::memcpy(row.data(),
                    old_grid_design +
                        static_cast<std::int64_t>(old_group) * old_columns,
                    old_row_bytes);
        const std::size_t old_group_index =
            static_cast<std::size_t>(old_group);
        (*remaining)[old_group_index] -= active_weights[input];
      }
      for (std::int32_t block = 0; block < block_count; ++block) {
        const std::int32_t width = block_widths[block];
        const float sign = block_signs[block];
        const float* values = nullptr;
        if (events) {
          values = block_event_values[block] +
                   input * static_cast<std::int64_t>(width);
        } else {
          auto& position = (*block_positions)[static_cast<std::size_t>(block)];
          while (position < block_lengths[block] &&
                 block_rows[block][position] < active_rows[input]) {
            ++position;
          }
          if (position < block_lengths[block] &&
              block_rows[block][position] == active_rows[input]) {
            values = block_grid_values[block] +
                     position * static_cast<std::int64_t>(width);
          }
        }
        if (values != nullptr) {
          const std::int32_t offset = block_offsets[block];
          for (std::int32_t inner = 0; inner < width; ++inner) {
            row[offset + inner] += sign * values[inner];
          }
        }
      }
      const double weight = events ? event_weights[input] : active_weights[input];
      const std::int64_t representative =
          events ? -(input + 1) : active_rows[input];
      std::int32_t* assigned = events
          ? output_event_groups + input
          : output_active_groups + input;
      if (!insert(weight, representative, assigned)) {
        return -3;
      }
    }
    if (!events) {
      for (std::int64_t old_group = 0; old_group < old_grid_groups; ++old_group) {
        const std::size_t old_group_index =
            static_cast<std::size_t>(old_group);
        double weight = (*remaining)[old_group_index];
        const double epsilon = std::numeric_limits<double>::epsilon();
        const double scale = std::max(
            1.0,
            std::abs(old_grid_weights[old_group]));
        double tolerance = 128.0 * epsilon * scale;
        if (weight < -tolerance) {
          std::int64_t removal_count = 0;
          for (std::int64_t input = 0; input < active_count; ++input) {
            removal_count +=
                old_grid_group_map[active_rows[input]] == old_group;
          }
          std::int64_t original_group_count = 0;
          for (std::int64_t input = 0; input < grid_count; ++input) {
            original_group_count += old_grid_group_map[input] == old_group;
          }
          const double operation_error =
              static_cast<double>(
                  original_group_count + removal_count + 1) * epsilon;
          const double gamma = operation_error < 0.5
              ? operation_error / (1.0 - operation_error)
              : 1.0;
          tolerance = (128.0 * epsilon + gamma) * scale;
        }
        if (weight < 0.0 && weight >= -tolerance) {
          weight = 0.0;
        }
        if (weight < 0.0 || !std::isfinite(weight)) {
          return -4;
        }
        std::fill(row.begin(), row.end(), 0.0F);
        std::memcpy(
            row.data(),
            old_grid_design + old_group * static_cast<std::int64_t>(old_columns),
            old_row_bytes);
        if (!insert(weight,
                    std::numeric_limits<std::int64_t>::min(),
                    output_background_groups + old_group)) {
          return -5;
        }
      }
    }
    return groups;
  };

  std::vector<double> remaining(
      old_grid_weights, old_grid_weights + old_grid_groups);
  std::vector<std::int64_t> unused_positions;
  const std::int64_t grouped_events = group_partition(
      true, event_rows, 0, &remaining, &unused_positions);
  if (grouped_events < 0) {
    return -5;
  }
  std::vector<std::int64_t> block_positions(
      static_cast<std::size_t>(block_count), 0);
  const std::int64_t grouped_grid = group_partition(
      false, active_count, grouped_events, &remaining, &block_positions);
  if (grouped_grid < 0) {
    // Preserve the inner failure class for diagnostics: -61 invalid old
    // event/grid mapping, -62 invalid old grid mapping, -63 active insert,
    // -64 mass conservation, -65 background insert.
    return -60 + grouped_grid;
  }
  *output_event_rows = grouped_events;
  return grouped_events + grouped_grid;
}

extern "C" std::int64_t certscr_group_float32_rows_pair(
    const float* matrix,
    const double* first_weights,
    const double* second_weights,
    const std::int64_t rows,
    const std::int64_t columns,
    float* output_matrix,
    double* output_first,
    double* output_second) {
  if (matrix == nullptr || first_weights == nullptr || second_weights == nullptr ||
      output_matrix == nullptr || output_first == nullptr ||
      output_second == nullptr || rows < 0 || columns < 1) {
    return -1;
  }
  std::unordered_map<std::uint64_t, std::int64_t> hash_head;
  hash_head.reserve(static_cast<std::size_t>(rows * 1.3) + 1);
  std::vector<std::int64_t> collision_next;
  collision_next.reserve(static_cast<std::size_t>(rows));
  std::int64_t groups = 0;
  const std::size_t row_bytes = static_cast<std::size_t>(columns) * sizeof(float);
  for (std::int64_t row = 0; row < rows; ++row) {
    const float* input = matrix + row * columns;
    const std::uint64_t hash = hash_row(input, columns);
    auto found = hash_head.find(hash);
    std::int64_t group = found == hash_head.end() ? -1 : found->second;
    while (group >= 0 &&
           std::memcmp(input, output_matrix + group * columns, row_bytes) != 0) {
      group = collision_next[static_cast<std::size_t>(group)];
    }
    if (group >= 0) {
      output_first[group] += first_weights[row];
      output_second[group] += second_weights[row];
      continue;
    }
    std::memcpy(output_matrix + groups * columns, input, row_bytes);
    output_first[groups] = first_weights[row];
    output_second[groups] = second_weights[row];
    collision_next.push_back(found == hash_head.end() ? -1 : found->second);
    hash_head[hash] = groups;
    ++groups;
  }
  return groups;
}

extern "C" std::int64_t certscr_sparse_kernel_block(
    const std::int64_t* base_indices,
    const std::int64_t* occurrence_times,
    const std::int64_t* end_times,
    const std::int64_t occurrences,
    const float* basis,
    const std::int32_t knots,
    const std::int32_t lag,
    std::int64_t* output_indices,
    float* output_values,
    const std::int64_t capacity) {
  if (base_indices == nullptr || occurrence_times == nullptr ||
      end_times == nullptr || basis == nullptr || output_indices == nullptr ||
      output_values == nullptr || occurrences < 0 || knots < 1 || lag < 1 ||
      capacity < 0) {
    return -1;
  }
  std::unordered_map<std::int64_t, std::int64_t> destination_group;
  destination_group.reserve(static_cast<std::size_t>(capacity * 1.3) + 1);
  std::vector<std::int64_t> destinations;
  destinations.reserve(static_cast<std::size_t>(capacity));
  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(capacity) * knots);
  for (std::int64_t occurrence = 0; occurrence < occurrences; ++occurrence) {
    const std::int64_t available = std::max<std::int64_t>(
        0, std::min<std::int64_t>(lag, end_times[occurrence] - occurrence_times[occurrence]));
    for (std::int64_t offset = 0; offset < available; ++offset) {
      const std::int64_t destination = base_indices[occurrence] + offset + 1;
      auto found = destination_group.find(destination);
      std::int64_t group;
      if (found == destination_group.end()) {
        group = static_cast<std::int64_t>(destinations.size());
        if (group >= capacity) {
          return -2;
        }
        destination_group.emplace(destination, group);
        destinations.push_back(destination);
        values.resize(static_cast<std::size_t>(group + 1) * knots, 0.0);
      } else {
        group = found->second;
      }
      for (std::int32_t knot = 0; knot < knots; ++knot) {
        values[static_cast<std::size_t>(group) * knots + knot] +=
            static_cast<double>(basis[static_cast<std::size_t>(knot) * lag + offset]);
      }
    }
  }
  std::vector<std::int64_t> order(destinations.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](const std::int64_t left, const std::int64_t right) {
    return destinations[static_cast<std::size_t>(left)] <
           destinations[static_cast<std::size_t>(right)];
  });
  for (std::size_t output = 0; output < order.size(); ++output) {
    const std::int64_t group = order[output];
    output_indices[output] = destinations[static_cast<std::size_t>(group)];
    for (std::int32_t knot = 0; knot < knots; ++knot) {
      output_values[output * knots + knot] = static_cast<float>(
          values[static_cast<std::size_t>(group) * knots + knot]);
    }
  }
  return static_cast<std::int64_t>(order.size());
}

namespace {

bool certscr_cloglog_terms(
    const double eta,
    double* loss,
    double* gradient,
    double* hessian) {
  if (!std::isfinite(eta)) {
    return false;
  }
  const double mu = std::exp(eta);
  if (mu < 1.0e-4) {
    // -log(exp(eta)) is evaluated as -eta so extremely rare-event
    // intercepts do not underflow before the Bernoulli expansion is applied.
    *loss = -eta + 0.5 * mu - (mu * mu) / 24.0;
    *gradient = -1.0 + 0.5 * mu - (mu * mu) / 12.0;
    *hessian = std::max(0.0, 0.5 * mu - (mu * mu) / 6.0);
    return std::isfinite(*loss) && std::isfinite(*gradient) &&
           std::isfinite(*hessian);
  }
  if (mu <= 50.0 && std::isfinite(mu)) {
    const double denominator = std::expm1(mu);
    *loss = -std::log(-std::expm1(-mu));
    *gradient = -mu / denominator;
    *hessian = std::max(
        0.0,
        -mu / denominator +
            (mu * mu) * std::exp(mu) / (denominator * denominator));
    return std::isfinite(*loss) && std::isfinite(*gradient) &&
           std::isfinite(*hessian);
  }
  if (eta > 0.0) {
    *loss = 0.0;
    *gradient = 0.0;
    *hessian = 0.0;
    return true;
  }
  return false;
}

bool certscr_prepared_state(
    const double* design,
    const std::int64_t rows,
    const std::int64_t event_rows,
    const std::int32_t columns,
    const double* event_weights,
    const double* grid_weights,
    const std::int32_t likelihood,
    const double* values,
    double* objective,
    std::vector<double>* gradient,
    std::vector<double>* hessian) {
  const bool derivatives = gradient != nullptr && hessian != nullptr;
  *objective = 0.0;
  if (derivatives) {
    std::fill(gradient->begin(), gradient->end(), 0.0);
    std::fill(hessian->begin(), hessian->end(), 0.0);
  }
  for (std::int64_t row = 0; row < rows; ++row) {
    const double* x = design + row * static_cast<std::int64_t>(columns);
    double eta = 0.0;
    for (std::int32_t column = 0; column < columns; ++column) {
      eta += x[column] * values[column];
    }
    double first = 0.0;
    double second = 0.0;
    if (row < event_rows) {
      const double weight = event_weights[row];
      if (weight <= 0.0) {
        continue;
      }
      if (likelihood == 0) {
        *objective -= weight * eta;
        first = -weight;
      } else {
        double loss = 0.0;
        double event_gradient = 0.0;
        double event_hessian = 0.0;
        if (!certscr_cloglog_terms(
                eta, &loss, &event_gradient, &event_hessian)) {
          return false;
        }
        *objective += weight * loss;
        first = weight * event_gradient;
        second = weight * event_hessian;
      }
    } else {
      const double weight = grid_weights[row - event_rows];
      if (weight <= 0.0) {
        continue;
      }
      const double mu = weight * std::exp(eta);
      if (!std::isfinite(mu)) {
        return false;
      }
      *objective += mu;
      first = mu;
      second = mu;
    }
    if (!derivatives) {
      continue;
    }
    for (std::int32_t left = 0; left < columns; ++left) {
      (*gradient)[left] += first * x[left];
      for (std::int32_t right = 0; right <= left; ++right) {
        (*hessian)[static_cast<std::size_t>(left) * columns + right] +=
            second * x[left] * x[right];
      }
    }
  }
  if (!std::isfinite(*objective)) {
    return false;
  }
  if (derivatives) {
    for (std::int32_t left = 0; left < columns; ++left) {
      if (!std::isfinite((*gradient)[left])) {
        return false;
      }
      for (std::int32_t right = 0; right < left; ++right) {
        (*hessian)[static_cast<std::size_t>(right) * columns + left] =
            (*hessian)[static_cast<std::size_t>(left) * columns + right];
      }
    }
  }
  return true;
}

bool certscr_cholesky_direction(
    const std::vector<double>& hessian,
    const std::vector<double>& gradient,
    const std::vector<std::int32_t>& active,
    const std::int32_t columns,
    std::vector<double>* direction) {
  const std::size_t width = active.size();
  if (width == 0) {
    return false;
  }
  std::vector<double> scale(width, 0.0);
  std::vector<double> factor(width * width, 0.0);
  std::vector<double> rhs(width, 0.0);
  for (std::size_t row = 0; row < width; ++row) {
    const std::int32_t source_row = active[row];
    const double diagonal = hessian[
        static_cast<std::size_t>(source_row) * columns + source_row];
    scale[row] = std::sqrt(std::max(
        diagonal, std::numeric_limits<double>::min()));
    rhs[row] = -gradient[source_row] / scale[row];
    for (std::size_t column = 0; column <= row; ++column) {
      const std::int32_t source_column = active[column];
      factor[row * width + column] =
          hessian[static_cast<std::size_t>(source_row) * columns + source_column] /
          (scale[row] * scale[column]);
    }
  }
  const double pivot_floor = std::numeric_limits<double>::epsilon() *
                             std::max<std::size_t>(1, width) * 64.0;
  for (std::size_t row = 0; row < width; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      double value = factor[row * width + column];
      for (std::size_t inner = 0; inner < column; ++inner) {
        value -= factor[row * width + inner] *
                 factor[column * width + inner];
      }
      if (row == column) {
        if (!(value > pivot_floor) || !std::isfinite(value)) {
          return false;
        }
        factor[row * width + column] = std::sqrt(value);
      } else {
        factor[row * width + column] =
            value / factor[column * width + column];
      }
    }
  }
  std::vector<double> solution = rhs;
  for (std::size_t row = 0; row < width; ++row) {
    for (std::size_t column = 0; column < row; ++column) {
      solution[row] -= factor[row * width + column] * solution[column];
    }
    solution[row] /= factor[row * width + row];
  }
  for (std::size_t reverse = width; reverse-- > 0;) {
    for (std::size_t column = reverse + 1; column < width; ++column) {
      solution[reverse] -=
          factor[column * width + reverse] * solution[column];
    }
    solution[reverse] /= factor[reverse * width + reverse];
  }
  std::fill(direction->begin(), direction->end(), 0.0);
  for (std::size_t row = 0; row < width; ++row) {
    (*direction)[active[row]] = solution[row] / scale[row];
  }
  return true;
}

bool certscr_rank_revealing_direction(
    const std::vector<double>& hessian,
    const std::vector<double>& gradient,
    const std::vector<std::int32_t>& active,
    const std::int32_t columns,
    std::vector<double>* direction) {
  // Pivoted Cholesky selects a maximal independent principal subspace of the
  // standardized Fisher matrix.  Solving on that subspace is the same Newton
  // step modulo non-identifiable null directions and needs no regularization.
  const std::size_t width = active.size();
  if (width == 0) {
    return false;
  }
  std::vector<double> scale(width, 0.0);
  std::vector<std::int32_t> permutation(width, 0);
  std::vector<double> diagonal(width, 0.0);
  std::vector<double> factor(width * width, 0.0);
  for (std::size_t row = 0; row < width; ++row) {
    permutation[row] = static_cast<std::int32_t>(row);
    const std::int32_t source = active[row];
    scale[row] = std::sqrt(std::max(
        hessian[static_cast<std::size_t>(source) * columns + source],
        std::numeric_limits<double>::min()));
    diagonal[row] = 1.0;
  }
  const double rank_tolerance =
      std::numeric_limits<double>::epsilon() *
      std::max<std::size_t>(1, width) * 64.0;
  std::size_t rank = 0;
  for (; rank < width; ++rank) {
    std::size_t pivot = rank;
    for (std::size_t row = rank + 1; row < width; ++row) {
      if (diagonal[row] > diagonal[pivot]) {
        pivot = row;
      }
    }
    if (!(diagonal[pivot] > rank_tolerance) ||
        !std::isfinite(diagonal[pivot])) {
      break;
    }
    if (pivot != rank) {
      std::swap(permutation[pivot], permutation[rank]);
      std::swap(diagonal[pivot], diagonal[rank]);
      for (std::size_t column = 0; column < rank; ++column) {
        std::swap(factor[pivot * width + column],
                  factor[rank * width + column]);
      }
    }
    factor[rank * width + rank] = std::sqrt(diagonal[rank]);
    const std::size_t original_pivot =
        static_cast<std::size_t>(permutation[rank]);
    for (std::size_t row = rank + 1; row < width; ++row) {
      const std::size_t original_row =
          static_cast<std::size_t>(permutation[row]);
      double value = hessian[
          static_cast<std::size_t>(active[original_row]) * columns +
          active[original_pivot]] /
          (scale[original_row] * scale[original_pivot]);
      for (std::size_t column = 0; column < rank; ++column) {
        value -= factor[row * width + column] *
                 factor[rank * width + column];
      }
      factor[row * width + rank] =
          value / factor[rank * width + rank];
      diagonal[row] -= factor[row * width + rank] *
                       factor[row * width + rank];
      if (diagonal[row] < 0.0 && diagonal[row] > -rank_tolerance) {
        diagonal[row] = 0.0;
      }
    }
  }
  if (rank == 0) {
    return false;
  }
  std::vector<double> solution(rank, 0.0);
  for (std::size_t row = 0; row < rank; ++row) {
    const std::size_t original =
        static_cast<std::size_t>(permutation[row]);
    solution[row] = -gradient[active[original]] / scale[original];
    for (std::size_t column = 0; column < row; ++column) {
      solution[row] -= factor[row * width + column] * solution[column];
    }
    solution[row] /= factor[row * width + row];
  }
  for (std::size_t reverse = rank; reverse-- > 0;) {
    for (std::size_t row = reverse + 1; row < rank; ++row) {
      solution[reverse] -=
          factor[row * width + reverse] * solution[row];
    }
    solution[reverse] /= factor[reverse * width + reverse];
  }
  std::fill(direction->begin(), direction->end(), 0.0);
  for (std::size_t row = 0; row < rank; ++row) {
    const std::size_t original =
        static_cast<std::size_t>(permutation[row]);
    (*direction)[active[original]] = solution[row] / scale[original];
  }
  return true;
}

bool certscr_sparse_delta_state(
    const double* base_design,
    const std::int64_t base_event_rows,
    const std::int64_t base_grid_rows,
    const std::int32_t base_columns,
    const double* residual_event_weights,
    const double* residual_grid_weights,
    const std::int32_t* grid_base_groups,
    const std::int64_t* grid_row_offsets,
    const std::int32_t* grid_columns,
    const float* grid_values,
    const double* active_grid_weights,
    const std::int64_t active_grid_rows,
    const std::int32_t* event_base_groups,
    const std::int64_t* event_row_offsets,
    const std::int32_t* event_columns,
    const float* event_values,
    const double* active_event_weights,
    const std::int64_t active_event_rows,
    const std::int32_t delta_columns,
    const std::int32_t likelihood,
    const double* values,
    double* objective,
    std::vector<double>* gradient,
    std::vector<double>* hessian) {
  const std::int32_t columns = base_columns + delta_columns;
  const bool derivatives = gradient != nullptr && hessian != nullptr;
  *objective = 0.0;
  if (derivatives) {
    std::fill(gradient->begin(), gradient->end(), 0.0);
    std::fill(hessian->begin(), hessian->end(), 0.0);
  }

  auto terms = [&](const double eta, const double weight, const bool event,
                   double* first, double* second) -> bool {
    *first = 0.0;
    *second = 0.0;
    if (!(weight > 0.0)) {
      return true;
    }
    if (!event) {
      const double mu = weight * std::exp(eta);
      if (!std::isfinite(mu)) {
        return false;
      }
      *objective += mu;
      *first = mu;
      *second = mu;
      return true;
    }
    if (likelihood == 0) {
      *objective -= weight * eta;
      *first = -weight;
      return true;
    }
    double loss = 0.0;
    double event_gradient = 0.0;
    double event_hessian = 0.0;
    if (!certscr_cloglog_terms(
            eta, &loss, &event_gradient, &event_hessian)) {
      return false;
    }
    *objective += weight * loss;
    *first = weight * event_gradient;
    *second = weight * event_hessian;
    return true;
  };

  auto base_row = [&](const double* x, const double weight,
                      const bool event) -> bool {
    if (!(weight > 0.0)) {
      return true;
    }
    double eta = 0.0;
    for (std::int32_t column = 0; column < base_columns; ++column) {
      eta += x[column] * values[column];
    }
    double first = 0.0;
    double second = 0.0;
    if (!terms(eta, weight, event, &first, &second)) {
      return false;
    }
    if (!derivatives) {
      return true;
    }
    for (std::int32_t left = 0; left < base_columns; ++left) {
      (*gradient)[left] += first * x[left];
      for (std::int32_t right = 0; right <= left; ++right) {
        (*hessian)[static_cast<std::size_t>(left) * columns + right] +=
            second * x[left] * x[right];
      }
    }
    return true;
  };

  for (std::int64_t row = 0; row < base_event_rows; ++row) {
    if (!base_row(
            base_design + row * static_cast<std::int64_t>(base_columns),
            residual_event_weights[row], true)) {
      return false;
    }
  }
  for (std::int64_t row = 0; row < base_grid_rows; ++row) {
    if (!base_row(
            base_design + (base_event_rows + row) *
                              static_cast<std::int64_t>(base_columns),
            residual_grid_weights[row], false)) {
      return false;
    }
  }

  auto active_row = [&](const double* x, const std::int64_t begin,
                        const std::int64_t end, const std::int32_t* sparse_columns,
                        const float* sparse_values, const double weight,
                        const bool event) -> bool {
    if (!(weight > 0.0)) {
      return true;
    }
    double eta = 0.0;
    for (std::int32_t column = 0; column < base_columns; ++column) {
      eta += x[column] * values[column];
    }
    for (std::int64_t cursor = begin; cursor < end; ++cursor) {
      if (sparse_columns[cursor] >= delta_columns) {
        continue;
      }
      eta += static_cast<double>(sparse_values[cursor]) *
             values[base_columns + sparse_columns[cursor]];
    }
    double first = 0.0;
    double second = 0.0;
    if (!terms(eta, weight, event, &first, &second)) {
      return false;
    }
    if (!derivatives) {
      return true;
    }
    for (std::int32_t left = 0; left < base_columns; ++left) {
      (*gradient)[left] += first * x[left];
      for (std::int32_t right = 0; right <= left; ++right) {
        (*hessian)[static_cast<std::size_t>(left) * columns + right] +=
            second * x[left] * x[right];
      }
    }
    for (std::int64_t cursor = begin; cursor < end; ++cursor) {
      if (sparse_columns[cursor] >= delta_columns) {
        continue;
      }
      const std::int32_t absolute = base_columns + sparse_columns[cursor];
      const double z = static_cast<double>(sparse_values[cursor]);
      (*gradient)[absolute] += first * z;
      for (std::int32_t right = 0; right < base_columns; ++right) {
        (*hessian)[static_cast<std::size_t>(absolute) * columns + right] +=
            second * z * x[right];
      }
      for (std::int64_t other = begin; other <= cursor; ++other) {
        if (sparse_columns[other] >= delta_columns) {
          continue;
        }
        const std::int32_t absolute_other =
            base_columns + sparse_columns[other];
        const std::int32_t high = std::max(absolute, absolute_other);
        const std::int32_t low = std::min(absolute, absolute_other);
        (*hessian)[static_cast<std::size_t>(high) * columns + low] +=
            second * z * static_cast<double>(sparse_values[other]);
      }
    }
    return true;
  };

  for (std::int64_t row = 0; row < active_grid_rows; ++row) {
    const std::int32_t group = grid_base_groups[row];
    if (group < 0 || group >= base_grid_rows ||
        !active_row(
            base_design + (base_event_rows + group) *
                              static_cast<std::int64_t>(base_columns),
            grid_row_offsets[row], grid_row_offsets[row + 1], grid_columns,
            grid_values, active_grid_weights[row], false)) {
      return false;
    }
  }
  for (std::int64_t row = 0; row < active_event_rows; ++row) {
    const std::int32_t group = event_base_groups[row];
    if (group < 0 || group >= base_event_rows ||
        !active_row(
            base_design + group * static_cast<std::int64_t>(base_columns),
            event_row_offsets[row], event_row_offsets[row + 1], event_columns,
            event_values, active_event_weights[row], true)) {
      return false;
    }
  }
  if (!std::isfinite(*objective)) {
    return false;
  }
  if (derivatives) {
    for (std::int32_t left = 0; left < columns; ++left) {
      if (!std::isfinite((*gradient)[left])) {
        return false;
      }
      for (std::int32_t right = 0; right < left; ++right) {
        (*hessian)[static_cast<std::size_t>(right) * columns + left] =
            (*hessian)[static_cast<std::size_t>(left) * columns + right];
      }
    }
  }
  return true;
}

}  // namespace

extern "C" std::int64_t certscr_group_sparse_delta_rows(
    const std::int32_t* input_base_groups,
    const std::int64_t* input_row_offsets,
    const std::int32_t* input_columns,
    const float* input_values,
    const double* input_weights,
    const std::int64_t rows,
    const std::int64_t nnz,
    std::int32_t* output_base_groups,
    std::int64_t* output_row_offsets,
    std::int32_t* output_columns,
    float* output_values,
    double* output_weights,
    std::int64_t* output_nnz) {
  if (input_base_groups == nullptr || input_row_offsets == nullptr ||
      input_columns == nullptr || input_values == nullptr ||
      input_weights == nullptr || output_base_groups == nullptr ||
      output_row_offsets == nullptr || output_columns == nullptr ||
      output_values == nullptr || output_weights == nullptr ||
      output_nnz == nullptr || rows < 0 || nnz < 0 ||
      input_row_offsets[0] != 0 || input_row_offsets[rows] != nnz) {
    return -1;
  }
  if (rows == 0) {
    output_row_offsets[0] = 0;
    *output_nnz = 0;
    return 0;
  }
  std::size_t capacity = 2;
  const std::size_t initial_expected = static_cast<std::size_t>(
      std::min<std::int64_t>(rows, 4'194'304));
  while (capacity < initial_expected + initial_expected / 2 + 1) {
    capacity <<= 1;
  }
  std::vector<std::int32_t> slots(capacity, -1);
  std::vector<std::uint64_t> slot_hashes(capacity, 0);
  std::vector<std::uint64_t> group_hashes;
  group_hashes.reserve(initial_expected);
  std::size_t mask = capacity - 1;
  auto rehash = [&]() {
    capacity <<= 1;
    mask = capacity - 1;
    slots.assign(capacity, -1);
    slot_hashes.assign(capacity, 0);
    for (std::size_t group = 0; group < group_hashes.size(); ++group) {
      const std::uint64_t hash = group_hashes[group];
      std::size_t slot = static_cast<std::size_t>(hash) & mask;
      while (slots[slot] >= 0) {
        slot = (slot + 1) & mask;
      }
      slots[slot] = static_cast<std::int32_t>(group);
      slot_hashes[slot] = hash;
    }
  };
  std::int64_t groups = 0;
  std::int64_t written_nnz = 0;
  output_row_offsets[0] = 0;
  for (std::int64_t row = 0; row < rows; ++row) {
    const std::int64_t begin = input_row_offsets[row];
    const std::int64_t end = input_row_offsets[row + 1];
    if (begin < 0 || end < begin || end > nnz ||
        input_base_groups[row] < 0 ||
        !(input_weights[row] > 0.0) || !std::isfinite(input_weights[row])) {
      return -2;
    }
    std::uint64_t hash = mix_word(
        static_cast<std::uint32_t>(input_base_groups[row]) ^
        (static_cast<std::uint64_t>(end - begin) << 32));
    for (std::int64_t cursor = begin; cursor < end; ++cursor) {
      if (input_columns[cursor] < 0 || !std::isfinite(input_values[cursor])) {
        return -2;
      }
      std::uint32_t value_bits = 0;
      std::memcpy(&value_bits, input_values + cursor, sizeof(value_bits));
      const std::uint64_t word =
          static_cast<std::uint32_t>(input_columns[cursor]) |
          (static_cast<std::uint64_t>(value_bits) << 32);
      hash = mix_word(hash ^ mix_word(word));
    }
    if ((group_hashes.size() + 1) * 10 >= capacity * 7) {
      rehash();
    }
    std::size_t slot = static_cast<std::size_t>(hash) & mask;
    std::int32_t matched = -1;
    while (slots[slot] >= 0) {
      const std::int32_t candidate = slots[slot];
      if (slot_hashes[slot] == hash &&
          output_base_groups[candidate] == input_base_groups[row]) {
        const std::int64_t candidate_begin = output_row_offsets[candidate];
        const std::int64_t candidate_end = output_row_offsets[candidate + 1];
        const std::int64_t width = end - begin;
        if (candidate_end - candidate_begin == width &&
            std::memcmp(output_columns + candidate_begin,
                        input_columns + begin,
                        static_cast<std::size_t>(width) * sizeof(std::int32_t)) == 0 &&
            std::memcmp(output_values + candidate_begin,
                        input_values + begin,
                        static_cast<std::size_t>(width) * sizeof(float)) == 0) {
          matched = candidate;
          break;
        }
      }
      slot = (slot + 1) & mask;
    }
    if (matched >= 0) {
      output_weights[matched] += input_weights[row];
      continue;
    }
    const std::int32_t group = static_cast<std::int32_t>(groups++);
    output_base_groups[group] = input_base_groups[row];
    output_weights[group] = input_weights[row];
    const std::int64_t width = end - begin;
    if (width > 0) {
      std::memmove(output_columns + written_nnz, input_columns + begin,
                   static_cast<std::size_t>(width) * sizeof(std::int32_t));
      std::memmove(output_values + written_nnz, input_values + begin,
                   static_cast<std::size_t>(width) * sizeof(float));
    }
    written_nnz += width;
    output_row_offsets[groups] = written_nnz;
    group_hashes.push_back(hash);
    slots[slot] = group;
    slot_hashes[slot] = hash;
  }
  *output_nnz = written_nnz;
  return groups;
}

extern "C" std::int64_t certscr_fit_prepared_cone(
    const double* design,
    const std::int64_t rows,
    const std::int64_t event_rows,
    const std::int32_t columns,
    const double* event_weights,
    const double* grid_weights,
    const std::int32_t constrained_start,
    const std::int32_t likelihood,
    const std::int32_t max_iterations,
    const double tolerance,
    const double* initial_values,
    double* output_values,
    double* output_objective,
    double* output_kkt,
    std::int32_t* output_iterations) {
  if (design == nullptr || event_weights == nullptr || grid_weights == nullptr ||
      initial_values == nullptr || output_values == nullptr ||
      output_objective == nullptr || output_kkt == nullptr ||
      output_iterations == nullptr || rows < 1 || event_rows < 0 ||
      event_rows > rows || columns < 1 || constrained_start < 1 ||
      constrained_start > columns || (likelihood != 0 && likelihood != 1) ||
      max_iterations < 1 || !(tolerance > 0.0)) {
    return -1;
  }
  std::vector<double> values(initial_values, initial_values + columns);
  for (std::int32_t column = constrained_start; column < columns; ++column) {
    if (!std::isfinite(values[column])) {
      return -2;
    }
    values[column] = std::max(0.0, values[column]);
  }
  std::vector<double> gradient(columns, 0.0);
  std::vector<double> hessian(
      static_cast<std::size_t>(columns) * columns, 0.0);
  std::vector<double> projected(columns, 0.0);
  std::vector<double> direction(columns, 0.0);
  std::vector<double> trial(columns, 0.0);
  std::vector<std::int32_t> active;
  active.reserve(columns);
  double objective = std::numeric_limits<double>::infinity();
  double kkt = std::numeric_limits<double>::infinity();
  std::int32_t iteration = 0;
  bool converged = false;
  for (iteration = 1; iteration <= max_iterations; ++iteration) {
    if (!certscr_prepared_state(
            design, rows, event_rows, columns, event_weights, grid_weights,
            likelihood, values.data(), &objective, &gradient, &hessian)) {
      break;
    }
    double boundary_scale = 1.0;
    for (std::int32_t column = constrained_start; column < columns; ++column) {
      boundary_scale = std::max(boundary_scale, std::abs(values[column]));
    }
    const double boundary_tolerance =
        std::numeric_limits<double>::epsilon() *
        std::max(1, columns - constrained_start) * boundary_scale;
    kkt = 0.0;
    active.clear();
    for (std::int32_t column = 0; column < columns; ++column) {
      const bool constrained = column >= constrained_start;
      const bool at_boundary = constrained && values[column] <= boundary_tolerance;
      projected[column] = at_boundary
                              ? std::min(gradient[column], 0.0)
                              : gradient[column];
      const double fisher = std::max(
          hessian[static_cast<std::size_t>(column) * columns + column],
          std::numeric_limits<double>::min());
      kkt = std::max(kkt, std::abs(projected[column]) / std::sqrt(fisher));
      if (!constrained || !at_boundary || gradient[column] < -boundary_tolerance) {
        active.push_back(column);
      }
    }
    if (std::isfinite(kkt) && kkt <= tolerance) {
      converged = true;
      break;
    }
    bool newton = certscr_cholesky_direction(
        hessian, gradient, active, columns, &direction);
    if (!newton) {
      newton = certscr_rank_revealing_direction(
          hessian, gradient, active, columns, &direction);
    }
    if (!newton) {
      // The full-rank path dominates these small grouped models.  A singular
      // Fisher system uses an exactly feasible projected-gradient direction;
      // final acceptance still requires the ordinary host KKT certificate.
      double row_norm = 0.0;
      for (std::int32_t row = 0; row < columns; ++row) {
        double sum = 0.0;
        for (std::int32_t column = 0; column < columns; ++column) {
          sum += std::abs(
              hessian[static_cast<std::size_t>(row) * columns + column]);
        }
        row_norm = std::max(row_norm, sum);
      }
      const double inverse_norm = 1.0 / std::max(row_norm, 1.0e-8);
      for (std::int32_t column = 0; column < columns; ++column) {
        direction[column] = -projected[column] * inverse_norm;
      }
    }
    for (std::int32_t column = constrained_start; column < columns; ++column) {
      if (values[column] <= boundary_tolerance && direction[column] < 0.0) {
        direction[column] = 0.0;
      }
    }
    double slope = 0.0;
    for (std::int32_t column = 0; column < columns; ++column) {
      slope += gradient[column] * direction[column];
    }
    if (!std::isfinite(slope) || slope >= 0.0) {
      for (std::int32_t column = 0; column < columns; ++column) {
        direction[column] = -projected[column];
      }
      slope = 0.0;
      for (std::int32_t column = 0; column < columns; ++column) {
        slope += gradient[column] * direction[column];
      }
    }
    if (!std::isfinite(slope) || slope >= 0.0) {
      break;
    }
    double step = 1.0;
    for (std::int32_t column = constrained_start; column < columns; ++column) {
      if (direction[column] < 0.0 && values[column] > boundary_tolerance) {
        step = std::min(step, -values[column] / direction[column]);
      }
    }
    step = std::max(step, 1.0e-12);
    bool accepted = false;
    for (std::int32_t line = 0; line < 40; ++line) {
      double actual_slope = 0.0;
      for (std::int32_t column = 0; column < columns; ++column) {
        trial[column] = values[column] + step * direction[column];
        if (column >= constrained_start) {
          trial[column] = std::max(0.0, trial[column]);
        }
        actual_slope += gradient[column] * (trial[column] - values[column]);
      }
      double trial_objective = 0.0;
      if (certscr_prepared_state(
              design, rows, event_rows, columns, event_weights, grid_weights,
              likelihood, trial.data(), &trial_objective, nullptr, nullptr) &&
          trial_objective <= objective + 1.0e-4 * actual_slope) {
        values.swap(trial);
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if (!accepted) {
      break;
    }
  }
  // Recompute at the returned point so diagnostics never describe the state
  // before the last accepted line-search update.
  if (!certscr_prepared_state(
          design, rows, event_rows, columns, event_weights, grid_weights,
          likelihood, values.data(), &objective, &gradient, &hessian)) {
    return 1;
  }
  double boundary_scale = 1.0;
  for (std::int32_t column = constrained_start; column < columns; ++column) {
    boundary_scale = std::max(boundary_scale, std::abs(values[column]));
  }
  const double boundary_tolerance =
      std::numeric_limits<double>::epsilon() *
      std::max(1, columns - constrained_start) * boundary_scale;
  kkt = 0.0;
  for (std::int32_t column = 0; column < columns; ++column) {
    const double component =
        column >= constrained_start && values[column] <= boundary_tolerance
            ? std::min(gradient[column], 0.0)
            : gradient[column];
    const double fisher = std::max(
        hessian[static_cast<std::size_t>(column) * columns + column],
        std::numeric_limits<double>::min());
    kkt = std::max(kkt, std::abs(component) / std::sqrt(fisher));
  }
  std::copy(values.begin(), values.end(), output_values);
  *output_objective = objective;
  *output_kkt = kkt;
  *output_iterations = std::min(iteration, max_iterations);
  return (converged || (std::isfinite(kkt) && kkt <= tolerance)) ? 0 : 1;
}

extern "C" std::int64_t certscr_fit_sparse_delta_cone(
    const double* base_design,
    const std::int64_t base_event_rows,
    const std::int64_t base_grid_rows,
    const std::int32_t base_columns,
    const double* residual_event_weights,
    const double* residual_grid_weights,
    const std::int32_t* grid_base_groups,
    const std::int64_t* grid_row_offsets,
    const std::int32_t* grid_columns,
    const float* grid_values,
    const double* active_grid_weights,
    const std::int64_t active_grid_rows,
    const std::int64_t grid_nnz,
    const std::int32_t* event_base_groups,
    const std::int64_t* event_row_offsets,
    const std::int32_t* event_columns,
    const float* event_values,
    const double* active_event_weights,
    const std::int64_t active_event_rows,
    const std::int64_t event_nnz,
    const std::int32_t stored_delta_columns,
    const std::int32_t delta_columns,
    const std::int32_t constrained_start,
    const std::int32_t likelihood,
    const std::int32_t max_iterations,
    const double tolerance,
    const double* initial_values,
    double* output_values,
    double* output_objective,
    double* output_kkt,
    std::int32_t* output_iterations) {
  const std::int32_t columns = base_columns + delta_columns;
  if (base_design == nullptr || residual_event_weights == nullptr ||
      residual_grid_weights == nullptr || grid_base_groups == nullptr ||
      grid_row_offsets == nullptr || grid_columns == nullptr ||
      grid_values == nullptr || active_grid_weights == nullptr ||
      event_base_groups == nullptr || event_row_offsets == nullptr ||
      event_columns == nullptr || event_values == nullptr ||
      active_event_weights == nullptr || initial_values == nullptr ||
      output_values == nullptr || output_objective == nullptr ||
      output_kkt == nullptr || output_iterations == nullptr ||
      base_event_rows < 0 || base_grid_rows < 0 || base_columns < 1 ||
      active_grid_rows < 0 || active_event_rows < 0 || grid_nnz < 0 ||
      event_nnz < 0 || stored_delta_columns < 1 || delta_columns < 1 ||
      delta_columns > stored_delta_columns || constrained_start < base_columns ||
      constrained_start > columns || (likelihood != 0 && likelihood != 1) ||
      max_iterations < 1 || !(tolerance > 0.0) ||
      grid_row_offsets[0] != 0 ||
      grid_row_offsets[active_grid_rows] != grid_nnz ||
      event_row_offsets[0] != 0 ||
      event_row_offsets[active_event_rows] != event_nnz) {
    return -1;
  }
  for (std::int64_t cursor = 0; cursor < grid_nnz; ++cursor) {
    if (grid_columns[cursor] < 0 ||
        grid_columns[cursor] >= stored_delta_columns ||
        !std::isfinite(grid_values[cursor])) {
      return -2;
    }
  }
  for (std::int64_t cursor = 0; cursor < event_nnz; ++cursor) {
    if (event_columns[cursor] < 0 ||
        event_columns[cursor] >= stored_delta_columns ||
        !std::isfinite(event_values[cursor])) {
      return -2;
    }
  }

  std::vector<double> values(initial_values, initial_values + columns);
  for (std::int32_t column = 0; column < columns; ++column) {
    if (!std::isfinite(values[column])) {
      return -2;
    }
    if (column >= constrained_start) {
      values[column] = std::max(0.0, values[column]);
    }
  }
  std::vector<double> gradient(columns, 0.0);
  std::vector<double> hessian(
      static_cast<std::size_t>(columns) * columns, 0.0);
  std::vector<double> projected(columns, 0.0);
  std::vector<double> direction(columns, 0.0);
  std::vector<double> trial(columns, 0.0);
  std::vector<std::int32_t> active;
  active.reserve(columns);
  auto state = [&](const double* point, double* objective,
                   std::vector<double>* score,
                   std::vector<double>* fisher) -> bool {
    return certscr_sparse_delta_state(
        base_design, base_event_rows, base_grid_rows, base_columns,
        residual_event_weights, residual_grid_weights, grid_base_groups,
        grid_row_offsets, grid_columns, grid_values, active_grid_weights,
        active_grid_rows, event_base_groups, event_row_offsets, event_columns,
        event_values, active_event_weights, active_event_rows, delta_columns,
        likelihood, point, objective, score, fisher);
  };

  double objective = std::numeric_limits<double>::infinity();
  double kkt = std::numeric_limits<double>::infinity();
  std::int32_t iteration = 0;
  bool converged = false;
  for (iteration = 1; iteration <= max_iterations; ++iteration) {
    if (!state(values.data(), &objective, &gradient, &hessian)) {
      break;
    }
    double boundary_scale = 1.0;
    for (std::int32_t column = constrained_start; column < columns; ++column) {
      boundary_scale = std::max(boundary_scale, std::abs(values[column]));
    }
    const double boundary_tolerance =
        std::numeric_limits<double>::epsilon() *
        std::max(1, columns - constrained_start) * boundary_scale;
    kkt = 0.0;
    active.clear();
    for (std::int32_t column = 0; column < columns; ++column) {
      const bool constrained = column >= constrained_start;
      const bool at_boundary = constrained && values[column] <= boundary_tolerance;
      projected[column] = at_boundary
                              ? std::min(gradient[column], 0.0)
                              : gradient[column];
      const double diagonal = std::max(
          hessian[static_cast<std::size_t>(column) * columns + column],
          std::numeric_limits<double>::min());
      kkt = std::max(kkt, std::abs(projected[column]) / std::sqrt(diagonal));
      if (!constrained || !at_boundary || gradient[column] < -boundary_tolerance) {
        active.push_back(column);
      }
    }
    if (std::isfinite(kkt) && kkt <= tolerance) {
      converged = true;
      break;
    }
    bool newton = certscr_cholesky_direction(
        hessian, gradient, active, columns, &direction);
    if (!newton) {
      newton = certscr_rank_revealing_direction(
          hessian, gradient, active, columns, &direction);
    }
    if (!newton) {
      double row_norm = 0.0;
      for (std::int32_t row = 0; row < columns; ++row) {
        double sum = 0.0;
        for (std::int32_t column = 0; column < columns; ++column) {
          sum += std::abs(
              hessian[static_cast<std::size_t>(row) * columns + column]);
        }
        row_norm = std::max(row_norm, sum);
      }
      const double inverse_norm = 1.0 / std::max(row_norm, 1.0e-8);
      for (std::int32_t column = 0; column < columns; ++column) {
        direction[column] = -projected[column] * inverse_norm;
      }
    }
    for (std::int32_t column = constrained_start; column < columns; ++column) {
      if (values[column] <= boundary_tolerance && direction[column] < 0.0) {
        direction[column] = 0.0;
      }
    }
    double slope = 0.0;
    for (std::int32_t column = 0; column < columns; ++column) {
      slope += gradient[column] * direction[column];
    }
    if (!std::isfinite(slope) || slope >= 0.0) {
      for (std::int32_t column = 0; column < columns; ++column) {
        direction[column] = -projected[column];
      }
      slope = 0.0;
      for (std::int32_t column = 0; column < columns; ++column) {
        slope += gradient[column] * direction[column];
      }
    }
    if (!std::isfinite(slope) || slope >= 0.0) {
      break;
    }
    double step = 1.0;
    for (std::int32_t column = constrained_start; column < columns; ++column) {
      if (direction[column] < 0.0 && values[column] > boundary_tolerance) {
        step = std::min(step, -values[column] / direction[column]);
      }
    }
    step = std::max(step, 1.0e-12);
    bool accepted = false;
    for (std::int32_t line = 0; line < 40; ++line) {
      double actual_slope = 0.0;
      for (std::int32_t column = 0; column < columns; ++column) {
        trial[column] = values[column] + step * direction[column];
        if (column >= constrained_start) {
          trial[column] = std::max(0.0, trial[column]);
        }
        actual_slope += gradient[column] * (trial[column] - values[column]);
      }
      double trial_objective = 0.0;
      if (state(trial.data(), &trial_objective, nullptr, nullptr) &&
          trial_objective <= objective + 1.0e-4 * actual_slope) {
        values.swap(trial);
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if (!accepted) {
      double row_norm = 0.0;
      for (std::int32_t row = 0; row < columns; ++row) {
        double sum = 0.0;
        for (std::int32_t column = 0; column < columns; ++column) {
          sum += std::abs(
              hessian[static_cast<std::size_t>(row) * columns + column]);
        }
        row_norm = std::max(row_norm, sum);
      }
      double pg_step = 1.0 / std::max(row_norm, 1.0e-8);
      for (std::int32_t line = 0; line < 40; ++line) {
        for (std::int32_t column = 0; column < columns; ++column) {
          trial[column] = values[column] - pg_step * projected[column];
          if (column >= constrained_start) {
            trial[column] = std::max(0.0, trial[column]);
          }
        }
        double trial_objective = 0.0;
        if (state(trial.data(), &trial_objective, nullptr, nullptr) &&
            trial_objective < objective) {
          values.swap(trial);
          accepted = true;
          break;
        }
        pg_step *= 0.5;
      }
      if (!accepted) {
        break;
      }
    }
  }
  if (!state(values.data(), &objective, &gradient, &hessian)) {
    return 1;
  }
  double boundary_scale = 1.0;
  for (std::int32_t column = constrained_start; column < columns; ++column) {
    boundary_scale = std::max(boundary_scale, std::abs(values[column]));
  }
  const double boundary_tolerance =
      std::numeric_limits<double>::epsilon() *
      std::max(1, columns - constrained_start) * boundary_scale;
  kkt = 0.0;
  for (std::int32_t column = 0; column < columns; ++column) {
    const double component =
        column >= constrained_start && values[column] <= boundary_tolerance
            ? std::min(gradient[column], 0.0)
            : gradient[column];
    const double diagonal = std::max(
        hessian[static_cast<std::size_t>(column) * columns + column],
        std::numeric_limits<double>::min());
    kkt = std::max(kkt, std::abs(component) / std::sqrt(diagonal));
  }
  std::copy(values.begin(), values.end(), output_values);
  *output_objective = objective;
  *output_kkt = kkt;
  *output_iterations = std::min(iteration, max_iterations);
  return (converged || (std::isfinite(kkt) && kkt <= tolerance)) ? 0 : 1;
}

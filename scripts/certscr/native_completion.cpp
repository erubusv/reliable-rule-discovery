#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <numeric>
#include <vector>

namespace {
std::uint64_t hash_row(const float* values, const std::int64_t width) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::int64_t column = 0; column < width; ++column) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, values + column, sizeof(bits));
    hash ^= static_cast<std::uint64_t>(bits);
    hash *= 1099511628211ULL;
  }
  return hash;
}
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
  for (std::int32_t index = 0; index < array_count; ++index) {
    if (lengths[index] < 0 || (lengths[index] > 0 && arrays[index] == nullptr)) {
      return -2;
    }
    for (std::int64_t position = 1; position < lengths[index]; ++position) {
      if (arrays[index][position] <= arrays[index][position - 1]) {
        return -4;
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
    output[written++] = next_value;
    for (std::int32_t index = 0; index < array_count; ++index) {
      auto& position = positions[static_cast<std::size_t>(index)];
      if (position < lengths[index] && arrays[index][position] == next_value) {
        ++position;
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
      output_weights[group] += weights[row];
      continue;
    }
    // ``matrix`` and ``output_matrix`` may alias in the solver-only in-place
    // compaction path.  Stable grouping writes only at/before the current
    // input row, so memmove is safe and cannot overwrite a future row.
    std::memmove(output_matrix + groups * columns, input, row_bytes);
    output_weights[groups] = weights[row];
    collision_next.push_back(found == hash_head.end() ? -1 : found->second);
    hash_head[hash] = groups;
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

}  // namespace

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

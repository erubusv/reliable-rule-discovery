#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <omp.h>
#include <unordered_map>
#include <vector>

namespace {

bool double_buffer(PyObject* object, Py_buffer* view, int dimensions, bool writable) {
    const int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS |
                      (writable ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object, view, flags) != 0) return false;
    if (view->ndim != dimensions || view->itemsize != static_cast<Py_ssize_t>(sizeof(double)) ||
        view->format == nullptr || std::strcmp(view->format, "d") != 0) {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_ValueError, "expected a C-contiguous float64 buffer");
        return false;
    }
    return true;
}

bool int64_buffer(PyObject* object, Py_buffer* view, int dimensions, bool writable) {
    const int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS |
                      (writable ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object, view, flags) != 0) return false;
    if (view->ndim != dimensions || view->itemsize != static_cast<Py_ssize_t>(sizeof(std::int64_t))) {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_ValueError, "expected a C-contiguous int64 buffer");
        return false;
    }
    return true;
}

bool integer_lookup_buffer(PyObject* object, Py_buffer* view) {
    const int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
    if (PyObject_GetBuffer(object, view, flags) != 0) return false;
    if (view->ndim != 1 ||
        (view->itemsize != static_cast<Py_ssize_t>(sizeof(std::int32_t)) &&
         view->itemsize != static_cast<Py_ssize_t>(sizeof(std::int64_t)))) {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_ValueError, "expected a C-contiguous int32/int64 lookup");
        return false;
    }
    return true;
}

std::uint64_t row_hash(const double* row, std::int64_t columns) {
    // FNV-1a over canonical IEEE-754 words.  Signed zero is canonicalized
    // because +0 and -0 define the same design row.  Hash collisions are
    // always resolved by an exact elementwise comparison below.
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::int64_t column = 0; column < columns; ++column) {
        std::uint64_t bits = 0;
        if (row[column] != 0.0) std::memcpy(&bits, row + column, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
        hash ^= hash >> 32;
    }
    return hash;
}

bool equal_row(const double* left, const double* right, std::int64_t columns) {
    for (std::int64_t column = 0; column < columns; ++column) {
        if (left[column] != right[column]) return false;
    }
    return true;
}

PyObject* aggregate_design_rows(PyObject*, PyObject* args) {
    PyObject *x_obj, *exposure_obj, *noevent_obj, *event_obj;
    PyObject* groups_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOOO|O", &x_obj, &exposure_obj, &noevent_obj,
                          &event_obj, &groups_obj)) {
        return nullptr;
    }
    Py_buffer x{}, exposure{}, noevent{}, event{};
    Py_buffer groups{};
    if (!double_buffer(x_obj, &x, 2, true)) return nullptr;
    if (!double_buffer(exposure_obj, &exposure, 1, true)) {
        PyBuffer_Release(&x);
        return nullptr;
    }
    if (!double_buffer(noevent_obj, &noevent, 1, true)) {
        PyBuffer_Release(&x);
        PyBuffer_Release(&exposure);
        return nullptr;
    }
    if (!double_buffer(event_obj, &event, 1, true)) {
        PyBuffer_Release(&x);
        PyBuffer_Release(&exposure);
        PyBuffer_Release(&noevent);
        return nullptr;
    }
    const bool retain_groups = groups_obj != Py_None;
    if (retain_groups && !int64_buffer(groups_obj, &groups, 1, true)) {
        PyBuffer_Release(&x);
        PyBuffer_Release(&exposure);
        PyBuffer_Release(&noevent);
        PyBuffer_Release(&event);
        return nullptr;
    }
    const std::int64_t rows = x.shape[0], columns = x.shape[1];
    if (exposure.shape[0] != rows || noevent.shape[0] != rows ||
        event.shape[0] != rows || (retain_groups && groups.shape[0] != rows)) {
        PyErr_SetString(PyExc_ValueError, "design aggregation weight shape mismatch");
        PyBuffer_Release(&x);
        PyBuffer_Release(&exposure);
        PyBuffer_Release(&noevent);
        PyBuffer_Release(&event);
        if (retain_groups) PyBuffer_Release(&groups);
        return nullptr;
    }
    auto* xp = static_cast<double*>(x.buf);
    auto* ep = static_cast<double*>(exposure.buf);
    auto* np = static_cast<double*>(noevent.buf);
    auto* yp = static_cast<double*>(event.buf);
    auto* group = retain_groups ? static_cast<std::int64_t*>(groups.buf) : nullptr;
    std::int64_t output = 0;
    bool allocation_failed = false;
    Py_BEGIN_ALLOW_THREADS
    try {
        std::unordered_map<std::uint64_t, std::vector<std::int64_t>> buckets;
        buckets.reserve(static_cast<std::size_t>(std::min<std::int64_t>(rows, 262144)));
        for (std::int64_t input = 0; input < rows; ++input) {
            const double* row = xp + input * columns;
            auto& representatives = buckets[row_hash(row, columns)];
            std::int64_t matched = -1;
            for (const auto candidate : representatives) {
                if (equal_row(row, xp + candidate * columns, columns)) {
                    matched = candidate;
                    break;
                }
            }
            if (matched >= 0) {
                if (group != nullptr) group[input] = matched;
                ep[matched] += ep[input];
                np[matched] += np[input];
                yp[matched] += yp[input];
                continue;
            }
            if (group != nullptr) group[input] = output;
            if (output != input) {
                std::memmove(xp + output * columns, row,
                             static_cast<std::size_t>(columns) * sizeof(double));
                ep[output] = ep[input];
                np[output] = np[input];
                yp[output] = yp[input];
            }
            representatives.push_back(output);
            ++output;
        }
    } catch (const std::bad_alloc&) {
        allocation_failed = true;
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&x);
    PyBuffer_Release(&exposure);
    PyBuffer_Release(&noevent);
    PyBuffer_Release(&event);
    if (retain_groups) PyBuffer_Release(&groups);
    if (allocation_failed) return PyErr_NoMemory();
    return PyLong_FromLongLong(output);
}

PyObject* moments(PyObject*, PyObject* args) {
    PyObject *x_obj, *first_obj, *second_obj, *gradient_obj, *hessian_obj;
    if (!PyArg_ParseTuple(args, "OOOOO", &x_obj, &first_obj, &second_obj, &gradient_obj, &hessian_obj)) {
        return nullptr;
    }
    Py_buffer x{}, first{}, second{}, gradient{}, hessian{};
    if (!double_buffer(x_obj, &x, 2, false)) return nullptr;
    if (!double_buffer(first_obj, &first, 1, false)) { PyBuffer_Release(&x); return nullptr; }
    if (!double_buffer(second_obj, &second, 1, false)) { PyBuffer_Release(&x); PyBuffer_Release(&first); return nullptr; }
    if (!double_buffer(gradient_obj, &gradient, 1, true)) { PyBuffer_Release(&x); PyBuffer_Release(&first); PyBuffer_Release(&second); return nullptr; }
    if (!double_buffer(hessian_obj, &hessian, 2, true)) { PyBuffer_Release(&x); PyBuffer_Release(&first); PyBuffer_Release(&second); PyBuffer_Release(&gradient); return nullptr; }
    const std::int64_t rows = x.shape[0], columns = x.shape[1];
    const bool valid = first.shape[0] == rows && second.shape[0] == rows &&
                       gradient.shape[0] == columns && hessian.shape[0] == columns &&
                       hessian.shape[1] == columns;
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "moment buffer shape mismatch");
        PyBuffer_Release(&x); PyBuffer_Release(&first); PyBuffer_Release(&second);
        PyBuffer_Release(&gradient); PyBuffer_Release(&hessian);
        return nullptr;
    }
    const auto* xp = static_cast<const double*>(x.buf);
    const auto* fp = static_cast<const double*>(first.buf);
    const auto* sp = static_cast<const double*>(second.buf);
    auto* gp = static_cast<double*>(gradient.buf);
    auto* hp = static_cast<double*>(hessian.buf);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (std::int64_t j = 0; j < columns; ++j) {
        long double sum = 0.0L;
        for (std::int64_t i = 0; i < rows; ++i) {
            sum += static_cast<long double>(xp[i * columns + j]) * fp[i];
        }
        gp[j] = static_cast<double>(sum);
    }
    #pragma omp parallel for schedule(static)
    for (std::int64_t flat = 0; flat < columns * columns; ++flat) {
        const std::int64_t j = flat / columns, k = flat % columns;
        long double sum = 0.0L;
        for (std::int64_t i = 0; i < rows; ++i) {
            sum += static_cast<long double>(xp[i * columns + j]) * sp[i] *
                   xp[i * columns + k];
        }
        hp[flat] = static_cast<double>(sum);
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&x); PyBuffer_Release(&first); PyBuffer_Release(&second);
    PyBuffer_Release(&gradient); PyBuffer_Release(&hessian);
    Py_RETURN_NONE;
}

PyObject* nonnegative_quadratic_gains(PyObject*, PyObject* args) {
    PyObject *gradient_obj, *hessian_obj, *output_obj;
    if (!PyArg_ParseTuple(args, "OOO", &gradient_obj, &hessian_obj, &output_obj)) {
        return nullptr;
    }
    Py_buffer gradient{}, hessian{}, output{};
    if (!double_buffer(gradient_obj, &gradient, 2, false)) return nullptr;
    if (!double_buffer(hessian_obj, &hessian, 3, false)) {
        PyBuffer_Release(&gradient);
        return nullptr;
    }
    if (!double_buffer(output_obj, &output, 1, true)) {
        PyBuffer_Release(&gradient);
        PyBuffer_Release(&hessian);
        return nullptr;
    }
    const std::int64_t batches = gradient.shape[0], dimension = gradient.shape[1];
    const bool valid = hessian.shape[0] == batches &&
                       hessian.shape[1] == dimension &&
                       hessian.shape[2] == dimension &&
                       output.shape[0] == batches && dimension >= 1 && dimension <= 16;
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "quadratic gain buffer shape mismatch");
        PyBuffer_Release(&gradient);
        PyBuffer_Release(&hessian);
        PyBuffer_Release(&output);
        return nullptr;
    }
    const auto* gp = static_cast<const double*>(gradient.buf);
    const auto* hp = static_cast<const double*>(hessian.buf);
    auto* op = static_cast<double*>(output.buf);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        const double* g = gp + batch * dimension;
        const double* h = hp + batch * dimension * dimension;
        double best = 0.0;
        const std::uint64_t limit = 1ULL << dimension;
        for (std::uint64_t mask = 1; mask < limit; ++mask) {
            int active[16];
            int count = 0;
            for (int index = 0; index < dimension; ++index) {
                if (mask & (1ULL << index)) active[count++] = index;
            }
            double system[16][17]{};
            for (int row = 0; row < count; ++row) {
                for (int column = 0; column < count; ++column) {
                    system[row][column] =
                        0.5 * (h[active[row] * dimension + active[column]] +
                               h[active[column] * dimension + active[row]]);
                }
                system[row][count] = -g[active[row]];
            }
            bool nonsingular = true;
            for (int pivot = 0; pivot < count; ++pivot) {
                int selected = pivot;
                for (int row = pivot + 1; row < count; ++row) {
                    if (std::abs(system[row][pivot]) >
                        std::abs(system[selected][pivot])) selected = row;
                }
                if (std::abs(system[selected][pivot]) <= 1.0e-18) {
                    nonsingular = false;
                    break;
                }
                if (selected != pivot) {
                    for (int column = pivot; column <= count; ++column) {
                        std::swap(system[pivot][column], system[selected][column]);
                    }
                }
                const double scale = system[pivot][pivot];
                for (int column = pivot; column <= count; ++column) {
                    system[pivot][column] /= scale;
                }
                for (int row = 0; row < count; ++row) {
                    if (row == pivot) continue;
                    const double factor = system[row][pivot];
                    for (int column = pivot; column <= count; ++column) {
                        system[row][column] -= factor * system[pivot][column];
                    }
                }
            }
            if (!nonsingular) continue;
            double delta[16]{};
            bool feasible = true;
            for (int row = 0; row < count; ++row) {
                const double value = system[row][count];
                if (value < -1.0e-10) {
                    feasible = false;
                    break;
                }
                delta[active[row]] = std::max(0.0, value);
            }
            if (!feasible) continue;
            for (int index = 0; index < dimension && feasible; ++index) {
                if (mask & (1ULL << index)) continue;
                double stationarity = g[index];
                for (int column = 0; column < dimension; ++column) {
                    stationarity += h[index * dimension + column] * delta[column];
                }
                if (stationarity < -1.0e-8) feasible = false;
            }
            if (!feasible) continue;
            double linear = 0.0, quadratic = 0.0;
            for (int row = 0; row < dimension; ++row) {
                linear += g[row] * delta[row];
                for (int column = 0; column < dimension; ++column) {
                    quadratic += delta[row] * h[row * dimension + column] *
                                 delta[column];
                }
            }
            best = std::max(best, -linear - 0.5 * quadratic);
        }
        op[batch] = best;
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&gradient);
    PyBuffer_Release(&hessian);
    PyBuffer_Release(&output);
    Py_RETURN_NONE;
}

PyObject* set_num_threads(PyObject*, PyObject* args) {
    int count;
    if (!PyArg_ParseTuple(args, "i", &count)) return nullptr;
    if (count < 1) {
        PyErr_SetString(PyExc_ValueError, "OpenMP thread count must be positive");
        return nullptr;
    }
    omp_set_dynamic(0);
    omp_set_num_threads(count);
    Py_RETURN_NONE;
}

PyObject* future_rows(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *spans_obj, *starts_obj, *ends_obj,
             *offsets_obj, *rows_obj, *row_spans_obj;
    int window, horizon;
    if (!PyArg_ParseTuple(args, "OOOOOOiiOO", &entities_obj, &times_obj, &spans_obj,
                          &starts_obj, &ends_obj, &offsets_obj, &window, &horizon,
                          &rows_obj, &row_spans_obj)) return nullptr;
    Py_buffer entities{}, times{}, spans{}, starts{}, ends{}, offsets{}, rows{}, row_spans{};
    if (!int64_buffer(entities_obj, &entities, 1, false)) return nullptr;
    if (!int64_buffer(times_obj, &times, 1, false)) { PyBuffer_Release(&entities); return nullptr; }
    if (!int64_buffer(spans_obj, &spans, 1, false)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); return nullptr;
    }
    if (!int64_buffer(starts_obj, &starts, 1, false)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&spans); return nullptr;
    }
    if (!int64_buffer(ends_obj, &ends, 1, false)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&spans);
        PyBuffer_Release(&starts); return nullptr;
    }
    if (!int64_buffer(offsets_obj, &offsets, 1, false)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&spans);
        PyBuffer_Release(&starts); PyBuffer_Release(&ends); return nullptr;
    }
    if (!int64_buffer(rows_obj, &rows, 1, true)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&spans);
        PyBuffer_Release(&starts); PyBuffer_Release(&ends); PyBuffer_Release(&offsets); return nullptr;
    }
    if (!int64_buffer(row_spans_obj, &row_spans, 1, true)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&spans);
        PyBuffer_Release(&starts); PyBuffer_Release(&ends); PyBuffer_Release(&offsets);
        PyBuffer_Release(&rows); return nullptr;
    }
    const auto count = entities.shape[0], entity_count = starts.shape[0];
    bool valid = horizon >= 1 && times.shape[0] == count && spans.shape[0] == count &&
                 ends.shape[0] == entity_count && offsets.shape[0] == entity_count + 1 &&
                 rows.shape[0] >= count * static_cast<std::int64_t>(horizon) &&
                 row_spans.shape[0] >= count * static_cast<std::int64_t>(horizon);
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "future-row buffer shape mismatch");
        goto future_fail;
    }
    {
        const auto* ep = static_cast<const std::int64_t*>(entities.buf);
        const auto* tp = static_cast<const std::int64_t*>(times.buf);
        const auto* sp = static_cast<const std::int64_t*>(spans.buf);
        const auto* startp = static_cast<const std::int64_t*>(starts.buf);
        const auto* endp = static_cast<const std::int64_t*>(ends.buf);
        const auto* offsetp = static_cast<const std::int64_t*>(offsets.buf);
        auto* output_rows = static_cast<std::int64_t*>(rows.buf);
        auto* output_spans = static_cast<std::int64_t*>(row_spans.buf);
        std::int64_t output = 0;
        Py_BEGIN_ALLOW_THREADS
        for (std::int64_t i = 0; i < count; ++i) {
            if (sp[i] > window) continue;
            const auto entity = ep[i];
            if (entity < 0 || entity >= entity_count) continue;
            const auto maximum = std::min<std::int64_t>(horizon, endp[entity] - tp[i]);
            const auto base = offsetp[entity] + tp[i] - startp[entity];
            for (std::int64_t lag = 1; lag <= maximum; ++lag) {
                output_rows[output] = base + lag;
                output_spans[output] = sp[i];
                ++output;
            }
        }
        Py_END_ALLOW_THREADS
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&spans);
        PyBuffer_Release(&starts); PyBuffer_Release(&ends); PyBuffer_Release(&offsets);
        PyBuffer_Release(&rows); PyBuffer_Release(&row_spans);
        return PyLong_FromLongLong(output);
    }
future_fail:
    PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&spans);
    PyBuffer_Release(&starts); PyBuffer_Release(&ends); PyBuffer_Release(&offsets);
    PyBuffer_Release(&rows); PyBuffer_Release(&row_spans);
    return nullptr;
}

PyObject* accumulate_kernel(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *starts_obj, *ends_obj, *offsets_obj,
             *basis_obj, *lookup_obj, *accumulator_obj;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &entities_obj, &times_obj,
                          &starts_obj, &ends_obj, &offsets_obj, &basis_obj,
                          &lookup_obj, &accumulator_obj)) return nullptr;
    Py_buffer entities{}, times{}, starts{}, ends{}, offsets{}, basis{}, lookup{},
              accumulator{};
    int acquired = 0;
    const auto release = [&]() {
        if (acquired >= 8) PyBuffer_Release(&accumulator);
        if (acquired >= 7) PyBuffer_Release(&lookup);
        if (acquired >= 6) PyBuffer_Release(&basis);
        if (acquired >= 5) PyBuffer_Release(&offsets);
        if (acquired >= 4) PyBuffer_Release(&ends);
        if (acquired >= 3) PyBuffer_Release(&starts);
        if (acquired >= 2) PyBuffer_Release(&times);
        if (acquired >= 1) PyBuffer_Release(&entities);
    };
    if (!int64_buffer(entities_obj, &entities, 1, false)) return nullptr;
    ++acquired;
    if (!int64_buffer(times_obj, &times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(starts_obj, &starts, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(ends_obj, &ends, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(offsets_obj, &offsets, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(basis_obj, &basis, 2, false)) { release(); return nullptr; }
    ++acquired;
    if (!integer_lookup_buffer(lookup_obj, &lookup)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(accumulator_obj, &accumulator, 2, true)) {
        release(); return nullptr;
    }
    ++acquired;
    const auto count = entities.shape[0], entity_count = starts.shape[0];
    const auto knots = basis.shape[0], lag = basis.shape[1];
    const bool valid = times.shape[0] == count && ends.shape[0] == entity_count &&
                       offsets.shape[0] == entity_count + 1 &&
                       accumulator.shape[0] >= 0 &&
                       accumulator.shape[1] == knots;
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "kernel accumulator shape mismatch");
        release();
        return nullptr;
    }
    const auto* ep = static_cast<const std::int64_t*>(entities.buf);
    const auto* tp = static_cast<const std::int64_t*>(times.buf);
    const auto* startp = static_cast<const std::int64_t*>(starts.buf);
    const auto* endp = static_cast<const std::int64_t*>(ends.buf);
    const auto* offsetp = static_cast<const std::int64_t*>(offsets.buf);
    const auto* bp = static_cast<const double*>(basis.buf);
    const auto* lookp64 = lookup.itemsize == static_cast<Py_ssize_t>(sizeof(std::int64_t))
                              ? static_cast<const std::int64_t*>(lookup.buf)
                              : nullptr;
    const auto* lookp32 = lookup.itemsize == static_cast<Py_ssize_t>(sizeof(std::int32_t))
                              ? static_cast<const std::int32_t*>(lookup.buf)
                              : nullptr;
    auto* out = static_cast<double*>(accumulator.buf);
    bool invalid = false;
    Py_BEGIN_ALLOW_THREADS
    for (std::int64_t i = 0; i < count && !invalid; ++i) {
        const auto entity = ep[i];
        if (entity < 0 || entity >= entity_count) {
            invalid = true;
            break;
        }
        const auto maximum = std::min<std::int64_t>(lag, endp[entity] - tp[i]);
        const auto base = offsetp[entity] + tp[i] - startp[entity];
        for (std::int64_t l = 1; l <= maximum; ++l) {
            const auto row = base + l;
            if (row < 0 || row >= lookup.shape[0]) {
                invalid = true;
                break;
            }
            const auto position = lookp64 ? lookp64[row] : lookp32[row];
            if (position < 0 || position >= accumulator.shape[0]) {
                invalid = true;
                break;
            }
            for (std::int64_t k = 0; k < knots; ++k) {
                out[position * knots + k] += bp[k * lag + l - 1];
            }
        }
    }
    Py_END_ALLOW_THREADS
    if (invalid) {
        PyErr_SetString(PyExc_ValueError, "kernel row missing from accumulator lookup");
        release();
        return nullptr;
    }
    release();
    Py_RETURN_NONE;
}

PyObject* fill_candidate_batch(PyObject*, PyObject* args) {
    PyObject *destination_obj, *maximum_rows_obj, *lookup_obj, *rows_list,
             *values_list, *batch_obj, *column_obj, *start_obj;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &destination_obj, &maximum_rows_obj,
                          &lookup_obj, &rows_list, &values_list, &batch_obj,
                          &column_obj, &start_obj))
        return nullptr;
    Py_buffer destination{}, maximum_rows{}, lookup{}, batch{}, column{};
    if (!double_buffer(destination_obj, &destination, 3, true)) return nullptr;
    if (!int64_buffer(maximum_rows_obj, &maximum_rows, 1, false)) {
        PyBuffer_Release(&destination);
        return nullptr;
    }
    if (!integer_lookup_buffer(lookup_obj, &lookup)) {
        PyBuffer_Release(&destination);
        PyBuffer_Release(&maximum_rows);
        return nullptr;
    }
    if (!int64_buffer(batch_obj, &batch, 1, false)) {
        PyBuffer_Release(&destination); PyBuffer_Release(&maximum_rows);
        PyBuffer_Release(&lookup); return nullptr;
    }
    if (!int64_buffer(column_obj, &column, 1, false)) {
        PyBuffer_Release(&destination); PyBuffer_Release(&maximum_rows);
        PyBuffer_Release(&lookup); PyBuffer_Release(&batch); return nullptr;
    }
    const auto start = PyLong_AsLongLong(start_obj);
    const auto count = PySequence_Size(rows_list);
    const bool valid = start >= 0 && count >= 0 &&
        PySequence_Size(values_list) == count && batch.shape[0] == count &&
        column.shape[0] == count && destination.shape[1] > 0 &&
        start + destination.shape[1] <= maximum_rows.shape[0];
    if (PyErr_Occurred() || !valid) {
        PyErr_SetString(PyExc_ValueError, "candidate fill metadata mismatch");
        PyBuffer_Release(&destination); PyBuffer_Release(&maximum_rows);
        PyBuffer_Release(&lookup); PyBuffer_Release(&batch); PyBuffer_Release(&column);
        return nullptr;
    }
    std::vector<Py_buffer> row_views(static_cast<std::size_t>(count));
    std::vector<Py_buffer> value_views(static_cast<std::size_t>(count));
    std::int64_t acquired = 0;
    for (std::int64_t item = 0; item < count; ++item) {
        PyObject* rows_obj = PySequence_GetItem(rows_list, item);
        PyObject* values_obj = PySequence_GetItem(values_list, item);
        const bool row_acquired = rows_obj &&
            int64_buffer(rows_obj, &row_views[item], 1, false);
        const bool value_acquired = row_acquired && values_obj &&
            double_buffer(values_obj, &value_views[item], 2, false);
        if (!row_acquired || !value_acquired) {
            if (row_acquired) PyBuffer_Release(&row_views[item]);
            Py_XDECREF(rows_obj); Py_XDECREF(values_obj);
            for (std::int64_t previous = 0; previous < acquired; ++previous) {
                PyBuffer_Release(&row_views[previous]);
                PyBuffer_Release(&value_views[previous]);
            }
            PyBuffer_Release(&destination); PyBuffer_Release(&maximum_rows);
            PyBuffer_Release(&lookup); PyBuffer_Release(&batch);
            PyBuffer_Release(&column);
            return nullptr;
        }
        Py_DECREF(rows_obj); Py_DECREF(values_obj);
        ++acquired;
        if (row_views[item].shape[0] != value_views[item].shape[0]) {
            PyErr_SetString(PyExc_ValueError, "candidate block row/value mismatch");
            for (std::int64_t previous = 0; previous < acquired; ++previous) {
                PyBuffer_Release(&row_views[previous]);
                PyBuffer_Release(&value_views[previous]);
            }
            PyBuffer_Release(&destination); PyBuffer_Release(&maximum_rows);
            PyBuffer_Release(&lookup); PyBuffer_Release(&batch);
            PyBuffer_Release(&column);
            return nullptr;
        }
    }
    const auto* maximum = static_cast<const std::int64_t*>(maximum_rows.buf);
    const auto* batchp = static_cast<const std::int64_t*>(batch.buf);
    const auto* columnp = static_cast<const std::int64_t*>(column.buf);
    const auto* lookup64 = lookup.itemsize == static_cast<Py_ssize_t>(sizeof(std::int64_t))
                               ? static_cast<const std::int64_t*>(lookup.buf)
                               : nullptr;
    const auto* lookup32 = lookup.itemsize == static_cast<Py_ssize_t>(sizeof(std::int32_t))
                               ? static_cast<const std::int32_t*>(lookup.buf)
                               : nullptr;
    auto* output = static_cast<double*>(destination.buf);
    const auto batches = destination.shape[0], rows = destination.shape[1],
               dimensions = destination.shape[2];
    const auto lower_row = maximum[start];
    const auto upper_row = maximum[start + rows - 1];
    bool invalid = false;
    Py_BEGIN_ALLOW_THREADS
    for (std::int64_t item = 0; item < count && !invalid; ++item) {
        const auto* block_rows = static_cast<const std::int64_t*>(row_views[item].buf);
        const auto* block_values = static_cast<const double*>(value_views[item].buf);
        const auto block_count = row_views[item].shape[0];
        const auto width = value_views[item].shape[1];
        const auto left = std::lower_bound(block_rows, block_rows + block_count,
                                           lower_row) - block_rows;
        const auto right = std::upper_bound(block_rows, block_rows + block_count,
                                            upper_row) - block_rows;
        if (batchp[item] < 0 || batchp[item] >= batches || columnp[item] < 0 ||
            columnp[item] + width > dimensions) {
            invalid = true;
            break;
        }
        for (std::int64_t index = left; index < right; ++index) {
            const auto raw_row = block_rows[index];
            if (raw_row < 0 || raw_row >= lookup.shape[0]) {
                invalid = true;
                break;
            }
            const auto global = lookup64 ? lookup64[raw_row] : lookup32[raw_row];
            const auto local = global - start;
            if (local < 0 || local >= rows) {
                invalid = true;
                break;
            }
            std::memcpy(
                output + (batchp[item] * rows + local) * dimensions + columnp[item],
                block_values + index * width,
                static_cast<std::size_t>(width) * sizeof(double));
        }
    }
    Py_END_ALLOW_THREADS
    for (std::int64_t item = 0; item < acquired; ++item) {
        PyBuffer_Release(&row_views[item]);
        PyBuffer_Release(&value_views[item]);
    }
    PyBuffer_Release(&destination); PyBuffer_Release(&maximum_rows);
    PyBuffer_Release(&lookup); PyBuffer_Release(&batch); PyBuffer_Release(&column);
    if (invalid) {
        PyErr_SetString(PyExc_ValueError, "candidate block is outside its tile");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* sparse_joint_moments(PyObject*, PyObject* args) {
    PyObject *rows_list, *values_list, *lookup_obj, *first_obj, *second_obj,
             *gradient_obj, *hessian_obj, *cross_obj;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &rows_list, &values_list,
                          &lookup_obj, &first_obj, &second_obj, &gradient_obj,
                          &hessian_obj, &cross_obj))
        return nullptr;
    Py_buffer lookup{}, first{}, second{}, gradient{}, hessian{}, cross{};
    if (!integer_lookup_buffer(lookup_obj, &lookup)) return nullptr;
    if (!double_buffer(first_obj, &first, 1, false)) {
        PyBuffer_Release(&lookup); return nullptr;
    }
    if (!double_buffer(second_obj, &second, 1, false)) {
        PyBuffer_Release(&lookup); PyBuffer_Release(&first); return nullptr;
    }
    if (!double_buffer(gradient_obj, &gradient, 1, true)) {
        PyBuffer_Release(&lookup); PyBuffer_Release(&first);
        PyBuffer_Release(&second); return nullptr;
    }
    if (!double_buffer(hessian_obj, &hessian, 2, true)) {
        PyBuffer_Release(&lookup); PyBuffer_Release(&first);
        PyBuffer_Release(&second); PyBuffer_Release(&gradient); return nullptr;
    }
    if (!double_buffer(cross_obj, &cross, 1, true)) {
        PyBuffer_Release(&lookup); PyBuffer_Release(&first);
        PyBuffer_Release(&second); PyBuffer_Release(&gradient);
        PyBuffer_Release(&hessian); return nullptr;
    }
    const auto block_count = PySequence_Size(rows_list);
    if (block_count < 1 || PySequence_Size(values_list) != block_count ||
        first.shape[0] != second.shape[0]) {
        PyErr_SetString(PyExc_ValueError, "invalid sparse joint block metadata");
        goto sparse_joint_fail;
    }
    {
        std::vector<Py_buffer> row_views(static_cast<std::size_t>(block_count));
        std::vector<Py_buffer> value_views(static_cast<std::size_t>(block_count));
        std::int64_t acquired = 0, width = -1;
        for (std::int64_t block = 0; block < block_count; ++block) {
            PyObject* rows_obj = PySequence_GetItem(rows_list, block);
            PyObject* values_obj = PySequence_GetItem(values_list, block);
            const bool row_acquired = rows_obj &&
                int64_buffer(rows_obj, &row_views[block], 1, false);
            const bool value_acquired = row_acquired && values_obj &&
                double_buffer(values_obj, &value_views[block], 2, false);
            Py_XDECREF(rows_obj); Py_XDECREF(values_obj);
            if (!row_acquired || !value_acquired) {
                if (row_acquired) PyBuffer_Release(&row_views[block]);
                for (std::int64_t previous = 0; previous < acquired; ++previous) {
                    PyBuffer_Release(&row_views[previous]);
                    PyBuffer_Release(&value_views[previous]);
                }
                goto sparse_joint_fail;
            }
            ++acquired;
            if (row_views[block].shape[0] != value_views[block].shape[0] ||
                (width >= 0 && value_views[block].shape[1] != width)) {
                PyErr_SetString(PyExc_ValueError, "inconsistent sparse joint block");
                for (std::int64_t previous = 0; previous < acquired; ++previous) {
                    PyBuffer_Release(&row_views[previous]);
                    PyBuffer_Release(&value_views[previous]);
                }
                goto sparse_joint_fail;
            }
            width = value_views[block].shape[1];
        }
        const auto dimension = block_count * width;
        if (gradient.shape[0] != dimension || cross.shape[0] != dimension ||
            hessian.shape[0] != dimension || hessian.shape[1] != dimension) {
            PyErr_SetString(PyExc_ValueError, "sparse joint output shape mismatch");
            for (std::int64_t block = 0; block < acquired; ++block) {
                PyBuffer_Release(&row_views[block]);
                PyBuffer_Release(&value_views[block]);
            }
            goto sparse_joint_fail;
        }
        const auto* lookup64 = lookup.itemsize == static_cast<Py_ssize_t>(sizeof(std::int64_t))
                                   ? static_cast<const std::int64_t*>(lookup.buf)
                                   : nullptr;
        const auto* lookup32 = lookup.itemsize == static_cast<Py_ssize_t>(sizeof(std::int32_t))
                                   ? static_cast<const std::int32_t*>(lookup.buf)
                                   : nullptr;
        const auto* firstp = static_cast<const double*>(first.buf);
        const auto* secondp = static_cast<const double*>(second.buf);
        auto* gp = static_cast<double*>(gradient.buf);
        auto* hp = static_cast<double*>(hessian.buf);
        auto* cp = static_cast<double*>(cross.buf);
        std::fill(gp, gp + dimension, 0.0);
        std::fill(hp, hp + dimension * dimension, 0.0);
        std::fill(cp, cp + dimension, 0.0);
        bool invalid = false;
        Py_BEGIN_ALLOW_THREADS
        for (std::int64_t block = 0; block < block_count && !invalid; ++block) {
            const auto* rows = static_cast<const std::int64_t*>(row_views[block].buf);
            const auto* values = static_cast<const double*>(value_views[block].buf);
            for (std::int64_t index = 0; index < row_views[block].shape[0]; ++index) {
                const auto raw = rows[index];
                if (raw < 0 || raw >= lookup.shape[0]) { invalid = true; break; }
                const auto position = lookup64 ? lookup64[raw] : lookup32[raw];
                if (position < 0 || position >= first.shape[0]) { invalid = true; break; }
                for (std::int64_t knot = 0; knot < width; ++knot) {
                    const auto column = block * width + knot;
                    const auto value = values[index * width + knot];
                    gp[column] += value * firstp[position];
                    cp[column] += value * secondp[position];
                }
            }
        }
        for (std::int64_t left_block = 0;
             left_block < block_count && !invalid; ++left_block) {
            const auto* left_rows = static_cast<const std::int64_t*>(row_views[left_block].buf);
            const auto* left_values = static_cast<const double*>(value_views[left_block].buf);
            for (std::int64_t right_block = left_block;
                 right_block < block_count && !invalid; ++right_block) {
                const auto* right_rows = static_cast<const std::int64_t*>(row_views[right_block].buf);
                const auto* right_values = static_cast<const double*>(value_views[right_block].buf);
                std::int64_t left_index = 0, right_index = 0;
                while (left_index < row_views[left_block].shape[0] &&
                       right_index < row_views[right_block].shape[0]) {
                    const auto left_row = left_rows[left_index];
                    const auto right_row = right_rows[right_index];
                    if (left_row < right_row) { ++left_index; continue; }
                    if (right_row < left_row) { ++right_index; continue; }
                    if (left_row < 0 || left_row >= lookup.shape[0]) {
                        invalid = true;
                        break;
                    }
                    const auto position = lookup64 ? lookup64[left_row] : lookup32[left_row];
                    if (position < 0 || position >= second.shape[0]) {
                        invalid = true;
                        break;
                    }
                    const auto weight = secondp[position];
                    for (std::int64_t left_knot = 0; left_knot < width; ++left_knot) {
                        const auto left_column = left_block * width + left_knot;
                        const auto left_value = left_values[left_index * width + left_knot];
                        const auto right_start = left_block == right_block
                            ? left_knot : 0;
                        for (std::int64_t right_knot = right_start;
                             right_knot < width; ++right_knot) {
                            const auto right_column = right_block * width + right_knot;
                            const auto value = left_value * weight *
                                right_values[right_index * width + right_knot];
                            hp[left_column * dimension + right_column] += value;
                            if (left_column != right_column)
                                hp[right_column * dimension + left_column] += value;
                        }
                    }
                    ++left_index;
                    ++right_index;
                }
            }
        }
        Py_END_ALLOW_THREADS
        for (std::int64_t block = 0; block < acquired; ++block) {
            PyBuffer_Release(&row_views[block]);
            PyBuffer_Release(&value_views[block]);
        }
        if (invalid) {
            PyErr_SetString(PyExc_ValueError, "sparse joint row lookup failed");
            goto sparse_joint_fail;
        }
    }
    PyBuffer_Release(&lookup); PyBuffer_Release(&first); PyBuffer_Release(&second);
    PyBuffer_Release(&gradient); PyBuffer_Release(&hessian); PyBuffer_Release(&cross);
    Py_RETURN_NONE;
sparse_joint_fail:
    PyBuffer_Release(&lookup); PyBuffer_Release(&first); PyBuffer_Release(&second);
    PyBuffer_Release(&gradient); PyBuffer_Release(&hessian); PyBuffer_Release(&cross);
    return nullptr;
}

PyObject* kernel_contributions(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *spans_obj, *starts_obj, *ends_obj,
             *offsets_obj, *basis_obj, *rows_obj, *values_obj;
    int window;
    if (!PyArg_ParseTuple(args, "OOOOOOOiOO", &entities_obj, &times_obj, &spans_obj,
                          &starts_obj, &ends_obj, &offsets_obj, &basis_obj, &window,
                          &rows_obj, &values_obj)) return nullptr;
    Py_buffer entities{}, times{}, spans{}, starts{}, ends{}, offsets{}, basis{}, rows{}, values{};
    int acquired = 0;
    const auto release_acquired = [&]() {
        if (acquired >= 8) PyBuffer_Release(&rows);
        if (acquired >= 7) PyBuffer_Release(&basis);
        if (acquired >= 6) PyBuffer_Release(&offsets);
        if (acquired >= 5) PyBuffer_Release(&ends);
        if (acquired >= 4) PyBuffer_Release(&starts);
        if (acquired >= 3) PyBuffer_Release(&spans);
        if (acquired >= 2) PyBuffer_Release(&times);
        if (acquired >= 1) PyBuffer_Release(&entities);
    };
    if (!int64_buffer(entities_obj, &entities, 1, false)) return nullptr;
    ++acquired;
    if (!int64_buffer(times_obj, &times, 1, false)) { release_acquired(); return nullptr; }
    ++acquired;
    if (!int64_buffer(spans_obj, &spans, 1, false)) { release_acquired(); return nullptr; }
    ++acquired;
    if (!int64_buffer(starts_obj, &starts, 1, false)) { release_acquired(); return nullptr; }
    ++acquired;
    if (!int64_buffer(ends_obj, &ends, 1, false)) { release_acquired(); return nullptr; }
    ++acquired;
    if (!int64_buffer(offsets_obj, &offsets, 1, false)) { release_acquired(); return nullptr; }
    ++acquired;
    if (!double_buffer(basis_obj, &basis, 2, false)) { release_acquired(); return nullptr; }
    ++acquired;
    if (!int64_buffer(rows_obj, &rows, 1, true)) { release_acquired(); return nullptr; }
    ++acquired;
    if (!double_buffer(values_obj, &values, 2, true)) { release_acquired(); return nullptr; }
    ++acquired;
    const auto count = entities.shape[0];
    const auto knots = basis.shape[0];
    const auto lag = basis.shape[1];
    const auto entity_count = starts.shape[0];
    if (times.shape[0] != count || spans.shape[0] != count || ends.shape[0] != entity_count ||
        offsets.shape[0] != entity_count + 1 || rows.shape[0] < count * lag ||
        values.shape[0] < count * lag || values.shape[1] != knots) {
        PyErr_SetString(PyExc_ValueError, "kernel contribution buffer shape mismatch");
        goto fail;
    }
    {
        const auto* ep = static_cast<const std::int64_t*>(entities.buf);
        const auto* tp = static_cast<const std::int64_t*>(times.buf);
        const auto* sp = static_cast<const std::int64_t*>(spans.buf);
        const auto* startp = static_cast<const std::int64_t*>(starts.buf);
        const auto* endp = static_cast<const std::int64_t*>(ends.buf);
        const auto* offsetp = static_cast<const std::int64_t*>(offsets.buf);
        const auto* bp = static_cast<const double*>(basis.buf);
        auto* rp = static_cast<std::int64_t*>(rows.buf);
        auto* vp = static_cast<double*>(values.buf);
        for (std::int64_t i = 0; i < count; ++i) {
            if (ep[i] < 0 || ep[i] >= entity_count) {
                PyErr_SetString(PyExc_ValueError, "completion entity is out of range");
                goto fail;
            }
        }
        std::int64_t output = 0;
        Py_BEGIN_ALLOW_THREADS
        for (std::int64_t i = 0; i < count; ++i) {
            if (sp[i] > window) continue;
            const auto entity = ep[i];
            const auto maximum = std::min<std::int64_t>(lag, endp[entity] - tp[i]);
            const auto base = offsetp[entity] + tp[i] - startp[entity];
            for (std::int64_t l = 1; l <= maximum; ++l) {
                rp[output] = base + l;
                for (std::int64_t k = 0; k < knots; ++k) vp[output * knots + k] = bp[k * lag + l - 1];
                ++output;
            }
        }
        Py_END_ALLOW_THREADS
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&spans);
        PyBuffer_Release(&starts); PyBuffer_Release(&ends); PyBuffer_Release(&offsets);
        PyBuffer_Release(&basis); PyBuffer_Release(&rows); PyBuffer_Release(&values);
        return PyLong_FromLongLong(output);
    }
fail:
    PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&spans);
    PyBuffer_Release(&starts); PyBuffer_Release(&ends); PyBuffer_Release(&offsets);
    PyBuffer_Release(&basis); PyBuffer_Release(&rows); PyBuffer_Release(&values);
    return nullptr;
}

PyObject* completion_events(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *offsets_obj, *output_entities_obj,
             *output_times_obj, *output_spans_obj;
    if (!PyArg_ParseTuple(args, "OOOOOO", &entities_obj, &times_obj, &offsets_obj,
                          &output_entities_obj, &output_times_obj, &output_spans_obj)) {
        return nullptr;
    }
    Py_buffer entities{}, times{}, offsets{}, output_entities{}, output_times{}, output_spans{};
    if (!int64_buffer(entities_obj, &entities, 1, false)) return nullptr;
    if (!int64_buffer(times_obj, &times, 1, false)) { PyBuffer_Release(&entities); return nullptr; }
    if (!int64_buffer(offsets_obj, &offsets, 1, false)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); return nullptr;
    }
    if (!int64_buffer(output_entities_obj, &output_entities, 1, true)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&offsets); return nullptr;
    }
    if (!int64_buffer(output_times_obj, &output_times, 1, true)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&offsets);
        PyBuffer_Release(&output_entities); return nullptr;
    }
    if (!int64_buffer(output_spans_obj, &output_spans, 1, true)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&offsets);
        PyBuffer_Release(&output_entities); PyBuffer_Release(&output_times); return nullptr;
    }
    const auto event_count = entities.shape[0];
    const auto source_count = offsets.shape[0] - 1;
    bool valid = times.shape[0] == event_count && source_count >= 1 && source_count <= 3 &&
                 output_entities.shape[0] >= event_count && output_times.shape[0] >= event_count &&
                 output_spans.shape[0] >= event_count;
    const auto* ep = static_cast<const std::int64_t*>(entities.buf);
    const auto* tp = static_cast<const std::int64_t*>(times.buf);
    const auto* op = static_cast<const std::int64_t*>(offsets.buf);
    if (valid) {
        valid = op[0] == 0 && op[source_count] == event_count;
        for (std::int64_t source = 0; source < source_count; ++source) {
            valid = valid && op[source] <= op[source + 1];
        }
    }
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "completion event buffer shape mismatch");
        goto completion_fail;
    }
    {
        auto* out_e = static_cast<std::int64_t*>(output_entities.buf);
        auto* out_t = static_cast<std::int64_t*>(output_times.buf);
        auto* out_s = static_cast<std::int64_t*>(output_spans.buf);
        std::vector<std::int64_t> position(source_count), end(source_count), group_end(source_count);
        for (std::int64_t source = 0; source < source_count; ++source) {
            position[source] = op[source];
            end[source] = op[source + 1];
        }
        std::int64_t output = 0;
        Py_BEGIN_ALLOW_THREADS
        while (true) {
            bool exhausted = false;
            std::int64_t candidate_entity = std::numeric_limits<std::int64_t>::min();
            for (std::int64_t source = 0; source < source_count; ++source) {
                if (position[source] >= end[source]) { exhausted = true; break; }
                candidate_entity = std::max(candidate_entity, ep[position[source]]);
            }
            if (exhausted) break;
            bool aligned = true;
            for (std::int64_t source = 0; source < source_count; ++source) {
                while (position[source] < end[source] && ep[position[source]] < candidate_entity) {
                    const auto skipped = ep[position[source]];
                    while (position[source] < end[source] && ep[position[source]] == skipped) ++position[source];
                }
                if (position[source] >= end[source]) { exhausted = true; break; }
                if (ep[position[source]] != candidate_entity) aligned = false;
            }
            if (exhausted) break;
            if (!aligned) continue;
            for (std::int64_t source = 0; source < source_count; ++source) {
                group_end[source] = position[source];
                while (group_end[source] < end[source] && ep[group_end[source]] == candidate_entity) {
                    ++group_end[source];
                }
            }
            std::vector<std::int64_t> cursor = position;
            std::vector<std::int64_t> latest(source_count, std::numeric_limits<std::int64_t>::min());
            while (true) {
                std::int64_t next_time = std::numeric_limits<std::int64_t>::max();
                for (std::int64_t source = 0; source < source_count; ++source) {
                    if (cursor[source] < group_end[source]) next_time = std::min(next_time, tp[cursor[source]]);
                }
                if (next_time == std::numeric_limits<std::int64_t>::max()) break;
                bool witnessed = true;
                std::int64_t minimum = std::numeric_limits<std::int64_t>::max();
                std::int64_t maximum = std::numeric_limits<std::int64_t>::min();
                for (std::int64_t source = 0; source < source_count; ++source) {
                    while (cursor[source] < group_end[source] && tp[cursor[source]] <= next_time) {
                        latest[source] = tp[cursor[source]++];
                    }
                    witnessed = witnessed && latest[source] != std::numeric_limits<std::int64_t>::min();
                    minimum = std::min(minimum, latest[source]);
                    maximum = std::max(maximum, latest[source]);
                }
                if (witnessed) {
                    out_e[output] = candidate_entity;
                    out_t[output] = next_time;
                    out_s[output] = maximum - minimum;
                    ++output;
                }
            }
            position = group_end;
        }
        Py_END_ALLOW_THREADS
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&offsets);
        PyBuffer_Release(&output_entities); PyBuffer_Release(&output_times); PyBuffer_Release(&output_spans);
        return PyLong_FromLongLong(output);
    }
completion_fail:
    PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&offsets);
    PyBuffer_Release(&output_entities); PyBuffer_Release(&output_times); PyBuffer_Release(&output_spans);
    return nullptr;
}

PyMethodDef methods[] = {
    {"moments", moments, METH_VARARGS, "Deterministic gradient/Fisher moments."},
    {"nonnegative_quadratic_gains", nonnegative_quadratic_gains, METH_VARARGS, "Exact batched nonnegative quadratic gains."},
    {"aggregate_design_rows", aggregate_design_rows, METH_VARARGS, "Losslessly aggregate identical design rows."},
    {"set_num_threads", set_num_threads, METH_VARARGS, "Set deterministic OpenMP worker count."},
    {"future_rows", future_rows, METH_VARARGS, "Strict-future footprint rows."},
    {"accumulate_kernel", accumulate_kernel, METH_VARARGS, "Accumulate newly admitted kernel completions."},
    {"fill_candidate_batch", fill_candidate_batch, METH_VARARGS, "Fill one hierarchy candidate tile."},
    {"sparse_joint_moments", sparse_joint_moments, METH_VARARGS, "Exact moments of sparse joint blocks."},
    {"kernel_contributions", kernel_contributions, METH_VARARGS, "Strict-future kernel contributions."},
    {"completion_events", completion_events, METH_VARARGS, "Latest-witness completion events."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_cpu_native", nullptr, -1, methods};

}  // namespace

PyMODINIT_FUNC PyInit__cpu_native() { return PyModule_Create(&module); }

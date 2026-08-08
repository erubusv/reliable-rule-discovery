#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <tuple>
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

bool int32_buffer(PyObject* object, Py_buffer* view, int dimensions, bool writable) {
    const int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS |
                      (writable ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object, view, flags) != 0) return false;
    if (view->ndim != dimensions ||
        view->itemsize != static_cast<Py_ssize_t>(sizeof(std::int32_t))) {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_ValueError, "expected a C-contiguous int32 buffer");
        return false;
    }
    return true;
}

bool uint8_buffer(PyObject* object, Py_buffer* view, int dimensions, bool writable) {
    const int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS |
                      (writable ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object, view, flags) != 0) return false;
    if (view->ndim != dimensions ||
        view->itemsize != static_cast<Py_ssize_t>(sizeof(std::uint8_t))) {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_ValueError, "expected a C-contiguous uint8 buffer");
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

std::uint64_t quotient_hash(std::int64_t group, const double* row,
                            std::int64_t columns) {
    // The group id is part of the exact quotient key. Hash collisions are
    // resolved by exact group/row comparisons below.
    std::uint64_t hash = row_hash(row, columns);
    std::uint64_t group_bits = 0;
    std::memcpy(&group_bits, &group, sizeof(group_bits));
    hash ^= group_bits + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
}

inline void cloglog_value_first(double eta, double& value, double& first) {
    const double clipped = std::max(-745.0, std::min(700.0, eta));
    const double x = std::exp(clipped);
    const double tiny = std::numeric_limits<double>::min();
    if (x < 1.0e-4) {
        value = -std::log(std::max(x, tiny)) + x / 2.0 - x * x / 24.0;
        first = -1.0 + x / 2.0 - x * x / 12.0;
        return;
    }
    if (x > 40.0) {
        const double tail = std::exp(-x);
        value = -std::log1p(-tail);
        first = -x * tail / std::max(1.0 - tail, tiny);
        return;
    }
    const double denominator = std::expm1(x);
    value = -std::log(-std::expm1(-x));
    first = -x / denominator;
}

inline void mixed_cloglog_value_gradient(double eta, double noevent, double event,
                                         double& value, double& gradient) {
    double event_value = 0.0, event_first = 0.0;
    cloglog_value_first(eta, event_value, event_first);
    const double intensity = std::exp(std::min(eta, 700.0));
    value = noevent * intensity + event * event_value;
    gradient = noevent * intensity + event * event_first;
}

PyObject* likelihood_value_eta_gradient(PyObject*, PyObject* args) {
    PyObject *x_obj, *beta_obj, *primary_obj, *event_obj, *eta_obj,
        *gradient_obj;
    int likelihood_mode = 0;
    if (!PyArg_ParseTuple(args, "OOOOiOO", &x_obj, &beta_obj, &primary_obj,
                          &event_obj, &likelihood_mode, &eta_obj,
                          &gradient_obj)) {
        return nullptr;
    }
    Py_buffer x{}, beta{}, primary{}, event{}, eta{}, gradient{};
    if (!double_buffer(x_obj, &x, 2, false)) return nullptr;
    if (!double_buffer(beta_obj, &beta, 1, false) ||
        !double_buffer(primary_obj, &primary, 1, false) ||
        !double_buffer(event_obj, &event, 1, false) ||
        !double_buffer(eta_obj, &eta, 1, true) ||
        !double_buffer(gradient_obj, &gradient, 1, true)) {
        if (beta.buf) PyBuffer_Release(&beta);
        if (primary.buf) PyBuffer_Release(&primary);
        if (event.buf) PyBuffer_Release(&event);
        if (eta.buf) PyBuffer_Release(&eta);
        if (gradient.buf) PyBuffer_Release(&gradient);
        PyBuffer_Release(&x);
        return nullptr;
    }
    const std::int64_t rows = x.shape[0];
    const std::int64_t columns = x.shape[1];
    const bool valid =
        (likelihood_mode == 1 || likelihood_mode == 2) &&
        beta.shape[0] == columns && primary.shape[0] == rows &&
        event.shape[0] == rows && eta.shape[0] == rows &&
        gradient.shape[0] == columns;
    if (!valid) {
        PyBuffer_Release(&x);
        PyBuffer_Release(&beta);
        PyBuffer_Release(&primary);
        PyBuffer_Release(&event);
        PyBuffer_Release(&eta);
        PyBuffer_Release(&gradient);
        PyErr_SetString(PyExc_ValueError,
                        "likelihood value/gradient buffers are misaligned");
        return nullptr;
    }
    const auto* design = static_cast<const double*>(x.buf);
    const auto* coefficients = static_cast<const double*>(beta.buf);
    const auto* primary_weight = static_cast<const double*>(primary.buf);
    const auto* event_weight = static_cast<const double*>(event.buf);
    auto* predictor = static_cast<double*>(eta.buf);
    auto* output_gradient = static_cast<double*>(gradient.buf);
    const int threads = std::max(1, omp_get_max_threads());
    std::vector<double> partial_nll(static_cast<std::size_t>(threads), 0.0);
    std::vector<double> partial_gradient(
        static_cast<std::size_t>(threads) * static_cast<std::size_t>(columns),
        0.0);
    double nll = 0.0;
    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel
    {
        const int thread = omp_get_thread_num();
        double local_nll = 0.0;
        double* local_gradient =
            partial_gradient.data() +
            static_cast<std::size_t>(thread) *
                static_cast<std::size_t>(columns);
#pragma omp for schedule(static)
        for (std::int64_t row = 0; row < rows; ++row) {
            const double* values = design + row * columns;
            double linear = 0.0;
            for (std::int64_t column = 0; column < columns; ++column)
                linear += values[column] * coefficients[column];
            predictor[row] = linear;
            double value = 0.0, first = 0.0;
            if (likelihood_mode == 1) {
                const double intensity = std::exp(std::min(linear, 700.0));
                value = primary_weight[row] * intensity -
                        event_weight[row] * linear;
                first = primary_weight[row] * intensity - event_weight[row];
            } else {
                mixed_cloglog_value_gradient(
                    linear, primary_weight[row], event_weight[row], value,
                    first);
            }
            local_nll += value;
            for (std::int64_t column = 0; column < columns; ++column)
                local_gradient[column] += values[column] * first;
        }
        partial_nll[static_cast<std::size_t>(thread)] = local_nll;
    }
    std::fill(output_gradient, output_gradient + columns, 0.0);
    for (int thread = 0; thread < threads; ++thread) {
        nll += partial_nll[static_cast<std::size_t>(thread)];
        const double* local_gradient =
            partial_gradient.data() +
            static_cast<std::size_t>(thread) *
                static_cast<std::size_t>(columns);
        for (std::int64_t column = 0; column < columns; ++column)
            output_gradient[column] += local_gradient[column];
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&x);
    PyBuffer_Release(&beta);
    PyBuffer_Release(&primary);
    PyBuffer_Release(&event);
    PyBuffer_Release(&eta);
    PyBuffer_Release(&gradient);
    return PyFloat_FromDouble(nll);
}

PyObject* design_column_cross(PyObject*, PyObject* args) {
    PyObject *x_obj, *output_obj;
    long long selected = -1;
    if (!PyArg_ParseTuple(args, "OLO", &x_obj, &selected, &output_obj))
        return nullptr;
    Py_buffer x{}, output{};
    if (!double_buffer(x_obj, &x, 2, false)) return nullptr;
    if (!double_buffer(output_obj, &output, 1, true)) {
        PyBuffer_Release(&x);
        return nullptr;
    }
    const std::int64_t rows = x.shape[0];
    const std::int64_t columns = x.shape[1];
    if (selected < 0 || selected >= columns || output.shape[0] != columns) {
        PyBuffer_Release(&x);
        PyBuffer_Release(&output);
        PyErr_SetString(PyExc_ValueError, "selected design column is invalid");
        return nullptr;
    }
    const auto* design = static_cast<const double*>(x.buf);
    auto* result = static_cast<double*>(output.buf);
    const int threads = std::max(1, omp_get_max_threads());
    std::vector<double> partial(
        static_cast<std::size_t>(threads) * static_cast<std::size_t>(columns),
        0.0);
    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel
    {
        const int thread = omp_get_thread_num();
        double* local =
            partial.data() + static_cast<std::size_t>(thread) *
                                 static_cast<std::size_t>(columns);
#pragma omp for schedule(static)
        for (std::int64_t row = 0; row < rows; ++row) {
            const double* values = design + row * columns;
            const double multiplier = values[selected];
            for (std::int64_t column = 0; column < columns; ++column)
                local[column] += values[column] * multiplier;
        }
    }
    std::fill(result, result + columns, 0.0);
    for (int thread = 0; thread < threads; ++thread) {
        const double* local =
            partial.data() + static_cast<std::size_t>(thread) *
                                 static_cast<std::size_t>(columns);
        for (std::int64_t column = 0; column < columns; ++column)
            result[column] += local[column];
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&x);
    PyBuffer_Release(&output);
    Py_RETURN_NONE;
}

PyObject* cloglog_mixed_conjugate(PyObject*, PyObject* args) {
    PyObject *dual_obj, *noevent_obj, *event_obj, *output_obj;
    if (!PyArg_ParseTuple(args, "OOOO", &dual_obj, &noevent_obj, &event_obj,
                          &output_obj)) {
        return nullptr;
    }
    Py_buffer dual{}, noevent{}, event{}, output{};
    if (!double_buffer(dual_obj, &dual, 1, false)) return nullptr;
    if (!double_buffer(noevent_obj, &noevent, 1, false)) {
        PyBuffer_Release(&dual);
        return nullptr;
    }
    if (!double_buffer(event_obj, &event, 1, false)) {
        PyBuffer_Release(&dual);
        PyBuffer_Release(&noevent);
        return nullptr;
    }
    if (!double_buffer(output_obj, &output, 1, true)) {
        PyBuffer_Release(&dual);
        PyBuffer_Release(&noevent);
        PyBuffer_Release(&event);
        return nullptr;
    }
    const std::int64_t size = dual.shape[0];
    if (noevent.shape[0] != size || event.shape[0] != size ||
        output.shape[0] != size) {
        PyErr_SetString(PyExc_ValueError, "cloglog conjugate shape mismatch");
        PyBuffer_Release(&dual);
        PyBuffer_Release(&noevent);
        PyBuffer_Release(&event);
        PyBuffer_Release(&output);
        return nullptr;
    }
    const auto* up = static_cast<const double*>(dual.buf);
    const auto* np = static_cast<const double*>(noevent.buf);
    const auto* ep = static_cast<const double*>(event.buf);
    auto* destination = static_cast<double*>(output.buf);
    std::int64_t mixed_count = 0;
    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(static) reduction(+ : mixed_count)
    for (std::int64_t index = 0; index < size; ++index) {
        const double noevent_weight = np[index];
        const double event_weight = ep[index];
        if (!(noevent_weight > 0.0 && event_weight > 0.0)) continue;
        ++mixed_count;
        const double u = up[index];
        const double lower_domain = -event_weight;
        if (u < lower_domain) {
            destination[index] = std::numeric_limits<double>::infinity();
            continue;
        }
        if (std::abs(u - lower_domain) <= 1.0e-14) {
            destination[index] = 0.0;
            continue;
        }
        double low = -50.0, high = 50.0;
        double value = 0.0, low_gradient = 0.0, high_gradient = 0.0;
        mixed_cloglog_value_gradient(low, noevent_weight, event_weight, value,
                                     low_gradient);
        mixed_cloglog_value_gradient(high, noevent_weight, event_weight, value,
                                     high_gradient);
        while (low_gradient > u && low > -740.0) {
            low *= 2.0;
            mixed_cloglog_value_gradient(low, noevent_weight, event_weight, value,
                                         low_gradient);
        }
        while (high_gradient < u && high < 700.0) {
            high *= 2.0;
            mixed_cloglog_value_gradient(high, noevent_weight, event_weight, value,
                                         high_gradient);
        }
        if (!(low_gradient <= u && u <= high_gradient)) {
            destination[index] = std::numeric_limits<double>::infinity();
            continue;
        }
        for (int iteration = 0; iteration < 100; ++iteration) {
            const double middle = 0.5 * (low + high);
            double gradient = 0.0;
            mixed_cloglog_value_gradient(middle, noevent_weight, event_weight,
                                         value, gradient);
            if (gradient < u) {
                low = middle;
            } else {
                high = middle;
            }
        }
        const double eta = 0.5 * (low + high);
        double gradient = 0.0;
        mixed_cloglog_value_gradient(eta, noevent_weight, event_weight, value,
                                     gradient);
        destination[index] = u * eta - value;
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&dual);
    PyBuffer_Release(&noevent);
    PyBuffer_Release(&event);
    PyBuffer_Release(&output);
    return PyLong_FromLongLong(mixed_count);
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
        // Hashing every wide float64 row dominated exact support
        // construction.  Hash values are independent, so compute them in
        // parallel, then retain the original deterministic input-order
        // insertion/equality/summation pass below.  The compact row order and
        // floating-point weight accumulation are therefore bit-for-bit
        // unchanged.
        std::vector<std::uint64_t> hashes(
            static_cast<std::size_t>(rows));
        #pragma omp parallel for schedule(static)
        for (std::int64_t input = 0; input < rows; ++input) {
            hashes[static_cast<std::size_t>(input)] =
                row_hash(xp + input * columns, columns);
        }
        std::unordered_map<std::uint64_t, std::vector<std::int64_t>> buckets;
        buckets.reserve(static_cast<std::size_t>(std::min<std::int64_t>(rows, 262144)));
        for (std::int64_t input = 0; input < rows; ++input) {
            const double* row = xp + input * columns;
            auto& representatives =
                buckets[hashes[static_cast<std::size_t>(input)]];
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

PyObject* accumulate_cluster_scores(PyObject*, PyObject* args) {
    PyObject *design_obj, *row_groups_obj, *clusters_obj, *multipliers_obj,
        *output_obj;
    if (!PyArg_ParseTuple(args, "OOOOO", &design_obj, &row_groups_obj,
                          &clusters_obj, &multipliers_obj, &output_obj)) {
        return nullptr;
    }
    Py_buffer design{}, row_groups{}, clusters{}, multipliers{}, output{};
    if (!double_buffer(design_obj, &design, 2, false)) return nullptr;
    if (!int32_buffer(row_groups_obj, &row_groups, 1, false) ||
        !int32_buffer(clusters_obj, &clusters, 1, false) ||
        !double_buffer(multipliers_obj, &multipliers, 1, false) ||
        !double_buffer(output_obj, &output, 2, true)) {
        if (row_groups.buf) PyBuffer_Release(&row_groups);
        if (clusters.buf) PyBuffer_Release(&clusters);
        if (multipliers.buf) PyBuffer_Release(&multipliers);
        if (output.buf) PyBuffer_Release(&output);
        PyBuffer_Release(&design);
        return nullptr;
    }
    const std::int64_t rows = row_groups.shape[0];
    const std::int64_t groups = design.shape[0];
    const std::int64_t columns = design.shape[1];
    const std::int64_t cluster_count = output.shape[0];
    const bool valid = clusters.shape[0] == rows && multipliers.shape[0] == rows &&
                       output.shape[1] == columns;
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "cluster score buffer shape mismatch");
        PyBuffer_Release(&design); PyBuffer_Release(&row_groups);
        PyBuffer_Release(&clusters); PyBuffer_Release(&multipliers);
        PyBuffer_Release(&output);
        return nullptr;
    }
    const auto* xp = static_cast<const double*>(design.buf);
    const auto* groupp = static_cast<const std::int32_t*>(row_groups.buf);
    const auto* clusterp = static_cast<const std::int32_t*>(clusters.buf);
    const auto* multiplierp = static_cast<const double*>(multipliers.buf);
    auto* outp = static_cast<double*>(output.buf);
    int invalid = 0;
    // Build a stable cluster-major row index once.  The former column-major
    // implementation scanned every active row once per *dense* design
    // column, even though an aggregated TPP row contains one baseline entry
    // and only the active rule-kernel entries.  Stable grouping lets one
    // worker own an output cluster (no atomics), visit each row once, and skip
    // exact zeros.  For every output cell the additions remain in original
    // row order, so this is deterministic and bitwise equivalent to the old
    // loop rather than an approximate reduction.
    std::vector<std::int64_t> counts(cluster_count, 0);
    std::vector<std::int64_t> offsets(cluster_count + 1, 0);
    std::vector<std::int64_t> cursor(cluster_count, 0);
    std::vector<std::int64_t> clustered_rows(rows, 0);
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto group = groupp[row];
        const auto cluster = clusterp[row];
        if (group < 0 || group >= groups || cluster < 0 ||
            cluster >= cluster_count) {
            invalid = 1;
            break;
        }
        ++counts[cluster];
    }
    if (!invalid) {
        for (std::int64_t cluster = 0; cluster < cluster_count; ++cluster)
            offsets[cluster + 1] = offsets[cluster] + counts[cluster];
        std::copy(offsets.begin(), offsets.end() - 1, cursor.begin());
        for (std::int64_t row = 0; row < rows; ++row)
            clustered_rows[cursor[clusterp[row]]++] = row;
        Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(dynamic, 32)
        for (std::int64_t cluster = 0; cluster < cluster_count; ++cluster) {
            auto* destination = outp + cluster * columns;
            for (std::int64_t item = offsets[cluster];
                 item < offsets[cluster + 1]; ++item) {
                const auto row = clustered_rows[item];
                const auto* source = xp +
                    static_cast<std::int64_t>(groupp[row]) * columns;
                const double multiplier = multiplierp[row];
                for (std::int64_t column = 0; column < columns; ++column) {
                    const double value = source[column];
                    if (value != 0.0)
                        destination[column] += multiplier * value;
                }
            }
        }
        Py_END_ALLOW_THREADS
    }
    PyBuffer_Release(&design); PyBuffer_Release(&row_groups);
    PyBuffer_Release(&clusters); PyBuffer_Release(&multipliers);
    PyBuffer_Release(&output);
    if (invalid) {
        PyErr_SetString(PyExc_ValueError, "cluster score index is invalid");
        return nullptr;
    }
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
    int requested_workers = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOOO|i", &entities_obj, &times_obj,
                          &starts_obj, &ends_obj, &offsets_obj, &basis_obj,
                          &lookup_obj, &accumulator_obj,
                          &requested_workers)) return nullptr;
    if (requested_workers < 0) {
        PyErr_SetString(PyExc_ValueError, "kernel worker count must be nonnegative");
        return nullptr;
    }
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
    bool allocation_failed = false;
    Py_BEGIN_ALLOW_THREADS
    try {
        std::vector<std::int64_t> boundaries(
            static_cast<std::size_t>(entity_count) + 1, 0);
        for (std::int64_t i = 0; i < count; ++i) {
            const auto entity = ep[i];
            if (entity < 0 || entity >= entity_count ||
                tp[i] < startp[entity] || tp[i] > endp[entity]) {
                invalid = true;
                break;
            }
            ++boundaries[static_cast<std::size_t>(entity) + 1];
        }
        if (!invalid) {
            for (std::int64_t entity = 0; entity < entity_count; ++entity) {
                boundaries[static_cast<std::size_t>(entity) + 1] +=
                    boundaries[static_cast<std::size_t>(entity)];
            }
            auto positions = boundaries;
            std::vector<std::int64_t> ordered(static_cast<std::size_t>(count));
            for (std::int64_t i = 0; i < count; ++i) {
                const auto entity = ep[i];
                ordered[static_cast<std::size_t>(
                    positions[static_cast<std::size_t>(entity)]++)] = i;
            }
            // A triangular M-knot basis is piecewise affine in integer lag.
            // Expanding every completion into ``lag`` rows creates an
            // O(completions * lag) temporary and repeats the same additions
            // whenever response intervals overlap.  Encode each basis row as
            // exact (up to machine-roundoff) affine runs and add those runs to
            // entity-local difference arrays.  The final prefix scan touches
            // each entity observation row once, so both memory and work are
            // independent of the number of overlapping completions.
            struct AffineRun {
                std::int64_t first;
                std::int64_t last;
                double slope;
                double intercept;
            };
            std::vector<std::vector<AffineRun>> runs(
                static_cast<std::size_t>(knots));
            for (std::int64_t k = 0; k < knots; ++k) {
                const auto* values = bp + k * lag;
                if (lag == 1) {
                    runs[static_cast<std::size_t>(k)].push_back(
                        {1, 1, 0.0, values[0]});
                    continue;
                }
                std::int64_t first = 1;
                double slope = values[1] - values[0];
                double intercept = values[0] - slope;
                for (std::int64_t l = 2; l < lag; ++l) {
                    const double next_slope = values[l] - values[l - 1];
                    const double scale = std::max(
                        {1.0, std::abs(slope), std::abs(next_slope)});
                    if (std::abs(next_slope - slope) <=
                        32.0 * std::numeric_limits<double>::epsilon() * scale) {
                        continue;
                    }
                    runs[static_cast<std::size_t>(k)].push_back(
                        {first, l, slope, intercept});
                    first = l + 1;
                    slope = next_slope;
                    intercept = values[l] - slope * static_cast<double>(l + 1);
                }
                runs[static_cast<std::size_t>(k)].push_back(
                    {first, lag, slope, intercept});
            }

            int invalid_parallel = 0;
            const int accumulation_workers = requested_workers > 0
                ? std::min(requested_workers, omp_get_max_threads())
                : omp_get_max_threads();
            #pragma omp parallel num_threads(accumulation_workers) reduction(|:invalid_parallel)
            {
                std::vector<double> slope_difference;
                std::vector<double> intercept_difference;
                std::vector<std::int64_t> coverage_difference;
                #pragma omp for schedule(dynamic, 64)
                for (std::int64_t entity = 0; entity < entity_count; ++entity) {
                    const auto begin = boundaries[static_cast<std::size_t>(entity)];
                    const auto finish = boundaries[static_cast<std::size_t>(entity) + 1];
                    if (begin == finish) continue;
                    const auto length = endp[entity] - startp[entity] + 1;
                    if (length <= 1) continue;
                    const auto scratch_size = static_cast<std::size_t>(
                        (length + 1) * knots);
                    slope_difference.assign(scratch_size, 0.0);
                    intercept_difference.assign(scratch_size, 0.0);
                    coverage_difference.assign(
                        static_cast<std::size_t>(length + 1), 0);
                    for (std::int64_t cursor = begin; cursor < finish; ++cursor) {
                        const auto i = ordered[static_cast<std::size_t>(cursor)];
                        const auto completion = tp[i] - startp[entity];
                        const auto maximum = std::min<std::int64_t>(
                            lag, endp[entity] - tp[i]);
                        if (maximum <= 0) continue;
                        ++coverage_difference[static_cast<std::size_t>(
                            completion + 1)];
                        --coverage_difference[static_cast<std::size_t>(
                            completion + maximum + 1)];
                        for (std::int64_t k = 0; k < knots; ++k) {
                            for (const auto& run : runs[static_cast<std::size_t>(k)]) {
                                const auto first_lag = run.first;
                                const auto last_lag = std::min(run.last, maximum);
                                if (first_lag > last_lag) continue;
                                const auto first_row = completion + first_lag;
                                const auto after_row = completion + last_lag + 1;
                                const auto first_index = static_cast<std::size_t>(
                                    first_row * knots + k);
                                const auto after_index = static_cast<std::size_t>(
                                    after_row * knots + k);
                                // f(row-completion) = slope*row + adjusted intercept.
                                const double adjusted_intercept =
                                    run.intercept -
                                    run.slope * static_cast<double>(completion);
                                slope_difference[first_index] += run.slope;
                                slope_difference[after_index] -= run.slope;
                                intercept_difference[first_index] += adjusted_intercept;
                                intercept_difference[after_index] -= adjusted_intercept;
                            }
                        }
                    }
                    std::vector<double> active_slope(
                        static_cast<std::size_t>(knots), 0.0);
                    std::vector<double> active_intercept(
                        static_cast<std::size_t>(knots), 0.0);
                    std::int64_t active_coverage = 0;
                    for (std::int64_t local_row = 1; local_row < length; ++local_row) {
                        const auto global_row = offsetp[entity] + local_row;
                        if (global_row < 0 || global_row >= lookup.shape[0]) {
                            invalid_parallel = 1;
                            continue;
                        }
                        const auto position =
                            lookp64 ? lookp64[global_row] : lookp32[global_row];
                        active_coverage += coverage_difference[
                            static_cast<std::size_t>(local_row)];
                        for (std::int64_t k = 0; k < knots; ++k) {
                            const auto scratch_index = static_cast<std::size_t>(
                                local_row * knots + k);
                            active_slope[static_cast<std::size_t>(k)] +=
                                slope_difference[scratch_index];
                            active_intercept[static_cast<std::size_t>(k)] +=
                                intercept_difference[scratch_index];
                            const double value =
                                active_slope[static_cast<std::size_t>(k)] *
                                    static_cast<double>(local_row) +
                                active_intercept[static_cast<std::size_t>(k)];
                            if (active_coverage == 0) {
                                // Affine difference sums can leave a few ulps
                                // after the final interval cancellation.  The
                                // exact integer coverage is authoritative and
                                // prevents those roundoff residues from being
                                // mistaken for a missing response row.
                                continue;
                            }
                            if (position >= 0 && position < accumulator.shape[0]) {
                                out[position * knots + k] += value;
                            } else {
                                invalid_parallel = 1;
                            }
                        }
                    }
                }
            }
            invalid = invalid_parallel != 0;
        }
    } catch (const std::bad_alloc&) {
        allocation_failed = true;
    }
    Py_END_ALLOW_THREADS
    if (allocation_failed) {
        release();
        return PyErr_NoMemory();
    }
    if (invalid) {
        PyErr_SetString(PyExc_ValueError, "kernel row missing from accumulator lookup");
        release();
        return nullptr;
    }
    release();
    Py_RETURN_NONE;
}

PyObject* kernel_touched_positions(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *starts_obj, *ends_obj, *offsets_obj,
             *lookup_obj, *marks_obj, *positions_obj;
    int horizon;
    if (!PyArg_ParseTuple(args, "OOOOOOiOO", &entities_obj, &times_obj,
                          &starts_obj, &ends_obj, &offsets_obj, &lookup_obj,
                          &horizon, &marks_obj, &positions_obj)) return nullptr;
    Py_buffer entities{}, times{}, starts{}, ends{}, offsets{}, lookup{}, marks{},
              positions{};
    int acquired = 0;
    const auto release = [&]() {
        if (acquired >= 8) PyBuffer_Release(&positions);
        if (acquired >= 7) PyBuffer_Release(&marks);
        if (acquired >= 6) PyBuffer_Release(&lookup);
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
    if (!integer_lookup_buffer(lookup_obj, &lookup)) { release(); return nullptr; }
    ++acquired;
    if (!uint8_buffer(marks_obj, &marks, 1, true)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(positions_obj, &positions, 1, true)) {
        release(); return nullptr;
    }
    ++acquired;
    const auto count = entities.shape[0], entity_count = starts.shape[0];
    const bool valid = horizon >= 0 && times.shape[0] == count &&
                       ends.shape[0] == entity_count &&
                       offsets.shape[0] == entity_count + 1 &&
                       marks.shape[0] <= positions.shape[0];
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "kernel touched-position shape mismatch");
        release();
        return nullptr;
    }
    const auto* ep = static_cast<const std::int64_t*>(entities.buf);
    const auto* tp = static_cast<const std::int64_t*>(times.buf);
    const auto* startp = static_cast<const std::int64_t*>(starts.buf);
    const auto* endp = static_cast<const std::int64_t*>(ends.buf);
    const auto* offsetp = static_cast<const std::int64_t*>(offsets.buf);
    const auto* lookp64 = lookup.itemsize == static_cast<Py_ssize_t>(sizeof(std::int64_t))
                              ? static_cast<const std::int64_t*>(lookup.buf)
                              : nullptr;
    const auto* lookp32 = lookup.itemsize == static_cast<Py_ssize_t>(sizeof(std::int32_t))
                              ? static_cast<const std::int32_t*>(lookup.buf)
                              : nullptr;
    auto* markp = static_cast<std::uint8_t*>(marks.buf);
    auto* output = static_cast<std::int64_t*>(positions.buf);
    const auto direct_denominator = std::max<std::int64_t>(1, 4LL * horizon);
    const bool record_directly = count < marks.shape[0] / direct_denominator;
    bool invalid = false;
    bool allocation_failed = false;
    std::int64_t written = 0;
    Py_BEGIN_ALLOW_THREADS
    try {
        std::vector<std::int64_t> boundaries(
            static_cast<std::size_t>(entity_count) + 1, 0);
        for (std::int64_t i = 0; i < count; ++i) {
            const auto entity = ep[i];
            if (entity < 0 || entity >= entity_count ||
                tp[i] < startp[entity] || tp[i] > endp[entity]) {
                invalid = true;
                break;
            }
            ++boundaries[static_cast<std::size_t>(entity) + 1];
        }
        if (!invalid) {
            for (std::int64_t entity = 0; entity < entity_count; ++entity) {
                boundaries[static_cast<std::size_t>(entity) + 1] +=
                    boundaries[static_cast<std::size_t>(entity)];
            }
            auto cursors = boundaries;
            std::vector<std::int64_t> ordered(static_cast<std::size_t>(count));
            for (std::int64_t i = 0; i < count; ++i) {
                const auto entity = ep[i];
                ordered[static_cast<std::size_t>(
                    cursors[static_cast<std::size_t>(entity)]++)] = i;
            }
            int invalid_parallel = 0;
            #pragma omp parallel for schedule(static) reduction(|:invalid_parallel)
            for (std::int64_t entity = 0; entity < entity_count; ++entity) {
                for (std::int64_t cursor = boundaries[static_cast<std::size_t>(entity)];
                     cursor < boundaries[static_cast<std::size_t>(entity) + 1];
                     ++cursor) {
                    const auto i = ordered[static_cast<std::size_t>(cursor)];
                    const auto maximum = std::min<std::int64_t>(
                        horizon, endp[entity] - tp[i]);
                    const auto base = offsetp[entity] + tp[i] - startp[entity];
                    for (std::int64_t lag = 1; lag <= maximum; ++lag) {
                        const auto row = base + lag;
                        if (row < 0 || row >= lookup.shape[0]) {
                            invalid_parallel = 1;
                            continue;
                        }
                        const auto position = lookp64 ? lookp64[row] : lookp32[row];
                        if (position < 0 || position >= marks.shape[0]) {
                            invalid_parallel = 1;
                            continue;
                        }
                        // Entity observation ranges are disjoint, so all
                        // writers of one mark belong to the same OpenMP
                        // iteration.  Record the first transition immediately
                        // instead of scanning the complete active footprint
                        // after every W increment.
                        if (markp[position] == 0) {
                            markp[position] = 1;
                            if (record_directly) {
                                std::int64_t slot = 0;
                                #pragma omp atomic capture
                                slot = written++;
                                output[slot] = position;
                            }
                        }
                    }
                }
            }
            invalid = invalid_parallel != 0;
            if (!invalid) {
                if (record_directly) {
                    std::sort(output, output + written);
                    for (std::int64_t index = 0; index < written; ++index)
                        markp[output[index]] = 0;
                } else {
                    for (std::int64_t position = 0; position < marks.shape[0];
                         ++position) {
                        if (markp[position] == 0) continue;
                        output[written++] = position;
                        markp[position] = 0;
                    }
                }
            }
        }
    } catch (const std::bad_alloc&) {
        allocation_failed = true;
    }
    Py_END_ALLOW_THREADS
    if (allocation_failed) {
        release();
        return PyErr_NoMemory();
    }
    if (invalid) {
        // Never leave scratch marks dirty after a fail-open exception.
        if (record_directly) {
            for (std::int64_t index = 0; index < written; ++index)
                markp[output[index]] = 0;
        } else {
            std::fill(markp, markp + marks.shape[0], static_cast<std::uint8_t>(0));
        }
        PyErr_SetString(PyExc_ValueError, "kernel touched-position lookup failed");
        release();
        return nullptr;
    }
    release();
    return PyLong_FromLongLong(written);
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

PyObject* fill_pricing_values(PyObject*, PyObject* args) {
    PyObject *query_obj, *rows_obj, *values_obj, *lookup_obj, *output_obj;
    if (!PyArg_ParseTuple(args, "OOOOO", &query_obj, &rows_obj, &values_obj,
                          &lookup_obj, &output_obj)) {
        return nullptr;
    }
    Py_buffer query{}, rows{}, values{}, lookup{}, output{};
    bool lookup_acquired = false;
    if (!int64_buffer(query_obj, &query, 1, false)) return nullptr;
    if (!int64_buffer(rows_obj, &rows, 1, false)) {
        PyBuffer_Release(&query);
        return nullptr;
    }
    if (!double_buffer(values_obj, &values, 2, false)) {
        PyBuffer_Release(&query); PyBuffer_Release(&rows);
        return nullptr;
    }
    if (!double_buffer(output_obj, &output, 2, true)) {
        PyBuffer_Release(&query); PyBuffer_Release(&rows);
        PyBuffer_Release(&values);
        return nullptr;
    }
    if (lookup_obj != Py_None) {
        if (!integer_lookup_buffer(lookup_obj, &lookup)) {
            PyBuffer_Release(&query); PyBuffer_Release(&rows);
            PyBuffer_Release(&values); PyBuffer_Release(&output);
            return nullptr;
        }
        lookup_acquired = true;
    }
    const bool valid = rows.shape[0] == values.shape[0] &&
                       output.shape[0] == query.shape[0] &&
                       output.shape[1] == values.shape[1];
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "pricing row/value scratch mismatch");
        PyBuffer_Release(&query); PyBuffer_Release(&rows);
        PyBuffer_Release(&values); PyBuffer_Release(&output);
        if (lookup_acquired) PyBuffer_Release(&lookup);
        return nullptr;
    }
    const auto* queryp = static_cast<const std::int64_t*>(query.buf);
    const auto* rowp = static_cast<const std::int64_t*>(rows.buf);
    const auto* valuep = static_cast<const double*>(values.buf);
    const auto* lookup64 = lookup_acquired &&
                                   lookup.itemsize == static_cast<Py_ssize_t>(sizeof(std::int64_t))
                               ? static_cast<const std::int64_t*>(lookup.buf)
                               : nullptr;
    const auto* lookup32 = lookup_acquired &&
                                   lookup.itemsize == static_cast<Py_ssize_t>(sizeof(std::int32_t))
                               ? static_cast<const std::int32_t*>(lookup.buf)
                               : nullptr;
    auto* out = static_cast<double*>(output.buf);
    const auto query_count = query.shape[0], source_count = rows.shape[0],
               width = values.shape[1];
    bool invalid = false;
    Py_BEGIN_ALLOW_THREADS
    std::fill(out, out + query_count * width, 0.0);
    if (lookup_acquired) {
        for (std::int64_t index = 0; index < query_count; ++index) {
            const auto raw = queryp[index];
            if (raw < 0 || raw >= lookup.shape[0]) {
                invalid = true;
                break;
            }
            const auto position = lookup64 ? lookup64[raw] : lookup32[raw];
            if (position < 0) continue;
            if (position >= source_count || rowp[position] != raw) {
                invalid = true;
                break;
            }
            std::memcpy(out + index * width, valuep + position * width,
                        static_cast<std::size_t>(width) * sizeof(double));
        }
    } else if (query_count && source_count) {
        auto cursor = std::lower_bound(rowp, rowp + source_count, queryp[0]);
        for (std::int64_t index = 0; index < query_count; ++index) {
            cursor = std::lower_bound(cursor, rowp + source_count, queryp[index]);
            if (cursor == rowp + source_count) break;
            if (*cursor != queryp[index]) continue;
            const auto position = cursor - rowp;
            std::memcpy(out + index * width, valuep + position * width,
                        static_cast<std::size_t>(width) * sizeof(double));
        }
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&query); PyBuffer_Release(&rows);
    PyBuffer_Release(&values); PyBuffer_Release(&output);
    if (lookup_acquired) PyBuffer_Release(&lookup);
    if (invalid) {
        PyErr_SetString(PyExc_ValueError, "pricing lookup is inconsistent");
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

inline void update_latest_primitive(
    std::array<std::array<std::int64_t, 3>, 3>& latest_time,
    std::array<std::array<std::int64_t, 3>, 3>& latest_id,
    std::int64_t source,
    std::int64_t time,
    std::int64_t primitive,
    std::int64_t source_count) {
    auto& times = latest_time[source];
    auto& ids = latest_id[source];
    std::int64_t slot = -1;
    for (std::int64_t index = 0; index < source_count; ++index) {
        if (ids[index] == primitive) {
            slot = index;
            break;
        }
    }
    if (slot < 0) {
        for (std::int64_t index = 0; index < source_count; ++index) {
            if (ids[index] == std::numeric_limits<std::int64_t>::min()) {
                slot = index;
                break;
            }
        }
    }
    if (slot < 0) slot = source_count - 1;
    ids[slot] = primitive;
    times[slot] = time;
    // At most q distinct IDs are sufficient for an exact system of distinct
    // representatives among q<=3 sources: at most q-1 newer IDs can be
    // consumed by the other sources.
    for (std::int64_t right = slot; right > 0; --right) {
        const auto left = right - 1;
        if (times[left] > times[right] ||
            (times[left] == times[right] && ids[left] < ids[right]))
            break;
        std::swap(times[left], times[right]);
        std::swap(ids[left], ids[right]);
    }
}

inline bool latest_distinct_span(
    const std::array<std::array<std::int64_t, 3>, 3>& latest_time,
    const std::array<std::array<std::int64_t, 3>, 3>& latest_id,
    std::int64_t source_count,
    std::int64_t& span) {
    const auto missing = std::numeric_limits<std::int64_t>::min();
    span = std::numeric_limits<std::int64_t>::max();
    std::int64_t best_total = std::numeric_limits<std::int64_t>::min();
    if (source_count == 1) {
        if (latest_id[0][0] == missing) return false;
        span = 0;
        return true;
    }
    if (source_count == 2) {
        for (std::int64_t first = 0; first < source_count; ++first) {
            if (latest_id[0][first] == missing) continue;
            for (std::int64_t second = 0; second < source_count; ++second) {
                if (latest_id[1][second] == missing ||
                    latest_id[0][first] == latest_id[1][second])
                    continue;
                const auto low = std::min(
                    latest_time[0][first], latest_time[1][second]);
                const auto high = std::max(
                    latest_time[0][first], latest_time[1][second]);
                const auto total =
                    latest_time[0][first] + latest_time[1][second];
                if (total > best_total ||
                    (total == best_total && high - low < span)) {
                    best_total = total;
                    span = high - low;
                }
            }
        }
        return span != std::numeric_limits<std::int64_t>::max();
    }
    for (std::int64_t first = 0; first < source_count; ++first) {
        if (latest_id[0][first] == missing) continue;
        for (std::int64_t second = 0; second < source_count; ++second) {
            if (latest_id[1][second] == missing ||
                latest_id[0][first] == latest_id[1][second])
                continue;
            for (std::int64_t third = 0; third < source_count; ++third) {
                if (latest_id[2][third] == missing ||
                    latest_id[0][first] == latest_id[2][third] ||
                    latest_id[1][second] == latest_id[2][third])
                    continue;
                const auto low = std::min(
                    latest_time[0][first],
                    std::min(latest_time[1][second], latest_time[2][third]));
                const auto high = std::max(
                    latest_time[0][first],
                    std::max(latest_time[1][second], latest_time[2][third]));
                const auto total = latest_time[0][first] +
                                   latest_time[1][second] +
                                   latest_time[2][third];
                if (total > best_total ||
                    (total == best_total && high - low < span)) {
                    best_total = total;
                    span = high - low;
                }
            }
        }
    }
    return span != std::numeric_limits<std::int64_t>::max();
}

PyObject* completion_events(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *primitive_obj, *offsets_obj, *output_entities_obj,
             *output_times_obj, *output_spans_obj;
    if (!PyArg_ParseTuple(args, "OOOOOOO", &entities_obj, &times_obj, &primitive_obj, &offsets_obj,
                          &output_entities_obj, &output_times_obj, &output_spans_obj)) {
        return nullptr;
    }
    Py_buffer entities{}, times{}, primitive{}, offsets{}, output_entities{}, output_times{}, output_spans{};
    if (!int64_buffer(entities_obj, &entities, 1, false)) return nullptr;
    if (!int64_buffer(times_obj, &times, 1, false)) { PyBuffer_Release(&entities); return nullptr; }
    if (!int64_buffer(primitive_obj, &primitive, 1, false)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); return nullptr;
    }
    if (!int64_buffer(offsets_obj, &offsets, 1, false)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&primitive); return nullptr;
    }
    if (!int64_buffer(output_entities_obj, &output_entities, 1, true)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&primitive);
        PyBuffer_Release(&offsets); return nullptr;
    }
    if (!int64_buffer(output_times_obj, &output_times, 1, true)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&primitive);
        PyBuffer_Release(&offsets);
        PyBuffer_Release(&output_entities); return nullptr;
    }
    if (!int64_buffer(output_spans_obj, &output_spans, 1, true)) {
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&primitive);
        PyBuffer_Release(&offsets);
        PyBuffer_Release(&output_entities); PyBuffer_Release(&output_times); return nullptr;
    }
    const auto event_count = entities.shape[0];
    const auto source_count = offsets.shape[0] - 1;
    bool valid = times.shape[0] == event_count && primitive.shape[0] == event_count &&
                 source_count >= 1 && source_count <= 3 &&
                 output_entities.shape[0] >= event_count && output_times.shape[0] >= event_count &&
                 output_spans.shape[0] >= event_count;
    const auto* ep = static_cast<const std::int64_t*>(entities.buf);
    const auto* tp = static_cast<const std::int64_t*>(times.buf);
    const auto* pp = static_cast<const std::int64_t*>(primitive.buf);
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
        std::array<std::int64_t, 3> position{}, end{}, group_end{};
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
            std::array<std::int64_t, 3> cursor{};
            std::array<std::array<std::int64_t, 3>, 3> latest_time{};
            std::array<std::array<std::int64_t, 3>, 3> latest_id{};
            for (auto& values : latest_time)
                values.fill(std::numeric_limits<std::int64_t>::min());
            for (auto& values : latest_id)
                values.fill(std::numeric_limits<std::int64_t>::min());
            for (std::int64_t source = 0; source < source_count; ++source)
                cursor[source] = position[source];
            while (true) {
                std::int64_t next_time = std::numeric_limits<std::int64_t>::max();
                for (std::int64_t source = 0; source < source_count; ++source) {
                    if (cursor[source] < group_end[source]) next_time = std::min(next_time, tp[cursor[source]]);
                }
                if (next_time == std::numeric_limits<std::int64_t>::max()) break;
                for (std::int64_t source = 0; source < source_count; ++source) {
                    while (cursor[source] < group_end[source] && tp[cursor[source]] <= next_time) {
                        update_latest_primitive(
                            latest_time, latest_id, source,
                            tp[cursor[source]], pp[cursor[source]], source_count);
                        ++cursor[source];
                    }
                }
                std::int64_t span = 0;
                if (latest_distinct_span(
                        latest_time, latest_id, source_count, span)) {
                    out_e[output] = candidate_entity;
                    out_t[output] = next_time;
                    out_s[output] = span;
                    ++output;
                }
            }
            position = group_end;
        }
        Py_END_ALLOW_THREADS
        PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&primitive);
        PyBuffer_Release(&offsets);
        PyBuffer_Release(&output_entities); PyBuffer_Release(&output_times); PyBuffer_Release(&output_spans);
        return PyLong_FromLongLong(output);
    }
completion_fail:
    PyBuffer_Release(&entities); PyBuffer_Release(&times); PyBuffer_Release(&primitive);
    PyBuffer_Release(&offsets);
    PyBuffer_Release(&output_entities); PyBuffer_Release(&output_times); PyBuffer_Release(&output_spans);
    return nullptr;
}

PyObject* ordered_completion_events(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *primitive_obj, *offsets_obj,
             *output_entities_obj, *output_times_obj, *output_spans_obj;
    if (!PyArg_ParseTuple(args, "OOOOOOO", &entities_obj, &times_obj,
                          &primitive_obj, &offsets_obj, &output_entities_obj,
                          &output_times_obj, &output_spans_obj)) {
        return nullptr;
    }
    Py_buffer entities{}, times{}, primitive{}, offsets{}, output_entities{},
              output_times{}, output_spans{};
    int acquired = 0;
    auto release = [&]() {
        if (acquired >= 7) PyBuffer_Release(&output_spans);
        if (acquired >= 6) PyBuffer_Release(&output_times);
        if (acquired >= 5) PyBuffer_Release(&output_entities);
        if (acquired >= 4) PyBuffer_Release(&offsets);
        if (acquired >= 3) PyBuffer_Release(&primitive);
        if (acquired >= 2) PyBuffer_Release(&times);
        if (acquired >= 1) PyBuffer_Release(&entities);
    };
    if (!int64_buffer(entities_obj, &entities, 1, false)) return nullptr;
    ++acquired;
    if (!int64_buffer(times_obj, &times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(primitive_obj, &primitive, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(offsets_obj, &offsets, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(output_entities_obj, &output_entities, 1, true)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(output_times_obj, &output_times, 1, true)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(output_spans_obj, &output_spans, 1, true)) { release(); return nullptr; }
    ++acquired;

    const auto event_count = entities.shape[0];
    const auto source_count = offsets.shape[0] - 1;
    const auto* ep = static_cast<const std::int64_t*>(entities.buf);
    const auto* tp = static_cast<const std::int64_t*>(times.buf);
    const auto* op = static_cast<const std::int64_t*>(offsets.buf);
    bool valid = times.shape[0] == event_count && primitive.shape[0] == event_count &&
                 (source_count == 2 || source_count == 3) && op[0] == 0 &&
                 op[source_count] == event_count &&
                 output_entities.shape[0] >= event_count &&
                 output_times.shape[0] >= event_count &&
                 output_spans.shape[0] >= event_count;
    for (std::int64_t source = 0; valid && source < source_count; ++source)
        valid = op[source] <= op[source + 1];
    if (!valid) {
        release();
        PyErr_SetString(PyExc_ValueError, "ordered completion buffer shape mismatch");
        return nullptr;
    }

    auto* out_e = static_cast<std::int64_t*>(output_entities.buf);
    auto* out_t = static_cast<std::int64_t*>(output_times.buf);
    auto* out_s = static_cast<std::int64_t*>(output_spans.buf);
    std::array<std::int64_t, 3> position{}, end{}, group_end{};
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
                while (position[source] < end[source] && ep[position[source]] == skipped)
                    ++position[source];
            }
            if (position[source] >= end[source]) { exhausted = true; break; }
            if (ep[position[source]] != candidate_entity) aligned = false;
        }
        if (exhausted) break;
        if (!aligned) continue;
        for (std::int64_t source = 0; source < source_count; ++source) {
            group_end[source] = position[source];
            while (group_end[source] < end[source] &&
                   ep[group_end[source]] == candidate_entity)
                ++group_end[source];
        }

        std::int64_t first_cursor = position[0];
        std::int64_t latest_first = -1;
        if (source_count == 2) {
            std::int64_t terminal_cursor = position[1];
            while (terminal_cursor < group_end[1]) {
                const auto terminal = tp[terminal_cursor];
                while (first_cursor < group_end[0] && tp[first_cursor] < terminal) {
                    latest_first = first_cursor++;
                }
                if (latest_first >= 0) {
                    out_e[output] = candidate_entity;
                    out_t[output] = terminal;
                    out_s[output] = terminal - tp[latest_first];
                    ++output;
                }
                while (terminal_cursor < group_end[1] &&
                       tp[terminal_cursor] == terminal)
                    ++terminal_cursor;
            }
        } else {
            std::int64_t middle_cursor = position[1];
            std::int64_t latest_valid_middle = -1;
            std::int64_t first_for_latest_middle = -1;
            std::int64_t terminal_cursor = position[2];
            while (terminal_cursor < group_end[2]) {
                const auto terminal = tp[terminal_cursor];
                while (middle_cursor < group_end[1] &&
                       tp[middle_cursor] < terminal) {
                    const auto middle = tp[middle_cursor];
                    while (first_cursor < group_end[0] && tp[first_cursor] < middle)
                        latest_first = first_cursor++;
                    if (latest_first >= 0) {
                        latest_valid_middle = middle_cursor;
                        first_for_latest_middle = latest_first;
                    }
                    ++middle_cursor;
                }
                if (latest_valid_middle >= 0 && first_for_latest_middle >= 0) {
                    out_e[output] = candidate_entity;
                    out_t[output] = terminal;
                    out_s[output] = terminal - tp[first_for_latest_middle];
                    ++output;
                }
                while (terminal_cursor < group_end[2] &&
                       tp[terminal_cursor] == terminal)
                    ++terminal_cursor;
            }
        }
        position = group_end;
    }
    Py_END_ALLOW_THREADS
    release();
    return PyLong_FromLongLong(output);
}

PyObject* observed_temporal_motifs(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *predicates_obj, *primitive_obj,
        *atomic_obj, *unordered_pair_obj, *ordered_pair_obj,
        *unordered_triplet_obj, *ordered_triplet_obj;
    int predicate_count = 0, q_max = 0, allow_unordered = 0, allow_ordered = 0;
    long long maximum_span = 0;
    if (!PyArg_ParseTuple(
            args, "OOOOiiLppOOOOO", &entities_obj, &times_obj,
            &predicates_obj, &primitive_obj, &predicate_count, &q_max,
            &maximum_span, &allow_unordered, &allow_ordered, &atomic_obj,
            &unordered_pair_obj, &ordered_pair_obj, &unordered_triplet_obj,
            &ordered_triplet_obj)) {
        return nullptr;
    }
    Py_buffer entities{}, times{}, predicates{}, primitive{}, atomic{},
        unordered_pair{}, ordered_pair{}, unordered_triplet{}, ordered_triplet{};
    int acquired = 0;
    auto release = [&]() {
        if (acquired >= 9) PyBuffer_Release(&ordered_triplet);
        if (acquired >= 8) PyBuffer_Release(&unordered_triplet);
        if (acquired >= 7) PyBuffer_Release(&ordered_pair);
        if (acquired >= 6) PyBuffer_Release(&unordered_pair);
        if (acquired >= 5) PyBuffer_Release(&atomic);
        if (acquired >= 4) PyBuffer_Release(&primitive);
        if (acquired >= 3) PyBuffer_Release(&predicates);
        if (acquired >= 2) PyBuffer_Release(&times);
        if (acquired >= 1) PyBuffer_Release(&entities);
    };
    if (!int64_buffer(entities_obj, &entities, 1, false)) return nullptr;
    ++acquired;
    if (!int64_buffer(times_obj, &times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(predicates_obj, &predicates, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(primitive_obj, &primitive, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(atomic_obj, &atomic, 1, true)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(unordered_pair_obj, &unordered_pair, 1, true)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(ordered_pair_obj, &ordered_pair, 1, true)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(unordered_triplet_obj, &unordered_triplet, 1, true)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(ordered_triplet_obj, &ordered_triplet, 1, true)) { release(); return nullptr; }
    ++acquired;

    const auto count = entities.shape[0];
    const std::int64_t pair_size =
        static_cast<std::int64_t>(predicate_count) * predicate_count;
    const std::int64_t triplet_size = pair_size * predicate_count;
    bool valid = predicate_count > 0 && q_max >= 1 && q_max <= 3 &&
                 maximum_span >= 0 && times.shape[0] == count &&
                 predicates.shape[0] == count && primitive.shape[0] == count &&
                 atomic.shape[0] == predicate_count &&
                 unordered_pair.shape[0] == pair_size &&
                 ordered_pair.shape[0] == pair_size &&
                 unordered_triplet.shape[0] == triplet_size &&
                 ordered_triplet.shape[0] == triplet_size;
    if (!valid) {
        release();
        PyErr_SetString(PyExc_ValueError, "observed motif buffer mismatch");
        return nullptr;
    }
    const auto* ep = static_cast<const std::int64_t*>(entities.buf);
    const auto* tp = static_cast<const std::int64_t*>(times.buf);
    const auto* pp = static_cast<const std::int64_t*>(predicates.buf);
    const auto* ip = static_cast<const std::int64_t*>(primitive.buf);
    auto* atom = static_cast<std::int64_t*>(atomic.buf);
    auto* upair = static_cast<std::int64_t*>(unordered_pair.buf);
    auto* opair = static_cast<std::int64_t*>(ordered_pair.buf);
    auto* utriplet = static_cast<std::int64_t*>(unordered_triplet.buf);
    auto* otriplet = static_cast<std::int64_t*>(ordered_triplet.buf);
    for (std::int64_t index = 0; valid && index < count; ++index) {
        valid = ep[index] >= 0 && pp[index] >= 0 && pp[index] < predicate_count &&
                (index == 0 || ep[index - 1] <= ep[index]);
    }
    if (!valid) {
        release();
        PyErr_SetString(PyExc_ValueError, "observed motif event stream is invalid");
        return nullptr;
    }

    struct PrimitiveGroup {
        std::int64_t time;
        std::int64_t primitive;
        std::vector<int> predicates;
    };
    auto pair_index = [predicate_count](int first, int second) {
        return static_cast<std::int64_t>(first) * predicate_count + second;
    };
    auto triplet_index = [predicate_count](int first, int second, int third) {
        return (static_cast<std::int64_t>(first) * predicate_count + second) *
                   predicate_count +
               third;
    };

    Py_BEGIN_ALLOW_THREADS
    std::int64_t left = 0;
    while (left < count) {
        std::int64_t right = left + 1;
        while (right < count && ep[right] == ep[left]) ++right;
        std::vector<PrimitiveGroup> groups;
        groups.reserve(static_cast<std::size_t>(right - left));
        std::unordered_map<std::int64_t, std::size_t> by_primitive;
        by_primitive.reserve(static_cast<std::size_t>(right - left));
        for (std::int64_t index = left; index < right; ++index) {
            ++atom[pp[index]];
            auto found = by_primitive.find(ip[index]);
            if (found == by_primitive.end()) {
                const auto position = groups.size();
                by_primitive.emplace(ip[index], position);
                groups.push_back(
                    PrimitiveGroup{tp[index], ip[index], {static_cast<int>(pp[index])}}
                );
            } else {
                auto& group = groups[found->second];
                // Provenance contract: one primitive occupies one tick.
                if (group.time != tp[index]) continue;
                const int predicate = static_cast<int>(pp[index]);
                if (std::find(group.predicates.begin(), group.predicates.end(), predicate) ==
                    group.predicates.end())
                    group.predicates.push_back(predicate);
            }
        }
        std::sort(groups.begin(), groups.end(), [](const auto& first, const auto& second) {
            return std::tie(first.time, first.primitive) <
                   std::tie(second.time, second.primitive);
        });
        const std::size_t size = groups.size();
        const auto missing = std::numeric_limits<std::int64_t>::min();
        if (allow_unordered && q_max >= 2) {
            // A finite predicate alphabet makes dynamic motif recognition
            // O(events * P^2), rather than enumerating every event triple.
            // ``latest[p]`` is the most recent *previous primitive* carrying
            // p. ``pair_first[a,b]`` is the largest possible minimum witness
            // time among previously observed distinct-primitive pairs.  It is
            // therefore sufficient for every maximum-span existence query.
            std::vector<std::int64_t> latest(predicate_count, missing);
            std::vector<std::int64_t> pair_first(pair_size, missing);
            for (const auto& group : groups) {
                if (q_max >= 3) {
                    for (int third : group.predicates) {
                        for (int first = 0; first < predicate_count; ++first) {
                            if (first == third) continue;
                            for (int second = first + 1; second < predicate_count;
                                 ++second) {
                                if (second == third) continue;
                                const auto witness =
                                    pair_first[pair_index(first, second)];
                                if (witness != missing &&
                                    group.time - witness <= maximum_span) {
                                    std::array<int, 3> values{first, second, third};
                                    std::sort(values.begin(), values.end());
                                    ++utriplet[triplet_index(
                                        values[0], values[1], values[2])];
                                }
                            }
                        }
                    }
                }
                // Do not update ``latest`` until the complete attribute set of
                // this primitive has been inspected: one primitive can never
                // witness two antecedent predicates by itself.
                for (int second : group.predicates) {
                    for (int first = 0; first < predicate_count; ++first) {
                        if (first == second || latest[first] == missing ||
                            group.time - latest[first] > maximum_span)
                            continue;
                        const int low = std::min(first, second);
                        const int high = std::max(first, second);
                        ++upair[pair_index(low, high)];
                        auto& witness = pair_first[pair_index(low, high)];
                        witness = std::max(witness, latest[first]);
                    }
                }
                for (int predicate : group.predicates)
                    latest[predicate] = std::max(latest[predicate], group.time);
            }
        }
        if (allow_ordered && q_max >= 2) {
            // Strict order is updated one tick at a time.  Predicates from any
            // primitive at the current tick query only states ending at an
            // earlier tick; they enter the dynamic state after all queries.
            std::vector<std::int64_t> latest(predicate_count, missing);
            std::vector<std::int64_t> pair_first(pair_size, missing);
            std::size_t tick_left = 0;
            while (tick_left < size) {
                std::size_t tick_right = tick_left + 1;
                while (tick_right < size &&
                       groups[tick_right].time == groups[tick_left].time)
                    ++tick_right;
                const auto tick = groups[tick_left].time;
                std::vector<std::uint8_t> present(predicate_count, 0);
                for (std::size_t index = tick_left; index < tick_right; ++index)
                    for (int predicate : groups[index].predicates)
                        present[predicate] = 1;
                if (q_max >= 3) {
                    for (int third = 0; third < predicate_count; ++third) {
                        if (!present[third]) continue;
                        for (int first = 0; first < predicate_count; ++first) {
                            if (first == third) continue;
                            for (int second = 0; second < predicate_count; ++second) {
                                if (second == first || second == third) continue;
                                const auto witness =
                                    pair_first[pair_index(first, second)];
                                if (witness != missing &&
                                    tick - witness <= maximum_span)
                                    ++otriplet[triplet_index(first, second, third)];
                            }
                        }
                    }
                }
                for (int second = 0; second < predicate_count; ++second) {
                    if (!present[second]) continue;
                    for (int first = 0; first < predicate_count; ++first) {
                        if (first == second || latest[first] == missing ||
                            tick - latest[first] > maximum_span)
                            continue;
                        ++opair[pair_index(first, second)];
                        auto& witness = pair_first[pair_index(first, second)];
                        witness = std::max(witness, latest[first]);
                    }
                }
                for (int predicate = 0; predicate < predicate_count; ++predicate)
                    if (present[predicate]) latest[predicate] = tick;
                tick_left = tick_right;
            }
        }
        left = right;
    }
    Py_END_ALLOW_THREADS
    release();
    Py_RETURN_NONE;
}

PyObject* completion_window_counts(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *primitive_obj, *offsets_obj, *ends_obj, *windows_obj,
             *counts_obj, *minimum_obj;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &entities_obj, &times_obj, &primitive_obj, &offsets_obj,
                          &ends_obj, &windows_obj, &counts_obj, &minimum_obj)) {
        return nullptr;
    }
    Py_buffer entities{}, times{}, primitive{}, offsets{}, ends{}, windows{}, counts{}, minimum{};
    int acquired = 0;
    auto release = [&]() {
        if (acquired >= 8) PyBuffer_Release(&minimum);
        if (acquired >= 7) PyBuffer_Release(&counts);
        if (acquired >= 6) PyBuffer_Release(&windows);
        if (acquired >= 5) PyBuffer_Release(&ends);
        if (acquired >= 4) PyBuffer_Release(&offsets);
        if (acquired >= 3) PyBuffer_Release(&primitive);
        if (acquired >= 2) PyBuffer_Release(&times);
        if (acquired >= 1) PyBuffer_Release(&entities);
    };
    if (!int64_buffer(entities_obj, &entities, 1, false)) return nullptr;
    ++acquired;
    if (!int64_buffer(times_obj, &times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(primitive_obj, &primitive, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(offsets_obj, &offsets, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(ends_obj, &ends, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(windows_obj, &windows, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(counts_obj, &counts, 1, true)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(minimum_obj, &minimum, 1, true)) {
        release(); return nullptr;
    }
    ++acquired;

    const auto event_count = entities.shape[0];
    const auto source_count = offsets.shape[0] - 1;
    const auto window_count = windows.shape[0];
    const auto* ep = static_cast<const std::int64_t*>(entities.buf);
    const auto* tp = static_cast<const std::int64_t*>(times.buf);
    const auto* pp = static_cast<const std::int64_t*>(primitive.buf);
    const auto* op = static_cast<const std::int64_t*>(offsets.buf);
    const auto* endp = static_cast<const std::int64_t*>(ends.buf);
    const auto* wp = static_cast<const std::int64_t*>(windows.buf);
    auto* cp = static_cast<std::int64_t*>(counts.buf);
    auto* mp = static_cast<std::int64_t*>(minimum.buf);
    bool valid = times.shape[0] == event_count && primitive.shape[0] == event_count &&
                 source_count >= 1 &&
                 source_count <= 3 && window_count >= 1 &&
                 counts.shape[0] == window_count &&
                 minimum.shape[0] == ends.shape[0] && op[0] == 0 &&
                 op[source_count] == event_count;
    for (std::int64_t source = 0; valid && source < source_count; ++source) {
        valid = op[source] <= op[source + 1];
    }
    for (std::int64_t index = 0; valid && index < window_count; ++index) {
        valid = wp[index] >= 0 && (index == 0 || wp[index - 1] < wp[index]);
    }
    for (std::int64_t index = 0; valid && index < event_count; ++index) {
        valid = ep[index] >= 0 && ep[index] < ends.shape[0];
    }
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "completion window-count buffer mismatch");
        release();
        return nullptr;
    }
    std::fill(cp, cp + window_count, 0);
    std::fill(
        mp, mp + minimum.shape[0],
        std::numeric_limits<std::int64_t>::max());
    {
        std::array<std::int64_t, 3> position{}, source_end{}, group_end{};
        std::vector<std::int64_t> difference(window_count + 1, 0);
        const bool uniform_windows =
            window_count > 1 && wp[0] == 0 && wp[1] > 0 &&
            wp[window_count - 1] == wp[1] * (window_count - 1);
        const auto uniform_step = uniform_windows ? wp[1] : 0;
        for (std::int64_t source = 0; source < source_count; ++source) {
            position[source] = op[source];
            source_end[source] = op[source + 1];
        }
        Py_BEGIN_ALLOW_THREADS
        while (true) {
            bool exhausted = false;
            std::int64_t candidate_entity = std::numeric_limits<std::int64_t>::min();
            for (std::int64_t source = 0; source < source_count; ++source) {
                if (position[source] >= source_end[source]) { exhausted = true; break; }
                candidate_entity = std::max(candidate_entity, ep[position[source]]);
            }
            if (exhausted) break;
            bool aligned = true;
            for (std::int64_t source = 0; source < source_count; ++source) {
                while (position[source] < source_end[source] &&
                       ep[position[source]] < candidate_entity) {
                    const auto skipped = ep[position[source]];
                    while (position[source] < source_end[source] &&
                           ep[position[source]] == skipped) ++position[source];
                }
                if (position[source] >= source_end[source]) { exhausted = true; break; }
                if (ep[position[source]] != candidate_entity) aligned = false;
            }
            if (exhausted) break;
            if (!aligned) continue;
            for (std::int64_t source = 0; source < source_count; ++source) {
                group_end[source] = position[source];
                while (group_end[source] < source_end[source] &&
                       ep[group_end[source]] == candidate_entity) ++group_end[source];
            }
            std::array<std::int64_t, 3> cursor{};
            std::array<std::array<std::int64_t, 3>, 3> latest_time{};
            std::array<std::array<std::int64_t, 3>, 3> latest_id{};
            for (auto& values : latest_time)
                values.fill(std::numeric_limits<std::int64_t>::min());
            for (auto& values : latest_id)
                values.fill(std::numeric_limits<std::int64_t>::min());
            for (std::int64_t source = 0; source < source_count; ++source)
                cursor[source] = position[source];
            while (true) {
                std::int64_t next_time = std::numeric_limits<std::int64_t>::max();
                for (std::int64_t source = 0; source < source_count; ++source) {
                    if (cursor[source] < group_end[source])
                        next_time = std::min(next_time, tp[cursor[source]]);
                }
                if (next_time == std::numeric_limits<std::int64_t>::max()) break;
                for (std::int64_t source = 0; source < source_count; ++source) {
                    while (cursor[source] < group_end[source] &&
                           tp[cursor[source]] <= next_time) {
                        update_latest_primitive(
                            latest_time, latest_id, source,
                            tp[cursor[source]], pp[cursor[source]], source_count);
                        ++cursor[source];
                    }
                }
                std::int64_t span = 0;
                if (latest_distinct_span(
                        latest_time, latest_id, source_count, span) &&
                    next_time < endp[candidate_entity]) {
                    mp[candidate_entity] = std::min(
                        mp[candidate_entity], span);
                    const std::int64_t* admitted = nullptr;
                    if (uniform_windows) {
                        const auto index =
                            span <= 0
                                ? 0
                                : (span + uniform_step - 1) / uniform_step;
                        admitted = index < window_count ? wp + index
                                                        : wp + window_count;
                    } else {
                        admitted =
                            std::lower_bound(wp, wp + window_count, span);
                    }
                    if (admitted != wp + window_count) ++difference[admitted - wp];
                }
            }
            position = group_end;
        }
        std::int64_t cumulative = 0;
        for (std::int64_t index = 0; index < window_count; ++index) {
            cumulative += difference[index];
            cp[index] = cumulative;
        }
        Py_END_ALLOW_THREADS
    }
    release();
    Py_RETURN_NONE;
}

PyObject* response_min_spans(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *spans_obj, *starts_obj, *ends_obj,
             *offsets_obj, *threshold_obj;
    int horizon;
    if (!PyArg_ParseTuple(args, "OOOOOOiO", &entities_obj, &times_obj, &spans_obj,
                          &starts_obj, &ends_obj, &offsets_obj, &horizon,
                          &threshold_obj)) {
        return nullptr;
    }
    Py_buffer entities{}, times{}, spans{}, starts{}, ends{}, offsets{}, threshold{};
    int acquired = 0;
    auto release = [&]() {
        if (acquired >= 7) PyBuffer_Release(&threshold);
        if (acquired >= 6) PyBuffer_Release(&offsets);
        if (acquired >= 5) PyBuffer_Release(&ends);
        if (acquired >= 4) PyBuffer_Release(&starts);
        if (acquired >= 3) PyBuffer_Release(&spans);
        if (acquired >= 2) PyBuffer_Release(&times);
        if (acquired >= 1) PyBuffer_Release(&entities);
    };
    if (!int64_buffer(entities_obj, &entities, 1, false)) return nullptr;
    ++acquired;
    if (!int64_buffer(times_obj, &times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(spans_obj, &spans, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(starts_obj, &starts, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(ends_obj, &ends, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(offsets_obj, &offsets, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int32_buffer(threshold_obj, &threshold, 1, true)) { release(); return nullptr; }
    ++acquired;
    const auto count = entities.shape[0];
    const auto entity_count = starts.shape[0];
    const auto* ep = static_cast<const std::int64_t*>(entities.buf);
    const auto* tp = static_cast<const std::int64_t*>(times.buf);
    const auto* sp = static_cast<const std::int64_t*>(spans.buf);
    const auto* startp = static_cast<const std::int64_t*>(starts.buf);
    const auto* endp = static_cast<const std::int64_t*>(ends.buf);
    const auto* offsetp = static_cast<const std::int64_t*>(offsets.buf);
    auto* out = static_cast<std::int32_t*>(threshold.buf);
    bool valid = horizon >= 0 && times.shape[0] == count && spans.shape[0] == count &&
                 ends.shape[0] == entity_count && offsets.shape[0] == entity_count + 1 &&
                 offsetp[entity_count] == threshold.shape[0];
    bool entity_sorted = true;
    for (std::int64_t index = 0; valid && index < count; ++index) {
        if (ep[index] < 0 || ep[index] >= entity_count || sp[index] < 0 ||
            sp[index] >= std::numeric_limits<std::int32_t>::max()) {
            valid = false;
            break;
        }
        const auto entity = ep[index];
        const auto length = std::min<std::int64_t>(horizon, endp[entity] - tp[index]);
        const auto base = offsetp[entity] + tp[index] - startp[entity];
        valid = tp[index] >= startp[entity] && tp[index] <= endp[entity] &&
                base >= 0 && base + std::max<std::int64_t>(0, length) <
                                 threshold.shape[0];
        if (index && ep[index] < ep[index - 1]) entity_sorted = false;
    }
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "response minimum-span buffer mismatch");
        release();
        return nullptr;
    }
    if (count < 100'000 || entity_count < 50'000) {
        Py_BEGIN_ALLOW_THREADS
        for (std::int64_t index = 0; index < count; ++index) {
            const auto entity = ep[index];
            const auto length = std::min<std::int64_t>(
                horizon, endp[entity] - tp[index]);
            const auto base = offsetp[entity] + tp[index] - startp[entity];
            for (std::int64_t lag = 1; lag <= length; ++lag) {
                out[base + lag] = std::min(
                    out[base + lag], static_cast<std::int32_t>(sp[index]));
            }
        }
        Py_END_ALLOW_THREADS
        release();
        Py_RETURN_NONE;
    }
    bool allocation_failed = false;
    Py_BEGIN_ALLOW_THREADS
    try {
        std::vector<std::int64_t> boundaries(
            static_cast<std::size_t>(entity_count) + 1, 0);
        for (std::int64_t index = 0; index < count; ++index)
            ++boundaries[static_cast<std::size_t>(ep[index]) + 1];
        for (std::int64_t entity = 0; entity < entity_count; ++entity)
            boundaries[static_cast<std::size_t>(entity) + 1] +=
                boundaries[static_cast<std::size_t>(entity)];
        std::vector<std::int64_t> ordered;
        if (!entity_sorted) {
            auto positions = boundaries;
            ordered.resize(static_cast<std::size_t>(count));
            for (std::int64_t index = 0; index < count; ++index) {
                const auto entity = ep[index];
                ordered[static_cast<std::size_t>(
                    positions[static_cast<std::size_t>(entity)]++)] = index;
            }
        }
        // Each completion covers [time+1, time+horizon].  Compute the minimum
        // witness span over those intervals with an entity-local monotone
        // queue instead of writing every completion/lag pair.  This is the
        // exact sliding-window minimum and reduces the long-lag path from
        // O(completions*horizon) to O(completions+covered rows).
        #pragma omp parallel for schedule(static)
        for (std::int64_t entity = 0; entity < entity_count; ++entity) {
            const auto begin = boundaries[static_cast<std::size_t>(entity)];
            const auto finish = boundaries[static_cast<std::size_t>(entity) + 1];
            if (begin == finish || horizon <= 0) continue;
            std::vector<std::int64_t> local;
            local.reserve(static_cast<std::size_t>(finish - begin));
            for (std::int64_t cursor = begin; cursor < finish; ++cursor)
                local.push_back(entity_sorted ? cursor : ordered[cursor]);
            if (!std::is_sorted(
                    local.begin(), local.end(),
                    [&](std::int64_t left, std::int64_t right) {
                        return tp[left] < tp[right] ||
                               (tp[left] == tp[right] && left < right);
                    })) {
                std::stable_sort(
                    local.begin(), local.end(),
                    [&](std::int64_t left, std::int64_t right) {
                        return tp[left] < tp[right];
                    });
            }
            const auto first_row = std::max<std::int64_t>(
                startp[entity], tp[local.front()] + 1);
            const auto last_row = std::min<std::int64_t>(
                endp[entity], tp[local.back()] + horizon);
            std::deque<std::int64_t> minimum;
            std::size_t cursor = 0;
            for (std::int64_t time = first_row; time <= last_row; ++time) {
                while (cursor < local.size() && tp[local[cursor]] < time) {
                    const auto index = local[cursor++];
                    while (!minimum.empty() && sp[minimum.back()] >= sp[index])
                        minimum.pop_back();
                    minimum.push_back(index);
                }
                const auto oldest = time - horizon;
                while (!minimum.empty() && tp[minimum.front()] < oldest)
                    minimum.pop_front();
                if (minimum.empty()) continue;
                const auto row = offsetp[entity] + time - startp[entity];
                out[row] = std::min(
                    out[row], static_cast<std::int32_t>(sp[minimum.front()]));
            }
        }
    } catch (const std::bad_alloc&) {
        allocation_failed = true;
    }
    Py_END_ALLOW_THREADS
    if (allocation_failed) {
        release();
        return PyErr_NoMemory();
    }
    release();
    Py_RETURN_NONE;
}

PyObject* continuous_single_block_moments(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *spans_obj, *candidate_starts_obj,
        *candidate_ends_obj, *candidate_windows_obj, *entity_ends_obj,
        *grid_offsets_obj, *row_times_obj, *knot_edges_obj, *knot_scales_obj,
        *prefix_first_obj, *prefix_second_obj, *group_run_starts_obj,
        *group_run_ids_obj, *current_x_obj, *current_columns_obj,
        *gradient_obj, *hessian_obj, *cross_obj;
    int requested_workers = 0, gradient_only = 0;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOOOOOOOOOOOOii", &entities_obj, &times_obj,
            &spans_obj, &candidate_starts_obj, &candidate_ends_obj,
            &candidate_windows_obj, &entity_ends_obj, &grid_offsets_obj,
            &row_times_obj, &knot_edges_obj, &knot_scales_obj,
            &prefix_first_obj, &prefix_second_obj, &group_run_starts_obj,
            &group_run_ids_obj, &current_x_obj, &current_columns_obj,
            &gradient_obj, &hessian_obj, &cross_obj,
            &requested_workers, &gradient_only)) {
        return nullptr;
    }
    Py_buffer entities{}, times{}, spans{}, candidate_starts{}, candidate_ends{},
        candidate_windows{}, entity_ends{}, grid_offsets{}, row_times{},
        knot_edges{}, knot_scales{}, prefix_first{}, prefix_second{},
        group_run_starts{}, group_run_ids{}, current_x{}, current_columns{},
        gradient{}, hessian{}, cross{};
    int acquired = 0;
    auto release = [&]() {
        if (acquired >= 20) PyBuffer_Release(&cross);
        if (acquired >= 19) PyBuffer_Release(&hessian);
        if (acquired >= 18) PyBuffer_Release(&gradient);
        if (acquired >= 17) PyBuffer_Release(&current_columns);
        if (acquired >= 16) PyBuffer_Release(&current_x);
        if (acquired >= 15) PyBuffer_Release(&group_run_ids);
        if (acquired >= 14) PyBuffer_Release(&group_run_starts);
        if (acquired >= 13) PyBuffer_Release(&prefix_second);
        if (acquired >= 12) PyBuffer_Release(&prefix_first);
        if (acquired >= 11) PyBuffer_Release(&knot_scales);
        if (acquired >= 10) PyBuffer_Release(&knot_edges);
        if (acquired >= 9) PyBuffer_Release(&row_times);
        if (acquired >= 8) PyBuffer_Release(&grid_offsets);
        if (acquired >= 7) PyBuffer_Release(&entity_ends);
        if (acquired >= 6) PyBuffer_Release(&candidate_windows);
        if (acquired >= 5) PyBuffer_Release(&candidate_ends);
        if (acquired >= 4) PyBuffer_Release(&candidate_starts);
        if (acquired >= 3) PyBuffer_Release(&spans);
        if (acquired >= 2) PyBuffer_Release(&times);
        if (acquired >= 1) PyBuffer_Release(&entities);
    };
    if (!int32_buffer(entities_obj, &entities, 1, false)) return nullptr;
    ++acquired;
    if (!int64_buffer(times_obj, &times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(spans_obj, &spans, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(candidate_starts_obj, &candidate_starts, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(candidate_ends_obj, &candidate_ends, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(candidate_windows_obj, &candidate_windows, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(entity_ends_obj, &entity_ends, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(grid_offsets_obj, &grid_offsets, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(row_times_obj, &row_times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(knot_edges_obj, &knot_edges, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(knot_scales_obj, &knot_scales, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(prefix_first_obj, &prefix_first, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(prefix_second_obj, &prefix_second, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(group_run_starts_obj, &group_run_starts, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int32_buffer(group_run_ids_obj, &group_run_ids, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(current_x_obj, &current_x, 2, false)) { release(); return nullptr; }
    ++acquired;
    if (!int32_buffer(current_columns_obj, &current_columns, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(gradient_obj, &gradient, 2, true)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(hessian_obj, &hessian, 3, true)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(cross_obj, &cross, 3, true)) { release(); return nullptr; }
    ++acquired;

    const auto completion_count = entities.shape[0];
    const auto candidate_count = candidate_starts.shape[0];
    const auto entity_count = entity_ends.shape[0];
    const auto row_count = row_times.shape[0];
    const auto knot_count = knot_scales.shape[0];
    const auto current_dimension = current_x.shape[1];
    const auto selected_count = current_columns.shape[0];
    bool valid = requested_workers >= 0 && times.shape[0] == completion_count &&
                 spans.shape[0] == completion_count &&
                 candidate_ends.shape[0] == candidate_count &&
                 candidate_windows.shape[0] == candidate_count &&
                 grid_offsets.shape[0] == entity_count + 1 &&
                 knot_edges.shape[0] == knot_count + 1 && knot_count > 0 &&
                 prefix_first.shape[0] == row_count + 1 &&
                 prefix_second.shape[0] == row_count + 1 &&
                 group_run_starts.shape[0] == group_run_ids.shape[0] &&
                 group_run_starts.shape[0] > 0 && current_x.shape[0] > 0 &&
                 gradient.shape[0] == candidate_count &&
                 gradient.shape[1] == knot_count &&
                 hessian.shape[0] == candidate_count &&
                 hessian.shape[1] == knot_count &&
                 hessian.shape[2] == knot_count &&
                 cross.shape[0] == candidate_count &&
                 cross.shape[1] == selected_count &&
                 cross.shape[2] == knot_count;
    const auto* offsetp = static_cast<const std::int64_t*>(grid_offsets.buf);
    const auto* edgep = static_cast<const std::int64_t*>(knot_edges.buf);
    const auto* run_startp = static_cast<const std::int64_t*>(group_run_starts.buf);
    const auto* run_idp = static_cast<const std::int32_t*>(group_run_ids.buf);
    const auto* columnp = static_cast<const std::int32_t*>(current_columns.buf);
    if (valid) {
        valid = offsetp[0] == 0 && offsetp[entity_count] == row_count &&
                run_startp[0] == 0 && edgep[0] >= 0;
    }
    for (std::int64_t index = 0; valid && index < candidate_count; ++index) {
        const auto begin = static_cast<const std::int64_t*>(candidate_starts.buf)[index];
        const auto finish = static_cast<const std::int64_t*>(candidate_ends.buf)[index];
        valid = begin >= 0 && begin <= finish && finish <= completion_count &&
                static_cast<const std::int64_t*>(candidate_windows.buf)[index] >= 0;
    }
    for (std::int64_t index = 0; valid && index < knot_count; ++index) {
        valid = edgep[index] < edgep[index + 1] &&
                std::isfinite(static_cast<const double*>(knot_scales.buf)[index]) &&
                static_cast<const double*>(knot_scales.buf)[index] > 0.0;
    }
    for (std::int64_t index = 0; valid && index < selected_count; ++index)
        valid = columnp[index] >= 0 && columnp[index] < current_dimension;
    for (std::int64_t index = 0; valid && index < group_run_starts.shape[0]; ++index) {
        valid = run_startp[index] >= 0 && run_startp[index] < row_count &&
                (index == 0 || run_startp[index] > run_startp[index - 1]) &&
                run_idp[index] >= 0 && run_idp[index] < current_x.shape[0];
    }
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "continuous moment buffer mismatch");
        release();
        return nullptr;
    }

    const auto* entityp = static_cast<const std::int32_t*>(entities.buf);
    const auto* timep = static_cast<const std::int64_t*>(times.buf);
    const auto* spanp = static_cast<const std::int64_t*>(spans.buf);
    const auto* candidate_startp = static_cast<const std::int64_t*>(candidate_starts.buf);
    const auto* candidate_endp = static_cast<const std::int64_t*>(candidate_ends.buf);
    const auto* windowp = static_cast<const std::int64_t*>(candidate_windows.buf);
    const auto* entity_endp = static_cast<const std::int64_t*>(entity_ends.buf);
    const auto* row_timep = static_cast<const std::int64_t*>(row_times.buf);
    const auto* scalep = static_cast<const double*>(knot_scales.buf);
    const auto* firstp = static_cast<const double*>(prefix_first.buf);
    const auto* secondp = static_cast<const double*>(prefix_second.buf);
    const auto* xp = static_cast<const double*>(current_x.buf);
    auto* gradientp = static_cast<double*>(gradient.buf);
    auto* hessianp = static_cast<double*>(hessian.buf);
    auto* crossp = static_cast<double*>(cross.buf);
    std::fill(gradientp, gradientp + candidate_count * knot_count, 0.0);
    std::fill(hessianp, hessianp + candidate_count * knot_count * knot_count, 0.0);
    std::fill(crossp, crossp + candidate_count * selected_count * knot_count, 0.0);

    int invalid = 0;
    const int workers = requested_workers > 0 ? requested_workers : omp_get_max_threads();
    // Keep the historical task partition, so fixed-order reductions stay
    // byte-compatible, while scheduling those immutable tasks on more cores.
    const int execution_workers = std::max(
        1, std::min(2 * workers, omp_get_num_procs()));

    // ``current_x`` is a dense view, but a design group contains only a
    // handful of nonzero selected parent columns. Compress those columns once
    // per fused wave. Selected indices retain their original ascending order,
    // so the accumulation below is identical to the dense scan with zeros
    // skipped.
    const auto current_group_count = current_x.shape[0];
    std::vector<std::int64_t> selected_offsets(
        static_cast<std::size_t>(current_group_count + 1), 0);
    std::vector<std::int32_t> selected_indices;
    std::vector<double> selected_values;

    // A candidate can contain far more completions than every other
    // candidate in the same wave. Parallelizing only over candidates leaves
    // one thread processing that large motif while the other threads become
    // idle. Split the packed completion ranges at entity boundaries. Tasks
    // for the same candidate then write to disjoint entity row ranges, so
    // there is no reduction, race, or change in floating-point evaluation.
    struct ProfileTask {
        std::int64_t candidate;
        std::int64_t begin;
        std::int64_t end;
    };
    std::int64_t total_work = 0;
    for (std::int64_t candidate = 0; candidate < candidate_count; ++candidate)
        total_work += candidate_endp[candidate] - candidate_startp[candidate];
    const auto target_tasks = std::max<std::int64_t>(1, 4LL * workers);
    const auto task_span = std::max<std::int64_t>(
        1, (total_work + target_tasks - 1) / target_tasks);
    std::vector<ProfileTask> tasks;
    std::vector<std::int64_t> candidate_task_offsets(
        static_cast<std::size_t>(candidate_count + 1), 0);
    tasks.reserve(static_cast<std::size_t>(target_tasks + candidate_count));
    for (std::int64_t candidate = 0; candidate < candidate_count; ++candidate) {
        candidate_task_offsets[static_cast<std::size_t>(candidate)] =
            static_cast<std::int64_t>(tasks.size());
        auto begin = candidate_startp[candidate];
        const auto end = candidate_endp[candidate];
        while (begin < end) {
            auto finish = std::min(end, begin + task_span);
            while (
                finish < end
                && entityp[finish] == entityp[finish - 1]
            )
                ++finish;
            tasks.push_back({candidate, begin, finish});
            begin = finish;
        }
    }
    candidate_task_offsets[static_cast<std::size_t>(candidate_count)] =
        static_cast<std::int64_t>(tasks.size());
    const auto task_count = static_cast<std::int64_t>(tasks.size());
    // A popular candidate may be split across several OpenMP tasks. Each task
    // owns a tiny accumulator; candidate outputs are reduced later in fixed
    // entity-range order. The previous shared ``+=`` was a data race.
    std::vector<double> task_gradient(
        static_cast<std::size_t>(task_count * knot_count), 0.0);
    std::vector<double> task_hessian(
        static_cast<std::size_t>(task_count * knot_count * knot_count), 0.0);
    std::vector<double> task_cross(
        static_cast<std::size_t>(task_count * selected_count * knot_count), 0.0);

    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(static) num_threads(execution_workers)
    for (std::int64_t group = 0; group < current_group_count; ++group) {
        std::int64_t count = 0;
        for (std::int64_t selected = 0; selected < selected_count; ++selected)
            count += xp[group * current_dimension + columnp[selected]] != 0.0;
        selected_offsets[static_cast<std::size_t>(group + 1)] = count;
    }
    for (std::int64_t group = 0; group < current_group_count; ++group)
        selected_offsets[static_cast<std::size_t>(group + 1)] +=
            selected_offsets[static_cast<std::size_t>(group)];
    selected_indices.resize(
        static_cast<std::size_t>(selected_offsets.back()));
    selected_values.resize(
        static_cast<std::size_t>(selected_offsets.back()));
#pragma omp parallel for schedule(static) num_threads(execution_workers)
    for (std::int64_t group = 0; group < current_group_count; ++group) {
        auto output = selected_offsets[static_cast<std::size_t>(group)];
        for (std::int64_t selected = 0; selected < selected_count; ++selected) {
            const double value = xp[group * current_dimension + columnp[selected]];
            if (value == 0.0) continue;
            selected_indices[static_cast<std::size_t>(output)] =
                static_cast<std::int32_t>(selected);
            selected_values[static_cast<std::size_t>(output)] = value;
            ++output;
        }
    }
#pragma omp parallel for schedule(dynamic, 1) num_threads(execution_workers) reduction(| : invalid)
    for (
        std::int64_t task_index = 0;
        task_index < static_cast<std::int64_t>(tasks.size());
        ++task_index
    ) {
        const auto& task = tasks[static_cast<std::size_t>(task_index)];
        const auto candidate = task.candidate;
        const auto completion_begin = task.begin;
        const auto completion_end = task.end;
        const auto maximum_span = windowp[candidate];
        auto* local_gradient =
            task_gradient.data() + task_index * knot_count;
        auto* local_hessian =
            task_hessian.data() + task_index * knot_count * knot_count;
        auto* local_cross =
            task_cross.data() + task_index * selected_count * knot_count;
        std::int64_t cursor = completion_begin;
        std::vector<std::int64_t> admitted;
        std::vector<std::int64_t> start_cursor(static_cast<std::size_t>(knot_count));
        std::vector<std::int64_t> end_cursor(static_cast<std::size_t>(knot_count));
        std::vector<double> active(static_cast<std::size_t>(knot_count));
        while (cursor < completion_end) {
            const auto entity = static_cast<std::int64_t>(entityp[cursor]);
            if (entity < 0 || entity >= entity_count) { invalid |= 1; break; }
            auto entity_finish = cursor + 1;
            while (entity_finish < completion_end && entityp[entity_finish] == entity)
                ++entity_finish;
            admitted.clear();
            admitted.reserve(static_cast<std::size_t>(entity_finish - cursor));
            for (auto item = cursor; item < entity_finish; ++item) {
                if (spanp[item] <= maximum_span && timep[item] < entity_endp[entity])
                    admitted.push_back(item);
            }
            cursor = entity_finish;
            if (admitted.empty()) continue;
            std::fill(start_cursor.begin(), start_cursor.end(), 0);
            std::fill(end_cursor.begin(), end_cursor.end(), 0);
            std::fill(active.begin(), active.end(), 0.0);
            const auto row_left = offsetp[entity];
            const auto row_right = offsetp[entity + 1];
            const auto terminal = entity_endp[entity] + 1;
            std::int64_t previous_row = row_left;
            auto* initial_run = std::upper_bound(
                run_startp,
                run_startp + group_run_starts.shape[0],
                previous_row);
            std::int64_t group_run =
                static_cast<std::int64_t>(initial_run - run_startp) - 1;
            for (;;) {
                std::int64_t next_time = std::numeric_limits<std::int64_t>::max();
                for (std::int64_t knot = 0; knot < knot_count; ++knot) {
                    if (start_cursor[knot] < static_cast<std::int64_t>(admitted.size())) {
                        const auto value = std::min(
                            terminal,
                            timep[admitted[static_cast<std::size_t>(start_cursor[knot])]] +
                                1 + edgep[knot]);
                        next_time = std::min(next_time, value);
                    }
                    if (end_cursor[knot] < static_cast<std::int64_t>(admitted.size())) {
                        const auto value = std::min(
                            terminal,
                            timep[admitted[static_cast<std::size_t>(end_cursor[knot])]] +
                                1 + edgep[knot + 1]);
                        next_time = std::min(next_time, value);
                    }
                }
                if (next_time == std::numeric_limits<std::int64_t>::max()) break;
                const auto* found = std::lower_bound(
                    row_timep + previous_row, row_timep + row_right, next_time);
                std::int64_t next_row = static_cast<std::int64_t>(found - row_timep);
                if (found == row_timep + row_right) {
                    if (next_time != terminal) { invalid |= 1; break; }
                    next_row = row_right;
                } else if (*found != next_time) {
                    invalid |= 1;
                    break;
                }
                if (next_row < previous_row) { invalid |= 1; break; }
                if (next_row > previous_row) {
                    const double sum_first = firstp[next_row] - firstp[previous_row];
                    const double sum_second = secondp[next_row] - secondp[previous_row];
                    for (std::int64_t left_knot = 0; left_knot < knot_count; ++left_knot) {
                        const double left_value = active[left_knot];
                        if (left_value == 0.0) continue;
                        local_gradient[left_knot] += left_value * sum_first;
                        if (!gradient_only) {
                            for (std::int64_t right_knot = 0; right_knot < knot_count; ++right_knot)
                                local_hessian[left_knot * knot_count + right_knot] +=
                                    left_value * active[right_knot] * sum_second;
                        }
                    }
                    if (!gradient_only && selected_count > 0) {
                        std::int64_t run = group_run;
                        auto segment_left = previous_row;
                        while (segment_left < next_row) {
                            const auto segment_right = std::min(
                                next_row,
                                run + 1 < group_run_starts.shape[0]
                                    ? run_startp[run + 1]
                                    : row_count);
                            const double local_second =
                                secondp[segment_right] - secondp[segment_left];
                            const auto group = static_cast<std::int64_t>(run_idp[run]);
                            const auto selected_begin =
                                selected_offsets[static_cast<std::size_t>(group)];
                            const auto selected_end =
                                selected_offsets[static_cast<std::size_t>(group + 1)];
                            for (auto nonzero = selected_begin;
                                 nonzero < selected_end;
                                 ++nonzero) {
                                const auto selected = static_cast<std::int64_t>(
                                    selected_indices[static_cast<std::size_t>(nonzero)]);
                                const double parent =
                                    selected_values[static_cast<std::size_t>(nonzero)];
                                for (std::int64_t knot = 0; knot < knot_count; ++knot)
                                    local_cross[selected * knot_count + knot] +=
                                        parent * active[knot] * local_second;
                            }
                            segment_left = segment_right;
                            if (
                                run + 1 < group_run_starts.shape[0] &&
                                segment_right == run_startp[run + 1])
                                ++run;
                        }
                        group_run = run;
                    }
                }
                for (std::int64_t knot = 0; knot < knot_count; ++knot) {
                    while (start_cursor[knot] < static_cast<std::int64_t>(admitted.size())) {
                        const auto value = std::min(
                            terminal,
                            timep[admitted[static_cast<std::size_t>(start_cursor[knot])]] +
                                1 + edgep[knot]);
                        if (value != next_time) break;
                        active[knot] += scalep[knot];
                        ++start_cursor[knot];
                    }
                    while (end_cursor[knot] < static_cast<std::int64_t>(admitted.size())) {
                        const auto value = std::min(
                            terminal,
                            timep[admitted[static_cast<std::size_t>(end_cursor[knot])]] +
                                1 + edgep[knot + 1]);
                        if (value != next_time) break;
                        active[knot] -= scalep[knot];
                        ++end_cursor[knot];
                    }
                }
                previous_row = next_row;
            }
        }
    }
    // Reduce independent task accumulators in deterministic candidate/entity
    // order. Candidates remain independent and can be reduced in parallel.
#pragma omp parallel for schedule(static) num_threads(execution_workers)
    for (std::int64_t candidate = 0; candidate < candidate_count; ++candidate) {
        auto* local_gradient = gradientp + candidate * knot_count;
        auto* local_hessian = hessianp + candidate * knot_count * knot_count;
        auto* local_cross = crossp + candidate * selected_count * knot_count;
        const auto task_begin =
            candidate_task_offsets[static_cast<std::size_t>(candidate)];
        const auto task_end =
            candidate_task_offsets[static_cast<std::size_t>(candidate + 1)];
        for (auto task_index = task_begin; task_index < task_end; ++task_index) {
            const auto* source_gradient =
                task_gradient.data() + task_index * knot_count;
            for (std::int64_t knot = 0; knot < knot_count; ++knot)
                local_gradient[knot] += source_gradient[knot];
            if (gradient_only) continue;
            const auto* source_hessian =
                task_hessian.data() + task_index * knot_count * knot_count;
            for (std::int64_t index = 0; index < knot_count * knot_count; ++index)
                local_hessian[index] += source_hessian[index];
            const auto* source_cross =
                task_cross.data() + task_index * selected_count * knot_count;
            for (std::int64_t index = 0; index < selected_count * knot_count; ++index)
                local_cross[index] += source_cross[index];
        }
    }
    Py_END_ALLOW_THREADS
    release();
    if (invalid) {
        PyErr_SetString(PyExc_ValueError,
                        "continuous completion boundary is not in the risk grid");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* continuous_single_block_profiles(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *spans_obj, *candidate_starts_obj,
        *candidate_ends_obj, *candidate_windows_obj, *entity_ends_obj,
        *grid_offsets_obj, *row_times_obj, *knot_edges_obj, *knot_scales_obj,
        *coefficients_obj, *output_obj;
    int requested_workers = 0;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOOOOOi", &entities_obj, &times_obj, &spans_obj,
            &candidate_starts_obj, &candidate_ends_obj, &candidate_windows_obj,
            &entity_ends_obj, &grid_offsets_obj, &row_times_obj,
            &knot_edges_obj, &knot_scales_obj, &coefficients_obj, &output_obj,
            &requested_workers)) {
        return nullptr;
    }
    Py_buffer entities{}, times{}, spans{}, candidate_starts{}, candidate_ends{},
        candidate_windows{}, entity_ends{}, grid_offsets{}, row_times{},
        knot_edges{}, knot_scales{}, coefficients{}, output{};
    int acquired = 0;
    auto release = [&]() {
        if (acquired >= 13) PyBuffer_Release(&output);
        if (acquired >= 12) PyBuffer_Release(&coefficients);
        if (acquired >= 11) PyBuffer_Release(&knot_scales);
        if (acquired >= 10) PyBuffer_Release(&knot_edges);
        if (acquired >= 9) PyBuffer_Release(&row_times);
        if (acquired >= 8) PyBuffer_Release(&grid_offsets);
        if (acquired >= 7) PyBuffer_Release(&entity_ends);
        if (acquired >= 6) PyBuffer_Release(&candidate_windows);
        if (acquired >= 5) PyBuffer_Release(&candidate_ends);
        if (acquired >= 4) PyBuffer_Release(&candidate_starts);
        if (acquired >= 3) PyBuffer_Release(&spans);
        if (acquired >= 2) PyBuffer_Release(&times);
        if (acquired >= 1) PyBuffer_Release(&entities);
    };
    if (!int32_buffer(entities_obj, &entities, 1, false)) return nullptr;
    ++acquired;
    if (!int64_buffer(times_obj, &times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(spans_obj, &spans, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(candidate_starts_obj, &candidate_starts, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(candidate_ends_obj, &candidate_ends, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(candidate_windows_obj, &candidate_windows, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(entity_ends_obj, &entity_ends, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(grid_offsets_obj, &grid_offsets, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(row_times_obj, &row_times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(knot_edges_obj, &knot_edges, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(knot_scales_obj, &knot_scales, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(coefficients_obj, &coefficients, 2, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(output_obj, &output, 2, true)) { release(); return nullptr; }
    ++acquired;

    const auto completion_count = entities.shape[0];
    const auto candidate_count = candidate_starts.shape[0];
    const auto entity_count = entity_ends.shape[0];
    const auto row_count = row_times.shape[0];
    const auto knot_count = knot_scales.shape[0];
    const auto* offsetp = static_cast<const std::int64_t*>(grid_offsets.buf);
    const auto* edgep = static_cast<const std::int64_t*>(knot_edges.buf);
    bool valid = requested_workers >= 0 && times.shape[0] == completion_count &&
                 spans.shape[0] == completion_count &&
                 candidate_ends.shape[0] == candidate_count &&
                 candidate_windows.shape[0] == candidate_count &&
                 grid_offsets.shape[0] == entity_count + 1 &&
                 knot_edges.shape[0] == knot_count + 1 && knot_count > 0 &&
                 coefficients.shape[0] == candidate_count &&
                 coefficients.shape[1] == knot_count &&
                 output.shape[0] == candidate_count &&
                 output.shape[1] == row_count && offsetp[0] == 0 &&
                 offsetp[entity_count] == row_count;
    for (std::int64_t index = 0; valid && index < candidate_count; ++index) {
        const auto begin = static_cast<const std::int64_t*>(candidate_starts.buf)[index];
        const auto finish = static_cast<const std::int64_t*>(candidate_ends.buf)[index];
        valid = begin >= 0 && begin <= finish && finish <= completion_count &&
                static_cast<const std::int64_t*>(candidate_windows.buf)[index] >= 0;
    }
    for (std::int64_t index = 0; valid && index < knot_count; ++index) {
        valid = edgep[index] < edgep[index + 1] &&
                std::isfinite(static_cast<const double*>(knot_scales.buf)[index]) &&
                static_cast<const double*>(knot_scales.buf)[index] > 0.0;
    }
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "continuous profile buffer mismatch");
        release();
        return nullptr;
    }
    const auto* entityp = static_cast<const std::int32_t*>(entities.buf);
    const auto* timep = static_cast<const std::int64_t*>(times.buf);
    const auto* spanp = static_cast<const std::int64_t*>(spans.buf);
    const auto* candidate_startp = static_cast<const std::int64_t*>(candidate_starts.buf);
    const auto* candidate_endp = static_cast<const std::int64_t*>(candidate_ends.buf);
    const auto* windowp = static_cast<const std::int64_t*>(candidate_windows.buf);
    const auto* entity_endp = static_cast<const std::int64_t*>(entity_ends.buf);
    const auto* row_timep = static_cast<const std::int64_t*>(row_times.buf);
    const auto* scalep = static_cast<const double*>(knot_scales.buf);
    const auto* coefficientp = static_cast<const double*>(coefficients.buf);
    auto* outputp = static_cast<double*>(output.buf);
    std::fill(outputp, outputp + candidate_count * row_count, 0.0);
    int invalid = 0;
    const int workers = requested_workers > 0 ? requested_workers : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers) reduction(| : invalid)
    for (std::int64_t candidate = 0; candidate < candidate_count; ++candidate) {
        const auto completion_begin = candidate_startp[candidate];
        const auto completion_end = candidate_endp[candidate];
        const auto maximum_span = windowp[candidate];
        auto* local_output = outputp + candidate * row_count;
        const auto* local_coefficients = coefficientp + candidate * knot_count;
        std::int64_t cursor = completion_begin;
        std::vector<std::int64_t> admitted;
        std::vector<std::int64_t> start_cursor(static_cast<std::size_t>(knot_count));
        std::vector<std::int64_t> end_cursor(static_cast<std::size_t>(knot_count));
        std::vector<double> active(static_cast<std::size_t>(knot_count));
        while (cursor < completion_end) {
            const auto entity = static_cast<std::int64_t>(entityp[cursor]);
            if (entity < 0 || entity >= entity_count) { invalid |= 1; break; }
            auto entity_finish = cursor + 1;
            while (entity_finish < completion_end && entityp[entity_finish] == entity)
                ++entity_finish;
            admitted.clear();
            admitted.reserve(static_cast<std::size_t>(entity_finish - cursor));
            for (auto item = cursor; item < entity_finish; ++item) {
                if (spanp[item] <= maximum_span && timep[item] < entity_endp[entity])
                    admitted.push_back(item);
            }
            cursor = entity_finish;
            if (admitted.empty()) continue;
            std::fill(start_cursor.begin(), start_cursor.end(), 0);
            std::fill(end_cursor.begin(), end_cursor.end(), 0);
            std::fill(active.begin(), active.end(), 0.0);
            const auto row_left = offsetp[entity];
            const auto row_right = offsetp[entity + 1];
            const auto terminal = entity_endp[entity] + 1;
            std::int64_t previous_row = row_left;
            for (;;) {
                std::int64_t next_time = std::numeric_limits<std::int64_t>::max();
                for (std::int64_t knot = 0; knot < knot_count; ++knot) {
                    if (start_cursor[knot] < static_cast<std::int64_t>(admitted.size()))
                        next_time = std::min(next_time, std::min(
                            terminal, timep[admitted[static_cast<std::size_t>(start_cursor[knot])]] + 1 + edgep[knot]));
                    if (end_cursor[knot] < static_cast<std::int64_t>(admitted.size()))
                        next_time = std::min(next_time, std::min(
                            terminal, timep[admitted[static_cast<std::size_t>(end_cursor[knot])]] + 1 + edgep[knot + 1]));
                }
                if (next_time == std::numeric_limits<std::int64_t>::max()) break;
                const auto* found = std::lower_bound(
                    row_timep + row_left, row_timep + row_right, next_time);
                std::int64_t next_row = static_cast<std::int64_t>(found - row_timep);
                if (found == row_timep + row_right) {
                    if (next_time != terminal) { invalid |= 1; break; }
                    next_row = row_right;
                } else if (*found != next_time || next_row < previous_row) {
                    invalid |= 1;
                    break;
                }
                if (next_row > previous_row) {
                    double value = 0.0;
                    for (std::int64_t knot = 0; knot < knot_count; ++knot)
                        value += active[knot] * local_coefficients[knot];
                    std::fill(local_output + previous_row, local_output + next_row, value);
                }
                for (std::int64_t knot = 0; knot < knot_count; ++knot) {
                    while (start_cursor[knot] < static_cast<std::int64_t>(admitted.size())) {
                        const auto value = std::min(
                            terminal, timep[admitted[static_cast<std::size_t>(start_cursor[knot])]] + 1 + edgep[knot]);
                        if (value != next_time) break;
                        active[knot] += scalep[knot];
                        ++start_cursor[knot];
                    }
                    while (end_cursor[knot] < static_cast<std::int64_t>(admitted.size())) {
                        const auto value = std::min(
                            terminal, timep[admitted[static_cast<std::size_t>(end_cursor[knot])]] + 1 + edgep[knot + 1]);
                        if (value != next_time) break;
                        active[knot] -= scalep[knot];
                        ++end_cursor[knot];
                    }
                }
                previous_row = next_row;
            }
        }
    }
    Py_END_ALLOW_THREADS
    release();
    if (invalid) {
        PyErr_SetString(PyExc_ValueError,
                        "continuous profile boundary is not in the risk grid");
        return nullptr;
    }
    Py_RETURN_NONE;
}



PyObject* continuous_additive_support_profiles(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *spans_obj, *component_starts_obj,
        *component_ends_obj, *component_windows_obj, *support_offsets_obj,
        *entity_ends_obj, *grid_offsets_obj, *row_times_obj, *knot_edges_obj,
        *knot_scales_obj, *coefficients_obj, *output_obj;
    int requested_workers = 0;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOOOOOOi", &entities_obj, &times_obj, &spans_obj,
            &component_starts_obj, &component_ends_obj, &component_windows_obj,
            &support_offsets_obj, &entity_ends_obj, &grid_offsets_obj,
            &row_times_obj, &knot_edges_obj, &knot_scales_obj,
            &coefficients_obj, &output_obj, &requested_workers)) {
        return nullptr;
    }
    Py_buffer entities{}, times{}, spans{}, component_starts{}, component_ends{},
        component_windows{}, support_offsets{}, entity_ends{}, grid_offsets{},
        row_times{}, knot_edges{}, knot_scales{}, coefficients{}, output{};
    int acquired = 0;
    auto release = [&]() {
        if (acquired >= 14) PyBuffer_Release(&output);
        if (acquired >= 13) PyBuffer_Release(&coefficients);
        if (acquired >= 12) PyBuffer_Release(&knot_scales);
        if (acquired >= 11) PyBuffer_Release(&knot_edges);
        if (acquired >= 10) PyBuffer_Release(&row_times);
        if (acquired >= 9) PyBuffer_Release(&grid_offsets);
        if (acquired >= 8) PyBuffer_Release(&entity_ends);
        if (acquired >= 7) PyBuffer_Release(&support_offsets);
        if (acquired >= 6) PyBuffer_Release(&component_windows);
        if (acquired >= 5) PyBuffer_Release(&component_ends);
        if (acquired >= 4) PyBuffer_Release(&component_starts);
        if (acquired >= 3) PyBuffer_Release(&spans);
        if (acquired >= 2) PyBuffer_Release(&times);
        if (acquired >= 1) PyBuffer_Release(&entities);
    };
    if (!int32_buffer(entities_obj, &entities, 1, false)) return nullptr;
    ++acquired;
    if (!int64_buffer(times_obj, &times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(spans_obj, &spans, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(component_starts_obj, &component_starts, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(component_ends_obj, &component_ends, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(component_windows_obj, &component_windows, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(support_offsets_obj, &support_offsets, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(entity_ends_obj, &entity_ends, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(grid_offsets_obj, &grid_offsets, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(row_times_obj, &row_times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(knot_edges_obj, &knot_edges, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(knot_scales_obj, &knot_scales, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(coefficients_obj, &coefficients, 2, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(output_obj, &output, 2, true)) { release(); return nullptr; }
    ++acquired;

    const auto completion_count = entities.shape[0];
    const auto component_count = component_starts.shape[0];
    const auto support_count = support_offsets.shape[0] - 1;
    const auto entity_count = entity_ends.shape[0];
    const auto row_count = row_times.shape[0];
    const auto knot_count = knot_scales.shape[0];
    const auto* offsetp = static_cast<const std::int64_t*>(grid_offsets.buf);
    const auto* edgep = static_cast<const std::int64_t*>(knot_edges.buf);
    const auto* support_offsetp =
        static_cast<const std::int64_t*>(support_offsets.buf);
    bool valid = requested_workers >= 0 && support_count >= 0 &&
                 times.shape[0] == completion_count &&
                 spans.shape[0] == completion_count &&
                 component_ends.shape[0] == component_count &&
                 component_windows.shape[0] == component_count &&
                 grid_offsets.shape[0] == entity_count + 1 &&
                 knot_edges.shape[0] == knot_count + 1 && knot_count > 0 &&
                 coefficients.shape[0] == component_count &&
                 coefficients.shape[1] == knot_count &&
                 output.shape[0] == support_count &&
                 output.shape[1] == row_count && offsetp[0] == 0 &&
                 offsetp[entity_count] == row_count &&
                 support_offsets.shape[0] >= 1 && support_offsetp[0] == 0 &&
                 support_offsetp[support_count] == component_count;
    for (std::int64_t index = 0; valid && index < component_count; ++index) {
        const auto begin = static_cast<const std::int64_t*>(component_starts.buf)[index];
        const auto finish = static_cast<const std::int64_t*>(component_ends.buf)[index];
        valid = begin >= 0 && begin <= finish && finish <= completion_count &&
                static_cast<const std::int64_t*>(component_windows.buf)[index] >= 0;
    }
    for (std::int64_t index = 0; valid && index < support_count; ++index)
        valid = support_offsetp[index] <= support_offsetp[index + 1];
    for (std::int64_t index = 0; valid && index < knot_count; ++index) {
        valid = edgep[index] < edgep[index + 1] &&
                std::isfinite(static_cast<const double*>(knot_scales.buf)[index]) &&
                static_cast<const double*>(knot_scales.buf)[index] > 0.0;
    }
    if (!valid) {
        PyErr_SetString(PyExc_ValueError,
                        "continuous additive profile buffer mismatch");
        release();
        return nullptr;
    }

    const auto* entityp = static_cast<const std::int32_t*>(entities.buf);
    const auto* timep = static_cast<const std::int64_t*>(times.buf);
    const auto* spanp = static_cast<const std::int64_t*>(spans.buf);
    const auto* component_startp =
        static_cast<const std::int64_t*>(component_starts.buf);
    const auto* component_endp =
        static_cast<const std::int64_t*>(component_ends.buf);
    const auto* windowp =
        static_cast<const std::int64_t*>(component_windows.buf);
    const auto* entity_endp = static_cast<const std::int64_t*>(entity_ends.buf);
    const auto* row_timep = static_cast<const std::int64_t*>(row_times.buf);
    const auto* scalep = static_cast<const double*>(knot_scales.buf);
    const auto* coefficientp = static_cast<const double*>(coefficients.buf);
    auto* outputp = static_cast<double*>(output.buf);
    std::fill(outputp, outputp + support_count * row_count, 0.0);
    int invalid = 0;
    const int workers = requested_workers > 0 ? requested_workers : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers) reduction(| : invalid)
    for (std::int64_t support = 0; support < support_count; ++support) {
        auto* local_output = outputp + support * row_count;
        std::vector<std::int64_t> admitted;
        std::vector<std::int64_t> start_cursor(static_cast<std::size_t>(knot_count));
        std::vector<std::int64_t> end_cursor(static_cast<std::size_t>(knot_count));
        std::vector<double> active(static_cast<std::size_t>(knot_count));
        for (auto component = support_offsetp[support];
             component < support_offsetp[support + 1]; ++component) {
            const auto completion_begin = component_startp[component];
            const auto completion_end = component_endp[component];
            const auto maximum_span = windowp[component];
            const auto* local_coefficients = coefficientp + component * knot_count;
            std::int64_t cursor = completion_begin;
            while (cursor < completion_end) {
                const auto entity = static_cast<std::int64_t>(entityp[cursor]);
                if (entity < 0 || entity >= entity_count) { invalid |= 1; break; }
                auto entity_finish = cursor + 1;
                while (entity_finish < completion_end && entityp[entity_finish] == entity)
                    ++entity_finish;
                admitted.clear();
                admitted.reserve(static_cast<std::size_t>(entity_finish - cursor));
                for (auto item = cursor; item < entity_finish; ++item) {
                    if (spanp[item] <= maximum_span && timep[item] < entity_endp[entity])
                        admitted.push_back(item);
                }
                cursor = entity_finish;
                if (admitted.empty()) continue;
                std::fill(start_cursor.begin(), start_cursor.end(), 0);
                std::fill(end_cursor.begin(), end_cursor.end(), 0);
                std::fill(active.begin(), active.end(), 0.0);
                const auto row_left = offsetp[entity];
                const auto row_right = offsetp[entity + 1];
                const auto terminal = entity_endp[entity] + 1;
                std::int64_t previous_row = row_left;
                for (;;) {
                    std::int64_t next_time = std::numeric_limits<std::int64_t>::max();
                    for (std::int64_t knot = 0; knot < knot_count; ++knot) {
                        if (start_cursor[knot] < static_cast<std::int64_t>(admitted.size()))
                            next_time = std::min(next_time, std::min(
                                terminal, timep[admitted[static_cast<std::size_t>(start_cursor[knot])]] + 1 + edgep[knot]));
                        if (end_cursor[knot] < static_cast<std::int64_t>(admitted.size()))
                            next_time = std::min(next_time, std::min(
                                terminal, timep[admitted[static_cast<std::size_t>(end_cursor[knot])]] + 1 + edgep[knot + 1]));
                    }
                    if (next_time == std::numeric_limits<std::int64_t>::max()) break;
                    const auto* found = std::lower_bound(
                        row_timep + previous_row, row_timep + row_right, next_time);
                    std::int64_t next_row = static_cast<std::int64_t>(found - row_timep);
                    if (found == row_timep + row_right) {
                        if (next_time != terminal) { invalid |= 1; break; }
                        next_row = row_right;
                    } else if (*found != next_time || next_row < previous_row) {
                        invalid |= 1;
                        break;
                    }
                    if (next_row > previous_row) {
                        double value = 0.0;
                        for (std::int64_t knot = 0; knot < knot_count; ++knot)
                            value += active[knot] * local_coefficients[knot];
                        if (value != 0.0) {
                            for (auto row = previous_row; row < next_row; ++row)
                                local_output[row] += value;
                        }
                    }
                    for (std::int64_t knot = 0; knot < knot_count; ++knot) {
                        while (start_cursor[knot] < static_cast<std::int64_t>(admitted.size())) {
                            const auto value = std::min(
                                terminal, timep[admitted[static_cast<std::size_t>(start_cursor[knot])]] + 1 + edgep[knot]);
                            if (value != next_time) break;
                            active[knot] += scalep[knot];
                            ++start_cursor[knot];
                        }
                        while (end_cursor[knot] < static_cast<std::int64_t>(admitted.size())) {
                            const auto value = std::min(
                                terminal, timep[admitted[static_cast<std::size_t>(end_cursor[knot])]] + 1 + edgep[knot + 1]);
                            if (value != next_time) break;
                            active[knot] -= scalep[knot];
                            ++end_cursor[knot];
                        }
                    }
                    previous_row = next_row;
                }
            }
        }
    }
    Py_END_ALLOW_THREADS
    release();
    if (invalid) {
        PyErr_SetString(PyExc_ValueError,
                        "continuous additive profile boundary is not in the risk grid");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* continuous_single_block_profile_distances(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *spans_obj, *candidate_starts_obj,
        *candidate_ends_obj, *candidate_windows_obj, *entity_ends_obj,
        *grid_offsets_obj, *row_times_obj, *knot_edges_obj, *knot_scales_obj,
        *coefficients_obj, *sqrt_fisher_obj, *current_obj, *references_obj,
        *output_obj;
    double tolerance = 0.0;
    int requested_workers = 0;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOOOOOOOdOi", &entities_obj, &times_obj, &spans_obj,
            &candidate_starts_obj, &candidate_ends_obj, &candidate_windows_obj,
            &entity_ends_obj, &grid_offsets_obj, &row_times_obj,
            &knot_edges_obj, &knot_scales_obj, &coefficients_obj,
            &sqrt_fisher_obj, &current_obj, &references_obj, &tolerance,
            &output_obj, &requested_workers)) {
        return nullptr;
    }
    Py_buffer entities{}, times{}, spans{}, candidate_starts{}, candidate_ends{},
        candidate_windows{}, entity_ends{}, grid_offsets{}, row_times{},
        knot_edges{}, knot_scales{}, coefficients{}, sqrt_fisher{}, current{},
        references{}, output{};
    int acquired = 0;
    auto release = [&]() {
        if (acquired >= 16) PyBuffer_Release(&output);
        if (acquired >= 15) PyBuffer_Release(&references);
        if (acquired >= 14) PyBuffer_Release(&current);
        if (acquired >= 13) PyBuffer_Release(&sqrt_fisher);
        if (acquired >= 12) PyBuffer_Release(&coefficients);
        if (acquired >= 11) PyBuffer_Release(&knot_scales);
        if (acquired >= 10) PyBuffer_Release(&knot_edges);
        if (acquired >= 9) PyBuffer_Release(&row_times);
        if (acquired >= 8) PyBuffer_Release(&grid_offsets);
        if (acquired >= 7) PyBuffer_Release(&entity_ends);
        if (acquired >= 6) PyBuffer_Release(&candidate_windows);
        if (acquired >= 5) PyBuffer_Release(&candidate_ends);
        if (acquired >= 4) PyBuffer_Release(&candidate_starts);
        if (acquired >= 3) PyBuffer_Release(&spans);
        if (acquired >= 2) PyBuffer_Release(&times);
        if (acquired >= 1) PyBuffer_Release(&entities);
    };
    if (!int32_buffer(entities_obj, &entities, 1, false)) return nullptr;
    ++acquired;
    if (!int64_buffer(times_obj, &times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(spans_obj, &spans, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(candidate_starts_obj, &candidate_starts, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(candidate_ends_obj, &candidate_ends, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(candidate_windows_obj, &candidate_windows, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(entity_ends_obj, &entity_ends, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(grid_offsets_obj, &grid_offsets, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(row_times_obj, &row_times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(knot_edges_obj, &knot_edges, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(knot_scales_obj, &knot_scales, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(coefficients_obj, &coefficients, 2, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(sqrt_fisher_obj, &sqrt_fisher, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(current_obj, &current, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(references_obj, &references, 2, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(output_obj, &output, 2, true)) { release(); return nullptr; }
    ++acquired;

    const auto completion_count = entities.shape[0];
    const auto candidate_count = candidate_starts.shape[0];
    const auto entity_count = entity_ends.shape[0];
    const auto row_count = row_times.shape[0];
    const auto knot_count = knot_scales.shape[0];
    const auto reference_count = references.shape[1];
    const auto* offsetp = static_cast<const std::int64_t*>(grid_offsets.buf);
    const auto* edgep = static_cast<const std::int64_t*>(knot_edges.buf);
    bool valid = requested_workers >= 0 && tolerance >= 0.0 &&
                 times.shape[0] == completion_count &&
                 spans.shape[0] == completion_count &&
                 candidate_ends.shape[0] == candidate_count &&
                 candidate_windows.shape[0] == candidate_count &&
                 grid_offsets.shape[0] == entity_count + 1 &&
                 knot_edges.shape[0] == knot_count + 1 && knot_count > 0 &&
                 coefficients.shape[0] == candidate_count &&
                 coefficients.shape[1] == knot_count &&
                 sqrt_fisher.shape[0] == row_count &&
                 current.shape[0] == row_count &&
                 references.shape[0] == row_count &&
                 reference_count > 0 &&
                 output.shape[0] == candidate_count &&
                 output.shape[1] >= 1 &&
                 offsetp[0] == 0 && offsetp[entity_count] == row_count;
    for (std::int64_t index = 0; valid && index < candidate_count; ++index) {
        const auto begin = static_cast<const std::int64_t*>(candidate_starts.buf)[index];
        const auto finish = static_cast<const std::int64_t*>(candidate_ends.buf)[index];
        valid = begin >= 0 && begin <= finish && finish <= completion_count &&
                static_cast<const std::int64_t*>(candidate_windows.buf)[index] >= 0;
    }
    for (std::int64_t index = 0; valid && index < knot_count; ++index) {
        valid = edgep[index] < edgep[index + 1] &&
                std::isfinite(static_cast<const double*>(knot_scales.buf)[index]) &&
                static_cast<const double*>(knot_scales.buf)[index] > 0.0;
    }
    if (!valid) {
        PyErr_SetString(PyExc_ValueError,
                        "continuous profile-distance buffer mismatch");
        release();
        return nullptr;
    }

    const auto* entityp = static_cast<const std::int32_t*>(entities.buf);
    const auto* timep = static_cast<const std::int64_t*>(times.buf);
    const auto* spanp = static_cast<const std::int64_t*>(spans.buf);
    const auto* candidate_startp = static_cast<const std::int64_t*>(candidate_starts.buf);
    const auto* candidate_endp = static_cast<const std::int64_t*>(candidate_ends.buf);
    const auto* windowp = static_cast<const std::int64_t*>(candidate_windows.buf);
    const auto* entity_endp = static_cast<const std::int64_t*>(entity_ends.buf);
    const auto* row_timep = static_cast<const std::int64_t*>(row_times.buf);
    const auto* scalep = static_cast<const double*>(knot_scales.buf);
    const auto* coefficientp = static_cast<const double*>(coefficients.buf);
    const auto* sqrtp = static_cast<const double*>(sqrt_fisher.buf);
    const auto* currentp = static_cast<const double*>(current.buf);
    const auto* referencep = static_cast<const double*>(references.buf);
    auto* outputp = static_cast<double*>(output.buf);
    const auto output_stride = output.shape[1];
    std::fill(outputp, outputp + candidate_count * output_stride, 0.0);
    std::vector<double> reference_from_current(
        static_cast<std::size_t>(reference_count), 0.0);
    bool first_reference_matches = true;
    for (std::int64_t row = 0; row < row_count; ++row) {
        const auto* local_references = referencep + row * reference_count;
        first_reference_matches =
            first_reference_matches && local_references[0] == currentp[row];
        for (std::int64_t reference = 1;
             reference < reference_count; ++reference) {
            const double difference = currentp[row] - local_references[reference];
            if (difference != 0.0)
                reference_from_current[static_cast<std::size_t>(reference)] +=
                    difference * difference;
        }
    }
    if (!first_reference_matches) {
        PyErr_SetString(PyExc_ValueError,
                        "first Fisher reference must be the current profile");
        release();
        return nullptr;
    }
    int invalid = 0;
    int allocation_failed = 0;
    const int workers = requested_workers > 0
        ? requested_workers : omp_get_max_threads();

    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers) reduction(| : invalid, allocation_failed)
    for (std::int64_t candidate = 0; candidate < candidate_count; ++candidate) {
        try {
            std::vector<double> increment(static_cast<std::size_t>(row_count), 0.0);
            const auto completion_begin = candidate_startp[candidate];
            const auto completion_end = candidate_endp[candidate];
            const auto maximum_span = windowp[candidate];
            const auto* local_coefficients =
                coefficientp + candidate * knot_count;
            std::int64_t cursor = completion_begin;
            std::vector<std::int64_t> admitted;
            std::vector<std::int64_t> start_cursor(
                static_cast<std::size_t>(knot_count));
            std::vector<std::int64_t> end_cursor(
                static_cast<std::size_t>(knot_count));
            std::vector<double> active(static_cast<std::size_t>(knot_count));
            while (cursor < completion_end) {
                const auto entity = static_cast<std::int64_t>(entityp[cursor]);
                if (entity < 0 || entity >= entity_count) {
                    invalid |= 1;
                    break;
                }
                auto entity_finish = cursor + 1;
                while (entity_finish < completion_end &&
                       entityp[entity_finish] == entity)
                    ++entity_finish;
                admitted.clear();
                admitted.reserve(
                    static_cast<std::size_t>(entity_finish - cursor));
                for (auto item = cursor; item < entity_finish; ++item) {
                    if (spanp[item] <= maximum_span &&
                        timep[item] < entity_endp[entity])
                        admitted.push_back(item);
                }
                cursor = entity_finish;
                if (admitted.empty()) continue;
                std::fill(start_cursor.begin(), start_cursor.end(), 0);
                std::fill(end_cursor.begin(), end_cursor.end(), 0);
                std::fill(active.begin(), active.end(), 0.0);
                const auto row_left = offsetp[entity];
                const auto row_right = offsetp[entity + 1];
                const auto terminal = entity_endp[entity] + 1;
                std::int64_t previous_row = row_left;
                for (;;) {
                    std::int64_t next_time =
                        std::numeric_limits<std::int64_t>::max();
                    for (std::int64_t knot = 0; knot < knot_count; ++knot) {
                        if (start_cursor[knot] <
                            static_cast<std::int64_t>(admitted.size()))
                            next_time = std::min(
                                next_time,
                                std::min(
                                    terminal,
                                    timep[admitted[static_cast<std::size_t>(
                                        start_cursor[knot])]] +
                                        1 + edgep[knot]));
                        if (end_cursor[knot] <
                            static_cast<std::int64_t>(admitted.size()))
                            next_time = std::min(
                                next_time,
                                std::min(
                                    terminal,
                                    timep[admitted[static_cast<std::size_t>(
                                        end_cursor[knot])]] +
                                        1 + edgep[knot + 1]));
                    }
                    if (next_time ==
                        std::numeric_limits<std::int64_t>::max())
                        break;
                    const auto* found = std::lower_bound(
                        row_timep + row_left, row_timep + row_right,
                        next_time);
                    std::int64_t next_row =
                        static_cast<std::int64_t>(found - row_timep);
                    if (found == row_timep + row_right) {
                        if (next_time != terminal) {
                            invalid |= 1;
                            break;
                        }
                        next_row = row_right;
                    } else if (*found != next_time ||
                               next_row < previous_row) {
                        invalid |= 1;
                        break;
                    }
                    if (next_row > previous_row) {
                        double value = 0.0;
                        for (std::int64_t knot = 0; knot < knot_count; ++knot)
                            value += active[knot] *
                                     local_coefficients[knot];
                        std::fill(
                            increment.begin() + previous_row,
                            increment.begin() + next_row, value);
                    }
                    for (std::int64_t knot = 0; knot < knot_count; ++knot) {
                        while (start_cursor[knot] <
                               static_cast<std::int64_t>(admitted.size())) {
                            const auto value = std::min(
                                terminal,
                                timep[admitted[static_cast<std::size_t>(
                                    start_cursor[knot])]] +
                                    1 + edgep[knot]);
                            if (value != next_time) break;
                            active[knot] += scalep[knot];
                            ++start_cursor[knot];
                        }
                        while (end_cursor[knot] <
                               static_cast<std::int64_t>(admitted.size())) {
                            const auto value = std::min(
                                terminal,
                                timep[admitted[static_cast<std::size_t>(
                                    end_cursor[knot])]] +
                                    1 + edgep[knot + 1]);
                            if (value != next_time) break;
                            active[knot] -= scalep[knot];
                            ++end_cursor[knot];
                        }
                    }
                    previous_row = next_row;
                }
            }
            auto* local_output = outputp + candidate * output_stride;
            bool has_increment = false;
            double best_distance = 0.0;
            for (std::int64_t row = 0; row < row_count; ++row) {
                const double raw_increment = increment[
                    static_cast<std::size_t>(row)] * sqrtp[row];
                const bool active_increment =
                    std::isfinite(raw_increment) &&
                    std::abs(raw_increment) > tolerance;
                has_increment = has_increment || active_increment;
                double candidate_value =
                    (active_increment ? raw_increment : 0.0) + currentp[row];
                if (!std::isfinite(candidate_value) ||
                    std::abs(candidate_value) <= tolerance)
                    candidate_value = 0.0;
                increment[static_cast<std::size_t>(row)] = candidate_value;
                const double difference = candidate_value - currentp[row];
                if (difference != 0.0)
                    best_distance += difference * difference;
            }
            if (!has_increment) {
                local_output[0] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            const double candidate_current_distance = best_distance;
            const double gamma =
                128.0 * std::numeric_limits<double>::epsilon() *
                static_cast<double>(std::max<std::int64_t>(1, row_count));
            for (std::int64_t reference = 1;
                 reference < reference_count; ++reference) {
                const double current_distance =
                    reference_from_current[static_cast<std::size_t>(reference)];
                const double root_gap = std::abs(
                    std::sqrt(std::max(0.0, current_distance)) -
                    std::sqrt(std::max(0.0, candidate_current_distance)));
                const double rounding_guard = gamma * (
                    1.0 + current_distance + candidate_current_distance +
                    2.0 * std::sqrt(
                        std::max(0.0, current_distance * candidate_current_distance)));
                const double lower =
                    std::max(0.0, root_gap * root_gap - rounding_guard);
                if (lower >= best_distance)
                    continue;
                double distance = 0.0;
                for (std::int64_t row = 0; row < row_count; ++row) {
                    const double difference =
                        increment[static_cast<std::size_t>(row)] -
                        referencep[row * reference_count + reference];
                    if (difference != 0.0)
                        distance += difference * difference;
                    if (distance >= best_distance)
                        break;
                }
                if (distance < best_distance)
                    best_distance = distance;
            }
            local_output[0] = best_distance;
        } catch (const std::bad_alloc&) {
            allocation_failed |= 1;
        }
    }
    Py_END_ALLOW_THREADS
    release();
    if (allocation_failed) {
        PyErr_NoMemory();
        return nullptr;
    }
    if (invalid) {
        PyErr_SetString(PyExc_ValueError,
                        "continuous profile-distance boundary is not in the risk grid");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* safe_shell_counts(PyObject*, PyObject* args) {
    PyObject *completion_offsets_obj, *completion_times_obj,
        *completion_spans_obj, *antecedent_indices_obj, *starts_obj, *ends_obj,
        *grid_offsets_obj, *windows_obj, *window_counts_obj, *signed_state_obj,
        *baseline_groups_obj, *event_state_obj, *output_obj;
    int horizon = 0, group_count = 0;
    double exposure = 0.0, weight = 0.0;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOOOOiiddO", &completion_offsets_obj,
            &completion_times_obj, &completion_spans_obj,
            &antecedent_indices_obj, &starts_obj, &ends_obj, &grid_offsets_obj,
            &windows_obj, &window_counts_obj, &signed_state_obj,
            &baseline_groups_obj, &event_state_obj, &horizon, &group_count,
            &exposure, &weight, &output_obj)) {
        return nullptr;
    }
    Py_buffer completion_offsets{}, completion_times{}, completion_spans{},
        antecedent_indices{}, starts{}, ends{}, grid_offsets{}, windows{},
        window_counts{}, signed_state{}, baseline_groups{}, event_state{}, output{};
    int acquired = 0;
    auto release = [&]() {
        if (acquired >= 13) PyBuffer_Release(&output);
        if (acquired >= 12) PyBuffer_Release(&event_state);
        if (acquired >= 11) PyBuffer_Release(&baseline_groups);
        if (acquired >= 10) PyBuffer_Release(&signed_state);
        if (acquired >= 9) PyBuffer_Release(&window_counts);
        if (acquired >= 8) PyBuffer_Release(&windows);
        if (acquired >= 7) PyBuffer_Release(&grid_offsets);
        if (acquired >= 6) PyBuffer_Release(&ends);
        if (acquired >= 5) PyBuffer_Release(&starts);
        if (acquired >= 4) PyBuffer_Release(&antecedent_indices);
        if (acquired >= 3) PyBuffer_Release(&completion_spans);
        if (acquired >= 2) PyBuffer_Release(&completion_times);
        if (acquired >= 1) PyBuffer_Release(&completion_offsets);
    };
    if (!int64_buffer(completion_offsets_obj, &completion_offsets, 2, false)) return nullptr;
    ++acquired;
    if (!int64_buffer(completion_times_obj, &completion_times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(completion_spans_obj, &completion_spans, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int32_buffer(antecedent_indices_obj, &antecedent_indices, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(starts_obj, &starts, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(ends_obj, &ends, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(grid_offsets_obj, &grid_offsets, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(windows_obj, &windows, 2, false)) { release(); return nullptr; }
    ++acquired;
    if (!int32_buffer(window_counts_obj, &window_counts, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!uint8_buffer(signed_state_obj, &signed_state, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int32_buffer(baseline_groups_obj, &baseline_groups, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!uint8_buffer(event_state_obj, &event_state, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(output_obj, &output, 5, true)) { release(); return nullptr; }
    ++acquired;

    const auto requested = antecedent_indices.shape[0];
    const auto entity_count = starts.shape[0];
    const auto packed_antecedents = completion_offsets.shape[0];
    const auto maximum_windows = windows.shape[1];
    const auto n_grid = signed_state.shape[0];
    bool valid = requested == windows.shape[0] && requested == window_counts.shape[0] &&
                 ends.shape[0] == entity_count && grid_offsets.shape[0] == entity_count + 1 &&
                 completion_offsets.shape[1] == entity_count + 1 &&
                 completion_times.shape[0] == completion_spans.shape[0] &&
                 baseline_groups.shape[0] == n_grid && event_state.shape[0] == n_grid &&
                 grid_offsets.shape[0] > 0 && grid_offsets.shape[0] == entity_count + 1 &&
                 static_cast<std::int64_t*>(grid_offsets.buf)[entity_count] == n_grid &&
                 output.shape[0] == requested && output.shape[1] == 4 &&
                 output.shape[2] == maximum_windows && output.shape[3] == 3 &&
                 output.shape[4] == group_count && horizon >= 0 && group_count > 0 &&
                 exposure > 0.0 && weight > 0.0;
    const auto* selected = static_cast<const std::int32_t*>(antecedent_indices.buf);
    const auto* counts = static_cast<const std::int32_t*>(window_counts.buf);
    for (std::int64_t index = 0; valid && index < requested; ++index) {
        valid = selected[index] >= 0 && selected[index] < packed_antecedents &&
                counts[index] > 0 && counts[index] <= maximum_windows;
    }
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "safe shell count buffer mismatch");
        release();
        return nullptr;
    }

    const auto* packed_offsets = static_cast<const std::int64_t*>(completion_offsets.buf);
    const auto* times = static_cast<const std::int64_t*>(completion_times.buf);
    const auto* spans = static_cast<const std::int64_t*>(completion_spans.buf);
    const auto* startp = static_cast<const std::int64_t*>(starts.buf);
    const auto* endp = static_cast<const std::int64_t*>(ends.buf);
    const auto* gridp = static_cast<const std::int64_t*>(grid_offsets.buf);
    const auto* wp = static_cast<const std::int64_t*>(windows.buf);
    const auto* state = static_cast<const std::uint8_t*>(signed_state.buf);
    const auto* groups = static_cast<const std::int32_t*>(baseline_groups.buf);
    const auto* events = static_cast<const std::uint8_t*>(event_state.buf);
    auto* out = static_cast<double*>(output.buf);
    std::fill(out, out + output.shape[0] * output.shape[1] * output.shape[2] *
                         output.shape[3] * output.shape[4], 0.0);

    bool invalid = false;
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t request = 0; request < requested; ++request) {
        const auto packed = static_cast<std::int64_t>(selected[request]);
        const auto* offsets = packed_offsets + packed * (entity_count + 1);
        const auto* local_windows = wp + request * maximum_windows;
        const auto local_window_count = static_cast<std::int64_t>(counts[request]);
        for (std::int64_t entity = 0; entity < entity_count; ++entity) {
            const auto begin = offsets[entity];
            const auto finish = offsets[entity + 1];
            if (begin == finish || horizon <= 0) continue;
            const auto first_time = std::max<std::int64_t>(startp[entity], times[begin] + 1);
            const auto last_time = std::min<std::int64_t>(endp[entity], times[finish - 1] + horizon);
            std::deque<std::int64_t> minimum;
            auto cursor = begin;
            for (std::int64_t time = first_time; time <= last_time; ++time) {
                while (cursor < finish && times[cursor] < time) {
                    while (!minimum.empty() && spans[minimum.back()] >= spans[cursor])
                        minimum.pop_back();
                    minimum.push_back(cursor++);
                }
                const auto oldest = time - horizon;
                while (!minimum.empty() && times[minimum.front()] < oldest)
                    minimum.pop_front();
                if (minimum.empty()) continue;
                const auto span = spans[minimum.front()];
                const auto* admitted = std::lower_bound(
                    local_windows, local_windows + local_window_count, span);
                if (admitted == local_windows + local_window_count) continue;
                const auto window = static_cast<std::int64_t>(admitted - local_windows);
                const auto row = gridp[entity] + time - startp[entity];
                if (row < 0 || row >= n_grid || state[row] > 3 || groups[row] < 0 ||
                    groups[row] >= group_count || events[row] > 1) {
                    // All buffers are immutable, so a benign shared flag is
                    // sufficient; no output from an invalid call is consumed.
                    // Use an OpenMP atomic write to avoid a data race.
                    #pragma omp atomic write
                    invalid = true;
                    continue;
                }
                const auto category = static_cast<std::int64_t>(state[row]);
                const auto group = static_cast<std::int64_t>(groups[row]);
                const auto base = (((request * 4 + category) * maximum_windows + window) * 3) * group_count + group;
                const auto event = weight * static_cast<double>(events[row]);
                out[base] += exposure * weight;
                out[base + group_count] += weight - event;
                out[base + 2 * group_count] += event;
            }
        }
    }
    Py_END_ALLOW_THREADS
    release();
    if (invalid) {
        PyErr_SetString(PyExc_ValueError, "safe shell count index is invalid");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* label_run_ends(PyObject*, PyObject* args) {
    PyObject *signed_state_obj, *baseline_groups_obj, *event_state_obj,
        *grid_offsets_obj, *output_obj;
    if (!PyArg_ParseTuple(args, "OOOOO", &signed_state_obj,
                          &baseline_groups_obj, &event_state_obj,
                          &grid_offsets_obj, &output_obj))
        return nullptr;
    Py_buffer signed_state{}, baseline_groups{}, event_state{}, grid_offsets{},
        output{};
    if (!uint8_buffer(signed_state_obj, &signed_state, 1, false)) return nullptr;
    if (!int32_buffer(baseline_groups_obj, &baseline_groups, 1, false)) {
        PyBuffer_Release(&signed_state); return nullptr;
    }
    if (!uint8_buffer(event_state_obj, &event_state, 1, false)) {
        PyBuffer_Release(&signed_state); PyBuffer_Release(&baseline_groups);
        return nullptr;
    }
    if (!int64_buffer(grid_offsets_obj, &grid_offsets, 1, false)) {
        PyBuffer_Release(&signed_state); PyBuffer_Release(&baseline_groups);
        PyBuffer_Release(&event_state); return nullptr;
    }
    if (!int32_buffer(output_obj, &output, 1, true)) {
        PyBuffer_Release(&signed_state); PyBuffer_Release(&baseline_groups);
        PyBuffer_Release(&event_state); PyBuffer_Release(&grid_offsets);
        return nullptr;
    }
    const auto rows = signed_state.shape[0];
    const auto entities = grid_offsets.shape[0] - 1;
    const auto* statep = static_cast<const std::uint8_t*>(signed_state.buf);
    const auto* groupp = static_cast<const std::int32_t*>(baseline_groups.buf);
    const auto* eventp = static_cast<const std::uint8_t*>(event_state.buf);
    const auto* offsetp = static_cast<const std::int64_t*>(grid_offsets.buf);
    auto* out = static_cast<std::int32_t*>(output.buf);
    const bool valid = rows <= std::numeric_limits<std::int32_t>::max() &&
                       baseline_groups.shape[0] == rows &&
                       event_state.shape[0] == rows && output.shape[0] == rows &&
                       entities >= 0 && offsetp[0] == 0 && offsetp[entities] == rows;
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "label run-end buffer mismatch");
        PyBuffer_Release(&signed_state); PyBuffer_Release(&baseline_groups);
        PyBuffer_Release(&event_state); PyBuffer_Release(&grid_offsets);
        PyBuffer_Release(&output); return nullptr;
    }
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (std::int64_t entity = 0; entity < entities; ++entity) {
        const auto begin = offsetp[entity];
        const auto finish = offsetp[entity + 1];
        if (begin == finish) continue;
        out[finish - 1] = static_cast<std::int32_t>(finish);
        for (std::int64_t row = finish - 2; row >= begin; --row) {
            out[row] = (statep[row] == statep[row + 1] &&
                        groupp[row] == groupp[row + 1] &&
                        eventp[row] == eventp[row + 1])
                           ? out[row + 1]
                           : static_cast<std::int32_t>(row + 1);
        }
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&signed_state); PyBuffer_Release(&baseline_groups);
    PyBuffer_Release(&event_state); PyBuffer_Release(&grid_offsets);
    PyBuffer_Release(&output);
    Py_RETURN_NONE;
}

PyObject* safe_shell_counts_sources(PyObject*, PyObject* args) {
    PyObject *source_offsets_obj, *source_times_obj, *source_primitives_obj,
        *antecedent_predicates_obj, *antecedent_orders_obj, *starts_obj,
        *ends_obj, *grid_offsets_obj, *windows_obj, *window_counts_obj,
        *signed_state_obj, *baseline_groups_obj, *event_state_obj,
        *run_ends_obj, *output_obj;
    int horizon = 0, group_count = 0;
    double exposure = 0.0, weight = 0.0;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOOOOOOiiddO", &source_offsets_obj,
            &source_times_obj, &source_primitives_obj,
            &antecedent_predicates_obj, &antecedent_orders_obj, &starts_obj,
            &ends_obj, &grid_offsets_obj, &windows_obj, &window_counts_obj,
            &signed_state_obj, &baseline_groups_obj, &event_state_obj,
            &run_ends_obj,
            &horizon, &group_count, &exposure, &weight, &output_obj)) {
        return nullptr;
    }
    Py_buffer source_offsets{}, source_times{}, source_primitives{},
        antecedent_predicates{}, antecedent_orders{}, starts{}, ends{},
        grid_offsets{}, windows{}, window_counts{}, signed_state{},
        baseline_groups{}, event_state{}, run_ends{}, output{};
    int acquired = 0;
    auto release = [&]() {
        if (acquired >= 15) PyBuffer_Release(&output);
        if (acquired >= 14) PyBuffer_Release(&run_ends);
        if (acquired >= 13) PyBuffer_Release(&event_state);
        if (acquired >= 12) PyBuffer_Release(&baseline_groups);
        if (acquired >= 11) PyBuffer_Release(&signed_state);
        if (acquired >= 10) PyBuffer_Release(&window_counts);
        if (acquired >= 9) PyBuffer_Release(&windows);
        if (acquired >= 8) PyBuffer_Release(&grid_offsets);
        if (acquired >= 7) PyBuffer_Release(&ends);
        if (acquired >= 6) PyBuffer_Release(&starts);
        if (acquired >= 5) PyBuffer_Release(&antecedent_orders);
        if (acquired >= 4) PyBuffer_Release(&antecedent_predicates);
        if (acquired >= 3) PyBuffer_Release(&source_primitives);
        if (acquired >= 2) PyBuffer_Release(&source_times);
        if (acquired >= 1) PyBuffer_Release(&source_offsets);
    };
    if (!int64_buffer(source_offsets_obj, &source_offsets, 2, false)) return nullptr;
    ++acquired;
    if (!int64_buffer(source_times_obj, &source_times, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(source_primitives_obj, &source_primitives, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int32_buffer(antecedent_predicates_obj, &antecedent_predicates, 2, false)) { release(); return nullptr; }
    ++acquired;
    if (!int32_buffer(antecedent_orders_obj, &antecedent_orders, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(starts_obj, &starts, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(ends_obj, &ends, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(grid_offsets_obj, &grid_offsets, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int64_buffer(windows_obj, &windows, 2, false)) { release(); return nullptr; }
    ++acquired;
    if (!int32_buffer(window_counts_obj, &window_counts, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!uint8_buffer(signed_state_obj, &signed_state, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int32_buffer(baseline_groups_obj, &baseline_groups, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!uint8_buffer(event_state_obj, &event_state, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!int32_buffer(run_ends_obj, &run_ends, 1, false)) { release(); return nullptr; }
    ++acquired;
    if (!double_buffer(output_obj, &output, 5, true)) { release(); return nullptr; }
    ++acquired;

    const auto predicate_count = source_offsets.shape[0];
    const auto entity_count = starts.shape[0];
    const auto requested = antecedent_orders.shape[0];
    const auto maximum_order = antecedent_predicates.shape[1];
    const auto maximum_windows = windows.shape[1];
    const auto n_grid = signed_state.shape[0];
    const auto* source_offsetp =
        static_cast<const std::int64_t*>(source_offsets.buf);
    const auto* source_timep =
        static_cast<const std::int64_t*>(source_times.buf);
    const auto* source_primitivep =
        static_cast<const std::int64_t*>(source_primitives.buf);
    const auto* predicatep =
        static_cast<const std::int32_t*>(antecedent_predicates.buf);
    const auto* orderp = static_cast<const std::int32_t*>(antecedent_orders.buf);
    const auto* startp = static_cast<const std::int64_t*>(starts.buf);
    const auto* endp = static_cast<const std::int64_t*>(ends.buf);
    const auto* gridp = static_cast<const std::int64_t*>(grid_offsets.buf);
    const auto* wp = static_cast<const std::int64_t*>(windows.buf);
    const auto* countp = static_cast<const std::int32_t*>(window_counts.buf);
    const auto* statep = static_cast<const std::uint8_t*>(signed_state.buf);
    const auto* groupp = static_cast<const std::int32_t*>(baseline_groups.buf);
    const auto* eventp = static_cast<const std::uint8_t*>(event_state.buf);
    const auto* run_endp = static_cast<const std::int32_t*>(run_ends.buf);
    auto* out = static_cast<double*>(output.buf);
    bool valid = predicate_count > 0 && maximum_order == 3 && requested > 0 &&
                 source_offsets.shape[1] == entity_count + 1 &&
                 source_times.shape[0] == source_primitives.shape[0] &&
                 antecedent_predicates.shape[0] == requested &&
                 windows.shape[0] == requested && window_counts.shape[0] == requested &&
                 ends.shape[0] == entity_count && grid_offsets.shape[0] == entity_count + 1 &&
                 baseline_groups.shape[0] == n_grid && event_state.shape[0] == n_grid &&
                 run_ends.shape[0] == n_grid &&
                 gridp[entity_count] == n_grid && output.shape[0] == requested &&
                 output.shape[1] == 4 && output.shape[2] == maximum_windows &&
                 output.shape[3] == 3 && output.shape[4] == group_count &&
                 horizon >= 0 && group_count > 0 && exposure > 0.0 && weight > 0.0;
    for (std::int64_t request = 0; valid && request < requested; ++request) {
        const auto order = static_cast<std::int64_t>(orderp[request]);
        const auto count = static_cast<std::int64_t>(countp[request]);
        valid = order >= 1 && order <= 3 && count >= 1 && count <= maximum_windows;
        for (std::int64_t source = 0; valid && source < order; ++source) {
            const auto predicate = static_cast<std::int64_t>(
                predicatep[request * maximum_order + source]);
            valid = predicate >= 0 && predicate < predicate_count;
        }
        const auto* local_windows = wp + request * maximum_windows;
        for (std::int64_t index = 0; valid && index < count; ++index)
            valid = local_windows[index] >= 0 &&
                    (index == 0 || local_windows[index - 1] < local_windows[index]);
    }
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "source safe shell buffer mismatch");
        release();
        return nullptr;
    }
    std::fill(out, out + output.shape[0] * output.shape[1] * output.shape[2] *
                         output.shape[3] * output.shape[4], 0.0);

    bool invalid = false;
    bool allocation_failed = false;
    Py_BEGIN_ALLOW_THREADS
    try {
        #pragma omp parallel for schedule(dynamic, 1)
        for (std::int64_t request = 0; request < requested; ++request) {
            const auto source_count = static_cast<std::int64_t>(orderp[request]);
            const auto local_window_count =
                static_cast<std::int64_t>(countp[request]);
            const auto* local_windows = wp + request * maximum_windows;
            std::array<std::int64_t, 3> predicates{};
            for (std::int64_t source = 0; source < source_count; ++source)
                predicates[source] = predicatep[request * maximum_order + source];
            for (std::int64_t entity = 0; entity < entity_count; ++entity) {
                std::array<std::int64_t, 3> cursor{}, finish{};
                bool present = true;
                for (std::int64_t source = 0; source < source_count; ++source) {
                    const auto base = predicates[source] * (entity_count + 1);
                    cursor[source] = source_offsetp[base + entity];
                    finish[source] = source_offsetp[base + entity + 1];
                    if (cursor[source] >= finish[source]) present = false;
                }
                if (!present || horizon <= 0) continue;
                std::array<std::array<std::int64_t, 3>, 3> latest_time{};
                std::array<std::array<std::int64_t, 3>, 3> latest_id{};
                for (auto& values : latest_time)
                    values.fill(std::numeric_limits<std::int64_t>::min());
                for (auto& values : latest_id)
                    values.fill(std::numeric_limits<std::int64_t>::min());
                std::vector<std::pair<std::int64_t, std::int64_t>> completions;
                completions.reserve(static_cast<std::size_t>(
                    finish[0] - cursor[0]));
                while (true) {
                    std::int64_t next_time =
                        std::numeric_limits<std::int64_t>::max();
                    for (std::int64_t source = 0; source < source_count; ++source)
                        if (cursor[source] < finish[source])
                            next_time = std::min(next_time, source_timep[cursor[source]]);
                    if (next_time == std::numeric_limits<std::int64_t>::max()) break;
                    for (std::int64_t source = 0; source < source_count; ++source) {
                        while (cursor[source] < finish[source] &&
                               source_timep[cursor[source]] <= next_time) {
                            update_latest_primitive(
                                latest_time, latest_id, source,
                                source_timep[cursor[source]],
                                source_primitivep[cursor[source]], source_count);
                            ++cursor[source];
                        }
                    }
                    std::int64_t span = 0;
                    if (next_time < endp[entity] && latest_distinct_span(
                            latest_time, latest_id, source_count, span))
                        completions.emplace_back(next_time, span);
                }
                if (completions.empty()) continue;
                const auto first_time = std::max<std::int64_t>(
                    startp[entity], completions.front().first + 1);
                const auto last_time = std::min<std::int64_t>(
                    endp[entity], completions.back().first + horizon);
                std::deque<std::size_t> minimum;
                std::size_t completion_cursor = 0;
                std::int64_t time = first_time;
                while (time <= last_time) {
                    while (completion_cursor < completions.size() &&
                           completions[completion_cursor].first < time) {
                        while (!minimum.empty() &&
                               completions[minimum.back()].second >=
                                   completions[completion_cursor].second)
                            minimum.pop_back();
                        minimum.push_back(completion_cursor++);
                    }
                    const auto oldest = time - horizon;
                    while (!minimum.empty() &&
                           completions[minimum.front()].first < oldest)
                        minimum.pop_front();
                    if (minimum.empty()) {
                        if (completion_cursor >= completions.size()) break;
                        time = std::max<std::int64_t>(
                            time + 1,
                            completions[completion_cursor].first + 1);
                        continue;
                    }
                    const auto span = completions[minimum.front()].second;
                    const auto* admitted = std::lower_bound(
                        local_windows, local_windows + local_window_count, span);
                    const auto next_add =
                        completion_cursor < completions.size()
                            ? completions[completion_cursor].first + 1
                            : last_time + 1;
                    const auto next_expiry =
                        completions[minimum.front()].first + horizon + 1;
                    const auto boundary = std::max<std::int64_t>(
                        time + 1,
                        std::min<std::int64_t>(
                            last_time + 1, std::min(next_add, next_expiry)));
                    if (admitted != local_windows + local_window_count) {
                        const auto window = static_cast<std::int64_t>(
                            admitted - local_windows);
                        auto row = gridp[entity] + time - startp[entity];
                        const auto after_row =
                            gridp[entity] + boundary - startp[entity];
                        while (row < after_row) {
                            if (row < 0 || row >= n_grid || statep[row] > 3 ||
                                groupp[row] < 0 || groupp[row] >= group_count ||
                                eventp[row] > 1 || run_endp[row] <= row ||
                                run_endp[row] > gridp[entity + 1]) {
                                #pragma omp atomic write
                                invalid = true;
                                break;
                            }
                            const auto run_after = std::min<std::int64_t>(
                                after_row, run_endp[row]);
                            const auto length = run_after - row;
                            const auto category =
                                static_cast<std::int64_t>(statep[row]);
                            const auto group =
                                static_cast<std::int64_t>(groupp[row]);
                            const auto base = (((request * 4 + category) *
                                                maximum_windows + window) * 3) *
                                              group_count + group;
                            const auto event = weight * static_cast<double>(
                                eventp[row]) * static_cast<double>(length);
                            out[base] += exposure * weight *
                                         static_cast<double>(length);
                            out[base + group_count] +=
                                weight * static_cast<double>(length) - event;
                            out[base + 2 * group_count] += event;
                            row = run_after;
                        }
                    }
                    time = boundary;
                }
            }
        }
    } catch (const std::bad_alloc&) {
        allocation_failed = true;
    }
    Py_END_ALLOW_THREADS
    release();
    if (allocation_failed) return PyErr_NoMemory();
    if (invalid) {
        PyErr_SetString(PyExc_ValueError, "source safe shell index is invalid");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* bounded_span_order(PyObject*, PyObject* args) {
    PyObject *spans_obj, *output_obj;
    long long maximum;
    if (!PyArg_ParseTuple(args, "OLO", &spans_obj, &maximum, &output_obj)) {
        return nullptr;
    }
    Py_buffer spans{}, output{};
    if (!int64_buffer(spans_obj, &spans, 1, false)) return nullptr;
    if (!int64_buffer(output_obj, &output, 1, true)) {
        PyBuffer_Release(&spans);
        return nullptr;
    }
    if (maximum < 0 || output.shape[0] < spans.shape[0]) {
        PyErr_SetString(PyExc_ValueError, "bounded span order buffer mismatch");
        PyBuffer_Release(&spans);
        PyBuffer_Release(&output);
        return nullptr;
    }
    const auto* input = static_cast<const std::int64_t*>(spans.buf);
    auto* ordered = static_cast<std::int64_t*>(output.buf);
    const auto count = spans.shape[0];
    std::int64_t admitted = 0;
    bool valid = true;
    bool allocation_failed = false;
    Py_BEGIN_ALLOW_THREADS
    try {
        std::vector<std::int64_t> offsets(
            static_cast<std::size_t>(maximum) + 1, 0);
        for (std::int64_t index = 0; index < count; ++index) {
            const auto span = input[index];
            if (span < 0) {
                valid = false;
                break;
            }
            if (span <= maximum) ++offsets[static_cast<std::size_t>(span)];
        }
        if (valid) {
            std::int64_t prefix = 0;
            for (auto& value : offsets) {
                const auto frequency = value;
                value = prefix;
                prefix += frequency;
            }
            admitted = prefix;
            for (std::int64_t index = 0; index < count; ++index) {
                const auto span = input[index];
                if (span <= maximum) {
                    ordered[offsets[static_cast<std::size_t>(span)]++] = index;
                }
            }
        }
    } catch (const std::bad_alloc&) {
        allocation_failed = true;
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&spans);
    PyBuffer_Release(&output);
    if (allocation_failed) return PyErr_NoMemory();
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "completion spans must be nonnegative");
        return nullptr;
    }
    return PyLong_FromLongLong(admitted);
}

PyObject* sorted_unique_union(PyObject*, PyObject* args) {
    PyObject *parts_obj, *output_obj;
    if (!PyArg_ParseTuple(args, "OO", &parts_obj, &output_obj)) return nullptr;
    PyObject* sequence = PySequence_Fast(parts_obj, "row parts must be a sequence");
    if (sequence == nullptr) return nullptr;
    Py_buffer output{};
    if (!int64_buffer(output_obj, &output, 1, true)) {
        Py_DECREF(sequence);
        return nullptr;
    }
    const auto part_count = PySequence_Fast_GET_SIZE(sequence);
    std::vector<Py_buffer> parts(static_cast<std::size_t>(part_count));
    std::int64_t capacity = 0;
    bool valid = true;
    Py_ssize_t acquired = 0;
    for (Py_ssize_t index = 0; index < part_count; ++index) {
        PyObject* item = PySequence_Fast_GET_ITEM(sequence, index);
        if (!int64_buffer(item, &parts[static_cast<std::size_t>(index)], 1, false)) {
            valid = false;
            break;
        }
        ++acquired;
        const auto& part = parts[static_cast<std::size_t>(index)];
        if (capacity > std::numeric_limits<std::int64_t>::max() - part.shape[0]) {
            PyErr_SetString(PyExc_OverflowError, "row union capacity overflow");
            valid = false;
            break;
        }
        capacity += part.shape[0];
    }
    if (valid && output.shape[0] < capacity) {
        PyErr_SetString(PyExc_ValueError, "row union output is too small");
        valid = false;
    }
    std::int64_t written = 0;
    if (valid) {
        auto* destination = static_cast<std::int64_t*>(output.buf);
        std::vector<std::int64_t> positions(static_cast<std::size_t>(part_count), 0);
        Py_BEGIN_ALLOW_THREADS
        for (;;) {
            bool found = false;
            std::int64_t next = std::numeric_limits<std::int64_t>::max();
            for (Py_ssize_t part_index = 0; part_index < part_count; ++part_index) {
                const auto& part = parts[static_cast<std::size_t>(part_index)];
                const auto position = positions[static_cast<std::size_t>(part_index)];
                if (position >= part.shape[0]) continue;
                const auto* values = static_cast<const std::int64_t*>(part.buf);
                if (position > 0 && values[position] <= values[position - 1]) {
                    valid = false;
                    break;
                }
                next = std::min(next, values[position]);
                found = true;
            }
            if (!valid || !found) break;
            destination[written++] = next;
            for (Py_ssize_t part_index = 0; part_index < part_count; ++part_index) {
                const auto& part = parts[static_cast<std::size_t>(part_index)];
                auto& position = positions[static_cast<std::size_t>(part_index)];
                const auto* values = static_cast<const std::int64_t*>(part.buf);
                if (position < part.shape[0] && values[position] == next) ++position;
            }
        }
        Py_END_ALLOW_THREADS
    }
    for (Py_ssize_t index = 0; index < acquired; ++index) {
        PyBuffer_Release(&parts[static_cast<std::size_t>(index)]);
    }
    PyBuffer_Release(&output);
    Py_DECREF(sequence);
    if (!valid) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "row parts must be strictly increasing");
        }
        return nullptr;
    }
    return PyLong_FromLongLong(written);
}

PyObject* aggregate_quotient_rows(PyObject*, PyObject* args) {
    PyObject *groups_obj, *values_obj, *exposure_obj, *noevent_obj, *event_obj;
    PyObject *output_exposure_obj, *output_noevent_obj, *output_event_obj;
    PyObject *removed_exposure_obj, *removed_noevent_obj, *removed_event_obj;
    int requested_workers = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOO|i", &groups_obj, &values_obj,
                          &exposure_obj, &noevent_obj, &event_obj,
                          &output_exposure_obj, &output_noevent_obj,
                          &output_event_obj, &removed_exposure_obj,
                          &removed_noevent_obj, &removed_event_obj,
                          &requested_workers)) {
        return nullptr;
    }
    if (requested_workers < 0) {
        PyErr_SetString(PyExc_ValueError, "quotient worker count must be nonnegative");
        return nullptr;
    }
    Py_buffer groups{}, values{}, exposure{}, noevent{}, event{};
    Py_buffer output_exposure{}, output_noevent{}, output_event{};
    Py_buffer removed_exposure{}, removed_noevent{}, removed_event{};
    if (!int64_buffer(groups_obj, &groups, 1, false)) return nullptr;
    if (!double_buffer(values_obj, &values, 2, false)) {
        PyBuffer_Release(&groups);
        return nullptr;
    }
    auto release_key = [&]() {
        PyBuffer_Release(&groups);
        PyBuffer_Release(&values);
    };
    if (!double_buffer(exposure_obj, &exposure, 1, false)) {
        release_key();
        return nullptr;
    }
    if (!double_buffer(noevent_obj, &noevent, 1, false)) {
        release_key();
        PyBuffer_Release(&exposure);
        return nullptr;
    }
    if (!double_buffer(event_obj, &event, 1, false)) {
        release_key();
        PyBuffer_Release(&exposure);
        PyBuffer_Release(&noevent);
        return nullptr;
    }
    auto release_inputs = [&]() {
        release_key();
        PyBuffer_Release(&exposure);
        PyBuffer_Release(&noevent);
        PyBuffer_Release(&event);
    };
    if (!double_buffer(output_exposure_obj, &output_exposure, 1, true)) {
        release_inputs();
        return nullptr;
    }
    if (!double_buffer(output_noevent_obj, &output_noevent, 1, true)) {
        release_inputs();
        PyBuffer_Release(&output_exposure);
        return nullptr;
    }
    if (!double_buffer(output_event_obj, &output_event, 1, true)) {
        release_inputs();
        PyBuffer_Release(&output_exposure);
        PyBuffer_Release(&output_noevent);
        return nullptr;
    }
    auto release_outputs = [&]() {
        PyBuffer_Release(&output_exposure);
        PyBuffer_Release(&output_noevent);
        PyBuffer_Release(&output_event);
    };
    if (!double_buffer(removed_exposure_obj, &removed_exposure, 1, true)) {
        release_inputs();
        release_outputs();
        return nullptr;
    }
    if (!double_buffer(removed_noevent_obj, &removed_noevent, 1, true)) {
        release_inputs();
        release_outputs();
        PyBuffer_Release(&removed_exposure);
        return nullptr;
    }
    if (!double_buffer(removed_event_obj, &removed_event, 1, true)) {
        release_inputs();
        release_outputs();
        PyBuffer_Release(&removed_exposure);
        PyBuffer_Release(&removed_noevent);
        return nullptr;
    }
    auto release_removed = [&]() {
        PyBuffer_Release(&removed_exposure);
        PyBuffer_Release(&removed_noevent);
        PyBuffer_Release(&removed_event);
    };

    const std::int64_t rows = values.shape[0];
    const std::int64_t columns = values.shape[1];
    if (groups.shape[0] != rows || exposure.shape[0] != rows ||
        noevent.shape[0] != rows || event.shape[0] != rows ||
        output_exposure.shape[0] < rows || output_noevent.shape[0] < rows ||
        output_event.shape[0] < rows ||
        removed_noevent.shape[0] != removed_exposure.shape[0] ||
        removed_event.shape[0] != removed_exposure.shape[0]) {
        PyErr_SetString(PyExc_ValueError, "quotient aggregation shape mismatch");
        release_inputs();
        release_outputs();
        release_removed();
        return nullptr;
    }

    auto* group = static_cast<std::int64_t*>(groups.buf);
    auto* xp = static_cast<double*>(values.buf);
    auto* ep = static_cast<double*>(exposure.buf);
    auto* np = static_cast<double*>(noevent.buf);
    auto* yp = static_cast<double*>(event.buf);
    auto* output_ep = static_cast<double*>(output_exposure.buf);
    auto* output_np = static_cast<double*>(output_noevent.buf);
    auto* output_yp = static_cast<double*>(output_event.buf);
    auto* removed_ep = static_cast<double*>(removed_exposure.buf);
    auto* removed_np = static_cast<double*>(removed_noevent.buf);
    auto* removed_yp = static_cast<double*>(removed_event.buf);
    const std::int64_t group_count = removed_exposure.shape[0];
    std::int64_t output_count = 0;
    bool allocation_failed = false;
    bool invalid_group = false;

    Py_BEGIN_ALLOW_THREADS
    try {
        std::fill(removed_ep, removed_ep + group_count, 0.0);
        std::fill(removed_np, removed_np + group_count, 0.0);
        std::fill(removed_yp, removed_yp + group_count, 0.0);
        // Compute all three old-state sufficient statistics in one stable
        // input-order pass. This replaces three np.bincount traversals and an
        // additional touched-count allocation in the Python hot loop.
        for (std::int64_t input = 0; input < rows; ++input) {
            const auto old = group[input];
            if (old < 0 || old >= group_count) {
                invalid_group = true;
                break;
            }
            removed_ep[old] += ep[input];
            removed_np[old] += np[input];
            removed_yp[old] += yp[input];
        }
        if (invalid_group) throw std::runtime_error("invalid quotient group");
        // Equal quotient keys necessarily share a hash, so independent hash
        // shards can be merged in parallel without approximation. Within each
        // shard, stable input order preserves deterministic summation.
        const int available_workers = requested_workers > 0
            ? std::min(requested_workers, omp_get_max_threads())
            : omp_get_max_threads();
        const int shard_count = std::max(
            1, std::min<int>(available_workers, static_cast<int>(rows)));
        std::vector<std::uint64_t> hashes(static_cast<std::size_t>(rows));
        std::vector<std::int64_t> shard_sizes(
            static_cast<std::size_t>(shard_count), 0);
        #pragma omp parallel for schedule(static) num_threads(shard_count)
        for (std::int64_t input = 0; input < rows; ++input) {
            hashes[static_cast<std::size_t>(input)] = quotient_hash(
                group[input], xp + input * columns, columns);
        }
        for (std::int64_t input = 0; input < rows; ++input) {
            const int shard = static_cast<int>(
                hashes[static_cast<std::size_t>(input)] %
                static_cast<std::uint64_t>(shard_count));
            ++shard_sizes[static_cast<std::size_t>(shard)];
        }
        std::vector<std::int64_t> shard_offsets(
            static_cast<std::size_t>(shard_count + 1), 0);
        for (int shard = 0; shard < shard_count; ++shard) {
            shard_offsets[static_cast<std::size_t>(shard + 1)] =
                shard_offsets[static_cast<std::size_t>(shard)] +
                shard_sizes[static_cast<std::size_t>(shard)];
        }
        std::vector<std::int64_t> cursor = shard_offsets;
        std::vector<std::int64_t> order(static_cast<std::size_t>(rows));
        for (std::int64_t input = 0; input < rows; ++input) {
            const int shard = static_cast<int>(
                hashes[static_cast<std::size_t>(input)] %
                static_cast<std::uint64_t>(shard_count));
            order[static_cast<std::size_t>(
                cursor[static_cast<std::size_t>(shard)]++)] = input;
        }

        struct ShardResult {
            std::vector<std::int64_t> representatives;
            std::vector<std::int64_t> collision_next;
        };
        std::vector<ShardResult> results(static_cast<std::size_t>(shard_count));
        #pragma omp parallel for schedule(static) num_threads(shard_count)
        for (int shard = 0; shard < shard_count; ++shard) {
            auto& result = results[static_cast<std::size_t>(shard)];
            const auto begin = shard_offsets[static_cast<std::size_t>(shard)];
            const auto end = shard_offsets[static_cast<std::size_t>(shard + 1)];
            const auto reserve = static_cast<std::size_t>(end - begin);
            result.representatives.reserve(reserve);
            result.collision_next.reserve(reserve);
            // One unordered-map node per observed hash and one compact
            // int64 link per exact key.  The previous map-of-vectors allocated
            // a separate heap vector for almost every unique row, which made
            // the safe quotient itself an OOM risk on high-cardinality
            // footprints.  A linked collision chain retains exact equality
            // resolution without per-key allocator objects.
            std::unordered_map<std::uint64_t, std::int64_t> buckets;
            buckets.reserve(std::min<std::size_t>(reserve, 262144));
            for (std::int64_t position = begin; position < end; ++position) {
                const auto input = order[static_cast<std::size_t>(position)];
                const auto hash = hashes[static_cast<std::size_t>(input)];
                std::int64_t matched = -1;
                const auto bucket = buckets.find(hash);
                auto local = (
                    bucket == buckets.end() ? -1 : bucket->second
                );
                while (local >= 0) {
                    const auto representative = result.representatives[
                        static_cast<std::size_t>(local)];
                    if (group[input] == group[representative] &&
                        equal_row(xp + input * columns,
                                  xp + representative * columns, columns)) {
                        matched = local;
                        break;
                    }
                    local = result.collision_next[
                        static_cast<std::size_t>(local)];
                }
                if (matched >= 0) {
                    const auto output = begin + matched;
                    output_ep[output] += ep[input];
                    output_np[output] += np[input];
                    output_yp[output] += yp[input];
                } else {
                    const auto local = static_cast<std::int64_t>(
                        result.representatives.size());
                    result.representatives.push_back(input);
                    result.collision_next.push_back(
                        bucket == buckets.end() ? -1 : bucket->second
                    );
                    if (bucket == buckets.end()) {
                        buckets.emplace(hash, local);
                    } else {
                        bucket->second = local;
                    }
                    const auto output = begin + local;
                    output_ep[output] = ep[input];
                    output_np[output] = np[input];
                    output_yp[output] = yp[input];
                }
            }
        }
        for (int shard = 0; shard < shard_count; ++shard) {
            const auto begin = shard_offsets[static_cast<std::size_t>(shard)];
            const auto count = results[static_cast<std::size_t>(shard)]
                                   .representatives.size();
            for (std::size_t local = 0; local < count; ++local) {
                output_ep[output_count] = output_ep[begin + local];
                output_np[output_count] = output_np[begin + local];
                output_yp[output_count] = output_yp[begin + local];
                ++output_count;
            }
        }
    } catch (const std::bad_alloc&) {
        allocation_failed = true;
    } catch (const std::runtime_error&) {
        // The only runtime_error above is an invalid group. It is reported
        // after reacquiring the GIL.
    }
    Py_END_ALLOW_THREADS

    release_inputs();
    release_outputs();
    release_removed();
    if (allocation_failed) return PyErr_NoMemory();
    if (invalid_group) {
        PyErr_SetString(PyExc_ValueError, "quotient group is out of range");
        return nullptr;
    }
    return PyLong_FromLongLong(output_count);
}

double cloglog_event_loss_from_intensity(double intensity) {
    constexpr double tiny = std::numeric_limits<double>::min();
    if (intensity < 1.0e-4) {
        return -std::log(std::max(intensity, tiny)) + intensity / 2.0 -
               intensity * intensity / 24.0;
    }
    if (intensity > 40.0) {
        return -std::log1p(-std::exp(-intensity));
    }
    return -std::log(-std::expm1(-intensity));
}

PyObject* entity_loss_contrast(PyObject*, PyObject* args) {
    PyObject *null_eta_obj, *full_eta_obj, *null_baseline_obj,
        *full_baseline_obj, *rows_obj, *design_groups_obj,
        *baseline_groups_obj, *offsets_obj, *entity_weights_obj,
        *target_rows_obj, *target_counts_obj, *output_obj;
    double tick_exposure = 0.0;
    int likelihood = 0, requested_workers = 0;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOOOdiiO", &null_eta_obj, &full_eta_obj,
            &null_baseline_obj, &full_baseline_obj, &rows_obj,
            &design_groups_obj, &baseline_groups_obj, &offsets_obj,
            &entity_weights_obj, &target_rows_obj, &target_counts_obj,
            &tick_exposure, &likelihood, &requested_workers, &output_obj)) {
        return nullptr;
    }
    if (!(tick_exposure > 0.0) || (likelihood != 0 && likelihood != 1) ||
        requested_workers < 1) {
        PyErr_SetString(PyExc_ValueError, "invalid entity loss contrast arguments");
        return nullptr;
    }
    Py_buffer null_eta{}, full_eta{}, null_baseline{}, full_baseline{}, rows{};
    Py_buffer design_groups{}, baseline_groups{}, offsets{}, entity_weights{};
    Py_buffer target_rows{}, target_counts{}, output{};
    int acquired = 0;
    auto release = [&]() {
        if (acquired >= 12) PyBuffer_Release(&output);
        if (acquired >= 11) PyBuffer_Release(&target_counts);
        if (acquired >= 10) PyBuffer_Release(&target_rows);
        if (acquired >= 9) PyBuffer_Release(&entity_weights);
        if (acquired >= 8) PyBuffer_Release(&offsets);
        if (acquired >= 7) PyBuffer_Release(&baseline_groups);
        if (acquired >= 6) PyBuffer_Release(&design_groups);
        if (acquired >= 5) PyBuffer_Release(&rows);
        if (acquired >= 4) PyBuffer_Release(&full_baseline);
        if (acquired >= 3) PyBuffer_Release(&null_baseline);
        if (acquired >= 2) PyBuffer_Release(&full_eta);
        if (acquired >= 1) PyBuffer_Release(&null_eta);
    };
    if (!double_buffer(null_eta_obj, &null_eta, 1, false)) return nullptr;
    acquired = 1;
    if (!double_buffer(full_eta_obj, &full_eta, 1, false)) { release(); return nullptr; }
    acquired = 2;
    if (!double_buffer(null_baseline_obj, &null_baseline, 1, false)) { release(); return nullptr; }
    acquired = 3;
    if (!double_buffer(full_baseline_obj, &full_baseline, 1, false)) { release(); return nullptr; }
    acquired = 4;
    if (!int64_buffer(rows_obj, &rows, 1, false)) { release(); return nullptr; }
    acquired = 5;
    if (!int32_buffer(design_groups_obj, &design_groups, 1, false)) { release(); return nullptr; }
    acquired = 6;
    if (!int32_buffer(baseline_groups_obj, &baseline_groups, 1, false)) { release(); return nullptr; }
    acquired = 7;
    if (!int64_buffer(offsets_obj, &offsets, 1, false)) { release(); return nullptr; }
    acquired = 8;
    if (!double_buffer(entity_weights_obj, &entity_weights, 1, false)) { release(); return nullptr; }
    acquired = 9;
    if (!int64_buffer(target_rows_obj, &target_rows, 1, false)) { release(); return nullptr; }
    acquired = 10;
    if (!double_buffer(target_counts_obj, &target_counts, 1, false)) { release(); return nullptr; }
    acquired = 11;
    if (!double_buffer(output_obj, &output, 1, true)) { release(); return nullptr; }
    acquired = 12;

    const auto active_count = rows.shape[0];
    const auto entity_count = entity_weights.shape[0];
    const auto target_count = target_rows.shape[0];
    if (full_eta.shape[0] != null_eta.shape[0] ||
        full_baseline.shape[0] != null_baseline.shape[0] ||
        design_groups.shape[0] != active_count ||
        baseline_groups.shape[0] != active_count ||
        offsets.shape[0] != entity_count + 1 ||
        target_counts.shape[0] != target_count || output.shape[0] != entity_count) {
        release();
        PyErr_SetString(PyExc_ValueError, "entity loss contrast shape mismatch");
        return nullptr;
    }
    const auto* null_ep = static_cast<const double*>(null_eta.buf);
    const auto* full_ep = static_cast<const double*>(full_eta.buf);
    const auto* null_bp = static_cast<const double*>(null_baseline.buf);
    const auto* full_bp = static_cast<const double*>(full_baseline.buf);
    const auto* rowp = static_cast<const std::int64_t*>(rows.buf);
    const auto* designp = static_cast<const std::int32_t*>(design_groups.buf);
    const auto* baselinep = static_cast<const std::int32_t*>(baseline_groups.buf);
    const auto* offsetp = static_cast<const std::int64_t*>(offsets.buf);
    const auto* weightp = static_cast<const double*>(entity_weights.buf);
    const auto* targetp = static_cast<const std::int64_t*>(target_rows.buf);
    const auto* countp = static_cast<const double*>(target_counts.buf);
    auto* outp = static_cast<double*>(output.buf);
    int invalid = 0;
    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(static) num_threads(requested_workers) reduction(| : invalid)
    for (std::int64_t entity = 0; entity < entity_count; ++entity) {
        const auto row_left = offsetp[entity];
        const auto row_right = offsetp[entity + 1];
        const auto* active_begin = std::lower_bound(rowp, rowp + active_count, row_left);
        const auto* active_end = std::lower_bound(active_begin, rowp + active_count, row_right);
        const auto* target_cursor = std::lower_bound(targetp, targetp + target_count, row_left);
        const auto* target_end = std::lower_bound(target_cursor, targetp + target_count, row_right);
        std::int64_t target_index = target_cursor - targetp;
        double total = 0.0;
        for (const auto* cursor = active_begin; cursor != active_end; ++cursor) {
            const auto active = static_cast<std::int64_t>(cursor - rowp);
            const auto design = static_cast<std::int64_t>(designp[active]);
            const auto baseline = static_cast<std::int64_t>(baselinep[active]);
            if (design < 0 || design >= null_eta.shape[0] || baseline < 0 ||
                baseline >= null_baseline.shape[0]) {
                invalid |= 1;
                continue;
            }
            while (target_index < target_end - targetp && targetp[target_index] < *cursor)
                ++target_index;
            const double event =
                (target_index < target_end - targetp && targetp[target_index] == *cursor)
                    ? countp[target_index]
                    : 0.0;
            const double weight = weightp[entity];
            const double exposure = tick_exposure * weight;
            const double null_value_eta = null_ep[design];
            const double full_value_eta = full_ep[design];
            double null_loss = 0.0, full_loss = 0.0;
            if (likelihood == 0) {
                null_loss = exposure * std::exp(null_value_eta) - event * null_value_eta;
                full_loss = exposure * std::exp(full_value_eta) - event * full_value_eta;
            } else {
                const double null_intensity =
                    std::exp(std::max(-745.0, std::min(700.0, null_value_eta)));
                const double full_intensity =
                    std::exp(std::max(-745.0, std::min(700.0, full_value_eta)));
                const double noevent = exposure - event;
                null_loss = noevent * null_intensity +
                            event * cloglog_event_loss_from_intensity(null_intensity);
                full_loss = noevent * full_intensity +
                            event * cloglog_event_loss_from_intensity(full_intensity);
            }
            const double default_difference =
                weight * (null_bp[baseline] - full_bp[baseline]);
            total += null_loss - full_loss - default_difference;
        }
        outp[entity] = total;
    }
    Py_END_ALLOW_THREADS
    release();
    if (invalid) {
        PyErr_SetString(PyExc_ValueError, "entity loss contrast index is invalid");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* dependency_row_derivatives(PyObject*, PyObject* args) {
    PyObject *group_eta_obj, *rows_obj, *design_groups_obj,
        *baseline_groups_obj, *offsets_obj, *starts_obj, *entity_weights_obj,
        *dependency_codes_obj, *target_rows_obj, *target_counts_obj,
        *baseline_first_obj, *first_obj, *entity_cluster_obj,
        *time_cluster_obj, *default_first_obj;
    long long origin = 0;
    double tick_exposure = 0.0;
    int likelihood = 0, requested_workers = 0;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOOOLdiiOOOO", &group_eta_obj, &rows_obj,
            &design_groups_obj, &baseline_groups_obj, &offsets_obj,
            &starts_obj, &entity_weights_obj, &dependency_codes_obj,
            &target_rows_obj, &target_counts_obj, &baseline_first_obj,
            &origin, &tick_exposure, &likelihood, &requested_workers,
            &first_obj, &entity_cluster_obj, &time_cluster_obj,
            &default_first_obj)) {
        return nullptr;
    }
    if (!(tick_exposure > 0.0) || (likelihood != 0 && likelihood != 1) ||
        requested_workers < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid dependency derivative arguments");
        return nullptr;
    }
    Py_buffer group_eta{}, rows{}, design_groups{}, baseline_groups{}, offsets{};
    Py_buffer starts{}, entity_weights{}, dependency_codes{}, target_rows{};
    Py_buffer target_counts{}, baseline_first{}, first{}, entity_cluster{};
    Py_buffer time_cluster{}, default_first{};
    int acquired = 0;
    auto release = [&]() {
        if (acquired >= 15) PyBuffer_Release(&default_first);
        if (acquired >= 14) PyBuffer_Release(&time_cluster);
        if (acquired >= 13) PyBuffer_Release(&entity_cluster);
        if (acquired >= 12) PyBuffer_Release(&first);
        if (acquired >= 11) PyBuffer_Release(&baseline_first);
        if (acquired >= 10) PyBuffer_Release(&target_counts);
        if (acquired >= 9) PyBuffer_Release(&target_rows);
        if (acquired >= 8) PyBuffer_Release(&dependency_codes);
        if (acquired >= 7) PyBuffer_Release(&entity_weights);
        if (acquired >= 6) PyBuffer_Release(&starts);
        if (acquired >= 5) PyBuffer_Release(&offsets);
        if (acquired >= 4) PyBuffer_Release(&baseline_groups);
        if (acquired >= 3) PyBuffer_Release(&design_groups);
        if (acquired >= 2) PyBuffer_Release(&rows);
        if (acquired >= 1) PyBuffer_Release(&group_eta);
    };
    if (!double_buffer(group_eta_obj, &group_eta, 1, false)) return nullptr;
    acquired = 1;
    if (!int64_buffer(rows_obj, &rows, 1, false)) { release(); return nullptr; }
    acquired = 2;
    if (!int32_buffer(design_groups_obj, &design_groups, 1, false)) { release(); return nullptr; }
    acquired = 3;
    if (!int32_buffer(baseline_groups_obj, &baseline_groups, 1, false)) { release(); return nullptr; }
    acquired = 4;
    if (!int64_buffer(offsets_obj, &offsets, 1, false)) { release(); return nullptr; }
    acquired = 5;
    if (!int64_buffer(starts_obj, &starts, 1, false)) { release(); return nullptr; }
    acquired = 6;
    if (!double_buffer(entity_weights_obj, &entity_weights, 1, false)) { release(); return nullptr; }
    acquired = 7;
    if (!int32_buffer(dependency_codes_obj, &dependency_codes, 1, false)) { release(); return nullptr; }
    acquired = 8;
    if (!int64_buffer(target_rows_obj, &target_rows, 1, false)) { release(); return nullptr; }
    acquired = 9;
    if (!double_buffer(target_counts_obj, &target_counts, 1, false)) { release(); return nullptr; }
    acquired = 10;
    if (!double_buffer(baseline_first_obj, &baseline_first, 1, false)) { release(); return nullptr; }
    acquired = 11;
    if (!double_buffer(first_obj, &first, 1, true)) { release(); return nullptr; }
    acquired = 12;
    if (!int32_buffer(entity_cluster_obj, &entity_cluster, 1, true)) { release(); return nullptr; }
    acquired = 13;
    if (!int32_buffer(time_cluster_obj, &time_cluster, 1, true)) { release(); return nullptr; }
    acquired = 14;
    if (!double_buffer(default_first_obj, &default_first, 1, true)) { release(); return nullptr; }
    acquired = 15;

    const auto active_count = rows.shape[0];
    const auto entity_count = entity_weights.shape[0];
    const auto target_count = target_rows.shape[0];
    if (design_groups.shape[0] != active_count ||
        baseline_groups.shape[0] != active_count || offsets.shape[0] != entity_count + 1 ||
        starts.shape[0] != entity_count || dependency_codes.shape[0] != entity_count ||
        target_counts.shape[0] != target_count || first.shape[0] != active_count ||
        entity_cluster.shape[0] != active_count || time_cluster.shape[0] != active_count ||
        default_first.shape[0] != active_count) {
        release();
        PyErr_SetString(PyExc_ValueError, "dependency derivative shape mismatch");
        return nullptr;
    }
    const auto* etap = static_cast<const double*>(group_eta.buf);
    const auto* rowp = static_cast<const std::int64_t*>(rows.buf);
    const auto* designp = static_cast<const std::int32_t*>(design_groups.buf);
    const auto* baselinep = static_cast<const std::int32_t*>(baseline_groups.buf);
    const auto* offsetp = static_cast<const std::int64_t*>(offsets.buf);
    const auto* startp = static_cast<const std::int64_t*>(starts.buf);
    const auto* weightp = static_cast<const double*>(entity_weights.buf);
    const auto* dependencyp = static_cast<const std::int32_t*>(dependency_codes.buf);
    const auto* targetp = static_cast<const std::int64_t*>(target_rows.buf);
    const auto* countp = static_cast<const double*>(target_counts.buf);
    const auto* baseline_firstp = static_cast<const double*>(baseline_first.buf);
    auto* firstp = static_cast<double*>(first.buf);
    auto* entityp = static_cast<std::int32_t*>(entity_cluster.buf);
    auto* timep = static_cast<std::int32_t*>(time_cluster.buf);
    auto* defaultp = static_cast<double*>(default_first.buf);
    int invalid = 0;
    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(static) num_threads(requested_workers > 0 ? requested_workers : omp_get_max_threads()) reduction(| : invalid)
    for (std::int64_t entity = 0; entity < entity_count; ++entity) {
        const auto left = offsetp[entity];
        const auto right = offsetp[entity + 1];
        const auto* active_begin = std::lower_bound(rowp, rowp + active_count, left);
        const auto* active_end = std::lower_bound(active_begin, rowp + active_count, right);
        const auto* target_cursor = std::lower_bound(targetp, targetp + target_count, left);
        const auto* target_end = std::lower_bound(target_cursor, targetp + target_count, right);
        std::int64_t target_index = target_cursor - targetp;
        for (const auto* cursor = active_begin; cursor != active_end; ++cursor) {
            const auto active = static_cast<std::int64_t>(cursor - rowp);
            const auto design = static_cast<std::int64_t>(designp[active]);
            const auto baseline = static_cast<std::int64_t>(baselinep[active]);
            const auto calendar = startp[entity] + *cursor - left - origin;
            if (design < 0 || design >= group_eta.shape[0] || baseline < 0 ||
                baseline >= baseline_first.shape[0] || calendar < 0 ||
                calendar > std::numeric_limits<std::int32_t>::max()) {
                invalid |= 1;
                continue;
            }
            while (target_index < target_end - targetp && targetp[target_index] < *cursor)
                ++target_index;
            const double event =
                (target_index < target_end - targetp && targetp[target_index] == *cursor)
                    ? countp[target_index]
                    : 0.0;
            const double weight = weightp[entity];
            const double exposure = tick_exposure * weight;
            const double eta = etap[design];
            double derivative = 0.0;
            if (likelihood == 0) {
                // Match the canonical Poisson likelihood exactly.  A finite
                // fitted eta is part of the solver contract; silently
                // clipping only this dependency derivative would change the
                // declared sandwich complexity at extreme iterates.
                derivative = exposure * std::exp(eta) - event;
            } else {
                double event_value = 0.0, event_derivative = 0.0;
                cloglog_value_first(eta, event_value, event_derivative);
                const double intensity =
                    std::exp(std::max(-745.0, std::min(700.0, eta)));
                derivative = (exposure - event) * intensity + event * event_derivative;
            }
            firstp[active] = derivative;
            entityp[active] = dependencyp[entity];
            timep[active] = static_cast<std::int32_t>(calendar);
            defaultp[active] = baseline_firstp[baseline] * weight;
        }
    }
    Py_END_ALLOW_THREADS
    release();
    if (invalid) {
        PyErr_SetString(PyExc_ValueError, "dependency derivative index is invalid");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* subtract_group_weights(PyObject*, PyObject* args) {
    PyObject *groups_obj, *exposure_obj, *noevent_obj, *event_obj;
    PyObject *output_exposure_obj, *output_noevent_obj, *output_event_obj;
    if (!PyArg_ParseTuple(args, "OOOOOOO", &groups_obj, &exposure_obj,
                          &noevent_obj, &event_obj, &output_exposure_obj,
                          &output_noevent_obj, &output_event_obj)) {
        return nullptr;
    }
    Py_buffer groups{}, exposure{}, noevent{}, event{};
    Py_buffer output_exposure{}, output_noevent{}, output_event{};
    if (!int64_buffer(groups_obj, &groups, 1, false)) return nullptr;
    if (!double_buffer(exposure_obj, &exposure, 1, false)) {
        PyBuffer_Release(&groups);
        return nullptr;
    }
    if (!double_buffer(noevent_obj, &noevent, 1, false)) {
        PyBuffer_Release(&groups);
        PyBuffer_Release(&exposure);
        return nullptr;
    }
    if (!double_buffer(event_obj, &event, 1, false)) {
        PyBuffer_Release(&groups);
        PyBuffer_Release(&exposure);
        PyBuffer_Release(&noevent);
        return nullptr;
    }
    auto release_inputs = [&]() {
        PyBuffer_Release(&groups);
        PyBuffer_Release(&exposure);
        PyBuffer_Release(&noevent);
        PyBuffer_Release(&event);
    };
    if (!double_buffer(output_exposure_obj, &output_exposure, 1, true)) {
        release_inputs();
        return nullptr;
    }
    if (!double_buffer(output_noevent_obj, &output_noevent, 1, true)) {
        release_inputs();
        PyBuffer_Release(&output_exposure);
        return nullptr;
    }
    if (!double_buffer(output_event_obj, &output_event, 1, true)) {
        release_inputs();
        PyBuffer_Release(&output_exposure);
        PyBuffer_Release(&output_noevent);
        return nullptr;
    }
    const auto rows = groups.shape[0];
    const auto group_count = output_exposure.shape[0];
    if (exposure.shape[0] != rows || noevent.shape[0] != rows ||
        event.shape[0] != rows || output_noevent.shape[0] != group_count ||
        output_event.shape[0] != group_count) {
        PyErr_SetString(PyExc_ValueError, "group weight shape mismatch");
        release_inputs();
        PyBuffer_Release(&output_exposure);
        PyBuffer_Release(&output_noevent);
        PyBuffer_Release(&output_event);
        return nullptr;
    }
    auto* group = static_cast<std::int64_t*>(groups.buf);
    auto* ep = static_cast<double*>(exposure.buf);
    auto* np = static_cast<double*>(noevent.buf);
    auto* yp = static_cast<double*>(event.buf);
    auto* output_ep = static_cast<double*>(output_exposure.buf);
    auto* output_np = static_cast<double*>(output_noevent.buf);
    auto* output_yp = static_cast<double*>(output_event.buf);
    bool invalid_group = false;
    Py_BEGIN_ALLOW_THREADS
    // Stable input-order subtraction exactly matches three np.add.at calls,
    // but traverses the index vector once and avoids ufunc scatter overhead.
    for (std::int64_t input = 0; input < rows; ++input) {
        const auto destination = group[input];
        if (destination < 0 || destination >= group_count) {
            invalid_group = true;
            break;
        }
        output_ep[destination] -= ep[input];
        output_np[destination] -= np[input];
        output_yp[destination] -= yp[input];
    }
    Py_END_ALLOW_THREADS
    release_inputs();
    PyBuffer_Release(&output_exposure);
    PyBuffer_Release(&output_noevent);
    PyBuffer_Release(&output_event);
    if (invalid_group) {
        PyErr_SetString(PyExc_ValueError, "group weight index is out of range");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* completion_entity_offsets(PyObject*, PyObject* args) {
    PyObject *entities_obj, *pattern_starts_obj, *pattern_ends_obj, *output_obj;
    std::int64_t entity_count = 0;
    int workers = 1;
    if (!PyArg_ParseTuple(args, "OOOLiO", &entities_obj, &pattern_starts_obj,
                          &pattern_ends_obj, &entity_count, &workers,
                          &output_obj)) {
        return nullptr;
    }
    Py_buffer entities{}, pattern_starts{}, pattern_ends{}, output{};
    if (!int32_buffer(entities_obj, &entities, 1, false)) return nullptr;
    if (!int64_buffer(pattern_starts_obj, &pattern_starts, 1, false)) {
        PyBuffer_Release(&entities);
        return nullptr;
    }
    if (!int64_buffer(pattern_ends_obj, &pattern_ends, 1, false)) {
        PyBuffer_Release(&entities);
        PyBuffer_Release(&pattern_starts);
        return nullptr;
    }
    if (!int64_buffer(output_obj, &output, 2, true)) {
        PyBuffer_Release(&entities);
        PyBuffer_Release(&pattern_starts);
        PyBuffer_Release(&pattern_ends);
        return nullptr;
    }
    auto release = [&]() {
        PyBuffer_Release(&entities);
        PyBuffer_Release(&pattern_starts);
        PyBuffer_Release(&pattern_ends);
        PyBuffer_Release(&output);
    };
    const auto patterns = pattern_starts.shape[0];
    if (patterns < 1 || entity_count < 1 || workers < 1 ||
        pattern_ends.shape[0] != patterns ||
        output.shape[0] != patterns || output.shape[1] != entity_count + 1) {
        PyErr_SetString(PyExc_ValueError,
                        "completion offset buffers do not align");
        release();
        return nullptr;
    }
    const auto* ep = static_cast<const std::int32_t*>(entities.buf);
    const auto* sp = static_cast<const std::int64_t*>(pattern_starts.buf);
    const auto* xp = static_cast<const std::int64_t*>(pattern_ends.buf);
    auto* op = static_cast<std::int64_t*>(output.buf);
    const auto completion_count = entities.shape[0];
    int invalid = 0;
    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(static) num_threads(workers) reduction(| : invalid)
    for (std::int64_t pattern = 0; pattern < patterns; ++pattern) {
        const auto left = sp[pattern];
        const auto right = xp[pattern];
        auto* row = op + pattern * (entity_count + 1);
        if (left < 0 || right < left || right > completion_count) {
            invalid |= 1;
            continue;
        }
        auto cursor = left;
        row[0] = left;
        for (std::int64_t entity = 0; entity < entity_count; ++entity) {
            while (cursor < right && ep[cursor] == entity) ++cursor;
            if (cursor < right &&
                (ep[cursor] < entity || ep[cursor] >= entity_count)) {
                invalid |= 1;
            }
            row[entity + 1] = cursor;
        }
        if (cursor != right) invalid |= 1;
    }
    Py_END_ALLOW_THREADS
    release();
    if (invalid) {
        PyErr_SetString(PyExc_ValueError,
                        "completion entities are not sorted within pattern");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* completion_entity_profiles(PyObject*, PyObject* args) {
    PyObject *entities_obj, *times_obj, *spans_obj, *pattern_starts_obj,
        *pattern_ends_obj, *entity_ends_obj, *counts_obj,
        *output_entities_obj, *output_minimum_obj;
    int workers = 1;
    if (!PyArg_ParseTuple(args, "OOOOOOiOOO", &entities_obj, &times_obj,
                          &spans_obj, &pattern_starts_obj, &pattern_ends_obj,
                          &entity_ends_obj, &workers, &counts_obj,
                          &output_entities_obj, &output_minimum_obj)) {
        return nullptr;
    }
    Py_buffer entities{}, times{}, spans{}, pattern_starts{}, pattern_ends{},
        entity_ends{}, counts{}, output_entities{}, output_minimum{};
    int acquired = 0;
    auto release = [&]() {
        if (acquired >= 9) PyBuffer_Release(&output_minimum);
        if (acquired >= 8) PyBuffer_Release(&output_entities);
        if (acquired >= 7) PyBuffer_Release(&counts);
        if (acquired >= 6) PyBuffer_Release(&entity_ends);
        if (acquired >= 5) PyBuffer_Release(&pattern_ends);
        if (acquired >= 4) PyBuffer_Release(&pattern_starts);
        if (acquired >= 3) PyBuffer_Release(&spans);
        if (acquired >= 2) PyBuffer_Release(&times);
        if (acquired >= 1) PyBuffer_Release(&entities);
    };
    if (!int32_buffer(entities_obj, &entities, 1, false)) return nullptr;
    acquired = 1;
    if (!int64_buffer(times_obj, &times, 1, false)) { release(); return nullptr; }
    acquired = 2;
    if (!int64_buffer(spans_obj, &spans, 1, false)) { release(); return nullptr; }
    acquired = 3;
    if (!int64_buffer(pattern_starts_obj, &pattern_starts, 1, false)) {
        release(); return nullptr;
    }
    acquired = 4;
    if (!int64_buffer(pattern_ends_obj, &pattern_ends, 1, false)) {
        release(); return nullptr;
    }
    acquired = 5;
    if (!int64_buffer(entity_ends_obj, &entity_ends, 1, false)) {
        release(); return nullptr;
    }
    acquired = 6;
    if (!int64_buffer(counts_obj, &counts, 1, true)) { release(); return nullptr; }
    acquired = 7;
    if (!int32_buffer(output_entities_obj, &output_entities, 1, true)) {
        release(); return nullptr;
    }
    acquired = 8;
    if (!int64_buffer(output_minimum_obj, &output_minimum, 1, true)) {
        release(); return nullptr;
    }
    acquired = 9;
    const auto patterns = pattern_starts.shape[0];
    const auto completion_count = entities.shape[0];
    const auto entity_count = entity_ends.shape[0];
    if (workers < 1 || patterns < 1 || entity_count < 1 ||
        times.shape[0] != completion_count || spans.shape[0] != completion_count ||
        pattern_ends.shape[0] != patterns || counts.shape[0] != patterns ||
        output_entities.shape[0] != completion_count ||
        output_minimum.shape[0] != completion_count) {
        PyErr_SetString(PyExc_ValueError,
                        "completion profile buffers do not align");
        release();
        return nullptr;
    }
    const auto* ep = static_cast<const std::int32_t*>(entities.buf);
    const auto* tp = static_cast<const std::int64_t*>(times.buf);
    const auto* sp = static_cast<const std::int64_t*>(spans.buf);
    const auto* pp = static_cast<const std::int64_t*>(pattern_starts.buf);
    const auto* xp = static_cast<const std::int64_t*>(pattern_ends.buf);
    const auto* endp = static_cast<const std::int64_t*>(entity_ends.buf);
    auto* countp = static_cast<std::int64_t*>(counts.buf);
    auto* output_ep = static_cast<std::int32_t*>(output_entities.buf);
    auto* output_mp = static_cast<std::int64_t*>(output_minimum.buf);
    int invalid = 0;
    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(dynamic, 8) num_threads(workers) reduction(| : invalid)
    for (std::int64_t pattern = 0; pattern < patterns; ++pattern) {
        const auto left = pp[pattern];
        const auto right = xp[pattern];
        if (left < 0 || right < left || right > completion_count) {
            invalid |= 1;
            continue;
        }
        auto cursor = left;
        std::int64_t output_cursor = left;
        while (cursor < right) {
            const auto entity = ep[cursor];
            if (entity < 0 || entity >= entity_count) {
                invalid |= 1;
                break;
            }
            auto group_end = cursor + 1;
            while (group_end < right && ep[group_end] == entity) ++group_end;
            if (group_end < right && ep[group_end] < entity) invalid |= 1;
            auto minimum = std::numeric_limits<std::int64_t>::max();
            for (auto item = cursor; item < group_end; ++item) {
                if (sp[item] < 0) invalid |= 1;
                if (tp[item] < endp[entity])
                    minimum = std::min(minimum, sp[item]);
            }
            if (minimum != std::numeric_limits<std::int64_t>::max()) {
                output_ep[output_cursor] = entity;
                output_mp[output_cursor] = minimum;
                ++output_cursor;
            }
            cursor = group_end;
        }
        countp[pattern] = output_cursor - left;
    }
    Py_END_ALLOW_THREADS
    release();
    if (invalid) {
        PyErr_SetString(PyExc_ValueError,
                        "completion stream is invalid for entity profiling");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* candidate_entities_from_profiles(PyObject*, PyObject* args) {
    PyObject *profile_entities_obj, *minimum_spans_obj, *pattern_starts_obj,
        *pattern_counts_obj, *candidate_patterns_obj, *thresholds_obj,
        *output_offsets_obj, *output_entities_obj;
    int workers = 1;
    if (!PyArg_ParseTuple(args, "OOOOOOiOO", &profile_entities_obj,
                          &minimum_spans_obj, &pattern_starts_obj,
                          &pattern_counts_obj, &candidate_patterns_obj,
                          &thresholds_obj, &workers, &output_offsets_obj,
                          &output_entities_obj)) {
        return nullptr;
    }
    Py_buffer profile_entities{}, minimum_spans{}, pattern_starts{},
        pattern_counts{}, candidate_patterns{}, thresholds{}, output_offsets{},
        output_entities{};
    int acquired = 0;
    auto release = [&]() {
        if (acquired >= 8) PyBuffer_Release(&output_entities);
        if (acquired >= 7) PyBuffer_Release(&output_offsets);
        if (acquired >= 6) PyBuffer_Release(&thresholds);
        if (acquired >= 5) PyBuffer_Release(&candidate_patterns);
        if (acquired >= 4) PyBuffer_Release(&pattern_counts);
        if (acquired >= 3) PyBuffer_Release(&pattern_starts);
        if (acquired >= 2) PyBuffer_Release(&minimum_spans);
        if (acquired >= 1) PyBuffer_Release(&profile_entities);
    };
    if (!int32_buffer(profile_entities_obj, &profile_entities, 1, false))
        return nullptr;
    acquired = 1;
    if (!int64_buffer(minimum_spans_obj, &minimum_spans, 1, false)) {
        release(); return nullptr;
    }
    acquired = 2;
    if (!int64_buffer(pattern_starts_obj, &pattern_starts, 1, false)) {
        release(); return nullptr;
    }
    acquired = 3;
    if (!int64_buffer(pattern_counts_obj, &pattern_counts, 1, false)) {
        release(); return nullptr;
    }
    acquired = 4;
    if (!int32_buffer(candidate_patterns_obj, &candidate_patterns, 1, false)) {
        release(); return nullptr;
    }
    acquired = 5;
    if (!int64_buffer(thresholds_obj, &thresholds, 1, false)) {
        release(); return nullptr;
    }
    acquired = 6;
    if (!int64_buffer(output_offsets_obj, &output_offsets, 1, true)) {
        release(); return nullptr;
    }
    acquired = 7;
    if (!int32_buffer(output_entities_obj, &output_entities, 1, true)) {
        release(); return nullptr;
    }
    acquired = 8;
    const auto patterns = pattern_starts.shape[0];
    const auto candidates = candidate_patterns.shape[0];
    if (workers < 1 || patterns < 1 || candidates < 1 ||
        minimum_spans.shape[0] != profile_entities.shape[0] ||
        pattern_counts.shape[0] != patterns || thresholds.shape[0] != candidates ||
        output_offsets.shape[0] != candidates + 1) {
        PyErr_SetString(PyExc_ValueError,
                        "candidate profile buffers do not align");
        release();
        return nullptr;
    }
    const auto* entityp = static_cast<const std::int32_t*>(profile_entities.buf);
    const auto* minimump = static_cast<const std::int64_t*>(minimum_spans.buf);
    const auto* startp = static_cast<const std::int64_t*>(pattern_starts.buf);
    const auto* countp = static_cast<const std::int64_t*>(pattern_counts.buf);
    const auto* patternp = static_cast<const std::int32_t*>(candidate_patterns.buf);
    const auto* thresholdp = static_cast<const std::int64_t*>(thresholds.buf);
    auto* offsetp = static_cast<std::int64_t*>(output_offsets.buf);
    auto* outputp = static_cast<std::int32_t*>(output_entities.buf);
    const auto profile_capacity = profile_entities.shape[0];
    const auto output_capacity = output_entities.shape[0];
    int invalid = 0;
    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(static) num_threads(workers) reduction(| : invalid)
    for (std::int64_t candidate = 0; candidate < candidates; ++candidate) {
        const auto pattern = patternp[candidate];
        if (pattern < 0 || pattern >= patterns || thresholdp[candidate] < 0) {
            invalid |= 1;
            offsetp[candidate + 1] = 0;
            continue;
        }
        const auto left = startp[pattern];
        const auto right = left + countp[pattern];
        if (left < 0 || right < left || right > profile_capacity) {
            invalid |= 1;
            offsetp[candidate + 1] = 0;
            continue;
        }
        std::int64_t count = 0;
        for (auto item = left; item < right; ++item)
            count += minimump[item] <= thresholdp[candidate];
        offsetp[candidate + 1] = count;
    }
    offsetp[0] = 0;
    for (std::int64_t candidate = 0; candidate < candidates; ++candidate)
        offsetp[candidate + 1] += offsetp[candidate];
    if (offsetp[candidates] > output_capacity) invalid |= 1;
    if (!invalid) {
#pragma omp parallel for schedule(static) num_threads(workers) reduction(| : invalid)
        for (std::int64_t candidate = 0; candidate < candidates; ++candidate) {
            const auto pattern = patternp[candidate];
            const auto left = startp[pattern];
            const auto right = left + countp[pattern];
            auto destination = offsetp[candidate];
            for (auto item = left; item < right; ++item) {
                if (minimump[item] <= thresholdp[candidate])
                    outputp[destination++] = entityp[item];
            }
            if (destination != offsetp[candidate + 1]) invalid |= 1;
        }
    }
    Py_END_ALLOW_THREADS
    const auto total = offsetp[candidates];
    release();
    if (invalid) {
        PyErr_SetString(PyExc_ValueError,
                        "candidate profile index is invalid");
        return nullptr;
    }
    return PyLong_FromLongLong(total);
}


PyObject* dense_increment_sparse_distances(PyObject*, PyObject* args) {
    PyObject *dense_obj, *current_rows_obj, *current_values_obj,
        *right_rows_obj, *right_values_obj, *output_obj;
    double tolerance = 0.0;
    int workers = 1;
    if (!PyArg_ParseTuple(args, "OOOOOdOi", &dense_obj, &current_rows_obj,
                          &current_values_obj, &right_rows_obj,
                          &right_values_obj, &tolerance, &output_obj,
                          &workers)) {
        return nullptr;
    }
    Py_buffer dense{}, current_rows{}, current_values{}, output{};
    if (!double_buffer(dense_obj, &dense, 2, false)) return nullptr;
    if (!int64_buffer(current_rows_obj, &current_rows, 1, false)) {
        PyBuffer_Release(&dense);
        return nullptr;
    }
    if (!double_buffer(current_values_obj, &current_values, 1, false)) {
        PyBuffer_Release(&dense);
        PyBuffer_Release(&current_rows);
        return nullptr;
    }
    PyObject* right_rows_seq =
        PySequence_Fast(right_rows_obj, "right rows must be a sequence");
    PyObject* right_values_seq =
        PySequence_Fast(right_values_obj, "right values must be a sequence");
    if (right_rows_seq == nullptr || right_values_seq == nullptr) {
        Py_XDECREF(right_rows_seq);
        Py_XDECREF(right_values_seq);
        PyBuffer_Release(&dense);
        PyBuffer_Release(&current_rows);
        PyBuffer_Release(&current_values);
        return nullptr;
    }
    const auto right_count = PySequence_Fast_GET_SIZE(right_rows_seq);
    if (right_count != PySequence_Fast_GET_SIZE(right_values_seq)) {
        PyErr_SetString(PyExc_ValueError,
                        "right sparse profile sequences do not align");
        Py_DECREF(right_rows_seq);
        Py_DECREF(right_values_seq);
        PyBuffer_Release(&dense);
        PyBuffer_Release(&current_rows);
        PyBuffer_Release(&current_values);
        return nullptr;
    }
    std::vector<Py_buffer> right_rows(static_cast<std::size_t>(right_count));
    std::vector<Py_buffer> right_values(static_cast<std::size_t>(right_count));
    Py_ssize_t acquired_right = 0;
    auto release = [&]() {
        for (Py_ssize_t index = 0; index < acquired_right; ++index) {
            PyBuffer_Release(&right_values[static_cast<std::size_t>(index)]);
            PyBuffer_Release(&right_rows[static_cast<std::size_t>(index)]);
        }
        Py_DECREF(right_rows_seq);
        Py_DECREF(right_values_seq);
        PyBuffer_Release(&dense);
        PyBuffer_Release(&current_rows);
        PyBuffer_Release(&current_values);
    };
    for (Py_ssize_t index = 0; index < right_count; ++index) {
        if (!int64_buffer(PySequence_Fast_GET_ITEM(right_rows_seq, index),
                          &right_rows[static_cast<std::size_t>(index)], 1,
                          false)) {
            release();
            return nullptr;
        }
        if (!double_buffer(PySequence_Fast_GET_ITEM(right_values_seq, index),
                           &right_values[static_cast<std::size_t>(index)], 1,
                           false)) {
            PyBuffer_Release(&right_rows[static_cast<std::size_t>(index)]);
            release();
            return nullptr;
        }
        ++acquired_right;
        if (right_rows[static_cast<std::size_t>(index)].shape[0] !=
            right_values[static_cast<std::size_t>(index)].shape[0]) {
            PyErr_SetString(PyExc_ValueError,
                            "right sparse profile rows and values do not align");
            release();
            return nullptr;
        }
    }
    if (!double_buffer(output_obj, &output, 2, true)) {
        release();
        return nullptr;
    }
    const auto candidate_count = dense.shape[0];
    const auto row_count = dense.shape[1];
    const bool valid =
        current_rows.shape[0] == current_values.shape[0] &&
        output.shape[0] == candidate_count &&
        output.shape[1] == right_count && tolerance >= 0.0 && workers >= 0;
    if (!valid) {
        PyErr_SetString(PyExc_ValueError,
                        "dense/sparse Fisher distance buffers do not align");
        PyBuffer_Release(&output);
        release();
        return nullptr;
    }

    const auto* densep = static_cast<const double*>(dense.buf);
    const auto* current_rowp =
        static_cast<const std::int64_t*>(current_rows.buf);
    const auto* current_valuep =
        static_cast<const double*>(current_values.buf);
    auto* outputp = static_cast<double*>(output.buf);
    const auto current_count = current_rows.shape[0];
    const auto thread_count = workers > 0 ? workers : omp_get_max_threads();
    int invalid = 0;
    int allocation_failed = 0;
    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count) \
    reduction(| : invalid, allocation_failed)
    for (Py_ssize_t candidate = 0; candidate < candidate_count; ++candidate) {
        try {
            std::vector<std::int64_t> candidate_rows;
            std::vector<double> candidate_values;
            candidate_rows.reserve(static_cast<std::size_t>(
                std::min<Py_ssize_t>(row_count, current_count + 4096)));
            candidate_values.reserve(candidate_rows.capacity());
            Py_ssize_t current_position = 0;
            bool has_increment = false;
            const auto* increment = densep + candidate * row_count;
            for (Py_ssize_t row = 0; row < row_count; ++row) {
                if (current_position < current_count &&
                    current_rowp[current_position] < row) {
                    invalid |= 1;
                    break;
                }
                const bool has_current =
                    current_position < current_count &&
                    current_rowp[current_position] == row;
                const double raw_increment = increment[row];
                const bool has_active_increment =
                    std::isfinite(raw_increment) &&
                    std::abs(raw_increment) > tolerance;
                has_increment = has_increment || has_active_increment;
                if (!has_current && !has_active_increment) continue;
                double value = has_active_increment ? raw_increment : 0.0;
                if (has_current) {
                    value += current_valuep[current_position];
                    ++current_position;
                }
                if (std::isfinite(value) && std::abs(value) > tolerance) {
                    candidate_rows.push_back(static_cast<std::int64_t>(row));
                    candidate_values.push_back(value);
                }
            }
            if (current_position != current_count) invalid |= 1;
            if (!has_increment || invalid) {
                for (Py_ssize_t right = 0; right < right_count; ++right) {
                    outputp[candidate * right_count + right] =
                        std::numeric_limits<double>::quiet_NaN();
                }
                continue;
            }
            for (Py_ssize_t right = 0; right < right_count; ++right) {
                const auto& rr = right_rows[static_cast<std::size_t>(right)];
                const auto& rv = right_values[static_cast<std::size_t>(right)];
                const auto* rrp = static_cast<const std::int64_t*>(rr.buf);
                const auto* rvp = static_cast<const double*>(rv.buf);
                std::size_t left_position = 0;
                Py_ssize_t right_position = 0;
                double distance = 0.0;
                while (left_position < candidate_rows.size() &&
                       right_position < rr.shape[0]) {
                    const auto left_row = candidate_rows[left_position];
                    const auto right_row = rrp[right_position];
                    if (left_row == right_row) {
                        const double difference =
                            candidate_values[left_position] -
                            rvp[right_position];
                        distance += difference * difference;
                        ++left_position;
                        ++right_position;
                    } else if (left_row < right_row) {
                        const double value = candidate_values[left_position];
                        distance += value * value;
                        ++left_position;
                    } else {
                        const double value = rvp[right_position];
                        distance += value * value;
                        ++right_position;
                    }
                }
                while (left_position < candidate_rows.size()) {
                    const double value = candidate_values[left_position++];
                    distance += value * value;
                }
                while (right_position < rr.shape[0]) {
                    const double value = rvp[right_position++];
                    distance += value * value;
                }
                outputp[candidate * right_count + right] = distance;
            }
        } catch (const std::bad_alloc&) {
            allocation_failed |= 1;
        }
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&output);
    release();
    if (allocation_failed) return PyErr_NoMemory();
    if (invalid) {
        PyErr_SetString(PyExc_ValueError,
                        "sparse Fisher profile rows are outside the dense grid");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* sparse_squared_distances(PyObject*, PyObject* args) {
    PyObject *left_rows_obj, *left_values_obj, *right_rows_obj,
        *right_values_obj, *output_obj;
    int workers = 1;
    if (!PyArg_ParseTuple(args, "OOOOOi", &left_rows_obj, &left_values_obj,
                          &right_rows_obj, &right_values_obj, &output_obj,
                          &workers)) {
        return nullptr;
    }
    PyObject* left_rows_seq =
        PySequence_Fast(left_rows_obj, "left rows must be a sequence");
    PyObject* left_values_seq =
        PySequence_Fast(left_values_obj, "left values must be a sequence");
    PyObject* right_rows_seq =
        PySequence_Fast(right_rows_obj, "right rows must be a sequence");
    PyObject* right_values_seq =
        PySequence_Fast(right_values_obj, "right values must be a sequence");
    if (left_rows_seq == nullptr || left_values_seq == nullptr ||
        right_rows_seq == nullptr || right_values_seq == nullptr) {
        Py_XDECREF(left_rows_seq);
        Py_XDECREF(left_values_seq);
        Py_XDECREF(right_rows_seq);
        Py_XDECREF(right_values_seq);
        return nullptr;
    }
    const auto left_count = PySequence_Fast_GET_SIZE(left_rows_seq);
    const auto right_count = PySequence_Fast_GET_SIZE(right_rows_seq);
    if (left_count != PySequence_Fast_GET_SIZE(left_values_seq) ||
        right_count != PySequence_Fast_GET_SIZE(right_values_seq)) {
        PyErr_SetString(PyExc_ValueError,
                        "sparse profile row/value sequences do not align");
        Py_DECREF(left_rows_seq);
        Py_DECREF(left_values_seq);
        Py_DECREF(right_rows_seq);
        Py_DECREF(right_values_seq);
        return nullptr;
    }

    std::vector<Py_buffer> left_rows(static_cast<std::size_t>(left_count));
    std::vector<Py_buffer> left_values(static_cast<std::size_t>(left_count));
    std::vector<Py_buffer> right_rows(static_cast<std::size_t>(right_count));
    std::vector<Py_buffer> right_values(static_cast<std::size_t>(right_count));
    Py_buffer output{};
    Py_ssize_t acquired_left = 0;
    Py_ssize_t acquired_right = 0;
    bool output_acquired = false;
    auto release = [&]() {
        if (output_acquired) PyBuffer_Release(&output);
        for (Py_ssize_t index = 0; index < acquired_right; ++index) {
            PyBuffer_Release(&right_values[static_cast<std::size_t>(index)]);
            PyBuffer_Release(&right_rows[static_cast<std::size_t>(index)]);
        }
        for (Py_ssize_t index = 0; index < acquired_left; ++index) {
            PyBuffer_Release(&left_values[static_cast<std::size_t>(index)]);
            PyBuffer_Release(&left_rows[static_cast<std::size_t>(index)]);
        }
        Py_DECREF(left_rows_seq);
        Py_DECREF(left_values_seq);
        Py_DECREF(right_rows_seq);
        Py_DECREF(right_values_seq);
    };
    for (Py_ssize_t index = 0; index < left_count; ++index) {
        if (!int64_buffer(PySequence_Fast_GET_ITEM(left_rows_seq, index),
                          &left_rows[static_cast<std::size_t>(index)], 1,
                          false)) {
            release();
            return nullptr;
        }
        if (!double_buffer(PySequence_Fast_GET_ITEM(left_values_seq, index),
                           &left_values[static_cast<std::size_t>(index)], 1,
                           false)) {
            PyBuffer_Release(&left_rows[static_cast<std::size_t>(index)]);
            release();
            return nullptr;
        }
        ++acquired_left;
        if (left_rows[static_cast<std::size_t>(index)].shape[0] !=
            left_values[static_cast<std::size_t>(index)].shape[0]) {
            PyErr_SetString(PyExc_ValueError,
                            "left sparse profile rows and values do not align");
            release();
            return nullptr;
        }
    }
    for (Py_ssize_t index = 0; index < right_count; ++index) {
        if (!int64_buffer(PySequence_Fast_GET_ITEM(right_rows_seq, index),
                          &right_rows[static_cast<std::size_t>(index)], 1,
                          false)) {
            release();
            return nullptr;
        }
        if (!double_buffer(PySequence_Fast_GET_ITEM(right_values_seq, index),
                           &right_values[static_cast<std::size_t>(index)], 1,
                           false)) {
            PyBuffer_Release(&right_rows[static_cast<std::size_t>(index)]);
            release();
            return nullptr;
        }
        ++acquired_right;
        if (right_rows[static_cast<std::size_t>(index)].shape[0] !=
            right_values[static_cast<std::size_t>(index)].shape[0]) {
            PyErr_SetString(
                PyExc_ValueError,
                "right sparse profile rows and values do not align");
            release();
            return nullptr;
        }
    }
    if (!double_buffer(output_obj, &output, 2, true)) {
        release();
        return nullptr;
    }
    output_acquired = true;
    if (output.shape[0] != left_count || output.shape[1] != right_count ||
        workers < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "sparse distance output shape does not align");
        release();
        return nullptr;
    }

    auto* outputp = static_cast<double*>(output.buf);
    const auto pair_count = left_count * right_count;
    const auto thread_count = workers > 0 ? workers : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
    for (Py_ssize_t pair = 0; pair < pair_count; ++pair) {
        const auto left_index = pair / right_count;
        const auto right_index = pair % right_count;
        const auto& lr = left_rows[static_cast<std::size_t>(left_index)];
        const auto& lv = left_values[static_cast<std::size_t>(left_index)];
        const auto& rr = right_rows[static_cast<std::size_t>(right_index)];
        const auto& rv = right_values[static_cast<std::size_t>(right_index)];
        const auto* lrp = static_cast<const std::int64_t*>(lr.buf);
        const auto* lvp = static_cast<const double*>(lv.buf);
        const auto* rrp = static_cast<const std::int64_t*>(rr.buf);
        const auto* rvp = static_cast<const double*>(rv.buf);
        Py_ssize_t left_position = 0;
        Py_ssize_t right_position = 0;
        double distance = 0.0;
        while (left_position < lr.shape[0] &&
               right_position < rr.shape[0]) {
            if (lrp[left_position] == rrp[right_position]) {
                const double difference =
                    lvp[left_position] - rvp[right_position];
                distance += difference * difference;
                ++left_position;
                ++right_position;
            } else if (lrp[left_position] < rrp[right_position]) {
                distance += lvp[left_position] * lvp[left_position];
                ++left_position;
            } else {
                distance += rvp[right_position] * rvp[right_position];
                ++right_position;
            }
        }
        while (left_position < lr.shape[0]) {
            distance += lvp[left_position] * lvp[left_position];
            ++left_position;
        }
        while (right_position < rr.shape[0]) {
            distance += rvp[right_position] * rvp[right_position];
            ++right_position;
        }
        outputp[pair] = distance;
    }
    Py_END_ALLOW_THREADS
    release();
    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"moments", moments, METH_VARARGS, "Deterministic gradient/Fisher moments."},
    {"accumulate_cluster_scores", accumulate_cluster_scores, METH_VARARGS, "Accumulate row-design scores by dependency cluster."},
    {"likelihood_value_eta_gradient", likelihood_value_eta_gradient, METH_VARARGS, "Fused exact likelihood, predictor, and gradient."},
    {"design_column_cross", design_column_cross, METH_VARARGS, "Exact X'X column without a full Gram."},
    {"nonnegative_quadratic_gains", nonnegative_quadratic_gains, METH_VARARGS, "Exact batched nonnegative quadratic gains."},
    {"cloglog_mixed_conjugate", cloglog_mixed_conjugate, METH_VARARGS, "Exact mixed-row Bernoulli-cloglog conjugates."},
    {"aggregate_design_rows", aggregate_design_rows, METH_VARARGS, "Losslessly aggregate identical design rows."},
    {"aggregate_quotient_rows", aggregate_quotient_rows, METH_VARARGS, "Losslessly aggregate state-quotient rows in parallel."},
    {"entity_loss_contrast", entity_loss_contrast, METH_VARARGS, "Stream exact fitted-model NLL contrasts by entity."},
    {"dependency_row_derivatives", dependency_row_derivatives, METH_VARARGS, "Stream active-row dependency derivatives and cluster indices."},
    {"subtract_group_weights", subtract_group_weights, METH_VARARGS, "Subtract three weight vectors by integer group in one pass."},
    {"completion_entity_offsets", completion_entity_offsets, METH_VARARGS, "Build absolute entity offsets for compact completion waves."},
    {"completion_entity_profiles", completion_entity_profiles, METH_VARARGS, "Build exact strict-future minimum-span entity profiles."},
    {"candidate_entities_from_profiles", candidate_entities_from_profiles, METH_VARARGS, "Pack exact W-admissible candidate entities."},
    {"set_num_threads", set_num_threads, METH_VARARGS, "Set deterministic OpenMP worker count."},
    {"future_rows", future_rows, METH_VARARGS, "Strict-future footprint rows."},
    {"accumulate_kernel", accumulate_kernel, METH_VARARGS, "Accumulate newly admitted kernel completions."},
    {"kernel_touched_positions", kernel_touched_positions, METH_VARARGS, "Rows changed by newly admitted completions."},
    {"fill_candidate_batch", fill_candidate_batch, METH_VARARGS, "Fill one hierarchy candidate tile."},
    {"fill_pricing_values", fill_pricing_values, METH_VARARGS, "Gather one mutable hierarchy block into a pricing tile."},
    {"sparse_joint_moments", sparse_joint_moments, METH_VARARGS, "Exact moments of sparse joint blocks."},
    {"kernel_contributions", kernel_contributions, METH_VARARGS, "Strict-future kernel contributions."},
    {"completion_events", completion_events, METH_VARARGS, "Latest-witness completion events."},
    {"ordered_completion_events", ordered_completion_events, METH_VARARGS, "Strict ordered completion events."},
    {"observed_temporal_motifs", observed_temporal_motifs, METH_VARARGS, "Observed distinct-primitive temporal motif masks."},
    {"completion_window_counts", completion_window_counts, METH_VARARGS, "Streaming productive completion counts by W."},
    {"response_min_spans", response_min_spans, METH_VARARGS, "Dense exact minimum witness span per response row."},
    {"continuous_single_block_moments", continuous_single_block_moments, METH_VARARGS, "Exact interval-native continuous completion moments."},
    {"continuous_single_block_profiles", continuous_single_block_profiles, METH_VARARGS, "Exact interval-native continuous fitted profiles."},
    {"continuous_additive_support_profiles", continuous_additive_support_profiles, METH_VARARGS, "Exact fused additive-support continuous profiles."},
    {"continuous_single_block_profile_distances", continuous_single_block_profile_distances, METH_VARARGS, "Exact fused continuous fitted-profile distances."},
    {"dense_increment_sparse_distances", dense_increment_sparse_distances, METH_VARARGS, "Fused dense-increment distances to sparse profiles."},
    {"sparse_squared_distances", sparse_squared_distances, METH_VARARGS, "Batched squared distances between sorted sparse profiles."},
    {"safe_shell_counts", safe_shell_counts, METH_VARARGS, "Exact nested-W safe-bound counts from completion streams."},
    {"label_run_ends", label_run_ends, METH_VARARGS, "Exact run ends for immutable row labels."},
    {"safe_shell_counts_sources", safe_shell_counts_sources, METH_VARARGS, "Exact nested-W safe-bound counts fused from primitive source streams."},
    {"bounded_span_order", bounded_span_order, METH_VARARGS, "Stable counting order for bounded completion spans."},
    {"sorted_unique_union", sorted_unique_union, METH_VARARGS, "Merge sorted unique row arrays without sorting."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_cpu_native", nullptr, -1, methods};

}  // namespace

PyMODINIT_FUNC PyInit__cpu_native() { return PyModule_Create(&module); }

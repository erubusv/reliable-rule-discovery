#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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
        double sum = 0.0;
        for (std::int64_t i = 0; i < rows; ++i) sum += xp[i * columns + j] * fp[i];
        gp[j] = sum;
    }
    #pragma omp parallel for schedule(static)
    for (std::int64_t flat = 0; flat < columns * columns; ++flat) {
        const std::int64_t j = flat / columns, k = flat % columns;
        double sum = 0.0;
        for (std::int64_t i = 0; i < rows; ++i) {
            sum += xp[i * columns + j] * sp[i] * xp[i * columns + k];
        }
        hp[flat] = sum;
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&x); PyBuffer_Release(&first); PyBuffer_Release(&second);
    PyBuffer_Release(&gradient); PyBuffer_Release(&hessian);
    Py_RETURN_NONE;
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

PyMethodDef methods[] = {
    {"moments", moments, METH_VARARGS, "Deterministic gradient/Fisher moments."},
    {"kernel_contributions", kernel_contributions, METH_VARARGS, "Strict-future kernel contributions."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_cpu_native", nullptr, -1, methods};

}  // namespace

PyMODINIT_FUNC PyInit__cpu_native() { return PyModule_Create(&module); }

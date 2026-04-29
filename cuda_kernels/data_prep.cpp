/*
 * OpenMP-Parallelized Data Preprocessing (Step 7)
 * =================================================
 * CPU-side operations parallelized with OpenMP:
 *
 *  1. parallel_nan_insert     — parallel NaN placement in missing-data simulation
 *  2. parallel_frob_diff_norm — parallel ||A - B||_F reduction for accuracy eval
 *  3. parallel_cast_f32       — parallel float64 → float32 cast before H2D transfer
 *
 * ME759 concepts: #pragma omp parallel for, reduction clause, schedule(static),
 *                 private variables, thread-local RNG state
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

// ============================================================
//  1. Parallel NaN insertion
// ============================================================
//  Replaces the sequential loop:
//    input_copy.ravel()[indices] = np.nan
//  with a parallel write across `n_nan` positions.
//
//  index scatter-writes are independent (no aliasing) so the
//  parallel for is safe without synchronisation.

py::array_t<double> parallel_nan_insert(
    py::array_t<double, py::array::c_style | py::array::forcecast> data,
    py::array_t<long long, py::array::c_style | py::array::forcecast> indices)
{
    auto d_buf = data.request();
    auto i_buf = indices.request();

    double*    d_ptr = static_cast<double*>(d_buf.ptr);
    long long* i_ptr = static_cast<long long*>(i_buf.ptr);

    const int n_nan   = static_cast<int>(i_buf.size);
    const int n_total = static_cast<int>(d_buf.size);

    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_nan; i++) {
        d_ptr[i_ptr[i]] = nan_val;
    }

    return data;
}


// ============================================================
//  2. Parallel Frobenius norm  ||A - B||_F
// ============================================================
//  Replaces  np.linalg.norm(A - B)  with a parallel reduction.
//  Used in benchmark.py for completion accuracy:
//    accuracy = 1 - ||complete_data - full_data||_F / data_norm
//
//  OpenMP reduction(+:sum) aggregates per-thread partial sums.

double parallel_frob_diff_norm(
    py::array_t<double, py::array::c_style | py::array::forcecast> A,
    py::array_t<double, py::array::c_style | py::array::forcecast> B)
{
    auto a_buf = A.request();
    auto b_buf = B.request();

    if (a_buf.size != b_buf.size)
        throw std::invalid_argument("A and B must have the same number of elements");

    const double* a = static_cast<const double*>(a_buf.ptr);
    const double* b = static_cast<const double*>(b_buf.ptr);
    const int N     = static_cast<int>(a_buf.size);

    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < N; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }

    return std::sqrt(sum);
}


// ============================================================
//  3. Parallel float64 → float32 cast
// ============================================================
//  Parallelises the dtype conversion before pinned-memory H2D
//  transfer.  Returns a new float32 array.

py::array_t<float> parallel_cast_f32(
    py::array_t<double, py::array::c_style | py::array::forcecast> data)
{
    auto in_buf = data.request();
    const int N         = static_cast<int>(in_buf.size);
    const double* in_p  = static_cast<const double*>(in_buf.ptr);

    // Allocate output with same shape
    auto result = py::array_t<float>(in_buf.shape);
    float* out_p = static_cast<float*>(result.request().ptr);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        out_p[i] = static_cast<float>(in_p[i]);
    }

    return result;
}


// ============================================================
//  4. Query available threads
// ============================================================

int omp_thread_count() {
    int n = 1;
    #pragma omp parallel
    {
        #pragma omp single
        n = omp_get_num_threads();
    }
    return n;
}


// ============================================================
//  Python bindings
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("parallel_nan_insert", &parallel_nan_insert,
          "Parallel NaN insertion — scatter writes with omp parallel for");
    m.def("parallel_frob_diff_norm", &parallel_frob_diff_norm,
          "Parallel ||A-B||_F reduction — omp parallel for reduction(+:sum)");
    m.def("parallel_cast_f32", &parallel_cast_f32,
          "Parallel float64 → float32 cast — omp parallel for schedule(static)");
    m.def("omp_thread_count", &omp_thread_count,
          "Return number of OpenMP threads available");
}

/*
 * Fused Masked-Loss Reduction Kernel
 * ====================================
 * Fuses NaN masking + 3 Frobenius norm accumulations in one pass,
 * replacing 7+ separate PyTorch kernel launches with 2 CUDA kernels
 * and eliminating all intermediate tensor allocations.
 *
 * Forward: one thread per element (B*F), warp-shuffle + atomicAdd
 *   loss = ||Xc_m - x_omega||_F + ||Xc_m - dec_m||_F + ||dec_m - x_omega||_F
 *
 * Backward: elementwise, one thread per element
 *   d_Xc[i]      = mask * (d1/norm_d1 + d2/norm_d2)
 *   d_decoded[i] = mask * (-d2/norm_d2 + d3/norm_d3)
 *
 * ME759 concepts: warp-shuffle reduction, shared memory, atomicAdd,
 *                 kernel fusion, flat grid striding
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256


// ============================================================
//  Forward kernel: fused NaN mask + 3 squared-norm reduction
// ============================================================
//  Grid:  ceil(N / BLOCK_SIZE) blocks, strided loop over N = B*F
//  Block: BLOCK_SIZE threads
//  Smem:  3 * n_warps floats  (warp partial sums for d1², d2², d3²)

__global__ void __launch_bounds__(BLOCK_SIZE)
masked_loss_fwd_kernel(
    const float* __restrict__ x,        // (B, F) — has NaN
    const float* __restrict__ Xc,       // (B, F)
    const float* __restrict__ dec,      // (B, F)
    float* __restrict__ acc_d1,         // scalar accumulator
    float* __restrict__ acc_d2,
    float* __restrict__ acc_d3,
    int N)                              // = B * F
{
    const int n_warps = BLOCK_SIZE / 32;
    extern __shared__ float smem[];     // 3 * n_warps floats
    float* s1 = smem;
    float* s2 = smem + n_warps;
    float* s3 = smem + 2 * n_warps;

    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    float local_d1 = 0.f, local_d2 = 0.f, local_d3 = 0.f;

    // --- Strided loop: each thread covers multiple elements ---
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < N; i += gridDim.x * BLOCK_SIZE) {
        float xi = __ldg(&x[i]);
        float mask = isnan(xi) ? 0.f : 1.f;
        float xo   = isnan(xi) ? 0.f : xi;
        float xc   = __ldg(&Xc[i])  * mask;   // Xc_m
        float dc   = __ldg(&dec[i]) * mask;   // dec_m

        float d1 = xc - xo;
        float d2 = xc - dc;
        float d3 = dc - xo;

        local_d1 += d1 * d1;
        local_d2 += d2 * d2;
        local_d3 += d3 * d3;
    }

    // --- Warp-shuffle reduction (intra-warp) ---
    for (int off = 16; off > 0; off >>= 1) {
        local_d1 += __shfl_down_sync(0xFFFFFFFF, local_d1, off);
        local_d2 += __shfl_down_sync(0xFFFFFFFF, local_d2, off);
        local_d3 += __shfl_down_sync(0xFFFFFFFF, local_d3, off);
    }

    // --- Lane 0 writes warp partial sums to shared memory ---
    if (lane == 0) {
        s1[warp_id] = local_d1;
        s2[warp_id] = local_d2;
        s3[warp_id] = local_d3;
    }
    __syncthreads();

    // --- Warp 0 reduces across warp partials ---
    if (warp_id == 0) {
        local_d1 = (lane < n_warps) ? s1[lane] : 0.f;
        local_d2 = (lane < n_warps) ? s2[lane] : 0.f;
        local_d3 = (lane < n_warps) ? s3[lane] : 0.f;

        for (int off = 16; off > 0; off >>= 1) {
            local_d1 += __shfl_down_sync(0xFFFFFFFF, local_d1, off);
            local_d2 += __shfl_down_sync(0xFFFFFFFF, local_d2, off);
            local_d3 += __shfl_down_sync(0xFFFFFFFF, local_d3, off);
        }

        // Thread 0: atomic add to global accumulators
        if (lane == 0) {
            atomicAdd(acc_d1, local_d1);
            atomicAdd(acc_d2, local_d2);
            atomicAdd(acc_d3, local_d3);
        }
    }
}


// ============================================================
//  Backward kernel: elementwise gradient w.r.t. Xc and decoded
// ============================================================
//  Grid:  ceil(N / BLOCK_SIZE) blocks
//  Block: BLOCK_SIZE threads
//  No reduction — pure elementwise, fully parallel

__global__ void __launch_bounds__(BLOCK_SIZE)
masked_loss_bwd_kernel(
    const float* __restrict__ x,        // (B, F) — has NaN
    const float* __restrict__ Xc,       // (B, F)
    const float* __restrict__ dec,      // (B, F)
    float* __restrict__ d_Xc,           // (B, F) output
    float* __restrict__ d_dec,          // (B, F) output
    float norm_d1,                      // sqrt(sum_d1²), from forward
    float norm_d2,
    float norm_d3,
    int N)
{
    const float eps = 1e-8f;
    float inv1 = 1.f / (norm_d1 + eps);
    float inv2 = 1.f / (norm_d2 + eps);
    float inv3 = 1.f / (norm_d3 + eps);

    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= N) return;

    float xi   = __ldg(&x[i]);
    float mask = isnan(xi) ? 0.f : 1.f;
    float xo   = isnan(xi) ? 0.f : xi;
    float xc   = __ldg(&Xc[i])  * mask;
    float dc   = __ldg(&dec[i]) * mask;

    float d1 = xc - xo;
    float d2 = xc - dc;
    float d3 = dc - xo;

    // d_loss/d_Xc[i]      = mask * (d1/||d1|| + d2/||d2||)
    // d_loss/d_decoded[i] = mask * (-d2/||d2|| + d3/||d3||)
    d_Xc[i]  = mask * (d1 * inv1 + d2 * inv2);
    d_dec[i] = mask * (-d2 * inv2 + d3 * inv3);
}


// ============================================================
//  C++ wrapper — forward
// ============================================================

std::vector<torch::Tensor> masked_loss_forward(
    torch::Tensor x,        // (B, F), may have NaN, float32
    torch::Tensor Xc,       // (B, F), float32
    torch::Tensor decoded)  // (B, F), float32
{
    TORCH_CHECK(x.is_cuda(),       "x must be CUDA");
    TORCH_CHECK(Xc.is_cuda(),      "Xc must be CUDA");
    TORCH_CHECK(decoded.is_cuda(), "decoded must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32,       "x must be float32");
    TORCH_CHECK(Xc.dtype() == torch::kFloat32,      "Xc must be float32");
    TORCH_CHECK(decoded.dtype() == torch::kFloat32, "decoded must be float32");

    auto x_c   = x.contiguous();
    auto xc_c  = Xc.contiguous();
    auto dec_c = decoded.contiguous();

    const int N = x_c.numel();   // B * F

    // Zero-initialized accumulators on device
    auto sum_d1 = torch::zeros({1}, x.options());
    auto sum_d2 = torch::zeros({1}, x.options());
    auto sum_d3 = torch::zeros({1}, x.options());

    const int n_warps = BLOCK_SIZE / 32;
    const int blocks  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t smem = 3 * n_warps * sizeof(float);

    masked_loss_fwd_kernel<<<blocks, BLOCK_SIZE, smem>>>(
        x_c.data_ptr<float>(),
        xc_c.data_ptr<float>(),
        dec_c.data_ptr<float>(),
        sum_d1.data_ptr<float>(),
        sum_d2.data_ptr<float>(),
        sum_d3.data_ptr<float>(),
        N);

    // loss = sqrt(sum_d1) + sqrt(sum_d2) + sqrt(sum_d3)
    auto loss = torch::sqrt(sum_d1) + torch::sqrt(sum_d2) + torch::sqrt(sum_d3);
    loss = loss.squeeze(0);   // scalar

    return {loss, sum_d1, sum_d2, sum_d3};
}


// ============================================================
//  C++ wrapper — backward
// ============================================================

std::vector<torch::Tensor> masked_loss_backward(
    torch::Tensor x,
    torch::Tensor Xc,
    torch::Tensor decoded,
    double norm_d1,
    double norm_d2,
    double norm_d3)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");

    auto x_c   = x.contiguous();
    auto xc_c  = Xc.contiguous();
    auto dec_c = decoded.contiguous();

    const int N = x_c.numel();

    auto d_Xc  = torch::empty_like(xc_c);
    auto d_dec = torch::empty_like(dec_c);

    const int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    masked_loss_bwd_kernel<<<blocks, BLOCK_SIZE>>>(
        x_c.data_ptr<float>(),
        xc_c.data_ptr<float>(),
        dec_c.data_ptr<float>(),
        d_Xc.data_ptr<float>(),
        d_dec.data_ptr<float>(),
        static_cast<float>(norm_d1),
        static_cast<float>(norm_d2),
        static_cast<float>(norm_d3),
        N);

    return {d_Xc, d_dec};
}


// ============================================================
//  Python bindings
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &masked_loss_forward,
          "Fused masked-loss forward (warp-shuffle reduction)");
    m.def("backward", &masked_loss_backward,
          "Fused masked-loss backward (elementwise)");
}

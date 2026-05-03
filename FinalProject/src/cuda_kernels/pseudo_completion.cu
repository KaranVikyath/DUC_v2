/*
 * Fused Pseudo-Completion CUDA Kernels (Forward + Backward)
 * ==========================================================
 * Fuses per-feature matmul + bias + PReLU into single kernels,
 * eliminating intermediate tensor allocations and kernel launch overhead.
 *
 * Forward: For each feature f (one block) and batch element b (one thread):
 *   z[f][b] = dot(W[f][b][:], x[:, f]) + bias[f][b]
 *   out[b][f] = PReLU(z[f][b], alpha[f])
 *
 * Backward: Fuses PReLU backward + d_weight outer product + d_bias + d_prelu
 * reduction into a single kernel per feature.
 *
 * ME759 concepts: thread/block config, shared memory, kernel fusion,
 *                 warp shuffle reduction, __ldg read-only cache
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// ============================================================
//  Forward kernel
// ============================================================
//  Grid:  (feature_size,)     — one block per feature
//  Block: (min(batch_size, 512),) — threads stride over batch
//  Smem:  batch_size floats   — cached input column x[:, f]

__global__ void __launch_bounds__(512)
fused_pseudo_fwd(
    const float* __restrict__ x,        // (B, F)
    const float* __restrict__ weight,   // (F, B, B)
    const float* __restrict__ bias,     // (F, B)
    const float* __restrict__ prelu_w,  // (F,)
    float* __restrict__ out,            // (B, F)
    float* __restrict__ pre_act,        // (F, B) — saved for backward
    int B, int F)
{
    const int f = blockIdx.x;
    if (f >= F) return;

    extern __shared__ float sx[];  // B floats

    // --- Load input column x[:, f] into shared memory ---
    for (int i = threadIdx.x; i < B; i += blockDim.x)
        sx[i] = __ldg(&x[i * F + f]);
    __syncthreads();

    const float alpha = __ldg(&prelu_w[f]);
    const size_t wb = (size_t)f * B * B;   // weight base offset
    const size_t bb = (size_t)f * B;       // bias / pre_act base

    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        const float* wr = weight + wb + (size_t)b * B;

        // Dot product with 4-wide manual unroll
        float acc = 0.0f;
        int k = 0;
        for (; k + 3 < B; k += 4) {
            acc += __ldg(&wr[k    ]) * sx[k    ];
            acc += __ldg(&wr[k + 1]) * sx[k + 1];
            acc += __ldg(&wr[k + 2]) * sx[k + 2];
            acc += __ldg(&wr[k + 3]) * sx[k + 3];
        }
        for (; k < B; k++)
            acc += __ldg(&wr[k]) * sx[k];

        acc += __ldg(&bias[bb + b]);

        // Save pre-activation for backward pass
        pre_act[bb + b] = acc;

        // PReLU: max(0,z) + alpha * min(0,z)
        out[b * F + f] = (acc > 0.0f) ? acc : alpha * acc;
    }
}


// ============================================================
//  Backward kernel
// ============================================================
//  Fuses: PReLU backward → d_bias, d_prelu (warp-shuffle reduction),
//         d_weight (outer product per feature)
//
//  Grid:  (feature_size,)
//  Block: (min(batch_size, 512),)
//  Smem:  2*B + num_warps floats  (x_col, dz, warp partial sums)

__global__ void __launch_bounds__(512)
fused_pseudo_bwd(
    const float* __restrict__ grad_out,  // (B, F)
    const float* __restrict__ x_clean,   // (B, F)
    const float* __restrict__ prelu_w,   // (F,)
    const float* __restrict__ pre_act,   // (F, B)
    float* __restrict__ d_weight,        // (F, B, B)
    float* __restrict__ d_bias,          // (F, B)
    float* __restrict__ d_prelu,         // (F,)
    int B, int F)
{
    const int f = blockIdx.x;
    if (f >= F) return;

    extern __shared__ float smem[];
    float* s_x  = smem;              // [0   .. B)
    float* s_dz = smem + B;          // [B   .. 2B)
    float* s_wp = smem + 2 * B;      // [2B  .. 2B + num_warps)  warp partial sums

    const float alpha  = __ldg(&prelu_w[f]);
    const size_t bb    = (size_t)f * B;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;
    const int n_warps  = (blockDim.x + 31) >> 5;

    // ---- Pass 1: compute dz, write d_bias, accumulate d_prelu ----
    float local_dp = 0.0f;
    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        float xv = __ldg(&x_clean[b * F + f]);
        s_x[b] = xv;

        float go = __ldg(&grad_out[b * F + f]);
        float z  = pre_act[bb + b];
        float m  = (z > 0.0f) ? 1.0f : 0.0f;
        float dz = go * (m + alpha * (1.0f - m));

        s_dz[b] = dz;
        d_bias[bb + b] = dz;

        // d_prelu contribution: grad * z  where z <= 0
        if (z <= 0.0f)
            local_dp += go * z;
    }

    // ---- Warp-shuffle reduction for d_prelu ----
    for (int off = 16; off > 0; off >>= 1)
        local_dp += __shfl_down_sync(0xFFFFFFFF, local_dp, off);

    if (lane == 0)
        s_wp[warp_id] = local_dp;
    __syncthreads();

    // First warp finalises cross-warp reduction
    if (warp_id == 0) {
        float val = (lane < n_warps) ? s_wp[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, off);
        if (lane == 0)
            d_prelu[f] = val;
    }
    __syncthreads();          // s_x, s_dz still needed below

    // ---- Pass 2: d_weight[f][b][k] = dz[f][b] * x[k][f] ----
    const size_t wb = (size_t)f * B * B;
    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        float dz_b = s_dz[b];
        float* dw_row = d_weight + wb + (size_t)b * B;
        int k = 0;
        for (; k + 3 < B; k += 4) {
            dw_row[k    ] = dz_b * s_x[k    ];
            dw_row[k + 1] = dz_b * s_x[k + 1];
            dw_row[k + 2] = dz_b * s_x[k + 2];
            dw_row[k + 3] = dz_b * s_x[k + 3];
        }
        for (; k < B; k++)
            dw_row[k] = dz_b * s_x[k];
    }
}


// ============================================================
//  Hybrid forward kernel: fused bias + PReLU only
// ============================================================
//  Used AFTER torch.bmm (cuBLAS) computes the matmul.
//  Input: bmm_out (F, B) = W @ x  (already computed by cuBLAS)
//  Fuses: + bias → PReLU → transpose to (B, F) output
//  Also saves pre_act (F, B) for backward.
//
//  Grid:  (F,)   Block: (min(B, 512),)   Smem: 0

__global__ void __launch_bounds__(512)
fused_bias_prelu_fwd(
    const float* __restrict__ bmm_out,   // (F, B) — output of batched matmul
    const float* __restrict__ bias,      // (F, B)
    const float* __restrict__ prelu_w,   // (F,)
    float* __restrict__ out,             // (B, F)
    float* __restrict__ pre_act,         // (F, B)
    int B, int F)
{
    const int f = blockIdx.x;
    if (f >= F) return;

    const float alpha = __ldg(&prelu_w[f]);
    const size_t bb = (size_t)f * B;

    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        float z = __ldg(&bmm_out[bb + b]) + __ldg(&bias[bb + b]);
        pre_act[bb + b] = z;
        out[b * F + f] = (z > 0.0f) ? z : alpha * z;
    }
}


// ============================================================
//  Hybrid backward kernel: PReLU backward + d_prelu reduction
// ============================================================
//  Computes dz (PReLU backward), d_bias, d_prelu.
//  d_weight is handled by cuBLAS (outer product via bmm) in Python.
//
//  Grid:  (F,)   Block: (min(B, 512),)
//  Smem:  num_warps floats (for d_prelu reduction)

__global__ void __launch_bounds__(512)
fused_bias_prelu_bwd(
    const float* __restrict__ grad_out,  // (B, F)
    const float* __restrict__ prelu_w,   // (F,)
    const float* __restrict__ pre_act,   // (F, B)
    float* __restrict__ dz_out,          // (F, B) — PReLU backward result
    float* __restrict__ d_prelu,         // (F,)
    int B, int F)
{
    const int f = blockIdx.x;
    if (f >= F) return;

    extern __shared__ float s_wp[];      // num_warps floats

    const float alpha = __ldg(&prelu_w[f]);
    const size_t bb = (size_t)f * B;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int n_warps = (blockDim.x + 31) >> 5;

    float local_dp = 0.0f;
    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        float go = __ldg(&grad_out[b * F + f]);
        float z  = pre_act[bb + b];
        float m  = (z > 0.0f) ? 1.0f : 0.0f;
        float dz = go * (m + alpha * (1.0f - m));

        dz_out[bb + b] = dz;    // d_bias = dz, also used for d_weight

        if (z <= 0.0f)
            local_dp += go * z;
    }

    // Warp-shuffle + cross-warp reduction for d_prelu
    for (int off = 16; off > 0; off >>= 1)
        local_dp += __shfl_down_sync(0xFFFFFFFF, local_dp, off);

    if (lane == 0)
        s_wp[warp_id] = local_dp;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < n_warps) ? s_wp[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, off);
        if (lane == 0)
            d_prelu[f] = val;
    }
}


// ============================================================
//  C++ wrapper — hybrid forward (cuBLAS bmm done in Python,
//  this fuses bias + PReLU + transpose)
// ============================================================

std::vector<torch::Tensor> hybrid_bias_prelu_forward(
    torch::Tensor bmm_out,    // (F, B) — result of torch.bmm
    torch::Tensor bias,       // (F, B)
    torch::Tensor prelu_w)    // (F, 1) or (F,)
{
    TORCH_CHECK(bmm_out.is_cuda(), "bmm_out must be CUDA");
    TORCH_CHECK(bmm_out.dtype() == torch::kFloat32, "must be float32");

    const int F = bmm_out.size(0);
    const int B = bmm_out.size(1);

    auto bm_c = bmm_out.contiguous();
    auto b_c  = bias.contiguous();
    auto p_c  = prelu_w.contiguous().view({F});

    auto out     = torch::empty({B, F}, bmm_out.options());
    auto pre_act = torch::empty({F, B}, bmm_out.options());

    const int threads = std::min(B, 512);

    fused_bias_prelu_fwd<<<F, threads>>>(
        bm_c.data_ptr<float>(), b_c.data_ptr<float>(),
        p_c.data_ptr<float>(),
        out.data_ptr<float>(), pre_act.data_ptr<float>(),
        B, F);

    return {out, pre_act};
}


// ============================================================
//  C++ wrapper — hybrid backward (PReLU bwd + d_prelu reduction)
// ============================================================

std::vector<torch::Tensor> hybrid_bias_prelu_backward(
    torch::Tensor grad_out,   // (B, F)
    torch::Tensor prelu_w,    // (F, 1) or (F,)
    torch::Tensor pre_act)    // (F, B)
{
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");

    const int B = grad_out.size(0);
    const int F = grad_out.size(1);

    auto go_c = grad_out.contiguous();
    auto p_c  = prelu_w.contiguous().view({F});
    auto pa_c = pre_act.contiguous();

    auto dz_out = torch::empty({F, B}, grad_out.options());
    auto d_prelu = torch::empty({F}, grad_out.options());

    const int threads = std::min(B, 512);
    const int n_warps = (threads + 31) / 32;
    const size_t smem = n_warps * sizeof(float);

    fused_bias_prelu_bwd<<<F, threads, smem>>>(
        go_c.data_ptr<float>(), p_c.data_ptr<float>(),
        pa_c.data_ptr<float>(),
        dz_out.data_ptr<float>(), d_prelu.data_ptr<float>(),
        B, F);

    // dz_out serves as both d_bias and input for d_weight computation
    return {dz_out, d_prelu.unsqueeze(1)};
}


// ============================================================
//  C++ wrapper — fully-fused forward (naive, for reference)
// ============================================================

std::vector<torch::Tensor> fused_pseudo_completion_forward(
    torch::Tensor x,          // (B, F), NaN already replaced
    torch::Tensor weight,     // (F, B, B)
    torch::Tensor bias,       // (F, B)
    torch::Tensor prelu_w)    // (F, 1) or (F,)
{
    TORCH_CHECK(x.is_cuda(),                  "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(),             "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(),               "bias must be a CUDA tensor");
    TORCH_CHECK(prelu_w.is_cuda(),            "prelu_w must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    const int B = x.size(0);
    const int F = x.size(1);

    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();
    auto b_c = bias.contiguous();
    auto p_c = prelu_w.contiguous().view({F});

    auto out     = torch::empty({B, F}, x.options());
    auto pre_act = torch::empty({F, B}, x.options());

    const int threads = std::min(B, 512);
    const size_t smem = B * sizeof(float);

    fused_pseudo_fwd<<<F, threads, smem>>>(
        x_c.data_ptr<float>(), w_c.data_ptr<float>(),
        b_c.data_ptr<float>(), p_c.data_ptr<float>(),
        out.data_ptr<float>(), pre_act.data_ptr<float>(),
        B, F);

    return {out, pre_act};
}


// ============================================================
//  C++ wrapper — backward
// ============================================================

std::vector<torch::Tensor> fused_pseudo_completion_backward(
    torch::Tensor grad_out,   // (B, F)
    torch::Tensor x_clean,    // (B, F)
    torch::Tensor prelu_w,    // (F, 1) or (F,)
    torch::Tensor pre_act)    // (F, B)
{
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(grad_out.dtype() == torch::kFloat32, "grad_out must be float32");

    const int B = grad_out.size(0);
    const int F = grad_out.size(1);

    auto go_c = grad_out.contiguous();
    auto x_c  = x_clean.contiguous();
    auto p_c  = prelu_w.contiguous().view({F});
    auto pa_c = pre_act.contiguous();

    auto d_weight = torch::empty({F, B, B}, grad_out.options());
    auto d_bias   = torch::empty({F, B},    grad_out.options());
    auto d_prelu  = torch::empty({F},       grad_out.options());   // kernel writes final values

    const int threads  = std::min(B, 512);
    const int n_warps  = (threads + 31) / 32;
    const size_t smem  = (2 * B + n_warps) * sizeof(float);

    fused_pseudo_bwd<<<F, threads, smem>>>(
        go_c.data_ptr<float>(), x_c.data_ptr<float>(),
        p_c.data_ptr<float>(),  pa_c.data_ptr<float>(),
        d_weight.data_ptr<float>(), d_bias.data_ptr<float>(),
        d_prelu.data_ptr<float>(), B, F);

    return {d_weight, d_bias, d_prelu.unsqueeze(1)};
}


// ============================================================
//  Python bindings
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &fused_pseudo_completion_forward,
          "Fused pseudo-completion forward  (CUDA)");
    m.def("backward", &fused_pseudo_completion_backward,
          "Fused pseudo-completion backward (CUDA)");
    m.def("hybrid_forward",  &hybrid_bias_prelu_forward,
          "Hybrid bias+PReLU forward (after cuBLAS bmm)");
    m.def("hybrid_backward", &hybrid_bias_prelu_backward,
          "Hybrid PReLU backward + d_prelu reduction");
}

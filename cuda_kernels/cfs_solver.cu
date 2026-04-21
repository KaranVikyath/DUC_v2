/*
 * CFS Module — cuSOLVER SVD + Fused Projection
 * ==============================================
 * Replaces torch.svd_lowrank with direct cuSOLVER Jacobi SVD
 * (cusolverDnSgesvdj), then computes PZ = V_rank @ (V_rank^T @ Z)
 * via torch::mm (cuBLAS) avoiding the full (B,B) projection matrix.
 *
 * ME759 concepts: cuSOLVER library usage, cuBLAS GEMM,
 *                 column-major/row-major layout, CUDA streams
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <stdexcept>

// ============================================================
//  cuSOLVER handle — lazily created, reused across calls
// ============================================================

static cusolverDnHandle_t g_cusolver_handle = nullptr;

static cusolverDnHandle_t get_cusolver_handle() {
    if (!g_cusolver_handle) {
        cusolverStatus_t status = cusolverDnCreate(&g_cusolver_handle);
        TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS,
                    "cusolverDnCreate failed: ", status);
    }
    return g_cusolver_handle;
}


// ============================================================
//  Forward: cuSOLVER SVD + fused projection
// ============================================================
//
//  Input:  Z (B, F_enc) — latent embeddings, row-major
//  Output: PZ (B, F_enc), V_rank (B, rank), Coef (B, B)
//
//  Algorithm:
//    1. SVD(Zt) via cusolverDnSgesvdj where Zt = Z^T (F_enc, B)
//       - Row-major Z (B, F_enc) data IS column-major (F_enc, B)
//       - So pass Z.data with M=F_enc, N=B, lda=F_enc
//       - Outputs: U (F_enc, F_enc), S (F_enc,), V (B, F_enc)
//       - V[:, :rank] = right singular vectors of Zt = V_rank (B, rank)
//    2. PZ = V_rank @ (V_rank^T @ Z) — fused, avoids (B,B) matrix
//    3. Coef = V_rank @ V_rank^T — needed for clustering

std::vector<torch::Tensor> cfs_svd_project_forward(
    torch::Tensor Z,      // (B, F_enc), contiguous, CUDA, float32
    int rank)
{
    TORCH_CHECK(Z.is_cuda(), "Z must be a CUDA tensor");
    TORCH_CHECK(Z.dtype() == torch::kFloat32, "Z must be float32");
    TORCH_CHECK(Z.is_contiguous(), "Z must be contiguous");

    const int B = Z.size(0);      // batch size (e.g. 500)
    const int F = Z.size(1);      // encoded features (e.g. 40)
    const int minMN = std::min(F, B);

    TORCH_CHECK(rank <= minMN, "rank must be <= min(B, F_enc)");

    // --- Set cuSOLVER to use PyTorch's current CUDA stream ---
    auto handle = get_cusolver_handle();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cusolverDnSetStream(handle, stream);

    // --- Clone Z because gesvdj overwrites input ---
    // Row-major Z (B, F_enc) = column-major Zt (F_enc, B)
    // cuSOLVER sees: M=F_enc, N=B matrix in column-major
    auto A = Z.clone();   // (B, F_enc) contiguous — data = Zt col-major

    // --- Allocate SVD outputs ---
    // S: singular values (minMN,)
    auto S = torch::empty({minMN}, Z.options());

    // U: left singular vectors of Zt, (M, M) col-major = (F_enc, F_enc)
    // We don't need U, but gesvdj requires the buffer
    auto U = torch::empty({F, F}, Z.options());

    // V: right singular vectors of Zt, (N, minMN) col-major = (B, minMN)
    // This is what we want! V[:, :rank] are our projection vectors
    auto V = torch::empty({B * minMN}, Z.options());  // flat buffer

    // --- Configure Jacobi SVD ---
    gesvdjInfo_t gesvdj_params = nullptr;
    cusolverDnCreateGesvdjInfo(&gesvdj_params);
    cusolverDnXgesvdjSetTolerance(gesvdj_params, 1e-7);
    cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, 100);

    // --- Query workspace size ---
    int lwork = 0;
    cusolverStatus_t status = cusolverDnSgesvdj_bufferSize(
        handle,
        CUSOLVER_EIG_MODE_VECTOR,   // compute singular vectors
        1,                          // econ=1: economy SVD
        F,                          // M = rows of Zt = F_enc
        B,                          // N = cols of Zt = B
        A.data_ptr<float>(),        // device pointer to matrix
        F,                          // lda = M (column-major leading dim)
        S.data_ptr<float>(),        // singular values
        U.data_ptr<float>(),        // left singular vectors
        F,                          // ldu = M
        V.data_ptr<float>(),        // right singular vectors
        B,                          // ldv = N
        &lwork,
        gesvdj_params);
    TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS,
                "cusolverDnSgesvdj_bufferSize failed: ", status);

    // --- Allocate workspace + info ---
    auto work = torch::empty({lwork}, Z.options());
    auto devinfo = torch::zeros({1}, Z.options().dtype(torch::kInt32));

    // --- Compute SVD ---
    status = cusolverDnSgesvdj(
        handle,
        CUSOLVER_EIG_MODE_VECTOR,
        1,                          // econ
        F,                          // M
        B,                          // N
        A.data_ptr<float>(),        // input (overwritten)
        F,                          // lda
        S.data_ptr<float>(),        // singular values out
        U.data_ptr<float>(),        // U out (F_enc x F_enc col-major)
        F,                          // ldu
        V.data_ptr<float>(),        // V out (B x minMN col-major)
        B,                          // ldv
        work.data_ptr<float>(),
        lwork,
        devinfo.data_ptr<int>(),
        gesvdj_params);
    TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS,
                "cusolverDnSgesvdj failed: ", status);

    // Clean up gesvdj params
    cusolverDnDestroyGesvdjInfo(gesvdj_params);

    // --- Extract V_rank (B, rank) from V buffer ---
    // V buffer is column-major (B, minMN) with ldv=B
    // Column j occupies V[j*B .. (j+1)*B-1]
    // We want first `rank` columns → V_rank (B, rank)
    //
    // Wrap as torch tensor with column-major strides:
    //   element (i, j) at offset j*B + i → strides = {1, B}
    auto V_full = torch::from_blob(
        V.data_ptr<float>(),
        {B, minMN},             // shape
        {1, B},                 // strides: column-major
        Z.options()
    );
    // Take first `rank` columns and make contiguous (row-major)
    auto V_rank = V_full.narrow(1, 0, rank).contiguous();  // (B, rank)

    // --- Fused projection: PZ = V_rank @ (V_rank^T @ Z) ---
    // Step 1: VtZ = V_rank^T @ Z → (rank, F_enc)
    auto VtZ = torch::mm(V_rank.t(), Z);     // (rank, F_enc)
    // Step 2: PZ = V_rank @ VtZ → (B, F_enc)
    auto PZ = torch::mm(V_rank, VtZ);        // (B, F_enc)

    // --- Coef for clustering: V_rank @ V_rank^T → (B, B) ---
    auto Coef = torch::mm(V_rank, V_rank.t());  // (B, B) symmetric

    return {PZ, V_rank, Coef};
}


// ============================================================
//  Backward: approximate gradient (stop grad through SVD)
// ============================================================
//
//  d_Z = V_rank @ (V_rank^T @ grad_PZ)
//
//  This treats V as constant (detached). Justified because:
//  - CFS mode: total_loss = reconstruction_loss only
//  - SVD has no learnable parameters
//  - The primary gradient path (PZ = Coef @ Z) is preserved

std::vector<torch::Tensor> cfs_svd_project_backward(
    torch::Tensor grad_PZ,    // (B, F_enc)
    torch::Tensor V_rank)     // (B, rank) — saved from forward
{
    TORCH_CHECK(grad_PZ.is_cuda(), "grad_PZ must be CUDA");

    // d_Z = V_rank @ V_rank^T @ grad_PZ (fused, no B×B matrix)
    auto Vt_grad = torch::mm(V_rank.t(), grad_PZ);   // (rank, F_enc)
    auto d_Z = torch::mm(V_rank, Vt_grad);            // (B, F_enc)

    return {d_Z};
}


// ============================================================
//  Python bindings
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &cfs_svd_project_forward,
          "CFS SVD projection forward (cuSOLVER gesvdj + cuBLAS)");
    m.def("backward", &cfs_svd_project_backward,
          "CFS SVD projection backward (approximate, V detached)");
}

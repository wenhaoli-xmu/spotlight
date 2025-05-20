#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int BLOCK_N = 64;
constexpr int KH = 4;
constexpr int D = 4;
constexpr int QH = 28;
constexpr int G = QH / KH;

__global__ void attn_k4_q28_kernel(
    const uint32_t* __restrict__ q_ptr,
    const uint32_t* __restrict__ k_ptr,
    int16_t* __restrict__ o_ptr,
    int B, int KN
) {
    int b_idx = blockIdx.x;
    int n_block_idx = blockIdx.y;
    int n_start = n_block_idx * BLOCK_N;
    int n_local = threadIdx.x;
    int h = threadIdx.y;
    int n = n_start + n_local;

    extern __shared__ uint32_t shared_mem[];
    uint32_t* q_shared = shared_mem;
    uint32_t* k_shared = q_shared + G * KH * D;

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int num_threads = BLOCK_N * KH;

    for (int gh_idx = tid; gh_idx < G * KH; gh_idx += num_threads) {
        int g = gh_idx / KH;
        int h_load = gh_idx % KH;
        uint4 q_val = *reinterpret_cast<const uint4*>(
            q_ptr + b_idx * QH * D + (g * KH + h_load) * D
        );
        int shared_offset = (g * KH + h_load) * D;
        q_shared[shared_offset]     = q_val.x;
        q_shared[shared_offset + 1] = q_val.y;
        q_shared[shared_offset + 2] = q_val.z;
        q_shared[shared_offset + 3] = q_val.w;
    }

    for (int nh_idx = tid; nh_idx < BLOCK_N * KH; nh_idx += num_threads) {
        int n_local_load = nh_idx / KH;
        int h_load = nh_idx % KH;
        int n_global = n_start + n_local_load;
        int k_shared_offset = n_local_load * KH * D + h_load * D;

        if (n_global < KN) {
            uint4 k_val = *reinterpret_cast<const uint4*>(
                k_ptr + b_idx * KN * KH * D + n_global * KH * D + h_load * D
            );
            k_shared[k_shared_offset]     = k_val.x;
            k_shared[k_shared_offset + 1] = k_val.y;
            k_shared[k_shared_offset + 2] = k_val.z;
            k_shared[k_shared_offset + 3] = k_val.w;
        } else {
            k_shared[k_shared_offset]     = 0;
            k_shared[k_shared_offset + 1] = 0;
            k_shared[k_shared_offset + 2] = 0;
            k_shared[k_shared_offset + 3] = 0;
        }
    }

    __syncthreads();

    if (n < KN) {
        int accum = 0;
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            const int q_base = (g * KH + h) * D;
            const int k_base = (n_local * KH + h) * D;

            uint32_t q0 = q_shared[q_base];
            uint32_t q1 = q_shared[q_base + 1];
            uint32_t q2 = q_shared[q_base + 2];
            uint32_t q3 = q_shared[q_base + 3];

            uint32_t k0 = k_shared[k_base];
            uint32_t k1 = k_shared[k_base + 1];
            uint32_t k2 = k_shared[k_base + 2];
            uint32_t k3 = k_shared[k_base + 3];

            accum += __popc(~(q0 ^ k0)) 
                   + __popc(~(q1 ^ k1))
                   + __popc(~(q2 ^ k2))
                   + __popc(~(q3 ^ k3));
        }

        o_ptr[b_idx * KN * KH + n * KH + h] = static_cast<int16_t>(accum);
    }
}

}

torch::Tensor attn_k4_q28(torch::Tensor q_hash, torch::Tensor k_hash) {
    TORCH_CHECK(q_hash.is_cuda() && k_hash.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(q_hash.dtype() == torch::kUInt32 && k_hash.dtype() == torch::kUInt32, "Inputs must be uint32");
    TORCH_CHECK(q_hash.dim() == 4 && k_hash.dim() == 4, "Inputs must be 4D tensors");

    const int B = q_hash.size(0);
    const int QN = q_hash.size(1);
    const int KN = k_hash.size(1);

    TORCH_CHECK(QN == 1, "Only support single query");
    TORCH_CHECK(q_hash.size(2) == QH && k_hash.size(2) == KH, "Head dimension mismatch");
    TORCH_CHECK(q_hash.size(3) == D && k_hash.size(3) == D, "Feature dimension mismatch");

    torch::Device device = q_hash.device();
    TORCH_CHECK(k_hash.device() == device, "k_hash must be on the same device as q_hash");

    torch::DeviceGuard device_guard(device);

    torch::Tensor output = torch::zeros({B, KN, KH}, q_hash.options().dtype(torch::kInt16));

    dim3 blockDim(BLOCK_N, KH);
    dim3 gridDim(B, (KN + BLOCK_N - 1) / BLOCK_N);
    
    size_t shared_mem_size = (G * KH * D + BLOCK_N * KH * D) * sizeof(uint32_t);
    
    attn_k4_q28_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        q_hash.data_ptr<uint32_t>(),
        k_hash.data_ptr<uint32_t>(),
        output.data_ptr<int16_t>(),
        B, KN
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_k4_q28", &attn_k4_q28);
}
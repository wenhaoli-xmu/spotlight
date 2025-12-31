#include <cuda.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_UINT32(x) TORCH_CHECK(x.dtype() == torch::kInt32, #x " must be uint32 (int32 in torch)")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


__global__ void hamming_mask_kernel(
    const uint4* __restrict__ q_ptr, // [B, HQ, D/4] (Actually usually [B, 1, HQ, D])
    const uint4* __restrict__ k_ptr, // [B, N, HK, D/4]
    bool* __restrict__ output_mask,  // [B, HK, N]
    int B, int N, int HQ, int HK, int G_size, int threshold
) {    
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int hk = blockIdx.y;
    int b = blockIdx.z;

    if (n >= N) return;

    size_t k_idx = (size_t)b * N * HK + (size_t)n * HK + hk;
    uint4 k_vec = k_ptr[k_idx]; // 128-bit load

    bool is_selected = false;

    for (int g = 0; g < G_size; ++g) {
        int hq = hk * G_size + g;
        
        size_t q_idx = (size_t)b * HQ + hq;
        uint4 q_vec = q_ptr[q_idx];

        unsigned int dist = 0;
        dist += __popc(q_vec.x ^ k_vec.x);
        dist += __popc(q_vec.y ^ k_vec.y);
        dist += __popc(q_vec.z ^ k_vec.z);
        dist += __popc(q_vec.w ^ k_vec.w);

        if (dist > threshold) {
            is_selected = true;
            break;
        }
    }

    // Output shape: [B, HK, N]
    size_t out_idx = (size_t)b * HK * N + (size_t)hk * N + n;
    output_mask[out_idx] = is_selected;
}

// C++ Wrapper
torch::Tensor calc_hamming_mask_cuda(
    torch::Tensor query_bin, 
    torch::Tensor key_bins,
    int threshold
) {
    CHECK_CUDA(query_bin);
    CHECK_CUDA(key_bins);
    CHECK_CONTIGUOUS(query_bin);
    CHECK_CONTIGUOUS(key_bins);
    CHECK_UINT32(query_bin);
    CHECK_UINT32(key_bins);
    
    int B = key_bins.size(0);
    int N = key_bins.size(1);
    int HK = key_bins.size(2);
    int D = key_bins.size(3);

    int HQ = query_bin.numel() / (B * D); 

    TORCH_CHECK(D == 4, "Only D=4 (128-bit) is supported for this optimized kernel.");
    TORCH_CHECK(HQ % HK == 0, "HQ must be divisible by HK (GQA requirement).");
    
    int G_size = HQ / HK;

    auto options = torch::TensorOptions().dtype(torch::kBool).device(query_bin.device());
    torch::Tensor output_mask = torch::empty({B, HK, N}, options);

    const int threads = 256;
    const int blocks_n = (N + threads - 1) / threads;
    
    dim3 grid(blocks_n, HK, B);
    dim3 block(threads);

    hamming_mask_kernel<<<grid, block>>>(
        reinterpret_cast<const uint4*>(query_bin.data_ptr<int32_t>()),
        reinterpret_cast<const uint4*>(key_bins.data_ptr<int32_t>()),
        output_mask.data_ptr<bool>(),
        B, N, HQ, HK, G_size, threshold
    );

    return output_mask;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calc_hamming_mask", &calc_hamming_mask_cuda, "Fast Hamming mask (CUDA)", 
          py::arg("query_bin"), py::arg("key_bins"), py::arg("threshold") = 64);
}
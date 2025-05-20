#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void packbits_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint32_t* __restrict__ output,
    int total_packs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_packs) return;

    const __nv_bfloat16* in_ptr = input + idx * 32;
    uint32_t packed = 0;
    uint32_t sign;

    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        sign = (__bfloat162float(in_ptr[j]) > 0.0f);
        packed |= (sign << j);
    }

    output[idx] = packed;
}

torch::Tensor packbits(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "Input must be bfloat16");
    TORCH_CHECK(input.size(-1) % 32 == 0, "Last dimension must be divisible by 32");

    torch::Device device = input.device();

    torch::DeviceGuard device_guard(device);

    auto input_contig = input.contiguous();
    auto in_sizes = input.sizes();
    int last_dim = in_sizes.back();
    int num_packs = last_dim / 32;

    std::vector<int64_t> out_shape(in_sizes.begin(), in_sizes.end());
    out_shape.back() = num_packs;

    int total_packs = input.numel() / 32;
    torch::Tensor output = torch::empty({total_packs}, torch::dtype(torch::kUInt32).device(device));

    int threads = 512;
    int blocks = (total_packs + threads - 1) / threads;

    packbits_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(input_contig.data_ptr<at::BFloat16>()),
        output.data_ptr<uint32_t>(),
        total_packs
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA launch failed: ") + cudaGetErrorString(err));
    }

    return output.view(out_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("packbits", &packbits);
}
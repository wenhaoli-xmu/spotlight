#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define FLOAT4_WRITE(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT4_READ(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

__global__ void lru_update_kernel_optimized(
    const int* __restrict__ topk_indices,   
    int* __restrict__ lru_indices,          // [B, KH, allocated_size]
    int* __restrict__ lru_ptr,              
    const float* __restrict__ global_k,     
    const float* __restrict__ global_v,     
    float* __restrict__ cache_k,            
    float* __restrict__ cache_v,            
    const int B, 
    const int KH, 
    const int lru_logical_size,
    const int allocated_size,
    const int top_budget, 
    const int D, 
    const int MaxT
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    int batch_head_idx = b * KH + h;

    extern __shared__ int smem[];
    int* s_lru_indices = smem;                 
    int* s_diff_indices = &s_lru_indices[lru_logical_size]; 
    __shared__ int s_count;

    if (tid == 0) {
        s_count = 0;
    }

    int* global_lru_indices_ptr = lru_indices + (batch_head_idx * allocated_size);
    
    for (int i = tid; i < lru_logical_size; i += blockDim.x) {
        s_lru_indices[i] = global_lru_indices_ptr[i];
    }
    
    __syncthreads();


    const int* my_topk_ptr = topk_indices + (batch_head_idx * top_budget);
    for (int i = tid; i < top_budget; i += blockDim.x) {
        int candidate_idx = my_topk_ptr[i];
        if (candidate_idx >= 0) {
            bool exists = false;
            #pragma unroll
            for (int j = 0; j < lru_logical_size; j++) { // 遍历 1023 个
                if (s_lru_indices[j] == candidate_idx) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                int pos = atomicAdd(&s_count, 1);
                s_diff_indices[pos] = candidate_idx;
            }
        }
    }
    __syncthreads();

    int num_updates = s_count;
    if (num_updates == 0) return;

    int current_ptr = lru_ptr[batch_head_idx];

    if (tid == 0) {
        for (int i = 0; i < num_updates; i++) {
            int write_pos = (current_ptr + i) % lru_logical_size;
            global_lru_indices_ptr[write_pos] = s_diff_indices[i]; 
        }
        lru_ptr[batch_head_idx] = (current_ptr + num_updates) % lru_logical_size;
    }
    
    __syncthreads(); 

    int D_vec = D / 4; 
    int total_tasks = num_updates * D_vec;

    for (int task_id = tid; task_id < total_tasks; task_id += blockDim.x) {
        int update_idx = task_id / D_vec;
        int d_vec_idx = task_id % D_vec;
        int d_offset = d_vec_idx * 4;

        int token_global_idx = s_diff_indices[update_idx];
        
        int write_pos_logical = (current_ptr + update_idx) % lru_logical_size;

        long src_offset = ((long)b * MaxT * KH * D) + 
                          ((long)token_global_idx * KH * D) + 
                          ((long)h * D) + 
                          d_offset;

        long dst_offset = ((long)b * allocated_size * KH * D) + 
                          ((long)write_pos_logical * KH * D) + 
                          ((long)h * D) + 
                          d_offset;

        float4 k_val = FLOAT4_READ(global_k[src_offset]);
        FLOAT4_WRITE(cache_k[dst_offset]) = k_val;

        float4 v_val = FLOAT4_READ(global_v[src_offset]);
        FLOAT4_WRITE(cache_v[dst_offset]) = v_val;
    }
}

void lru_update_cuda(
    torch::Tensor topk_indices,
    torch::Tensor lru_indices,
    torch::Tensor lru_ptr,
    torch::Tensor global_k,
    torch::Tensor global_v,
    torch::Tensor cache_k,
    torch::Tensor cache_v,
    int lru_logical_size, 
    int top_budget
) {
    const int B = topk_indices.size(0);
    const int KH = topk_indices.size(1);
    const int MaxT = global_k.size(1);
    const int D = global_k.size(3);
    
    const int allocated_size = cache_k.size(1);

    TORCH_CHECK(lru_indices.size(2) == allocated_size, 
        "lru_indices last dim must match allocated_size (physical size) for memory alignment.");

    TORCH_CHECK(D % 4 == 0, "Hidden dimension D must be divisible by 4.");
    TORCH_CHECK(lru_logical_size <= allocated_size, "Logical size cannot exceed allocated size.");

    dim3 grid(B, KH);
    int threads = 128; 
    if (lru_logical_size > 256 || top_budget > 64) {
        threads = 256;
    }

    size_t smem_size = (lru_logical_size * sizeof(int)) + (top_budget * sizeof(int)) + sizeof(int);

    lru_update_kernel_optimized<<<grid, threads, smem_size>>>(
        topk_indices.data_ptr<int>(),
        lru_indices.data_ptr<int>(),
        lru_ptr.data_ptr<int>(),
        global_k.data_ptr<float>(),
        global_v.data_ptr<float>(),
        cache_k.data_ptr<float>(),
        cache_v.data_ptr<float>(),
        B, KH, 
        lru_logical_size,
        allocated_size,
        top_budget, D, MaxT
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("update", &lru_update_cuda, "LRU Cache Update Optimized (CUDA)");
}
#include <cuda.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// 假设 Head Dim 是 128 (float4 * 32) 或 64 等。
// 为了通用性，数据搬运部分我们用 float4 循环拷贝
#define MAX_SHARED_CANDIDATES 1024 // 每个 Head 每次最多更新多少个(受SharedMem限制)

__device__ __forceinline__ int popc_dist(uint4 a, uint4 b) {
    return __popc(a.x ^ b.x) + __popc(a.y ^ b.y) + __popc(a.z ^ b.z) + __popc(a.w ^ b.w);
}

// Fused Kernel
// Grid: [B, KH]
// Block: [256] or [128]
__global__ void fused_lru_update_kernel(
    const uint4* __restrict__ q_bins,       // [B, HQ, D/4] (Current Query)
    const uint4* __restrict__ k_bins,       // [B, MaxT, HK, D/4] (Full History)
    
    // LRU State Tensors (In-Place Modified)
    int* __restrict__ lru_ptr,              // [B, HK]  Current Ring Buffer Pointer
    bool* __restrict__ in_lru_mask,         // [B, HK, MaxT] Global Presence Mask
    int32_t* __restrict__ lru_indices,      // [B, HK, LRU_Size] Maps slot -> global_idx
    
    // Data Tensors (Source & Dest)
    const float* __restrict__ k_full,       // [B, MaxT, HK, HeadDim]
    const float* __restrict__ v_full,       // [B, MaxT, HK, HeadDim]
    float* __restrict__ k_lru,              // [B, LRU_Size, HK, HeadDim]
    float* __restrict__ v_lru,              // [B, LRU_Size, HK, HeadDim]

    // Constants
    int MaxT, int LRU_Size, int HeadDim, int G_size, int Threshold, int N_Current
) {
    // 1. Setup Indices
    int tid = threadIdx.x;
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y; // HK index

    // Shared Memory for Candidate Queue
    // 存储找到的需要更新的 global_index
    __shared__ int candidates[MAX_SHARED_CANDIDATES];
    __shared__ int candidate_count;
    
    // Shared Memory for Copy Tasks
    // task_src[i] -> task_dst[i]
    __shared__ int task_src[MAX_SHARED_CANDIDATES]; 
    __shared__ int task_dst[MAX_SHARED_CANDIDATES];

    if (tid == 0) {
        candidate_count = 0;
    }
    __syncthreads();

    // -----------------------------------------------------------
    // Phase 1: Search & Filter (Parallel)
    // -----------------------------------------------------------
    
    // Q vector for this head (Broadcast GQA: map HK -> HQ)
    // 假设 HQ = HK * G_size. 取当前 Head 对应的第一个 Query Head 或者做平均
    // 这里简单起见，取 group 里的第一个 query head
    int hq_idx = h_idx * G_size; 
    size_t q_offset = (size_t)b_idx * (gridDim.y * G_size) + hq_idx; 
    uint4 q_vec = q_bins[q_offset]; 

    // Iterate over all valid tokens in history
    for (int n = tid; n < N_Current; n += blockDim.x) {
        // 1. Check Mask first (Fast reject)
        size_t mask_idx = (size_t)b_idx * gridDim.y * MaxT + (size_t)h_idx * MaxT + n;
        
        if (in_lru_mask[mask_idx]) continue; // Already in cache, skip

        // 2. Compute Hamming
        size_t k_idx = (size_t)b_idx * MaxT * gridDim.y + (size_t)n * gridDim.y + h_idx;
        uint4 k_vec = k_bins[k_idx];
        
        int dist = popc_dist(q_vec, k_vec);

        // 3. Add to Queue if match
        if (dist <= Threshold) {
            int old = atomicAdd(&candidate_count, 1);
            if (old < MAX_SHARED_CANDIDATES) {
                candidates[old] = n;
            }
        }
    }
    __syncthreads();

    // -----------------------------------------------------------
    // Phase 2: Allocate & State Update (Serialized by Leader)
    // -----------------------------------------------------------
    int count = min(candidate_count, MAX_SHARED_CANDIDATES);
    
    if (tid == 0 && count > 0) {
        int ptr_idx = b_idx * gridDim.y + h_idx;
        int current_ptr = lru_ptr[ptr_idx];

        for (int i = 0; i < count; ++i) {
            int new_global_idx = candidates[i];
            
            // Double check (optional but safe): mask might have changed if logic complex
            // But here we are the only block processing this head.
            
            // 1. Identify Victim
            size_t slot_idx = (size_t)b_idx * LRU_Size * gridDim.y + (size_t)h_idx * LRU_Size + current_ptr; // Careful with stride
            // Layout assumption: [B, HK, LRU_Size] for indices? 
            // User provided lru_indices shape in Python: [B, LRU, HK] usually? 
            // Let's assume passed lru_indices is [B, HK, LRU_Size] for coalescing, 
            // BUT standard KV cache is [B, LRU, HK, D]. 
            // Let's stick to ptr logic.
            
            // Let's access linear lru_indices as [B, HK, LRU_Size] for simpler indexing
            size_t lru_indices_idx = (size_t)b_idx * gridDim.y * LRU_Size + (size_t)h_idx * LRU_Size + current_ptr;
            
            int old_global_idx = lru_indices[lru_indices_idx];

            // 2. Update Global Mask
            // Remove old
            if (old_global_idx != -1) {
                 size_t old_mask_idx = (size_t)b_idx * gridDim.y * MaxT + (size_t)h_idx * MaxT + old_global_idx;
                 in_lru_mask[old_mask_idx] = false;
            }
            // Add new
            size_t new_mask_idx = (size_t)b_idx * gridDim.y * MaxT + (size_t)h_idx * MaxT + new_global_idx;
            in_lru_mask[new_mask_idx] = true;

            // 3. Update Indices Array
            lru_indices[lru_indices_idx] = new_global_idx;

            // 4. Record Task for parallel copy
            task_src[i] = new_global_idx;
            task_dst[i] = current_ptr;

            // 5. Move Pointer
            current_ptr = (current_ptr + 1) % LRU_Size;
        }
        // Write back pointer
        lru_ptr[ptr_idx] = current_ptr;
    }
    __syncthreads();

    // -----------------------------------------------------------
    // Phase 3: Data Copy (Parallel)
    // -----------------------------------------------------------
    // 每个线程负责一部分数据的搬运
    // Total float elements to copy = count * HeadDim * 2 (K + V)
    
    if (count > 0) {
        // Flatten the copy loop
        int total_elements = count * HeadDim; // For one tensor
        
        for (int i = tid; i < total_elements; i += blockDim.x) {
            int task_idx = i / HeadDim;
            int dim_idx = i % HeadDim;

            int global_t_idx = task_src[task_idx];
            int lru_slot_idx = task_dst[task_idx];

            // Address mapping
            // Source: [B, MaxT, HK, D]
            size_t src_offset = (size_t)b_idx * MaxT * gridDim.y * HeadDim 
                              + (size_t)global_t_idx * gridDim.y * HeadDim 
                              + (size_t)h_idx * HeadDim + dim_idx;
            
            // Dest: [B, LRU, HK, D]
            size_t dst_offset = (size_t)b_idx * LRU_Size * gridDim.y * HeadDim 
                              + (size_t)lru_slot_idx * gridDim.y * HeadDim 
                              + (size_t)h_idx * HeadDim + dim_idx;

            // Copy Key
            k_lru[dst_offset] = k_full[src_offset];
            // Copy Value
            v_lru[dst_offset] = v_full[src_offset];
        }
    }
}

// Wrapper
void fused_lru_update(
    torch::Tensor q_bins, torch::Tensor k_bins,
    torch::Tensor lru_ptr, torch::Tensor in_lru_mask, torch::Tensor lru_indices,
    torch::Tensor k_full, torch::Tensor v_full,
    torch::Tensor k_lru, torch::Tensor v_lru,
    int threshold, int num_tokens
) {
    int B = k_bins.size(0);
    int MaxT = k_bins.size(1); // Capacity
    int HK = k_bins.size(2);
    int HQ = q_bins.size(1);
    int G_size = HQ / HK;
    int LRU_Size = lru_indices.size(2); // Assuming [B, HK, LRU_Size]
    int HeadDim = k_full.size(3);

    dim3 grid(B, HK);
    dim3 block(256);

    fused_lru_update_kernel<<<grid, block>>>(
        reinterpret_cast<const uint4*>(q_bins.data_ptr<int32_t>()),
        reinterpret_cast<const uint4*>(k_bins.data_ptr<int32_t>()),
        lru_ptr.data_ptr<int>(),
        in_lru_mask.data_ptr<bool>(),
        lru_indices.data_ptr<int32_t>(),
        k_full.data_ptr<float>(), v_full.data_ptr<float>(),
        k_lru.data_ptr<float>(), v_lru.data_ptr<float>(),
        MaxT, LRU_Size, HeadDim, G_size, threshold, num_tokens
    );
}
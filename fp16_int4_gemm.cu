#include <cuda.h>
#include <mma.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cstdio>
#include "common.h"


#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.
#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#define CHUNK_K 8

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B matrix
// in shared memory to minimize possible bank conflicts.
// Before performing the nvcuda::wmma::mma_sync operation, the warp must load the matrix
// data using the nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
// is not specified for that function, each lane in the warp can read one or multiple matrix
// elements from different matrix rows or columns.
// For shared memory, such access can result in bank conflicts if different rows / columns
// of the matrix map to the same bank. By shifting each row and column by a few bytes, we
// make sure that they map to different banks, thus reducing the number of possible bank
// conflicts.
// The number of 16 two-byte "half" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 16

using namespace nvcuda;
struct __device_builtin__ __builtin_align__(8) half4
{
    half x[4];
};

__global__ void compute_gemm(const half* A, 
    const uint16_t* q_weight, const uint8_t* q_zeros, const half* scales, 
    half* C, bool accum,
    const int M_TILES, const int N_TILES, const int K_TILES,
    const int Mr, const int Nr, const int Kr,
    int packed_zeros_height, int scales_height, int packed_weights_height) {
    extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];
    const int GLOBAL_MEM_STRIDE = Nr;

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Offset in shared memory from which the B matrix is stored.
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    // This pointer is used to access the C and D matrix tiles this warp computes.
    half* shmem_warp_tile_ptr = (half*)&shmem[0][0] +
        (warpId / 2) * SHMEM_STRIDE * K * 2 +
        (warpId % 2) * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and
    // from shared memory.
    half* shmem_warp_stream_ptr =
        (half*)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

    // Each CTA slides along the 128 x 128 tiles from the top left corner of the
    // matrix to the right and down, and selects the next tile to compute. Once
    // there's no such tile, all warps in this CTA exit.
    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i =
            ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES) {
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from to shared
        // memory.
        const size_t m_tile_index = (block_tile_i + warpId) * M;
        const size_t n_tile_index = block_tile_j * N;
        const size_t gmem_idx =
            m_tile_index * GLOBAL_MEM_STRIDE + n_tile_index;
        const half* src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory.
#pragma unroll
        for (int i = 0; i < K; i++) {

            if (m_tile_index + i < Mr && n_tile_index + laneId * sizeof(int2) / sizeof(half) < Nr)
            {
                *((int2*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                    *((int2*)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
                        laneId);
            }
            else
            {
                *((int2*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) = make_int2(0, 0);
            }
        }

        __syncthreads();

        // These fragments will accumulate the result of A and B matrix fragment
        // multiplications along the Kr dimension.
        wmma::fragment<wmma::accumulator, M, N, K, half> c[WARP_COL_TILES]
            [WARP_ROW_TILES];

        // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                if (accum)
                {
                    const half* tile_ptr =
                        shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                    wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
                }
                else
                {
                    wmma::fill_fragment(c[i][j], (half)0);
                }
            }
        }

        __syncthreads();

        const half* warp_ptrA = (&A[block_tile_i * M * Kr] + M * Kr * warpId);

        const size_t m_tile_index_a = (block_tile_i + warpId) * M;
        const size_t n_tile_index_b = (block_tile_j + warpId) * N;


        // Go through the global K dimension by a fixed step at a time.
#pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            // Copy slices of the A and B matrices to shared memory.
            size_t shmem_idxA = M * warpId;
            size_t shmem_idxB = N * warpId + shmem_idx_b_off;

            int2* lane_ptrA = (int2*)(warp_ptrA + tile_k * K) + laneId;

#pragma unroll
            for (int i = 0; i < (WARP_SIZE / 2); i++) {
                if (m_tile_index_a + i < Mr && tile_k * K + laneId * sizeof(int2) / sizeof(half) < Kr)
                {
                    // Copy 8 bytes at once in each lane.
                    *((int2*)&shmem[shmem_idxA][0] + laneId) = *lane_ptrA;

                    // Advance the global memory pointer and the shared memory index.
                    lane_ptrA = (int2*)((half*)lane_ptrA + Kr);
                }
                else
                {
                    *((int2*)&shmem[shmem_idxA][0] + laneId) = make_int2(0, 0);
                }

                if (n_tile_index_b + i < Nr && tile_k * K + laneId * sizeof(int2) / sizeof(half) < Kr)
                {
                    uint16_t weight4 = q_weight[(n_tile_index_b + i) * packed_weights_height + (tile_k * K) / 4 + laneId];

                    // 128 group size, 1 warp is handling 1 full group
                    int scale_zeros_index = (tile_k * K) / 128;
                    float scale = (float)scales[(n_tile_index_b + i) * scales_height + scale_zeros_index];

                    // two zero values fetched for two consecutive groups
                    uint8_t q_z = q_zeros[(n_tile_index_b + i) * packed_zeros_height + scale_zeros_index / 2];
                    float q_zf = (float)((q_z >> (4 * (scale_zeros_index % 2))) & 0xF);

                    half4 bVal;
#pragma unroll
                    for(int qi = 0; qi < 4; qi++)
                    {
                        float q_wt = (float)((weight4 >> (qi * 4)) & 0xF);
                        bVal.x[qi] = (half)(scale * (q_wt - q_zf));
                    }
                    
                    // Copy 8 bytes at once in each lane.
                    *((half4*)&shmem[shmem_idxB][0] + laneId) = bVal;
                }
                else
                {
                    *((int2*)&shmem[shmem_idxB][0] + laneId) = make_int2(0, 0);
                }

                shmem_idxA += 1;
                shmem_idxB += 1;

            }

            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
#pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major>
                    a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major>
                    b[WARP_ROW_TILES];

#pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
                    const half* tile_ptr = &shmem[shmem_idx_a][k_step * K];

                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

#pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be
                            // reused against the other A matrix fragments.
                            size_t shmem_idx_b = shmem_idx_b_off +
                                (WARP_ROW_TILES * N) * (warpId % 2) +
                                (j * N);
                            const half* tile_ptr = &shmem[shmem_idx_b][k_step * K];

                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
                        }

                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }

            __syncthreads();
        }

        // Store the D fragments to shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {

                half* tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Now that shared memory contains all the D tiles, stream them to global
        // memory.
        half* dst_gmem_warp_stream_ptr = &C[gmem_idx];

#pragma unroll
        for (int i = 0; i < K; i++) {
            if (m_tile_index + i < Mr && n_tile_index + laneId * sizeof(int2) / sizeof(half) < Nr)
            {
                *((int2*)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                    *((int2*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
            }
        }

        __syncthreads();
    }
}

static bool init = false;
void fp16_int4_gemm(cudaDeviceProp& deviceProp, cudaStream_t stream, int prompt_len, half* xout, half* x, QWeight &w, int inpSize, int opSize, bool accum)
{
    // matrix dimensions
    int Mr = prompt_len;
    int Kr = inpSize;
    int Nr = opSize;

    int M_GLOBAL = divUp(Mr, 128) * 128;
    int K_GLOBAL = divUp(Kr, 128) * 128;
    int N_GLOBAL = divUp(Nr, 128) * 128;

    int M_TILES = M_GLOBAL / M;
    int N_TILES = N_GLOBAL / N;
    int K_TILES = K_GLOBAL / K;

    enum {
    // Compute the right amount of shared memory to request.
    // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
    // per-CTA chunks
    // of the A and B matrices. Therefore, the right amount to request is the
    // maximum of those
    // two numbers.
    SHMEM_SZ = std::max(
        sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
        M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
            (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(half))
    };

    if (deviceProp.sharedMemPerBlockOptin < SHMEM_SZ)
    {
        printf(
            "WARNING: Current GPU does not have enough shared memory to run gemm kernel\n");
        return;
    }

    int scales_height = divUp(inpSize, 128); // groupsize 128
    int packed_wt_height = divUp(inpSize, 4); // int4 weights stored in int16 array
    int padded_zeros_height = divUp(scales_height, 8) * 8; // zeros are stored as int32 (which has 8 int4 values)
    int packed_zeros_height = divUp(padded_zeros_height, 2); // int4 zeros stored in int8 array  

    if (!init)
    {
        checkCudaErrors(cudaFuncSetAttribute(
            compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));

        init = true;
    }

    checkKernelErrors(
    (compute_gemm << <deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
        SHMEM_SZ, stream>>> (x, (uint16_t*)w.weight, (uint8_t*)w.zeros, w.scales, xout, accum, M_TILES, N_TILES, K_TILES, Mr, Nr, Kr,
                      packed_zeros_height, scales_height, packed_wt_height)));

}
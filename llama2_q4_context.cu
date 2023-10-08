/*
Inference for Llama-2 Transformer model in pure Cuda.

### INT4 - AWQ quantization version ###

1. First generate AWQ int-4 quantized weights following steps in https://github.com/mit-han-lab/llm-awq
 E.g:
  python -m awq.entry --model_path /path-to-model/Llama-2-7b-chat-hf --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_cache/llama2-7b-chat-metadata.pt
  python -m awq.entry --model_path /path-to-model/Llama-2-7b-chat-hf --w_bit 4 --q_group_size 128 --load_awq awq_cache/llama2-7b-chat-metadata.pt --q_backend real --dump_quant awq_weights/llama2-7b-awq.pt
 Note - AWQ scripts doesn't run on Windows. Use Linux or WSL.

2. Convert AWQ weights into individual weight binary files using convert_awq_to_bin.py

3. Convert/repack the weight binary files using the weight_packer.cpp utility.

4. Run this program pointing to the final weight file.
*/

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include "common.h"

// ----------------------------------------------------------------------------
// GPU kernels

static __global__ void copy_embedding_kernel(half* x, const half* __restrict__ table, int size, int* tokens, int* pPos)
{
    int ctxidx = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    int pos = *pPos + ctxidx;
    int token = tokens[pos];
    int table_index = index + token * size;
    x[index + ctxidx * size] = table[table_index];
}

// Single block - not enough parallelism for the GPU, but it's just 1% of total time
static __device__ void rmsnorm_core_kernel(half* o, half* x, half* weight, int size, int elementsPerThread, int ctxidx) {
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size) {
            float val = (float)x[index + ctxidx * size];
            ss += val * val;
        }
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss);

    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-6f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;

    // normalize
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size) {
            float val = (float)x[index + ctxidx * size];
            val *= ss * (float)weight[index];
            o[index + ctxidx * size] = (half)val;
        }
    }
}

static __global__ void rmsnorm_kernel(half* o, half* x, half* weight, int size, int elementsPerThread) {

    int ctxidx = blockIdx.x;
    rmsnorm_core_kernel(o, x, weight, size, elementsPerThread, ctxidx);
}

static __global__ void rmsnorm_kernel(half* o, half* x, half* weight, int size, int elementsPerThread, int ctxidx) {

    rmsnorm_core_kernel(o, x, weight, size, elementsPerThread, ctxidx);
}


// Only used for the final linear layer to get logits (for most other layers we use the INT4 version below)
__global__ void mat_vec_kernel(half* op, const half* ip, const half* wt, int n, int d, int numSerialLoads, 
    int ip_stride, int w_stride, int op_stride, int w_row_stride, float alpha, int ctxidx) {
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;
    const half* __restrict__ input = ip + blockIdx.y * ip_stride + ctxidx * n;
    const half* __restrict__ weight = wt + blockIdx.y * w_stride;
    half* output = op + blockIdx.y * op_stride;

    float sum = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + threadIdx.x) * 8;
        if (j < n) {
            half w[8];
            half ip[8];
            *((uint4 *)(&w)) = *((uint4 *)(&weight[index * w_row_stride + j]));
            *((uint4 *)(&ip)) = *((uint4 *)(&input[j]));
            for (int el = 0; el < 8; el++)
                sum += float(w[el]) * float(ip[el]);
        }
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);
    sum *= alpha;

    if (threadIdx.x == 0)
        output[index] = (half)sum;
}

// Simpler version of the above - handles non multiple of 8 dimensions too (used only by MHA block)
static __global__ void mat_vec_kernel_simple(half* op, half* ip, half* wt, int n, int numSerialElements,
    int ip_stride, int w_stride, int ip_row_stride, int w_row_stride, int op_row_stride, int op_plane_stride, float alpha, int *pPos) {

    int ctxidx = blockIdx.z;
    int op_stride = *pPos + 1 + ctxidx;
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= op_stride)
        return;

    const half* __restrict__ input = ip + blockIdx.y * ip_stride + ctxidx * ip_row_stride;
    const half* __restrict__ weight = wt + blockIdx.y * w_stride;
    half* output = op + blockIdx.y * op_plane_stride + ctxidx * op_row_stride;

    float sum = 0;
    for (int i = 0; i < numSerialElements; i++) {
        int j = i * 32 + threadIdx.x;
        if (j < n)
            sum += ((float)weight[index * w_row_stride + j]) * ((float)input[j]);
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);
    sum *= alpha;

    if (threadIdx.x == 0)
        output[index] = (half)sum;
}

// hardcoded for group-count = 128
static __global__ void mat_vec_kernel_int4(half* __restrict__ output, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, bool accum)
{
    int ctxidx = blockIdx.y;
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= opElements)
        return;

    float sum = 0;
    for (int ygq = 0; ygq * 128 + threadIdx.x * 4 < packed_weights_height; ygq++) {   // each iteration of this loop covers 8 x 128 elements in y dimension of weight matrix (weight matrix is column major)
        uint32_t packed_q_z = q_zeros[index * packed_zeros_height + ygq];

        // load weights in one go (32 elements from weight matrix loaded by each thread in one read)
        uint32_t loaded_packed_wts[4];
        *((uint4*)(&loaded_packed_wts[0])) = *((uint4*)(&q_weight[index * packed_weights_height + ygq * 128 + threadIdx.x * 4]));

        int group_y = ygq * 8 + (threadIdx.x / 4);
        float q_z = (float)(packed_q_z >> (4 * (threadIdx.x / 4)) & 0xF);
        float scale = (float)scales[index * scales_height + group_y];
        int y_base = ygq * 1024 + threadIdx.x * 32;

        for (int qi = 0; qi < 4; qi ++) {                 // each iteration of this loop covers 256 elements in y dimension of weight matrix
            int ys = y_base + qi * 8;
            if (ys < inputElements) {
                uint32_t packed_q_w = loaded_packed_wts[qi];
                half ip[8];
                *((uint4*)(&ip)) = *((uint4*)(&input[ys + ctxidx * inputElements]));

                for (int i = 0; i < 8; i++) {
                    float q_wt = (float)(packed_q_w & 0xF);
                    float w = (q_wt - q_z) * scale;
                    sum += w * float(ip[i]);
                    packed_q_w = (packed_q_w >> 4);
                }
            }
        }
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    if (threadIdx.x == 0) {
        output += ctxidx * opElements;

        if (accum)
            sum += (float)output[index];
        output[index] = (half)sum;
    }
}

// Here we make use of shared memory to achieve better memory access pattern, and transpose a 32x32 chunk of the matrix on the fly
// Again used only by the MHA block
static __global__ void vec_mat_kernel(half* op, const half* __restrict__ ip, const half* __restrict__ wt, int N, int *pPos, int w_stride, int op_stride, int w_row_stride, int op_row_stride, int ip_row_stride, int ip_plane_stride) {
    int ctxidx = blockIdx.z;
    int K = *pPos + 1 + ctxidx;
    const half* __restrict__ input = ip + blockIdx.y * ip_plane_stride + ctxidx * ip_row_stride;
    const half* __restrict__ weight = wt + blockIdx.y * w_stride;
    half* output = op + blockIdx.y * op_stride + ctxidx * op_row_stride;

    int start_n = blockIdx.x * 32;
    int i = start_n + threadIdx.y;

    // 2x for double buffering
    // +2 to avoid shared memory bank conflicts
    __shared__ half loaded_fragment[2][32][32 + 2];

    // OOB check
    if (i >= N)
        return;

    // load the first 32x32 fragment
    int n = start_n + threadIdx.x;
    int k = threadIdx.y;
    int offset = k * w_row_stride + n;
    loaded_fragment[0][threadIdx.y][threadIdx.x] = ((n < N) && (k < K)) ? weight[offset] : 0;

    float sum = 0;
    // Loop over the matrix row and vector elements
    for (int e = 0; ;) {
        __syncthreads();    // wait for the load

        int start_k = e * 32;
        if (start_k >= K) break;
        k = start_k + threadIdx.x;
        int buf_i = e & 1;
        sum += float(loaded_fragment[buf_i][threadIdx.x][threadIdx.y]) * ((k < K) ? (float) input[k] : 0);

        // load for the next iteration
        e++;
        start_k = e * 32;
        buf_i = e & 1;
        n = start_n + threadIdx.x;
        k = start_k + threadIdx.y;
        int offset = k * w_row_stride + n;
        loaded_fragment[buf_i][threadIdx.y][threadIdx.x] = ((n < N) && (k < K)) ? weight[offset] : 0;
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    if (threadIdx.x == 0)
        output[i] = (half)sum;
}

// Each block processes a single head
static __global__ void RoPERotation_kernel(half* sq, half* sk_base, int num_heads, int head_size, int *pPos, int loff) {
    int ctxidx = blockIdx.y;
    int pos = *pPos;
    half* sk = sk_base + loff + (pos + ctxidx) * num_heads * head_size;
    int h = blockIdx.x;
    half* q = sq + ctxidx * head_size * num_heads + h * head_size;
    half* k = sk + h * head_size;
    int i = threadIdx.x;
    int head_dim = (i * 2) % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    float q0 = q[i];
    float q1 = q[i + head_size/2];
    float k0 = k[i];
    float k1 = k[i + head_size / 2];
    q[i] = q0 * fcr - q1 * fci;
    q[i + head_size / 2] = q0 * fci + q1 * fcr;
    k[i] = k0 * fcr - k1 * fci;
    k[i + head_size / 2] = k0 * fci + k1 * fcr;
}

static __global__ void softmax_kernel(half* __restrict__ arr, int num_heads, int *pPos, int op_row_stride, int op_plane_stride) {
    __shared__ float att[MAX_SEQ_LEN];
    int ctxidx = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    int size = *pPos + 1 + ctxidx;

    // load input to shared memory
    for (int t = tid; t < size; t += step)
        att[t] = (float) arr[h * op_plane_stride + ctxidx * op_row_stride + t];
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    // find max value (for numerical stability)
    float max_val = tid < size ? att[tid] : 0;
    for (int i = tid + step; i < size; i += step)
        if (att[i] > max_val)
            max_val = att[i];

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        att[i] = expf(att[i] - max_val);
        sum += att[i];
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        arr[h * op_plane_stride + ctxidx * op_row_stride + t] = (half) (att[t] / sum);
}

static __global__ void silu_element_wise_mul_kernel(half* dest, half* src, int size) {
    int ctxidx = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = (float)dest[i + ctxidx*size];
        val *= 1.0f / (1.0f + expf(-val));
        val *= (float)src[i + ctxidx*size];
        dest[i + ctxidx*size] = (half)val;
    }
}

__global__ void argmax_kernel(half* __restrict__ x, int size, int* result, volatile int* pPos, int* pPosGpu, int prompt_len) {
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    int tid = threadIdx.x;
    int step = blockDim.x;

    // find local max value and its position
    float max_val = tid < size ? (float)x[tid] : -INFINITY;
    int   max_pos = tid < size ? tid : 0;
    for (int i = tid + step; i < size; i += step) {
        if ((float)x[i] > max_val) {
            max_val = x[i];
            max_pos = i;
        }
    }

    // find the global max value
    float global_max_val;
    global_max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = global_max_val;
    __syncthreads();
    global_max_val = shared_val;

    // possibility of race condition here, so we first write it to shared memory variable and then have just one thread to update the pointers.
    __shared__ int global_max_pos;
    if (max_val == global_max_val) {
        global_max_pos = max_pos;
    }
    __syncthreads();

    // write next token to the current token location
    if (threadIdx.x == 0) {
        int token_pos = *pPos;
        token_pos += prompt_len;

        result[token_pos] = global_max_pos;

        // update the token indices (unblocks the CPU)
        *pPos = token_pos;
        *pPosGpu = token_pos;
    }
}

// ----------------------------------------------------------------------------
// neural net blocks

void rmsnorm(cudaStream_t stream, int prompt_len, half* o, half* x, half* weight, int size) {
    int elementsPerThread = divUp(size, 1024);
    
    {
        rmsnorm_kernel <<< prompt_len, 1024, 0, stream>>> (o, x, weight, size, elementsPerThread);
    }
}

void rmsnorm_final(cudaStream_t stream, int prompt_len, half* o, half* x, half* weight, int size) {
    int elementsPerThread = divUp(size, 1024);

    // we need to run this only for the last token
    rmsnorm_kernel <<< 1, 1024, 0, stream>>> (o, x, weight, size, elementsPerThread, prompt_len - 1);
}

void matmul_classifier(cudaStream_t stream, int prompt_len, half* xout, half* x, half* w, int n, int d, int batch = 1, int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1, float alpha = 1.0f) {
    if ((n & 7) || (d & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(1); }
    int serialElements = divUp(n, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(d, 4), batch);
    if (w_row_stride == -1) w_row_stride = n;

    // need to run this only for the last token
    mat_vec_kernel <<<grid_dim, block_dim, 0, stream >>> (xout, x, w, n, d, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha, prompt_len - 1);
}

void matmul(cudaDeviceProp& deviceProp, cudaStream_t stream, int prompt_len, half* xout, half* x, QWeight &w, int inpSize, int opSize, bool accum = false) {
    if ((inpSize & 7) || (opSize & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(1); }
    // We are assuming a vector - matrix mul with col major matrix: height = inpSize,  width  = opSize
    int scales_height = divUp(inpSize, 128);
    int packed_wt_height = getPackedWeightHeight(inpSize);
    int packed_zeros_height = divUp(scales_height, 8);
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(opSize, 4), prompt_len);
    {
        mat_vec_kernel_int4 <<<grid_dim, block_dim, 0, stream >>> (xout, x, w.weight, w.zeros, w.scales, inpSize, opSize, packed_zeros_height, scales_height, packed_wt_height, accum);
    }
}

void RoPERotation(cudaStream_t stream, int prompt_len, half *q, half *k, int num_heads, int head_size, int* pPos, int loff) {
    
    {
        dim3 grid(num_heads, prompt_len);
        RoPERotation_kernel <<<grid, head_size / 2, 0, stream >>> (q, k, num_heads, head_size, pPos, loff);
    }
}

void MultiHeadAttention(cudaStream_t stream, int prompt_len, half *output, half *q, half *key_cache, half * value_cache, half *att, int num_heads, int head_size, int max_seq_len, int *pPos) {
    int dim = head_size * num_heads;
    // 1. Get attention scores
    int serialElements = divUp(head_size, 32);
    dim3 block_dim(32, 32);
    dim3 grid_dim1(divUp(max_seq_len, 32), num_heads, prompt_len);      // using max_seq_len instead of real seq_len here has measurable impact on perf (2%) :-/
    {
        mat_vec_kernel_simple <<< grid_dim1, block_dim, 0, stream >>> (att, q, key_cache, head_size, serialElements, head_size, head_size, dim, dim, max_seq_len, max_seq_len * max_seq_len, 1.0 / sqrt(head_size), pPos);
    }

    // 2. Run softmax kernel
    {
        dim3 grid(num_heads, prompt_len);
        softmax_kernel <<< grid, 1024, 0, stream >>> (att, num_heads, pPos, max_seq_len, max_seq_len * max_seq_len);
    }

    // 3. weighted sum of the values to get the final result
    {
        dim3 grid_dim2(divUp(head_size, 32), num_heads, prompt_len);
        vec_mat_kernel <<< grid_dim2, block_dim, 0, stream >>> (output, att, value_cache, head_size, pPos, head_size, head_size, dim, dim, max_seq_len, max_seq_len * max_seq_len);
    }
}

void siluElementwiseMul(cudaStream_t stream, int prompt_len, half *hb, half *hb2, int size) {
    {
        dim3 grid(divUp(size, 256), prompt_len);
        silu_element_wise_mul_kernel <<< grid, 256, 0, stream >>> (hb, hb2, size);
    }
}

void run_llama_network(int *pPos, Config* p, RunState* s, TransformerWeights* w, int prompt_len, cudaDeviceProp& deviceProp, cudaStream_t stream) {
    half* x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    {
        dim3 gridDim(divUp(dim, 256), prompt_len);
        copy_embedding_kernel <<<gridDim, 256, 0, stream >>> (x, w->token_embedding_table, dim, s->shared_data->tokens, pPos);
    }

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(stream, prompt_len, s->xb, x, w->layers[l].rms_att_weight, dim);

        // we directly store (key, value) at this time step (pos) to our kv cache
        int loff = l * p->seq_len * dim; // kv cache layer offset for convenience
        half* key_cache = s->key_cache + loff + (s->shared_data->pos * dim);
        half* val_cache = s->value_cache + loff + (s->shared_data->pos * dim);

        // qkv matmuls for this position (opt: can be done in single kernel as batch of 3)
        /*matmul(deviceProp, stream, prompt_len, s->q, s->xb, w->layers[l].wq_q, dim, dim);
        matmul(deviceProp, stream, prompt_len, key_cache, s->xb, w->layers[l].wq_k, dim, dim);
        matmul(deviceProp, stream, prompt_len, val_cache, s->xb, w->layers[l].wq_v, dim, dim);*/
        fp16_int4_gemm(deviceProp, stream, prompt_len, s->q, s->xb, w->layers[l].wq_q, dim, dim);
        fp16_int4_gemm(deviceProp, stream, prompt_len, key_cache, s->xb, w->layers[l].wq_k, dim, dim);
        fp16_int4_gemm(deviceProp, stream, prompt_len, val_cache, s->xb, w->layers[l].wq_v, dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        // also save the output (key, value) at this time step (pos) to our kv cache
        RoPERotation(stream, prompt_len, s->q, s->key_cache, p->n_heads, head_size, pPos, loff);

        // apply MHA using the query and the key-value cache
        MultiHeadAttention(stream, prompt_len, s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, p->n_heads, head_size, p->seq_len, pPos);

        // final matmul to get the output of the attention fused with residual connection back into x
        //matmul(deviceProp, stream, prompt_len, s->x, s->xb, w->layers[l].wq_o, dim, dim, true);
        fp16_int4_gemm(deviceProp, stream, prompt_len, s->x, s->xb, w->layers[l].wq_o, dim, dim, true);

        // ffn rmsnorm
        rmsnorm(stream, prompt_len, s->xb, x, w->layers[l].rms_ffn_weight, dim);

        // apply gate and up proj (opt: can be done in single kernel as batch of 2)
        /*matmul(deviceProp, stream, prompt_len, s->hb, s->xb, w->layers[l].wq_gate, dim, hidden_dim);
        matmul(deviceProp, stream, prompt_len, s->hb2, s->xb, w->layers[l].wq_up, dim, hidden_dim);*/
        fp16_int4_gemm(deviceProp, stream, prompt_len, s->hb, s->xb, w->layers[l].wq_gate, dim, hidden_dim);
        fp16_int4_gemm(deviceProp, stream, prompt_len, s->hb2, s->xb, w->layers[l].wq_up, dim, hidden_dim);

        // apply F.silu activation on hb and multiply it with hb2
        siluElementwiseMul(stream, prompt_len, s->hb, s->hb2, hidden_dim);
        //matmul(deviceProp, stream, prompt_len, s->x, s->hb, w->layers[l].wq_down, hidden_dim, dim, true);
        fp16_int4_gemm(deviceProp, stream, prompt_len, s->x, s->hb, w->layers[l].wq_down, hidden_dim, dim, true);
    }

    // final rmsnorm
    rmsnorm_final(stream, prompt_len, x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul_classifier(stream, prompt_len, s->logits, x, w->wcls, p->dim, p->vocab_size);
}

void transformer_ctx(Config* p, RunState* s, TransformerWeights* w, int prompt_len, cudaDeviceProp& deviceProp, cudaStream_t stream) {
#if DUMP_PER_TOKEN_TIMINGS == 1
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
#endif

    run_llama_network(s->pos, p, s, w, prompt_len, deviceProp, stream);

    // sample the next token using greedy argmax sampling: take the token with the highest probability (not included in the graph because of gen_token variable)
    argmax_kernel <<<1, 1024, 0, stream>>> (s->logits, p->vocab_size, &(s->shared_data->tokens[0]), &(s->shared_data->pos), s->pos, prompt_len);

#if DUMP_PER_TOKEN_TIMINGS == 1
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf(" t: %g ", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
}
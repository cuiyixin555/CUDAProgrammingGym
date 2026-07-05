// MIT License

// Copyright (c) 2026 CUI Xin (崔 欣)

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef FP16_SCALAR_PRODUCT_KERNEL_H
#define FP16_SCALAR_PRODUCT_KERNEL_H

#include <cuda_fp16.h>
#include <helper_cuda.h>

#define NUM_OF_BLOCKS 128
#define NUM_OF_THREADS 128

__forceinline__ __device__ void reduceInShared_intrinsics(half2 *const v) {
  if (threadIdx.x < 64)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 64]);
  __syncthreads();
  if (threadIdx.x < 32)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 32]);
  __syncthreads();
  if (threadIdx.x < 16)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 16]);
  __syncthreads();
  if (threadIdx.x < 8)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 8]);
  __syncthreads();
  if (threadIdx.x < 4)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 4]);
  __syncthreads();
  if (threadIdx.x < 2)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 2]);
  __syncthreads();
  if (threadIdx.x < 1)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 1]);
  __syncthreads();
}

__forceinline__ __device__ void reduceInShared_native(half2 *const v) {
  if (threadIdx.x < 64)
    v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 64];
  __syncthreads();
  if (threadIdx.x < 32)
    v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 32];
  __syncthreads();
  if (threadIdx.x < 16)
    v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 16];
  __syncthreads();
  if (threadIdx.x < 8)
    v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 8];
  __syncthreads();
  if (threadIdx.x < 4)
    v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 4];
  __syncthreads();
  if (threadIdx.x < 2)
    v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 2];
  __syncthreads();
  if (threadIdx.x < 1)
    v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 1];
  __syncthreads();
}

__global__ void scalarProductKernel_intrinsics(half2 const *const a,
                                               half2 const *const b,
                                               float *const results,
                                               size_t const size) {
  const int stride = gridDim.x * blockDim.x;
  __shared__ half2 shArray[NUM_OF_THREADS];

  shArray[threadIdx.x] = __float2half2_rn(0.f);
  half2 value = __float2half2_rn(0.f);

  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += stride) {
    value = __hfma2(a[i], b[i], value);
  }

  shArray[threadIdx.x] = value;
  __syncthreads();
  reduceInShared_intrinsics(shArray);

  if (threadIdx.x == 0) {
    half2 result = shArray[0];
    float f_result = __low2float(result) + __high2float(result);
    results[blockIdx.x] = f_result;
  }
}

__global__ void scalarProductKernel_native(half2 const *const a,
                                           half2 const *const b,
                                           float *const results,
                                           size_t const size) {
  const int stride = gridDim.x * blockDim.x;
  __shared__ half2 shArray[NUM_OF_THREADS];

  half2 value(0.f, 0.f);
  shArray[threadIdx.x] = value;

  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += stride) {
    value = a[i] * b[i] + value;
  }

  shArray[threadIdx.x] = value;
  __syncthreads();
  reduceInShared_native(shArray);

  if (threadIdx.x == 0) {
    half2 result = shArray[0];
    float f_result = (float)result.y + (float)result.x;
    results[blockIdx.x] = f_result;
  }
}

void generateInput(half2 *a, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    half2 temp;
    temp.x = static_cast<float>(rand() % 4);
    temp.y = static_cast<float>(rand() % 2);
    a[i] = temp;
  }
}

#endif // FP16_SCALAR_PRODUCT_KERNEL_H
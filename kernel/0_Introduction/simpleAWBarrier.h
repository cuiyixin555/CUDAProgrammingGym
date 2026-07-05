// MIT License

// Copyright (c) 2026 CUI Xin (崔 欣)

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef SIMPLE_AW_BARRIER_H
#define SIMPLE_AW_BARRIER_H

// Includes, system
#include <stdio.h>

// Includes CUDA
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h> // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h> // helper functions for CUDA error check

namespace cg = cooperative_groups;

#if __CUDA_ARCH__ >= 700
template <bool writeSquareRoot>
__device__ void
reduceBlockData(cuda::barrier<cuda::thread_scope_block> &barrier,
                cg::thread_block_tile<32> &tile32, double &threadSum,
                double *result) {
  extern __shared__ double tmp[];

#pragma unroll
  for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
    threadSum += tile32.shfl_down(threadSum, offset);
  }
  if (tile32.thread_rank() == 0) {
    tmp[tile32.meta_group_rank()] = threadSum;
  }

  auto token = barrier.arrive();

  barrier.wait(std::move(token));

  // The warp 0 will perform last round of reduction
  if (tile32.meta_group_rank() == 0) {
    double beta = tile32.thread_rank() < tile32.meta_group_size()
                      ? tmp[tile32.thread_rank()]
                      : 0.0;

#pragma unroll
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      beta += tile32.shfl_down(beta, offset);
    }

    if (tile32.thread_rank() == 0) {
      if (writeSquareRoot)
        *result = sqrt(beta);
      else
        *result = beta;
    }
  }
}
#endif

__global__ void normVecByDotProductAWBarrier(float *vecA, float *vecB,
                                             double *partialResults, int size) {
#if __CUDA_ARCH__ >= 700
  cg::thread_block cta = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

  if (threadIdx.x == 0) {
    init(&barrier, blockDim.x);
  }

  cg::sync(cta);

  double threadSum = 0.0;
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    threadSum += (double)(vecA[i] * vecB[i]);
  }

  // Each thread block performs reduction of partial dotProducts and writes to
  // global mem.
  reduceBlockData<false>(barrier, tile32, threadSum,
                         &partialResults[blockIdx.x]);

  cg::sync(grid);

  // One block performs the final summation of partial dot products
  // of all the thread blocks and writes the sqrt of final dot product.
  if (blockIdx.x == 0) {
    threadSum = 0.0;
    for (int i = cta.thread_rank(); i < gridDim.x; i += cta.size()) {
      threadSum += partialResults[i];
    }
    reduceBlockData<true>(barrier, tile32, threadSum, &partialResults[0]);
  }

  cg::sync(grid);

  const double finalValue = partialResults[0];

  // Perform normalization of vecA & vecB.
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    vecA[i] = (float)vecA[i] / finalValue;
    vecB[i] = (float)vecB[i] / finalValue;
  }
#endif
}

int runNormVecByDotProductAWBarrier(int argc, char **argv, int deviceId);

int runNormVecByDotProductAWBarrier(int argc, char **argv, int deviceId) {
  float *vecA, *d_vecA;
  float *vecB, *d_vecB;
  double *d_partialResults;
  int size = 10000000;

  checkCudaErrors(cudaMallocHost(&vecA, sizeof(float) * size));
  checkCudaErrors(cudaMallocHost(&vecB, sizeof(float) * size));

  checkCudaErrors(cudaMalloc(&d_vecA, sizeof(float) * size));
  checkCudaErrors(cudaMalloc(&d_vecB, sizeof(float) * size));

  float baseVal = 2.0;
  for (int i = 0; i < size; i++) {
    vecA[i] = vecB[i] = baseVal;
  }

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  checkCudaErrors(cudaMemcpyAsync(d_vecA, vecA, sizeof(float) * size,
                                  cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_vecB, vecB, sizeof(float) * size,
                                  cudaMemcpyHostToDevice, stream));

  // Kernel configuration, where a one-dimensional
  // grid and one-dimensional blocks are configured.
  int minGridSize = 0, blockSize = 0;
  checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
      &minGridSize, &blockSize, (void *)normVecByDotProductAWBarrier, 0, size));

  int smemSize = ((blockSize / 32) + 1) * sizeof(double);

  int numBlocksPerSm = 0;
  checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, normVecByDotProductAWBarrier, blockSize, smemSize));

  int multiProcessorCount = 0;
  checkCudaErrors(cudaDeviceGetAttribute(
      &multiProcessorCount, cudaDevAttrMultiProcessorCount, deviceId));

  minGridSize = multiProcessorCount * numBlocksPerSm;
  checkCudaErrors(cudaMalloc(&d_partialResults, minGridSize * sizeof(double)));

  printf("Launching normVecByDotProductAWBarrier kernel with numBlocks = %d "
         "blockSize = %d\n",
         minGridSize, blockSize);

  dim3 dimGrid(minGridSize, 1, 1), dimBlock(blockSize, 1, 1);

  void *kernelArgs[] = {(void *)&d_vecA, (void *)&d_vecB,
                        (void *)&d_partialResults, (void *)&size};

  checkCudaErrors(
      cudaLaunchCooperativeKernel((void *)normVecByDotProductAWBarrier, dimGrid,
                                  dimBlock, kernelArgs, smemSize, stream));

  checkCudaErrors(cudaMemcpyAsync(vecA, d_vecA, sizeof(float) * size,
                                  cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  float expectedResult = (baseVal / sqrt(size * baseVal * baseVal));
  unsigned int matches = 0;
  for (int i = 0; i < size; i++) {
    if ((vecA[i] - expectedResult) > 0.00001) {
      printf("mismatch at i = %d\n", i);
      break;
    } else {
      matches++;
    }
  }

  printf("Result = %s\n", matches == size ? "PASSED" : "FAILED");
  checkCudaErrors(cudaFree(d_vecA));
  checkCudaErrors(cudaFree(d_vecB));
  checkCudaErrors(cudaFree(d_partialResults));

  checkCudaErrors(cudaFreeHost(vecA));
  checkCudaErrors(cudaFreeHost(vecB));
  return matches == size;
}

#endif // SIMPLE_AW_BARRIER_H
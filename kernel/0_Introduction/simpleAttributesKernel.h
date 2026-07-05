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

// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

#ifndef SIMPLE_ATTRIBUTES_KERNEL_H
#define SIMPLE_ATTRIBUTES_KERNEL_H

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

cudaAccessPolicyWindow initAccessPolicyWindow(void) {
  cudaAccessPolicyWindow accessPolicyWindow = {0};
  accessPolicyWindow.base_ptr = (void *)0;
  accessPolicyWindow.num_bytes = 0;
  accessPolicyWindow.hitRatio = 0.f;
  accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  return accessPolicyWindow;
}

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param data  input data in global memory
//! @param dataSize  input data size
//! @param bigData  input bigData in global memory
//! @param bigDataSize  input bigData size
//! @param hitcount how many data access are done within block
////////////////////////////////////////////////////////////////////////////////
static __global__ void kernCacheSegmentTest(int *data, int dataSize, int *trash,
                                            int bigDataSize, int hitCount) {
  __shared__ unsigned int hit;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int tID = row * blockDim.y + col;
  uint32_t psRand = tID;

  atomicExch(&hit, 0);
  __syncthreads();
  while (hit < hitCount) {
    psRand ^= psRand << 13;
    psRand ^= psRand >> 17;
    psRand ^= psRand << 5;

    int idx = tID - psRand;
    if (idx < 0) {
      idx = -idx;
    }

    if ((tID % 2) == 0) {
      data[psRand % dataSize] = data[psRand % dataSize] + data[idx % dataSize];
    } else {
      trash[psRand % bigDataSize] =
          trash[psRand % bigDataSize] + trash[idx % bigDataSize];
    }

    atomicAdd(&hit, 1);
  }
}

#endif // SIMPLE_ATTRIBUTES_KERNEL_H
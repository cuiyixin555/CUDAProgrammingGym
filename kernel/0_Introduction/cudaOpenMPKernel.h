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

#ifndef CUDA_OPENMP_KERNEL_H
#define CUDA_OPENMP_KERNEL_H

#include <cuda_runtime.h>

// a simple kernel that simply increments each array element by b
__global__ void kernelAddConstant(int *g_a, const int b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_a[idx] += b;
}

// a predicate that checks whether each array element is set to its index plus b
inline int correctResult(int *data, const int n, const int b) {
  for (int i = 0; i < n; i++)
    if (data[i] != i + b)
      return 0;

  return 1;
}

#endif // CUDA_OPENMP_KERNEL_H
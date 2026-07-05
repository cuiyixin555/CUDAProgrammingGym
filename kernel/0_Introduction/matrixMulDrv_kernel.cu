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

#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]

template <int block_size, typename size_type>
__device__ void matrixMul(float *C, float *A, float *B, size_type wA,
                          size_type wB) {
  size_type bx = blockIdx.x;
  size_type by = blockIdx.y;
  size_type tx = threadIdx.x;
  size_type ty = threadIdx.y;

  size_type aBegin = wA * block_size * by;
  size_type aEnd = aBegin + wA - 1;
  size_type aStep = block_size;
  size_type bBegin = block_size * bx;
  size_type bStep = block_size * wB;

  float Csub = 0;

  for (size_type a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    __shared__ float As[block_size][block_size];
    __shared__ float Bs[block_size][block_size];

    AS(ty, tx) = A[a + wA * ty + tx];
    BS(ty, tx) = B[b + wB * ty + tx];

    __syncthreads();

#pragma unroll
    for (size_type k = 0; k < block_size; ++k)
      Csub += AS(ty, k) * BS(k, tx);

    __syncthreads();
  }

  size_type c = wB * block_size * by + block_size * bx;
  C[c + wB * ty + tx] = Csub;
}

extern "C" __global__ void matrixMul_bs8_64bit(float *C, float *A, float *B,
                                               size_t wA, size_t wB) {
  matrixMul<8, size_t>(C, A, B, wA, wB);
}

extern "C" __global__ void matrixMul_bs16_64bit(float *C, float *A, float *B,
                                                size_t wA, size_t wB) {
  matrixMul<16, size_t>(C, A, B, wA, wB);
}

extern "C" __global__ void matrixMul_bs32_64bit(float *C, float *A, float *B,
                                                size_t wA, size_t wB) {
  matrixMul<32, size_t>(C, A, B, wA, wB);
}

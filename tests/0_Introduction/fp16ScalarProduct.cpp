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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "fp16ScalarProductKernel.h"

int main(int argc, char *argv[]) {
  srand((unsigned int)time(NULL));
  size_t size = NUM_OF_BLOCKS * NUM_OF_THREADS * 16;

  half2 *vec[2];
  half2 *devVec[2];

  float *results;
  float *devResults;

  int devID = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp devProp;
  checkCudaErrors(cudaGetDeviceProperties(&devProp, devID));

  if (devProp.major < 5 || (devProp.major == 5 && devProp.minor < 3)) {
    printf(
        "ERROR: fp16ScalarProduct requires GPU devices with compute SM 5.3 or "
        "higher.\n");
    return EXIT_WAIVED;
  }

  for (int i = 0; i < 2; ++i) {
    checkCudaErrors(cudaMallocHost((void **)&vec[i], size * sizeof *vec[i]));
    checkCudaErrors(cudaMalloc((void **)&devVec[i], size * sizeof *devVec[i]));
  }

  checkCudaErrors(
      cudaMallocHost((void **)&results, NUM_OF_BLOCKS * sizeof *results));
  checkCudaErrors(
      cudaMalloc((void **)&devResults, NUM_OF_BLOCKS * sizeof *devResults));

  for (int i = 0; i < 2; ++i) {
    generateInput(vec[i], size);
    checkCudaErrors(cudaMemcpy(devVec[i], vec[i], size * sizeof *vec[i],
                               cudaMemcpyHostToDevice));
  }

  scalarProductKernel_native<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(
      devVec[0], devVec[1], devResults, size);

  checkCudaErrors(cudaMemcpy(results, devResults,
                             NUM_OF_BLOCKS * sizeof *results,
                             cudaMemcpyDeviceToHost));

  float result_native = 0;
  for (int i = 0; i < NUM_OF_BLOCKS; ++i) {
    result_native += results[i];
  }
  printf("Result native operators\t: %f \n", result_native);

  scalarProductKernel_intrinsics<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(
      devVec[0], devVec[1], devResults, size);

  checkCudaErrors(cudaMemcpy(results, devResults,
                             NUM_OF_BLOCKS * sizeof *results,
                             cudaMemcpyDeviceToHost));

  float result_intrinsics = 0;
  for (int i = 0; i < NUM_OF_BLOCKS; ++i) {
    result_intrinsics += results[i];
  }
  printf("Result intrinsics\t: %f \n", result_intrinsics);

  printf("&&&& fp16ScalarProduct %s\n",
         (fabs(result_intrinsics - result_native) < 0.00001) ? "PASSED"
                                                             : "FAILED");

  for (int i = 0; i < 2; ++i) {
    checkCudaErrors(cudaFree(devVec[i]));
    checkCudaErrors(cudaFreeHost(vec[i]));
  }

  checkCudaErrors(cudaFree(devResults));
  checkCudaErrors(cudaFreeHost(results));

  return EXIT_SUCCESS;
}

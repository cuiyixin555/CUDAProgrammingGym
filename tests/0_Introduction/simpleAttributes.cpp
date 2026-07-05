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

#include "simpleAttributesKernel.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdio.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  bool bTestResult = true;
  cudaAccessPolicyWindow accessPolicyWindow;
  cudaDeviceProp deviceProp;
  cudaStreamAttrValue streamAttrValue;
  cudaStream_t stream;
  cudaStreamAttrID streamAttrID;
  dim3 threads(32, 32);
  int *dataDevicePointer;
  int *dataHostPointer;
  int dataSize;
  int *bigDataDevicePointer;
  int *bigDataHostPointer;
  int bigDataSize;
  StopWatchInterface *timer = 0;

  printf("%s Starting...\n\n", argv[0]);

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  int devID = findCudaDevice(argc, (const char **)argv);
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  // Get device properties
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
  dim3 blocks(deviceProp.maxGridSize[1], 1);

  // Make sure device the l2 optimization
  if (deviceProp.persistingL2CacheMaxSize == 0) {
    printf("Waiving execution as device %d does not support persisting L2 "
           "Caching\n",
           devID);
    exit(EXIT_WAIVED);
  }

  // Create stream to assiocate with window
  checkCudaErrors(cudaStreamCreate(&stream));

  // Set the amount of l2 cache that will be persisting to maximum the device
  // can support
  checkCudaErrors(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                     deviceProp.persistingL2CacheMaxSize));

  // Stream attribute to set
  streamAttrID = cudaStreamAttributeAccessPolicyWindow;

  // Default window
  streamAttrValue.accessPolicyWindow = initAccessPolicyWindow();
  accessPolicyWindow = initAccessPolicyWindow();

  // Allocate size of both buffers
  bigDataSize = (deviceProp.l2CacheSize * 4) / sizeof(int);
  dataSize = (deviceProp.l2CacheSize / 4) / sizeof(int);

  // Allocate data
  checkCudaErrors(cudaMallocHost(&dataHostPointer, dataSize * sizeof(int)));
  checkCudaErrors(
      cudaMallocHost(&bigDataHostPointer, bigDataSize * sizeof(int)));

  for (int i = 0; i < bigDataSize; ++i) {
    if (i < dataSize) {
      dataHostPointer[i] = i;
    }

    bigDataHostPointer[bigDataSize - i - 1] = i;
  }

  checkCudaErrors(
      cudaMalloc((void **)&dataDevicePointer, dataSize * sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void **)&bigDataDevicePointer, bigDataSize * sizeof(int)));
  checkCudaErrors(cudaMemcpyAsync(dataDevicePointer, dataHostPointer,
                                  dataSize * sizeof(int),
                                  cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(bigDataDevicePointer, bigDataHostPointer,
                                  bigDataSize * sizeof(int),
                                  cudaMemcpyHostToDevice, stream));

  // Make a window for the buffer of interest
  accessPolicyWindow.base_ptr = (void *)dataDevicePointer;
  accessPolicyWindow.num_bytes = dataSize * sizeof(int);
  accessPolicyWindow.hitRatio = 1.f;
  accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  streamAttrValue.accessPolicyWindow = accessPolicyWindow;

  // Assign window to stream
  checkCudaErrors(
      cudaStreamSetAttribute(stream, streamAttrID, &streamAttrValue));

  // Demote any previous persisting lines
  checkCudaErrors(cudaCtxResetPersistingL2Cache());

  checkCudaErrors(cudaStreamSynchronize(stream));
  kernCacheSegmentTest<<<blocks, threads, 0, stream>>>(
      dataDevicePointer, dataSize, bigDataDevicePointer, bigDataSize, 0xAFFFF);

  checkCudaErrors(cudaStreamSynchronize(stream));
  // check if kernel execution generated and error
  getLastCudaError("Kernel execution failed");

  // Free memory
  checkCudaErrors(cudaFreeHost(dataHostPointer));
  checkCudaErrors(cudaFreeHost(bigDataHostPointer));
  checkCudaErrors(cudaFree(dataDevicePointer));
  checkCudaErrors(cudaFree(bigDataDevicePointer));

  sdkStopTimer(&timer);
  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  sdkDeleteTimer(&timer);

  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { runTest(argc, argv); }

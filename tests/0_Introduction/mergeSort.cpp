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

/*
 * Based on "Designing efficient sorting algorithms for manycore GPUs"
 * by Nadathur Satish, Mark Harris, and Michael Garland
 * http://mgarland.org/files/papers/gpusort-ipdps09.pdf
 *
 * Victor Podlozhnyuk 09/24/2009
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mergeSort_common.h"
#include "mergeSortKernel.h"

////////////////////////////////////////////////////////////////////////////////
// Validate sorted keys array (check for integrity and proper order)
////////////////////////////////////////////////////////////////////////////////
extern "C" uint validateSortedKeys(uint *resKey, uint *srcKey, uint batchSize,
                                   uint arrayLength, uint numValues,
                                   uint sortDir) {
  uint *srcHist;
  uint *resHist;

  if (arrayLength < 2) {
    printf("validateSortedKeys(): arrays too short, exiting...\n");
    return 1;
  }

  printf("...inspecting keys array: ");
  srcHist = (uint *)malloc(numValues * sizeof(uint));
  resHist = (uint *)malloc(numValues * sizeof(uint));

  int flag = 1;

  for (uint j = 0; j < batchSize; j++, srcKey += arrayLength, resKey += arrayLength) {
    memset(srcHist, 0, numValues * sizeof(uint));
    memset(resHist, 0, numValues * sizeof(uint));

    for (uint i = 0; i < arrayLength; i++) {
      if ((srcKey[i] < numValues) && (resKey[i] < numValues)) {
        srcHist[srcKey[i]]++;
        resHist[resKey[i]]++;
      } else {
        fprintf(stderr,
                "***Set %u source/result key arrays are not limited properly***\n",
                j);
        flag = 0;
        goto brk;
      }
    }

    for (uint i = 0; i < numValues; i++)
      if (srcHist[i] != resHist[i]) {
        fprintf(stderr,
                "***Set %u source/result keys histograms do not match***\n", j);
        flag = 0;
        goto brk;
      }

    for (uint i = 0; i < arrayLength - 1; i++)
      if ((sortDir && (resKey[i] > resKey[i + 1])) ||
          (!sortDir && (resKey[i] < resKey[i + 1]))) {
        fprintf(stderr, "***Set %u result key array is not ordered properly***\n",
                j);
        flag = 0;
        goto brk;
      }
  }

brk:
  free(resHist);
  free(srcHist);

  if (flag)
    printf("OK\n");

  return flag;
}

extern "C" void fillValues(uint *val, uint N) {
  for (uint i = 0; i < N; i++)
    val[i] = i;
}

extern "C" int validateSortedValues(uint *resKey, uint *resVal, uint *srcKey,
                                    uint batchSize, uint arrayLength) {
  int correctFlag = 1, stableFlag = 1;

  printf("...inspecting keys and values array: ");

  for (uint i = 0; i < batchSize; i++, resKey += arrayLength, resVal += arrayLength) {
    for (uint j = 0; j < arrayLength; j++) {
      if (resKey[j] != srcKey[resVal[j]])
        correctFlag = 0;

      if ((j < arrayLength - 1) && (resKey[j] == resKey[j + 1]) &&
          (resVal[j] > resVal[j + 1]))
        stableFlag = 0;
    }
  }

  printf(correctFlag ? "OK\n" : "***corrupted!!!***\n");
  printf(stableFlag ? "...stability property: stable!\n"
                    : "...stability property: NOT stable\n");

  return correctFlag;
}

////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  uint *h_SrcKey, *h_SrcVal, *h_DstKey, *h_DstVal;
  uint *d_SrcKey, *d_SrcVal, *d_BufKey, *d_BufVal, *d_DstKey, *d_DstVal;
  StopWatchInterface *hTimer = NULL;

  const uint N = 4 * 1048576;
  const uint DIR = 1;
  const uint numValues = 65536;

  printf("%s Starting...\n\n", argv[0]);

  int dev = findCudaDevice(argc, (const char **)argv);

  if (dev == -1) {
    return EXIT_FAILURE;
  }

  printf("Allocating and initializing host arrays...\n\n");
  sdkCreateTimer(&hTimer);
  h_SrcKey = (uint *)malloc(N * sizeof(uint));
  h_SrcVal = (uint *)malloc(N * sizeof(uint));
  h_DstKey = (uint *)malloc(N * sizeof(uint));
  h_DstVal = (uint *)malloc(N * sizeof(uint));

  srand(2009);

  for (uint i = 0; i < N; i++) {
    h_SrcKey[i] = rand() % numValues;
  }

  fillValues(h_SrcVal, N);

  printf("Allocating and initializing CUDA arrays...\n\n");
  checkCudaErrors(cudaMalloc((void **)&d_DstKey, N * sizeof(uint)));
  checkCudaErrors(cudaMalloc((void **)&d_DstVal, N * sizeof(uint)));
  checkCudaErrors(cudaMalloc((void **)&d_BufKey, N * sizeof(uint)));
  checkCudaErrors(cudaMalloc((void **)&d_BufVal, N * sizeof(uint)));
  checkCudaErrors(cudaMalloc((void **)&d_SrcKey, N * sizeof(uint)));
  checkCudaErrors(cudaMalloc((void **)&d_SrcVal, N * sizeof(uint)));
  checkCudaErrors(cudaMemcpy(d_SrcKey, h_SrcKey, N * sizeof(uint),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(uint),
                             cudaMemcpyHostToDevice));

  printf("Initializing GPU merge sort...\n");
  initMergeSort();

  printf("Running GPU merge sort...\n");
  checkCudaErrors(cudaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  mergeSort(d_DstKey, d_DstVal, d_BufKey, d_BufVal, d_SrcKey, d_SrcVal, N, DIR);
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  printf("Time: %f ms\n", sdkGetTimerValue(&hTimer));

  printf("Reading back GPU merge sort results...\n");
  checkCudaErrors(cudaMemcpy(h_DstKey, d_DstKey, N * sizeof(uint),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_DstVal, d_DstVal, N * sizeof(uint),
                             cudaMemcpyDeviceToHost));

  printf("Inspecting the results...\n");
  uint keysFlag =
      validateSortedKeys(h_DstKey, h_SrcKey, 1, N, numValues, DIR);

  uint valuesFlag = validateSortedValues(h_DstKey, h_DstVal, h_SrcKey, 1, N);

  printf("Shutting down...\n");
  closeMergeSort();
  sdkDeleteTimer(&hTimer);
  checkCudaErrors(cudaFree(d_SrcVal));
  checkCudaErrors(cudaFree(d_SrcKey));
  checkCudaErrors(cudaFree(d_BufVal));
  checkCudaErrors(cudaFree(d_BufKey));
  checkCudaErrors(cudaFree(d_DstVal));
  checkCudaErrors(cudaFree(d_DstKey));
  free(h_DstVal);
  free(h_DstKey);
  free(h_SrcVal);
  free(h_SrcKey);

  exit((keysFlag && valuesFlag) ? EXIT_SUCCESS : EXIT_FAILURE);
}

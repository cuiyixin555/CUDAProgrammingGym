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

#include "simpleAWBarrier.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdio.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("%s starting...\n", argv[0]);

  // This will pick the best possible CUDA capable device
  int dev = findCudaDevice(argc, (const char **)argv);

  int major = 0;
  checkCudaErrors(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));

  // Arrive-Wait Barrier require a GPU of Volta (SM7X) architecture or higher.
  if (major < 7) {
    printf("simpleAWBarrier requires SM 7.0 or higher.  Exiting...\n");
    exit(EXIT_WAIVED);
  }

  int supportsCooperativeLaunch = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&supportsCooperativeLaunch,
                                         cudaDevAttrCooperativeLaunch, dev));

  if (!supportsCooperativeLaunch) {
    printf("\nSelected GPU (%d) does not support Cooperative Kernel Launch, "
           "Waiving the run\n",
           dev);
    exit(EXIT_WAIVED);
  }

  int testResult = runNormVecByDotProductAWBarrier(argc, argv, dev);

  printf("%s completed, returned %s\n", argv[0], testResult ? "OK" : "ERROR!");
  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

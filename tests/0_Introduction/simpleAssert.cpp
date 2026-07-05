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

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <strings.h>
#include <sys/utsname.h>
#endif

// Includes, system
#include <cassert>
#include <stdio.h>

// Includes CUDA
#include "simpleAssertKernel.h"
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h> // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h> // helper functions for CUDA error check

const char *sampleName = "simpleAssert";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("%s starting...\n", sampleName);

  runTest(argc, argv);

  printf("%s completed, returned %s\n", sampleName,
         testResult ? "OK" : "ERROR!");
  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void runTest(int argc, char **argv) {
  int Nblocks = 2;
  int Nthreads = 32;
  cudaError_t error;

#ifndef _WIN32
  utsname OS_System_Type;
  uname(&OS_System_Type);

  printf("OS_System_Type.release = %s\n", OS_System_Type.release);

  if (!strcasecmp(OS_System_Type.sysname, "Darwin")) {
    printf("simpleAssert is not current supported on Mac OSX\n\n");
    exit(EXIT_SUCCESS);
  } else {
    printf("OS Info: <%s>\n\n", OS_System_Type.version);
  }

#endif

  // This will pick the best possible CUDA capable device
  findCudaDevice(argc, (const char **)argv);

  // Kernel configuration, where a one-dimensional
  // grid and one-dimensional blocks are configured.
  dim3 dimGrid(Nblocks);
  dim3 dimBlock(Nthreads);

  printf("Launch kernel to generate assertion failures\n");
  testKernel<<<dimGrid, dimBlock>>>(60);

  // Synchronize (flushes assert output).
  printf("\n-- Begin assert output\n\n");
  error = cudaDeviceSynchronize();
  printf("\n-- End assert output\n\n");

  // Check for errors and failed asserts in asynchronous kernel launch.
  if (error == cudaErrorAssert) {
    printf("Device assert failed as expected, "
           "CUDA error message is: %s\n\n",
           cudaGetErrorString(error));
  }

  testResult = error == cudaErrorAssert;
}

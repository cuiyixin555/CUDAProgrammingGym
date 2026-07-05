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

/**
 * This sample illustrates basic usage of cooperative groups within the thread
 * block. The code launches a single thread block, creates a cooperative group
 * of all threads in the block, and a set of tiled partition cooperative
 * groups. For each, it uses a generic reduction function to calculate the sum
 * of all the ranks in that group.
 */

#ifndef SIMPLE_COOPERATIVE_GROUPS_KERNEL_H
#define SIMPLE_COOPERATIVE_GROUPS_KERNEL_H

#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

/**
 * Calculates the sum of val across the group g. The workspace array, x,
 * must be large enough to contain g.size() integers.
 */
__device__ int sumReduction(cg::thread_group g, int *x, int val) {
  int lane = g.thread_rank();

  for (int i = g.size() / 2; i > 0; i /= 2) {
    x[lane] = val;
    g.sync();

    if (lane < i)
      val += x[lane + i];

    g.sync();
  }

  if (g.thread_rank() == 0)
    return val;
  else
    return -1;
}

/**
 * Creates cooperative groups and performs reductions.
 */
__global__ void cgkernel() {
  cg::thread_block threadBlockGroup = cg::this_thread_block();
  int threadBlockGroupSize = threadBlockGroup.size();

  extern __shared__ int workspace[];

  int input, output, expectedOutput;

  input = threadBlockGroup.thread_rank();
  expectedOutput = (threadBlockGroupSize - 1) * threadBlockGroupSize / 2;

  output = sumReduction(threadBlockGroup, workspace, input);

  if (threadBlockGroup.thread_rank() == 0) {
    printf(
        " Sum of all ranks 0..%d in threadBlockGroup is %d (expected %d)\n\n",
        (int)threadBlockGroup.size() - 1, output, expectedOutput);

    printf(" Now creating %d groups, each of size 16 threads:\n\n",
           (int)threadBlockGroup.size() / 16);
  }

  threadBlockGroup.sync();

  cg::thread_block_tile<16> tiledPartition16 =
      cg::tiled_partition<16>(threadBlockGroup);

  int workspaceOffset =
      threadBlockGroup.thread_rank() - tiledPartition16.thread_rank();

  input = tiledPartition16.thread_rank();
  expectedOutput = 15 * 16 / 2;

  output = sumReduction(tiledPartition16, workspace + workspaceOffset, input);

  if (tiledPartition16.thread_rank() == 0)
    printf("   Sum of all ranks 0..15 in this tiledPartition16 group is %d "
           "(expected %d)\n",
           output, expectedOutput);
}

#endif // SIMPLE_COOPERATIVE_GROUPS_KERNEL_H

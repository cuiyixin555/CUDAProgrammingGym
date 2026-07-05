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

////////////////////////////////////////////////////////////////////////////////
// Shortcut definitions
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;

#define SHARED_SIZE_LIMIT 1024U
#define SAMPLE_STRIDE 128

////////////////////////////////////////////////////////////////////////////////
// Extensive sort validation routine
////////////////////////////////////////////////////////////////////////////////
extern "C" uint validateSortedKeys(uint *resKey, uint *srcKey, uint batchSize,
                                   uint arrayLength, uint numValues,
                                   uint sortDir);

extern "C" void fillValues(uint *val, uint N);

extern "C" int validateSortedValues(uint *resKey, uint *resVal, uint *srcKey,
                                    uint batchSize, uint arrayLength);

////////////////////////////////////////////////////////////////////////////////
// CUDA merge sort
////////////////////////////////////////////////////////////////////////////////
extern "C" void initMergeSort(void);

extern "C" void closeMergeSort(void);

extern "C" void mergeSort(uint *dstKey, uint *dstVal, uint *bufKey,
                          uint *bufVal, uint *srcKey, uint *srcVal, uint N,
                          uint sortDir);

////////////////////////////////////////////////////////////////////////////////
// CPU "emulation"
////////////////////////////////////////////////////////////////////////////////
extern "C" void mergeSortHost(uint *dstKey, uint *dstVal, uint *bufKey,
                              uint *bufVal, uint *srcKey, uint *srcVal, uint N,
                              uint sortDir);

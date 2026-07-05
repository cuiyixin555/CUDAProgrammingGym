// MIT License

// Copyright (c) 2026 CUI Xin (崔 欣)

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


// These are helper functions for the SDK samples (string parsing,
// timers, image helpers, etc)
#ifndef COMMON_HELPER_FUNCTIONS_H_
#define COMMON_HELPER_FUNCTIONS_H_

#ifdef WIN32
#pragma warning(disable : 4996)
#endif

// includes, project
#include <assert.h>
#include <exception.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// includes, timer, string parsing, image helpers
#include <helper_image.h>  // helper functions for image compare, dump, data comparisons
#include <helper_string.h>  // helper functions for string parsing
#include <helper_timer.h>   // helper functions for timers

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#endif  // COMMON_HELPER_FUNCTIONS_H_

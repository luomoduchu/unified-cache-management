/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#ifndef CUDA_COMPAT_H
#define CUDA_COMPAT_H

#include <cstddef>
#include <cuda_runtime.h>

#if !defined(CUDART_VERSION) || (CUDART_VERSION < 12080)
#error "dev-sandbox CUDA copy cases require CUDA 12.8 or newer (cudaMemcpyBatchAsync)."
#endif

namespace cuda_compat {

/* cudaMemcpyBatchAsync changed shape between CUDA 12.8 and CUDA 13:
 *
 *   12.8 : (void** dsts, void** srcs, size_t* sizes, size_t count,
 *           cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs,
 *           size_t* failIdx, cudaStream_t stream)
 *
 *   13.x : (void* const* dsts, const void* const* srcs, const size_t* sizes, size_t count,
 *           cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs,
 *           cudaStream_t stream)
 *
 * CUDA 13 dropped the failIdx out-parameter and const-qualified the pointer
 * arrays. Both forms are called natively -- there is no degraded path, so a
 * future release that reshapes the call again fails to compile instead of
 * quietly measuring something else.
 *
 * The wrapper keeps the 12.8 parameter list so the four call sites read the
 * same on every CUDA version. CUDA 13 cannot report a failing index, so
 * failIdx is left untouched there. */
inline cudaError_t MemcpyBatchAsync(void* dsts[], void* srcs[], size_t sizes[], size_t count,
                                    cudaMemcpyAttributes attrs[], size_t attrsIdxs[],
                                    size_t numAttrs, size_t* failIdx, cudaStream_t stream)
{
#if CUDART_VERSION >= 13000
    (void)failIdx;
    return ::cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, stream);
#else
    return ::cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx,
                                  stream);
#endif
}

}  // namespace cuda_compat

#endif

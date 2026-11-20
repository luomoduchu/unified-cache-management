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
#include <cstdio>
#include <cuda_runtime.h>

#if defined(CUDART_VERSION) && (CUDART_VERSION < 12080)
#error "dev-sandbox CUDA copy cases require CUDA 12.8 or newer (cudaMemcpyAttributes)."
#endif

namespace cuda_compat {

/* cudaMemcpyBatchAsync arrived in CUDA 12.8 as
 *   (dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx, stream)
 * and that prototype did not survive the next major release -- CUDA 13 rejects
 * the call with "no matching function for call". Every batch memcpy in this
 * project goes through this wrapper so the version pin lives in one place.
 *
 * For a CUDA version whose native form is not pinned here the copies run as a
 * plain per-copy loop. That keeps the tool buildable, but it is not the same
 * thing being measured: cudaMemcpyAttributes carries a weak-consistency /
 * overlap-with-compute hint that a bare cudaMemcpyAsync cannot express, so the
 * fallback says so once on stderr instead of silently turning a "BatchCE" row
 * into a plain CE row.
 *
 * Unpinning a newer release is a one-line change: add its CUDART_VERSION range
 * to the condition below once the prototype in <cuda_runtime_api.h> is known. */
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12080) && (CUDART_VERSION < 13000)
#define CUDA_COMPAT_BATCH_MEMCPY_NATIVE 1
#else
#define CUDA_COMPAT_BATCH_MEMCPY_NATIVE 0
#endif

inline void WarnBatchMemcpyFallbackOnce()
{
    static bool warned = false;
    if (warned) { return; }
    warned = true;
    std::fprintf(stderr,
                 "[dev-sandbox] cudaMemcpyBatchAsync is not pinned for CUDART_VERSION=%d; batch "
                 "cases fall back to a per-copy loop and their numbers are not batched-enqueue "
                 "numbers.\n",
                 static_cast<int>(CUDART_VERSION));
}

/* The fallback cannot recover the transfer direction from the argument list, so
 * take it from the location hints the call sites already set. Returning
 * cudaMemcpyDefault when the hints are absent keeps the copy correct under
 * unified addressing instead of guessing a direction. */
inline cudaMemcpyKind InferCopyKind(const cudaMemcpyAttributes* attrs, size_t numAttrs)
{
    if (attrs == nullptr || numAttrs == 0) { return cudaMemcpyDefault; }
    const auto src = attrs[0].srcLocHint.type;
    const auto dst = attrs[0].dstLocHint.type;
    if ((src == cudaMemLocationTypeHost) && (dst == cudaMemLocationTypeDevice)) {
        return cudaMemcpyHostToDevice;
    }
    if ((src == cudaMemLocationTypeDevice) && (dst == cudaMemLocationTypeHost)) {
        return cudaMemcpyDeviceToHost;
    }
    return cudaMemcpyDefault;
}

/* Same parameter list as CUDA 12.8 so call sites stay readable. */
inline cudaError_t MemcpyBatchAsync(void* dsts[], void* srcs[], size_t sizes[], size_t count,
                                    cudaMemcpyAttributes attrs[], size_t attrsIdxs[],
                                    size_t numAttrs, size_t* failIdx, cudaStream_t stream)
{
#if CUDA_COMPAT_BATCH_MEMCPY_NATIVE
    return ::cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx,
                                  stream);
#else
    const auto kind = InferCopyKind(attrs, numAttrs);
    WarnBatchMemcpyFallbackOnce();
    for (size_t i = 0; i < count; ++i) {
        const auto status = ::cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], kind, stream);
        if (status != cudaSuccess) {
            if (failIdx != nullptr) { *failIdx = i; }
            return status;
        }
    }
    return cudaSuccess;
#endif
}

}  // namespace cuda_compat

#endif

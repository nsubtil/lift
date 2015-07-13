/*
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * Copyright (c) 2015, Roche Molecular Systems Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the copyright holders nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#ifndef ENABLE_TBB_BACKEND
#define ENABLE_TBB_BACKEND 1
#endif

#include <thrust/device_vector.h>

#include <thrust/system/cuda/vector.h>

#if ENABLE_TBB_BACKEND
#include <thrust/system/tbb/vector.h>
#endif

#include <thrust/execution_policy.h>

namespace lift {

enum target_system
{
    // host denotes the host CPU and is not meant to be used as a compute backend
    host,
    cuda,
#if ENABLE_TBB_BACKEND
    intel_tbb,
#endif
};


template <target_system system>
struct backend_policy
{ };


template <>
struct backend_policy<cuda>
{
    static inline decltype(thrust::cuda::par)& execution_policy(void)
    {
        return thrust::cuda::par;
    }
};

#if ENABLE_TBB_BACKEND
template <>
struct backend_policy<intel_tbb>
{
    static inline decltype(thrust::tbb::par)& execution_policy(void)
    {
        return thrust::tbb::par;
    }
};
#endif

} // namespace lift

// ugly macro hackery to force arbitrary device function / method instantiation
// note: we intentionally never instantiate device functions for the host system
#define __FUNC_CUDA(fun) auto *ptr_cuda = fun<lift::cuda>;
#define __METHOD_CUDA(base, method) auto ptr_cuda = &base<lift::cuda>::method;

#if ENABLE_TBB_BACKEND
#define __FUNC_TBB(fun) auto *ptr_TBB= fun<lift::intel_tbb>;
#define __METHOD_TBB(base, method) auto ptr_TBB = &base<lift::intel_tbb>::method;
#else
#define __FUNC_TBB(fun) ;
#define __METHOD_TBB(base, method) ;
#endif

// free function instantiation
#define INSTANTIATE(fun) \
        namespace __ ## fun ## __instantiation {    \
            __FUNC_CUDA(fun);                       \
            __FUNC_TBB(fun);                        \
    }

// method instantiation
#define METHOD_INSTANTIATE(base, method) \
        namespace __ ## base ## __ ## method ## __instantiation {   \
            __METHOD_CUDA(base, method);                            \
            __METHOD_TBB(base, method);                             \
    }

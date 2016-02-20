/*
 * Lift
 *
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

#include <lift/types.h>
#include <lift/backends.h>

#include <thrust/iterator/iterator_adaptor.h>

namespace lift {

// wrap an iterator in a thrust iterator
// this provides functionality to dereference a device iterator on the host
// and makes lift iterators work with thrust functions
template <target_system system, typename T, typename Iterator>
struct thrust_iterator_adaptor
    : public thrust::iterator_adaptor<thrust_iterator_adaptor<system, T, Iterator>,
                                      Iterator,
                                      T,
                                      typename backend_policy<system>::tag>
{
    typedef thrust::iterator_adaptor<thrust_iterator_adaptor<system, T, Iterator>,
                                     Iterator,
                                     T,
                                     typename backend_policy<system>::tag> base;
    using base::base;
};

// wrap a pointer in a thrust pointer
// same as above
template <target_system system, typename T>
struct thrust_pointer_adaptor
        : public thrust::pointer<T, typename backend_policy<system>::tag>
{
    typedef thrust::pointer<T, typename backend_policy<system>::tag> base;
    using base::base;
};

} // namespace lift

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

#include "pointer.h"

namespace lift {

template <target_system system, typename T>
struct __ldg_loader
{ };

template <typename T>
struct __ldg_loader<host, T>
{
    LIFT_HOST_DEVICE static inline T load(const T* pointer)
    {
        return *pointer;
    }
};

template <typename T>
struct __ldg_loader<cuda, T>
{
    LIFT_HOST_DEVICE static inline T load(const T* pointer)
    {
        return __ldg(pointer);
    }
};

// wrap a read-only memory pointer and access it through __ldg() on the GPU
template <target_system system, typename value_type, typename index_type = uint32>
struct ldg_pointer : public pointer<system, value_type, index_type>
{
    typedef pointer<system, value_type, index_type>     base;

    typedef typename base::reference_type               reference_type;
    typedef typename base::size_type                    size_type;

    using base::base;

    // ldg pointers are read-only
    LIFT_HOST_DEVICE reference_type at(size_type pos) = delete;
    LIFT_HOST_DEVICE reference_type operator[] (size_type pos) = delete;
    LIFT_HOST_DEVICE reference_type front() = delete;
    LIFT_HOST_DEVICE reference_type back() = delete;

    LIFT_HOST_DEVICE value_type at(size_type pos) const
    {
        return __ldg_loader<system, value_type>::load(&base::storage[pos]);
    }

    LIFT_HOST_DEVICE value_type operator[] (size_type pos) const
    {
        return __ldg_loader<system, value_type>::load(&base::storage[pos]);
    }

    LIFT_HOST_DEVICE value_type front() const
    {
        return __ldg_loader<system, value_type>::load(&base::storage[0]);
    }

    LIFT_HOST_DEVICE value_type back() const
    {
        return __ldg_loader<system, value_type>::load(&base::storage[base::storage_size - 1]);
    }

    // return a pointer to a memory range within this pointer
    LIFT_HOST_DEVICE ldg_pointer range(const size_type offset, size_type len = size_type(-1)) const
    {
        ldg_pointer ret;
        ret.storage = base::storage + offset;

        if (len == size_type(-1))
        {
            len = base::storage_size - offset;
        }

        ret.storage_size = len;

        return ret;
    }

    // pointer arithmetic
    // note that we don't do any bounds checking
    LIFT_HOST_DEVICE ldg_pointer operator+(off_t offset) const
    {
        ldg_pointer ret;
        ret.storage = base::storage + offset;
        ret.storage_size = base::storage_size - offset;

        return ret;
    }

    LIFT_HOST_DEVICE ldg_pointer operator-(off_t offset) const
    {
        ldg_pointer ret;
        ret.storage = base::storage - offset;
        ret.storage_size = base::storage_size + offset;

        return ret;
    }

    // return a truncated pointer
    LIFT_HOST_DEVICE ldg_pointer truncate(size_t new_size)
    {
        ldg_pointer ret;
        ret.storage = base::storage;
        ret.storage_size = new_size;

        return ret;
    }
};

template <target_system system, typename T, typename I>
LIFT_HOST_DEVICE const ldg_pointer<system, T, I> make_ldg_pointer(const pointer<system, T, I> p)
{
    return p;
}

} // namespace lift

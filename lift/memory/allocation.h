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

#include "../types.h"
#include "../backends.h"
#include "../decorators.h"

#include "default_allocator.h"
#include "pointer.h"
#include "thrust_wrappers.h"
#include "type_assignment_checks.h"

namespace lift {

// tagged mutable memory pointer interface
// exposes some of the std::vector interface, except for anything where the semantics differ from a plain pointer
template <target_system system,
          typename T,
          typename _index_type = uint32,
          typename allocator = typename default_memory_allocator<system>::type>
struct allocation : public pointer<system, T, _index_type>
{
    typedef pointer<system, T, _index_type> base;

    typedef typename base::pointer_type pointer_type;
    typedef typename base::value_type   value_type;
    typedef typename base::size_type    size_type;
    typedef typename base::index_type   index_type;

    typedef allocator                   allocator_type;

    using base::base;
    using base::storage;
    using base::storage_size;

    LIFT_HOST_DEVICE allocation()
        : base()
    { }

    allocation(size_type count)
        : base()
    {
        resize(count);
    }

    LIFT_HOST_DEVICE allocation(const allocation& other)
    {
        storage = other.storage;
        storage_size = other.storage_size;
    }

    // create allocation from pointer
    // this should probably be disallowed and done differently
    template <typename value_type>
    allocation(const pointer<system, value_type, index_type>& other)
    {
        *this = other;
    }

    // initializer list semantics: make a copy of the contents
    allocation(const std::initializer_list<value_type>& l)
        : base()
    {
        resize(l.size());
        index_type offset = 0;
        for(auto i : l)
        {
            base::poke(offset, i);
            offset++;
        }
    }

    // assign allocation from pointer
    // this should probably be disallowed and done differently
    template <typename value_type>
    allocation& operator=(const pointer<system, value_type, index_type>& other)
    {
        storage = other.data();
        storage_size = other.size();

        return *this;
    }

    // initializer list semantics: make a copy of the contents
    allocation& operator=(const std::initializer_list<value_type>& l)
    {
        resize(l.size());
        index_type offset = 0;
        for(auto i : l)
        {
            base::poke(offset, i);
            offset++;
        }

        return *this;
    }

    virtual void resize(size_type count)
    {
        size_type old_storage_size;
        pointer_type old_storage;

        old_storage = storage;
        old_storage_size = storage_size;

        storage = (pointer_type) allocator_type().allocate(sizeof(value_type) * count);

        if (old_storage != nullptr)
        {
            device_memory_copy((void *)storage, old_storage, sizeof(value_type) * std::min(count, old_storage_size));
            allocator_type().deallocate((pointer_type)old_storage);
        }

        storage_size = count;
    }

    // release the memory allocation
    virtual void free(void)
    {
        if (storage)
        {
            allocator_type().deallocate(storage);
        }

        storage = nullptr;
        storage_size = 0;
    }

    // cross-memory-space copy from another pointer
    // note that this does not handle copies across different GPUs
    template <typename other_allocation>
    void copy(const other_allocation& other)
    {
        __internal::check_value_type_assignment_compatible<value_type, typename other_allocation::value_type>();

        resize(other.size());

        if (system == cuda)
        {
            // copying to GPU...
            if (target_system(other_allocation::system_tag) == cuda)
            {
                // ... from the GPU
                cudaMemcpy((void *) base::data(), other.data(), sizeof(value_type) * other.size(), cudaMemcpyDeviceToDevice);
            } else {
                // ... from the host
                cudaMemcpy((void *) base::data(), other.data(), sizeof(value_type) * other.size(), cudaMemcpyHostToDevice);
            }
        } else {
            // copying to host...
            if (target_system(other_allocation::system_tag) == cuda)
            {
                // ... from the GPU
                cudaMemcpy((void *) base::data(), other.data(), sizeof(value_type) * other.size(), cudaMemcpyDeviceToHost);
            } else {
                // ... from the host
                memcpy((void *) base::data(), other.data(), sizeof(value_type) * other.size());
            }
        }
    }

protected:
    void device_memory_copy(void *dst, const void *src, size_t size)
    {
        if (system == cuda)
        {
            cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
        } else {
            memcpy(dst, src, size);
        }
    }
};

} // namespace lift

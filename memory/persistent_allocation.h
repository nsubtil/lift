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
#include "allocation.h"
#include "thrust_wrappers.h"
#include "type_assignment_checks.h"

namespace lift {

// same as allocation, except no reallocation happens when the size shrinks
// implements reserve() and capacity()
template <target_system system,
          typename T,
          typename _index_type = uint32,
          typename allocator = typename default_memory_allocator<system>::type>
struct persistent_allocation : public allocation<system, T, _index_type>
{
    typedef allocation<system, T, _index_type> base;

    typedef typename base::pointer_type pointer_type;
    typedef typename base::value_type   value_type;
    typedef typename base::size_type    size_type;
    typedef typename base::index_type   index_type;

    typedef allocator                   allocator_type;

    using base::storage;
    using base::storage_size;

    LIFT_HOST_DEVICE persistent_allocation()
        : base(), storage_capacity(0)
    { }

    persistent_allocation(size_type count)
        : base(), storage_capacity(0)
    {
        resize(count);
    }

    LIFT_HOST_DEVICE persistent_allocation(const persistent_allocation& other)
        : base(), storage_capacity(0)
    {
        *this = other;
    }

    // initializer list semantics: make a copy of the contents
    persistent_allocation(const std::initializer_list<value_type>& l)
        : base(), storage_capacity(0)
    {
        *this = l;
    }

    LIFT_HOST_DEVICE persistent_allocation& operator=(const persistent_allocation& other)
    {
        storage = other.storage;
        storage_size = other.storage_size;
        storage_capacity = other.storage_capacity;

        return *this;
    }

    // initializer list semantics: make a copy of the contents
    persistent_allocation& operator=(const std::initializer_list<value_type>& l)
    {
        resize(l.size());
        index_type offset = 0;
        for(auto i : l)
        {
            storage_write(offset, i);
            offset++;
        }

        return *this;
    }

    virtual void resize(size_type count) override
    {
        if (count <= storage_capacity)
        {
            storage_size = count;
            return;
        }

        size_type old_storage_size;
        pointer_type old_storage;

        old_storage = storage;
        old_storage_size = storage_size;

        storage = (pointer_type) allocator_type().allocate(sizeof(value_type) * count);

        if (old_storage != nullptr)
        {
            base::device_memory_copy((void *)storage, old_storage, sizeof(value_type) * std::min(count, old_storage_size));
            allocator_type().deallocate((pointer_type)old_storage);
        }

        storage_size = count;
        storage_capacity = count;
    }

    void reserve(size_type count)
    {
        if (count <= storage_capacity)
        {
            return;
        }

        pointer_type old_storage = storage;

        storage = (pointer_type) allocator_type().allocate(sizeof(value_type) * count);

        if (old_storage != nullptr)
        {
            base::device_memory_copy((void *)storage, old_storage, sizeof(value_type) * storage_size);
            allocator_type().deallocate((pointer_type)old_storage);
        }

        storage_capacity = count;
    }

    void shrink_to_fit(void)
    {
        if (storage_size == storage_capacity)
        {
            return;
        }

        size_type old_storage_capacity;
        pointer_type old_storage;

        old_storage = storage;
        old_storage_capacity = storage_capacity;

        storage = (pointer_type) allocator_type().allocate(sizeof(value_type) * storage_size);

        base::device_memory_copy((void *)storage, old_storage, sizeof(value_type) * storage_size);
        allocator_type().deallocate((pointer_type)old_storage);

        storage_capacity = storage_size;
    }

    // release the memory allocation
    virtual void free(void) override
    {
        if (storage)
        {
            allocator_type().deallocate(storage);
        }

        storage = nullptr;
        storage_size = 0;
        storage_capacity = 0;
    }

    void push_back(const value_type& value)
    {
        resize(storage_size + 1);
        storage[storage_size - 1] = value_type(value);
    }

    void clear(void)
    {
        storage_size = 0;
    }

protected:
    size_t storage_capacity;
};

} // namespace lift

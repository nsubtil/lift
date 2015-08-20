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

#include <algorithm>

#include "default_allocator.h"
#include "thrust_wrappers.h"
#include "const_memory_pointer.h"
#include "type_assignment_checks.h"

namespace lift {

// tagged mutable memory pointer interface
// exposes some of the std::vector interface, except for anything where the semantics differ from a plain pointer
template <target_system system,
          typename T,
          typename _index_type = uint32,
          typename allocator = typename default_memory_allocator<system>::type>
struct memory_pointer
{
    enum {
        system_tag = system,
    };

    enum {
        mutable_tag = 1,
    };

    typedef T                                          value_type;
    typedef _index_type                                index_type;
    typedef allocator                                  allocator_type;
    typedef index_type                                 size_type;
    typedef index_type                                 difference_type;
    typedef T&                                         reference;
    typedef const T&                                   const_reference;
    typedef T*                                         pointer;
    typedef const T*                                   const_pointer;
    typedef T*                                         iterator;
    typedef const T*                                   const_iterator;

    // thrust-compatible iterators
    typedef thrust_iterator_adaptor<system, value_type, iterator>         thrust_iterator;
    typedef thrust_iterator_adaptor<system, value_type, const_iterator>   thrust_const_iterator;

    // default ctor makes an invalid pointer
    LIFT_HOST_DEVICE memory_pointer()
        : storage(nullptr), storage_size(0)
    { }

    template <typename pointer_type>
    LIFT_HOST_DEVICE memory_pointer(pointer_type storage, size_type storage_size)
    {
        memory::check_value_type_assignment_compatible<value_type, std::remove_reference<decltype(*storage)>::type>();
        this->storage = (pointer) storage;
        this->storage_size = storage_size;
    }

    // size ctor resizes the pointer
    memory_pointer(size_type count)
        : storage(nullptr), storage_size(0)
    {
        resize(count);
    }

    // pointer semantics: copy ctor makes a copy of a pointer
    // note that if the allocators differ, resize() will break
    template <typename other_T, typename other_allocator>
    LIFT_HOST_DEVICE memory_pointer(const memory_pointer<system, other_T, index_type, other_allocator>& other)
        : storage(other.data()), storage_size(other.size())
    { }

    // initializer list semantics: make a copy of the contents
    memory_pointer(const std::initializer_list<value_type>& l)
        : storage(nullptr), storage_size(0)
    {
        resize(l.size());
        index_type offset = 0;
        for(auto i : l)
        {
            storage_write(offset, i);
            offset++;
        }
    }

    // pointer semantics: make a copy of a pointer
    // create a bad pointer when assigning across systems
    template <target_system other_system, typename other_T, typename other_allocator>
    LIFT_HOST_DEVICE memory_pointer& operator=(const memory_pointer<other_system, other_T, index_type, other_allocator>& other)
    {
        if (other_system != system)
        {
            storage = nullptr;
            storage_size = 0;
        } else {
            storage = other.data();
            storage_size = other.size();
        }

        return *this;
    }

    // initializer list semantics: make a copy of the contents
    memory_pointer& operator=(const std::initializer_list<value_type>& l)
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

    LIFT_HOST_DEVICE iterator begin()
    {
        return iterator(storage);
    }

    LIFT_HOST_DEVICE const_iterator begin() const
    {
        return const_iterator(storage);
    }

    LIFT_HOST_DEVICE const_iterator cbegin() const
    {
        return const_iterator(storage);
    }

    LIFT_HOST_DEVICE iterator end()
    {
        return iterator(storage + storage_size);
    }

    LIFT_HOST_DEVICE const_iterator end() const
    {
        return const_iterator(storage + storage_size);
    }

    LIFT_HOST_DEVICE const_iterator cend() const
    {
        return const_iterator(storage + storage_size);
    }

    // thrust-compatible iterators
    LIFT_HOST_DEVICE thrust_iterator t_begin()
    {
        return thrust_iterator(storage);
    }

    LIFT_HOST_DEVICE thrust_const_iterator t_begin() const
    {
        return thrust_const_iterator(storage);
    }

    LIFT_HOST_DEVICE thrust_iterator t_end()
    {
        return thrust_iterator(storage + storage_size);
    }

    LIFT_HOST_DEVICE thrust_const_iterator t_end() const
    {
        return thrust_const_iterator(storage + storage_size);
    }

    // TODO: reverse iterators?

    LIFT_HOST_DEVICE size_type size() const
    {
        return storage_size;
    }

    // TODO: reserve/max_size/capacity interface

    LIFT_HOST_DEVICE bool empty() const
    {
        return storage_size == 0;
    }

#if LIFT_DEVICE_COMPILATION
    // device accessors and iterators

    // note: at() does not throw on the device
    // bounds checking is not performed in GPU code
    LIFT_DEVICE reference at(size_type pos)
    {
        return storage[pos];
    }

    LIFT_DEVICE const_reference at(size_type pos) const
    {
        return storage[pos];
    }

    LIFT_DEVICE reference operator[] (size_type pos)
    {
        return storage[pos];
    }

    LIFT_DEVICE const_reference operator[] (size_type pos) const
    {
        return storage[pos];
    }

    LIFT_DEVICE reference front()
    {
        return storage[0];
    }

    LIFT_DEVICE const_reference front() const
    {
        return storage[0];
    }

    LIFT_DEVICE reference back()
    {
        return storage[storage_size - 1];
    }

    LIFT_DEVICE const_reference back() const
    {
        return storage[storage_size - 1];
    }
#else
    // note: accessor methods on the host return a value, not a reference
    LIFT_HOST value_type at(size_type pos) const
    {
        return storage_read(pos);
    }

    LIFT_HOST value_type operator[] (size_type pos) const
    {
        return storage_read(pos);
    }

    // we don't implement front() or back() on the host
#endif

    LIFT_HOST_DEVICE pointer data()
    {
        return storage;
    }

    LIFT_HOST_DEVICE const_pointer data() const
    {
        return storage;
    }

    void resize(size_type count)
    {
        size_type old_storage_size;
        pointer old_storage;

        old_storage = storage;
        old_storage_size = storage_size;

        storage = (pointer) allocator_type().allocate(sizeof(value_type) * count);

        if (old_storage != nullptr)
        {
            device_memory_copy((void *)storage, old_storage, sizeof(value_type) * std::min(count, old_storage_size));
            allocator_type().deallocate((pointer)old_storage);
        }

        storage_size = count;
    }

    // cross-memory-space copy from another pointer
    // note that this does not handle copies across different GPUs
    template <typename other_pointer>
    void copy(const other_pointer& other)
    {
        memory::check_value_type_assignment_compatible<value_type, typename other_pointer::value_type>();

        resize(other.size());

        if (system == cuda)
        {
            // copying to GPU...
            if (target_system(other_pointer::system_tag) == cuda)
            {
                // ... from the GPU
                cudaMemcpy((void *) data(), other.data(), sizeof(value_type) * other.size(), cudaMemcpyDeviceToDevice);
            } else {
                // ... from the host
                cudaMemcpy((void *) data(), other.data(), sizeof(value_type) * other.size(), cudaMemcpyHostToDevice);
            }
        } else {
            // copying to host...
            if (target_system(other_pointer::system_tag) == cuda)
            {
                // ... from the GPU
                cudaMemcpy((void *) data(), other.data(), sizeof(value_type) * other.size(), cudaMemcpyDeviceToHost);
            } else {
                // ... from the host
                memcpy((void *) data(), other.data(), sizeof(value_type) * other.size());
            }
        }
    }

#if 0
    // memory transfer from another identical pointer
    // this is equivalent to a copy if the target systems differ and a pointer assignment if they match
    // note that unlike copy(), the pointer types here must also match in value_type constness
    template <typename other_pointer>
    void transfer(const other_pointer& other)
    {
        check_value_type_assignment_compatible<value_type, typename other_pointer::value_type>();

        if (system == target_system(other_pointer::system_tag))
        {
            storage = (pointer) other.data();
            storage_size = other.size();
        } else {
            copy(other);
        }
    }
#endif

    // release the memory that this pointer points to
    void free(void)
    {
        if (storage)
        {
            allocator_type().deallocate(storage);
        }

        storage = nullptr;
        storage_size = 0;
    }

    // conversion to const_memory_pointer
    LIFT_HOST_DEVICE operator const_memory_pointer<system, value_type, index_type>() const
    {
        return const_memory_pointer<system, value_type, index_type>(*this);
    }

protected:
    // these are generally assumed to be slow operations
    value_type storage_read(size_type pos) const
    {
        if (system == cuda)
        {
            value_type v = value_type();
            cudaMemcpy((void *) &v, &storage[pos], sizeof(value_type), cudaMemcpyDeviceToHost);
            return v;
        } else {
            return storage[pos];
        }
    }

    void storage_write(size_type pos, const value_type value)
    {
        if (system == cuda)
        {
            cudaMemcpy(&storage[pos], &value, sizeof(value_type), cudaMemcpyHostToDevice);
        } else {
            storage[pos] = value;
        }
    }

    void device_memory_copy(void *dst, const void *src, size_t size)
    {
        if (system == cuda)
        {
            cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
        } else {
            memcpy(dst, src, size);
        }
    }

    pointer storage;
    size_type storage_size;
};

#if 0
// a memory pointer that releases memory when destroyed
template <target_system system,
          typename T,
          typename index_type = uint32,
          typename Allocator = typename default_memory_allocator<system, T, index_type>::type>
struct scoped_memory_pointer : public memory_pointer<system, T, index_type, Allocator>
{
    typedef memory_pointer<system, T, index_type, Allocator>        base;
    // "inherit" the allocator_type to ease syntax below
    typedef typename base::allocator_type                           allocator_type;

    using base::base;

    scoped_memory_pointer() = default;

    // scoped_memory_pointer owns memory
    // delete all pointer semantic ctors and operators
    template <typename other_T, typename other_allocator>
    scoped_memory_pointer(const memory_pointer<system, other_T, index_type, other_allocator>& other) = delete;
    template <typename other_allocator>
    scoped_memory_pointer(const memory_pointer<system, typename base::value_type, typename base::index_type, other_allocator>& other) = delete;

    template <target_system other_system, typename other_T, typename other_allocator>
    base& operator=(const memory_pointer<other_system, other_T, typename base::index_type, other_allocator>& other) = delete;

    // delete all operators that manipulate memory directly
    template <target_system other_system, typename other_allocator>
    void transfer(const memory_pointer<other_system, typename base::value_type, typename base::index_type, other_allocator>& other) = delete;
    void free(void) = delete;

    ~scoped_memory_pointer()
    {
        base::free();
    }
};
#endif

} // namespace lift

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

#include "type_assignment_checks.h"

namespace lift {

// tagged memory pointer interface, const version
// exposes some of the std::vector interface, except for anything that allocates memory
// or anything where the semantics differ from a plain pointer
template <target_system system,
          typename T,
          typename _index_type = uint32>
struct const_memory_pointer
{
    enum {
        system_tag = system,
    };

    enum {
        mutable_tag = 0,
    };

    typedef const T                                    value_type;
    typedef _index_type                                index_type;
    typedef index_type                                 size_type;
    typedef index_type                                 difference_type;
    typedef const T&                                   const_reference;
    typedef const T*                                   const_pointer;
    typedef const T*                                   const_iterator;

    // thrust-compatible iterators
    typedef thrust_iterator_adaptor<system, value_type, const_iterator>   thrust_const_iterator;

    LIFT_HOST_DEVICE const_memory_pointer()
        : storage(nullptr), storage_size(0)
    { }

    template <target_system other_system>
    LIFT_HOST_DEVICE const_memory_pointer(const const_memory_pointer<other_system, value_type, index_type>& other)
    {
        if (system == other_system)
        {
            storage = other.data();
            storage_size = other.size();
        } else {
            // create a bad pointer when assigning across systems
            storage = nullptr;
            storage_size = 0;
        }
    }

    // cross-space execution warnings will trigger if pointer is a scoped_memory_pointer
    // this is expected, so disable the warning here
    __lift_hd_warning_disable__
    template <typename pointer>
    LIFT_HOST_DEVICE const_memory_pointer(const pointer& other)
    {
        if (system == target_system(pointer::system_tag))
        {
            storage = other.data();
            storage_size = other.size();
        } else {
            storage = nullptr;
            storage_size = 0;
        }
    }

    template <target_system other_system>
    LIFT_HOST_DEVICE const_memory_pointer& operator=(const const_memory_pointer<other_system, value_type, index_type>& other)
    {
        if (system == other_system)
        {
            storage = other.data();
            storage_size = other.size();
        } else {
            // create a bad pointer when assigning across systems
            storage = nullptr;
            storage_size = 0;
        }

        return *this;
    }

    LIFT_HOST_DEVICE const_iterator begin() const
    {
        return const_iterator(storage);
    }

    LIFT_HOST_DEVICE const_iterator end() const
    {
        return const_iterator(storage + storage_size);
    }

    LIFT_HOST_DEVICE const_iterator cbegin() const
    {
        return const_iterator(storage);
    }

    LIFT_HOST_DEVICE const_iterator cend() const
    {
        return const_iterator(storage + storage_size);
    }

    // thrust-compatible iterators
    LIFT_HOST_DEVICE thrust_const_iterator t_begin() const
    {
        return thrust_const_iterator(storage);
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
    LIFT_DEVICE const_reference at(size_type pos) const
    {
        return storage[pos];
    }

    LIFT_DEVICE const_reference operator[] (size_type pos) const
    {
        return storage[pos];
    }

    LIFT_DEVICE const_reference front() const
    {
        return storage[0];
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

    LIFT_HOST_DEVICE const_pointer data() const
    {
        return storage;
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

    const_pointer storage;
    size_type storage_size;
};

} // namespace lift

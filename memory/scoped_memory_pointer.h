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

#include "memory_pointer.h"

namespace lift {

// a memory pointer that releases memory when destroyed
// this is only implemented for the host; it converts to the equivalent memory_pointer for kernels
template <target_system system,
          typename T,
          typename index_type = uint32,
          typename allocator = typename default_memory_allocator<system>::type>
class scoped_memory_pointer
{
public:

    typedef memory_pointer<system, T, index_type, allocator>        memptr;
    typedef const_memory_pointer<system, T, index_type>             const_memptr;

    enum {
        system_tag = system,
    };

    scoped_memory_pointer()
        : storage(memptr())
    { }

    // size ctor resizes the pointer
    scoped_memory_pointer(typename memptr::size_type count)
        : storage(count)
    { }

    // initializer list semantics: make a copy of the contents
    scoped_memory_pointer(const std::initializer_list<typename memptr::value_type>& l)
        : storage(l)
    { }

    // initializer list semantics: make a copy of the contents
    scoped_memory_pointer& operator=(const std::initializer_list<typename memptr::value_type>& l)
    {
        storage = l;
        return *this;
    }

    // no assignment operator
    scoped_memory_pointer& operator=(const scoped_memory_pointer& other) = delete;

    ~scoped_memory_pointer()
    {
        storage.free();
    }

    typename memptr::iterator begin()
    {
        return storage.begin();
    }

    typename memptr::const_iterator begin() const
    {
        return storage.begin();
    }

    typename memptr::const_iterator cbegin() const
    {
        return storage.cbegin();
    }

    typename memptr::iterator end()
    {
        return storage.end();
    }

    typename memptr::const_iterator end() const
    {
        return storage.end();
    }

    typename memptr::const_iterator cend() const
    {
        return storage.cend();
    }

    typename memptr::thrust_iterator t_begin()
    {
        return storage.t_begin();
    }

    typename memptr::thrust_const_iterator t_begin() const
    {
        return storage.t_begin();
    }

    typename memptr::thrust_iterator t_end()
    {
        return storage.t_end();
    }

    typename memptr::thrust_const_iterator t_end() const
    {
        return storage.t_end();
    }

    // TODO: reverse iterators?

    typename memptr::size_type size() const
    {
        return storage.size();
    }

    // TODO: reserve/max_size/capacity interface

    bool empty() const
    {
        return storage.empty();
    }

    // note: accessor methods on the host return a value, not a reference
    typename memptr::value_type at(typename memptr::size_type pos) const
    {
        return storage.at(pos);
    }

    typename memptr::value_type operator[] (typename memptr::size_type pos) const
    {
        return storage[pos];
    }

    // we don't implement front() or back() on the host

    typename memptr::pointer data()
    {
        return storage.data();
    }

    typename memptr::const_pointer data() const
    {
        return storage.data();
    }

    void resize(typename memptr::size_type count)
    {
        storage.resize(count);
    }

    template <typename other_pointer>
    void copy(const other_pointer& other)
    {
        storage.copy(other);
    }

    // convert to memory_pointer
    operator memptr()
    {
        return storage;
    }

    operator const_memptr() const
    {
        const_memptr ret;
        ret = storage;
        return ret;
    }

private:
    memptr storage;
};

} // namespace lift

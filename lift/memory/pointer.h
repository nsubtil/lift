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

#include <cuda_runtime.h>

#include <initializer_list>

#include <lift/types.h>
#include <lift/backends.h>
#include <lift/decorators.h>

#include <lift/memory/thrust_wrappers.h>
#include <lift/memory/type_assignment_checks.h>

namespace lift {

/**
 * Base class for tagged memory pointer implementation. Contains most of the
 * common functionality for pointers that is not system-specific.
 *
 * \tparam system       Defines the target memory space for this pointer
 * \tparam T            The underlying data type the pointer points at
 * \tparam _index_type  Integral type used in offset calculations. Most GPU
 *                      architectures lack full hardware support for 64-bit
 *                      integer math, and so can benefit from having a 32-bit
 *                      index even when building 64-bit code.
 */
template <target_system system,
          typename T,
          typename _index_type>
struct tagged_pointer_base
{
    enum {
        system_tag = system,
    };

    typedef T                                          value_type;
    typedef const T                                    const_value_type;
    typedef _index_type                                index_type;
    typedef index_type                                 size_type;
    typedef index_type                                 difference_type;
    typedef T&                                         reference_type;
    typedef const T&                                   const_reference_type;
    typedef T*                                         pointer_type;
    typedef const T*                                   const_pointer_type;
    typedef T*                                         iterator_type;
    typedef const T*                                   const_iterator_type;

    /// Thrust-compatible iterator types
    typedef thrust_iterator_adaptor<system,       value_type, iterator_type>         thrust_iterator_type;
    typedef thrust_iterator_adaptor<system, const value_type, const_iterator_type>   thrust_const_iterator_type;

    /// The default constructor initializes the pointer to null
    LIFT_HOST_DEVICE tagged_pointer_base()
        : storage(nullptr), storage_size(0)
    { }

    /// Construct a tagged pointer from a raw pointer
    LIFT_HOST_DEVICE tagged_pointer_base(T *storage, size_type storage_size)
       : storage(storage), storage_size(storage_size)
    { }

    /// Copy constructor creates a copy of the pointer.
    /// If the target system mismatches, however, a null pointer is created.
    template <target_system other_system, typename other_value_type>
    LIFT_HOST_DEVICE tagged_pointer_base(tagged_pointer_base<other_system, other_value_type, index_type>& other)
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

    template <typename other_pointer>
    LIFT_HOST_DEVICE tagged_pointer_base(other_pointer& other)
    {
        if (system == target_system(other_pointer::system_tag))
        {
            storage = other.data();
            storage_size = other.size();
        } else {
            storage = nullptr;
            storage_size = 0;
        }
    }

    /// Copy constructor creates a copy of the pointer.
    /// If the target system mismatches, however, a null pointer is created.
    template <target_system other_system, typename other_value_type>
    LIFT_HOST_DEVICE tagged_pointer_base& operator=(tagged_pointer_base<other_system, other_value_type, index_type>& other)
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

    // cross-memory-space copy from another pointer
    // note that this does not handle copies across different GPUs
    template <target_system other_system, typename other_value_type>
    LIFT_HOST_DEVICE void copy(const tagged_pointer_base<other_system, other_value_type, index_type>& other)
    {
        __internal::check_value_type_assignment_compatible<value_type, other_value_type>();

        // Make sure enough space for copy on destination
        assert(storage_size >= other.size());

        if (system == cuda)
        {
            // copying to GPU...
            if (other_system == cuda)
            {
                // ... from the GPU
                cudaMemcpy((void *) data(), other.data(), sizeof(value_type) * other.size(), cudaMemcpyDeviceToDevice);
            } else {
                // ... from the host
                cudaMemcpy((void *) data(), other.data(), sizeof(value_type) * other.size(), cudaMemcpyHostToDevice);
            }
        } else {
            // copying to host...
            if (other_system == cuda)
            {
                // ... from the GPU
                cudaMemcpy((void *) data(), other.data(), sizeof(value_type) * other.size(), cudaMemcpyDeviceToHost);
            } else {
                // ... from the host
                memcpy((void *) data(), other.data(), sizeof(value_type) * other.size());
            }
        }
    }

    LIFT_HOST_DEVICE const_reference_type at(size_type pos) const
    {
        return storage[pos];
    }

    LIFT_HOST_DEVICE reference_type at(size_type pos)
    {
        return storage[pos];
    }

    LIFT_HOST_DEVICE const_reference_type operator[] (size_type pos) const
    {
        return storage[pos];
    }

    LIFT_HOST_DEVICE reference_type operator[] (size_type pos)
    {
        return storage[pos];
    }

    LIFT_HOST_DEVICE const_reference_type front() const
    {
        return storage[0];
    }

    LIFT_HOST_DEVICE reference_type front()
    {
        return storage[0];
    }

    LIFT_HOST_DEVICE const_reference_type back() const
    {
        return &storage[storage_size - 1];
    }

    LIFT_HOST_DEVICE reference_type back()
    {
        return &storage[storage_size - 1];
    }

    LIFT_HOST_DEVICE const_iterator_type begin() const
    {
        return const_iterator_type(storage);
    }

    LIFT_HOST_DEVICE iterator_type begin()
    {
        return iterator_type(storage);
    }

    LIFT_HOST_DEVICE const_iterator_type end() const
    {
        return const_iterator_type(storage + storage_size);
    }

    LIFT_HOST_DEVICE iterator_type end()
    {
        return iterator_type(storage + storage_size);
    }

    LIFT_HOST_DEVICE const_iterator_type cbegin() const
    {
        return const_iterator_type(storage);
    }

    LIFT_HOST_DEVICE const_iterator_type cend() const
    {
        return const_iterator_type(storage + storage_size);
    }

    /// Returns a Thrust-compatible iterator pointing at the base address of the pointer
    LIFT_HOST_DEVICE thrust_const_iterator_type t_begin() const
    {
        return thrust_const_iterator_type(storage);
    }

    /// Returns a Thrust-compatible iterator pointing at the base address of the pointer
    LIFT_HOST_DEVICE thrust_iterator_type t_begin()
    {
        return thrust_iterator_type(storage);
    }

    /// Returns a Thrust-compatible iterator pointing at the end of the memory region
    /// covered by this pointer object
    LIFT_HOST_DEVICE thrust_const_iterator_type t_end() const
    {
        return thrust_const_iterator_type(storage + storage_size);
    }

    /// Returns a Thrust-compatible iterator pointing at the end of the memory region
    /// covered by this pointer object
    LIFT_HOST_DEVICE thrust_iterator_type t_end()
    {
        return thrust_iterator_type(storage + storage_size);
    }

    // TODO: reverse iterators?

    LIFT_HOST_DEVICE size_type size() const
    {
        return storage_size;
    }

    LIFT_HOST_DEVICE bool empty() const
    {
        return storage_size == 0;
    }

    LIFT_HOST_DEVICE pointer_type data() const
    {
        return storage;
    }

protected:
    pointer_type storage;
    size_type storage_size;
};

/**
 * Lift's tagged pointer class.
 *
 * The type of a pointer includes a system tag, which identifies each pointer based on the
 * memory space it belongs to. When dereferencing the pointer, the type of the index is also
 * selectable at compile-time. This is because GPU implementations can benefit from 32-bit
 * indexing even in 64-bit mode, as certain index calculations can be slow.
 *
 * Lift pointers are sized (meaning they track the amount of memory they point at), but
 * no bounds-checking is performed. They implement C-style pointer semantics, allowing
 * assignment from compatible pointer types as well as pointer arithmetic.
 *
 * Note that despite the fact that pointers are tagged with the memory space they belong to,
 * Lift does not directly handle cross-device dereferencing in the pointer dereference operator.
 * Directly dereferencing a GPU pointer on the CPU (or vice-versa) yields undefined behavior.
 * Lift pointers provide an explicit interface for cross-memory-space reads/writes
 * (peek() / poke()).
 *
 */
template <target_system system,
          typename T,
          typename _index_type = uint32>
struct pointer : public tagged_pointer_base<system, T, _index_type>
{
// the following declarations are meant for documenting the interface
// all of these exist in template specializations only
#if DOXYGEN_ONLY
    typedef tagged_pointer_base<host, T, _index_type>   base;

    typedef typename base::reference_type               reference_type;
    typedef typename base::const_reference_type         const_reference_type;
    typedef typename base::value_type                   value_type;
    typedef typename base::index_type                   index_type;
    typedef typename base::size_type                    size_type;
    typedef typename base::iterator_type                iterator_type;
    typedef typename base::const_iterator_type          const_iterator_type;

    using base::base;

    // return a pointer to a memory range within this pointer
    LIFT_HOST_DEVICE pointer range(const size_type offset, size_type len = size_type(-1)) const;

    // pointer arithmetic
    // note that we don't do any bounds checking
    LIFT_HOST_DEVICE pointer operator+(off_t offset) const;
    LIFT_HOST_DEVICE pointer operator-(off_t offset) const;
    // return a truncated pointer
    LIFT_HOST_DEVICE pointer truncate(size_t new_size);

    // read a value behind this memory pointer
    // note: this is slow for cuda pointers!
    value_type peek(index_type pos);

    // poke a value behind this memory pointer
    // note: this is slow for cuda pointers!
    void poke(index_type pos, const value_type value);
#endif
};

/**
 * CPU version of pointer.
 */
template <typename T,
          typename _index_type>
struct pointer<host, T, _index_type> : public tagged_pointer_base<host, T, _index_type>
{
    typedef tagged_pointer_base<host, T, _index_type>   base;

    typedef typename base::reference_type               reference_type;
    typedef typename base::const_reference_type         const_reference_type;
    typedef typename base::value_type                   value_type;
    typedef typename base::index_type                   index_type;
    typedef typename base::size_type                    size_type;
    typedef typename base::iterator_type                iterator_type;
    typedef typename base::const_iterator_type          const_iterator_type;

    using base::base;

    // return a pointer to a memory range within this pointer
    LIFT_HOST_DEVICE pointer range(const size_type offset, size_type len = size_type(-1)) const
    {
        pointer ret;
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
    LIFT_HOST_DEVICE pointer operator+(off_t offset) const
    {
        pointer ret;
        ret.storage = base::storage + offset;
        ret.storage_size = base::storage_size - offset;

        return ret;
    }

    LIFT_HOST_DEVICE pointer operator-(off_t offset) const
    {
        pointer ret;
        ret.storage = base::storage - offset;
        ret.storage_size = base::storage_size + offset;

        return ret;
    }

    // return a truncated pointer
    LIFT_HOST_DEVICE pointer truncate(size_t new_size)
    {
        pointer ret;
        ret.storage = base::storage;
        ret.storage_size = new_size;

        return ret;
    }

    // read a value behind this memory pointer
    // note: this is slow for cuda pointers!
    value_type peek(index_type pos) const
    {
        return base::storage[pos];
    }

    // poke a value behind this memory pointer
    // note: this is slow for cuda pointers!
    void poke(index_type pos, const value_type value)
    {
        base::storage[pos] = value;
    }
};

template <typename T,
          typename _index_type>
struct pointer<cuda, T, _index_type> : public tagged_pointer_base<cuda, T, _index_type>
{
    typedef tagged_pointer_base<cuda, T, _index_type>   base;

    typedef typename base::reference_type               reference_type;
    typedef typename base::const_reference_type         const_reference_type;
    typedef typename base::value_type                   value_type;
    typedef typename base::index_type                   index_type;
    typedef typename base::size_type                    size_type;
    typedef typename base::iterator_type                iterator_type;
    typedef typename base::const_iterator_type          const_iterator_type;

    using base::base;

    // return a pointer to a memory range within this pointer
    LIFT_HOST_DEVICE pointer range(const size_type offset, size_type len = size_type(-1)) const
    {
        pointer ret;
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
    LIFT_HOST_DEVICE pointer operator+(off_t offset) const
    {
        pointer ret;
        ret.storage = base::storage + offset;
        ret.storage_size = base::storage_size - offset;

        return ret;
    }

    LIFT_HOST_DEVICE pointer operator-(off_t offset) const
    {
        pointer ret;
        ret.storage = base::storage - offset;
        ret.storage_size = base::storage_size + offset;

        return ret;
    }

    // return a truncated pointer
    LIFT_HOST_DEVICE pointer truncate(size_t new_size)
    {
        pointer ret;
        ret.storage = base::storage;
        ret.storage_size = new_size;

        return ret;
    }

    // read a value behind this memory pointer
    // note: this is slow!
    value_type peek(const size_type pos) const
    {
        return storage_read(pos);
    }

    // poke a value behind this memory pointer
    // note: this is slow!
    void poke(const size_type pos, const value_type value)
    {
        cudaMemcpy(&base::storage[pos], &value, sizeof(value_type), cudaMemcpyHostToDevice);
    }

    // shortcut for peek, intended mostly for debug code
    // note: this is slow!
    value_type operator() (const index_type idx)
    {
        return peek(idx);
    }

protected:
    // this is slow!
    value_type storage_read(size_type pos) const
    {
        value_type v = value_type();
        cudaMemcpy((void *) &v, &base::storage[pos], sizeof(value_type), cudaMemcpyDeviceToHost);
        return v;
    }
};

} // namespace lift

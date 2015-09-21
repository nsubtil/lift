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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#if ENABLE_TBB_BACKEND
#include <thrust/system/tbb/vector.h>
#endif

#include "types.h"
#include "backends.h"
#include "decorators.h"

namespace lift {

// backend vector type chooser
// for now we use thrust, need to reimplement
template <target_system system, typename T>
struct backend_vector_type
{ };

template <typename T>
struct backend_vector_type<host, T>
{
    typedef thrust::host_vector<T> base_vector_type;
};

template <typename T>
struct backend_vector_type<cuda, T>
{
    typedef thrust::device_vector<T> base_vector_type;
};

template <typename T, typename IndexType = uint64>
struct vector_view
{
    // note: vector_view is *not* a container, it wraps a pointer to T

    // vector_view implements a subset of the std::vector interface that can be used on both host and device
    // the following types/methods from std::vector are *not* present
    //
    // allocator_type
    // operator=()
    // resize()
    // reserve()
    // shrink_to_fit()
    // assign()
    // push_back()
    // pop_back()
    // insert()
    // erase()
    // swap()
    // clear()
    // emplace()
    // emplace_back()
    // get_allocator()

    typedef T*                                                          iterator;
    typedef const T*                                                    const_iterator;
    typedef typename thrust::iterator_traits<iterator>::value_type      value_type;
    typedef typename thrust::iterator_traits<iterator>::reference       reference;
    typedef typename thrust::iterator_traits<const_iterator>::reference const_reference;
    typedef typename thrust::iterator_traits<iterator>::pointer         pointer;
    typedef typename thrust::iterator_traits<const_iterator>::pointer   const_pointer;
    typedef typename thrust::reverse_iterator<iterator>                 reverse_iterator;
    typedef typename thrust::reverse_iterator<const_iterator>           const_reverse_iterator;
    typedef typename thrust::iterator_traits<iterator>::difference_type difference_type;
    typedef IndexType                                                   size_type;

    CUDA_HOST_DEVICE vector_view()
        : m_vec(nullptr), m_size(0)
    { }

    CUDA_HOST_DEVICE vector_view(iterator vec, size_type size)
        : m_vec(vec), m_size(size)
    { }

    CUDA_HOST_DEVICE iterator begin()
    {
        return iterator(m_vec);
    }

    CUDA_HOST_DEVICE const_iterator begin() const
    {
        return const_iterator(m_vec);
    }

    CUDA_HOST_DEVICE const_iterator cbegin() const
    {
        return const_iterator(m_vec);
    }

    CUDA_HOST_DEVICE iterator end()
    {
        return iterator(m_vec + m_size);
    }

    CUDA_HOST_DEVICE const_iterator end() const
    {
        return const_iterator(m_vec + m_size);
    }


    CUDA_HOST_DEVICE const_iterator cend() const
    {
        return const_iterator(m_vec + m_size);
    }

    CUDA_HOST_DEVICE reverse_iterator rbegin()
    {
        return reverse_iterator(end());
    }

    CUDA_HOST_DEVICE const_reverse_iterator rbegin() const
    {
        return const_reverse_iterator(end());
    }

    CUDA_HOST_DEVICE const_reverse_iterator crbegin() const
    {
        return const_reverse_iterator(end());
    }

    CUDA_HOST_DEVICE reverse_iterator rend()
    {
        return reverse_iterator(begin());
    }

    CUDA_HOST_DEVICE const_reverse_iterator rend() const
    {
        return const_reverse_iterator(begin());
    }

    CUDA_HOST_DEVICE const_reverse_iterator crend() const
    {
        return const_reverse_iterator(begin());
    }

    CUDA_HOST_DEVICE size_type size() const
    {
        return m_size;
    }

    CUDA_HOST_DEVICE size_type max_size() const
    {
        return m_size;
    }

    CUDA_HOST_DEVICE size_type capacity() const
    {
        return m_size;
    }

    CUDA_HOST_DEVICE bool empty() const
    {
        return m_size == 0;
    }

    CUDA_HOST_DEVICE reference operator[](size_type n)
    {
        return m_vec[n];
    }

    CUDA_HOST_DEVICE const_reference operator[](size_type n) const
    {
        return m_vec[n];
    }

    CUDA_HOST_DEVICE reference at(size_type n)
    {
        return m_vec[n];
    }

    CUDA_HOST_DEVICE const_reference at(size_type n) const
    {
        return m_vec[n];
    }

    CUDA_HOST_DEVICE reference front()
    {
        return m_vec[0];
    }

    CUDA_HOST_DEVICE const_reference front() const
    {
        return m_vec[0];
    }

    CUDA_HOST_DEVICE reference back()
    {
        return m_vec[m_size - 1];
    }

    CUDA_HOST_DEVICE const_reference back() const
    {
        return m_vec[m_size - 1];
    }

    CUDA_HOST_DEVICE value_type* data() noexcept
    {
        return (value_type *)(m_vec);
    }

    CUDA_HOST_DEVICE const value_type* data() const noexcept
    {
        return (const value_type *)(m_vec);
    }

private:
    size_type m_size;
    iterator m_vec;
};

// our vector container
template <target_system system, typename T>
struct vector : public backend_vector_type<system, T>::base_vector_type
{
    typedef typename backend_vector_type<system, T>::base_vector_type base;
    using base::base;

    typedef vector_view<T> view;
    typedef vector_view<const T> const_view;

    operator view()
    {
        return view(base::size() ? thrust::raw_pointer_cast(base::data()) : nullptr,
                    base::size());
    }

    operator const_view() const
    {
        return const_view(base::size() ? thrust::raw_pointer_cast(base::data()) : nullptr,
                          base::size());
    }

    view range(off_t offset, size_t size = size_t(-1))
    {
        if (base::size() == 0)
        {
            return view(nullptr, 0);
        }

        if (size == size_t(-1))
        {
            size = base::size() - offset;
        }

        return view(thrust::raw_pointer_cast(base::data()) + offset, size);
    }

    const_view const_range(off_t offset, size_t size = size_t(-1)) const
    {
        if (base::size() == 0)
        {
            return const_view(nullptr, 0);
        }

        if (size == size_t(-1))
        {
            size = base::size() - offset;
        }

        return const_view(thrust::raw_pointer_cast(base::data()) + offset, size);
    }

    const_view range(off_t offset, size_t size = size_t(-1)) const
    {
        return const_range(offset, size);
    }

    // assignment from a host vector view
    void copy_from_view(const typename vector<host, T>::const_view& other)
    {
        base::assign(other.begin(), other.end());
    }
};

template <typename T> using d_vector = vector<cuda, T>;
template <typename T> using h_vector = vector<host, T>;

template <typename T, uint32 stride, typename IndexType = uint64>
struct strided_iterator
{
    typedef T*                                                          iterator;
    typedef const T*                                                    const_iterator;
    typedef typename thrust::iterator_traits<iterator>::value_type      value_type;
    typedef typename thrust::iterator_traits<iterator>::reference       reference;
    typedef typename thrust::iterator_traits<const_iterator>::reference const_reference;
    typedef typename thrust::iterator_traits<iterator>::pointer         pointer;
    typedef typename thrust::iterator_traits<const_iterator>::pointer   const_pointer;
    typedef typename thrust::reverse_iterator<iterator>                 reverse_iterator;
    typedef typename thrust::reverse_iterator<const_iterator>           const_reverse_iterator;
    typedef typename thrust::iterator_traits<iterator>::difference_type difference_type;
    typedef IndexType                                                   size_type;

    CUDA_HOST_DEVICE
    strided_iterator() = default;

    CUDA_HOST_DEVICE
    strided_iterator(T *base)
        : m_vec(base)
    { }

    CUDA_HOST_DEVICE inline size_type offset(size_type elem) const
    {
        return elem * stride;
    }

    CUDA_HOST_DEVICE reference operator[](size_type n)
    {
        return m_vec[offset(n)];
    }

    CUDA_HOST_DEVICE const_reference operator[](size_type n) const
    {
        return m_vec[offset(n)];
    }

    CUDA_HOST_DEVICE reference at(size_type n)
    {
        return m_vec[offset(n)];
    }

    CUDA_HOST_DEVICE const_reference at(size_type n) const
    {
        return m_vec[offset(n)];
    }

    T *m_vec;
};

} // namespace lift

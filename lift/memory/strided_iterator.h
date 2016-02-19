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

#include <iterator>

namespace lift {

template <typename T, uint32 stride, typename IndexType = uint64>
struct strided_iterator
{
    typedef T*                                                          iterator;
    typedef const T*                                                    const_iterator;
    typedef typename std::iterator_traits<iterator>::value_type         value_type;
    typedef typename std::iterator_traits<iterator>::reference          reference;
    typedef typename std::iterator_traits<const_iterator>::reference    const_reference;
    typedef typename std::iterator_traits<iterator>::pointer            pointer;
    typedef typename std::iterator_traits<const_iterator>::pointer      const_pointer;
    typedef typename std::reverse_iterator<iterator>                    reverse_iterator;
    typedef typename std::reverse_iterator<const_iterator>              const_reverse_iterator;
    typedef typename std::iterator_traits<iterator>::difference_type    difference_type;
    typedef IndexType                                                   size_type;

    LIFT_HOST_DEVICE
    strided_iterator() = default;

    LIFT_HOST_DEVICE
    strided_iterator(T *base)
        : m_vec(base)
    { }

    LIFT_HOST_DEVICE inline size_type offset(size_type elem) const
    {
        return elem * stride;
    }

    LIFT_HOST_DEVICE reference operator[](size_type n)
    {
        return m_vec[offset(n)];
    }

    LIFT_HOST_DEVICE const_reference operator[](size_type n) const
    {
        return m_vec[offset(n)];
    }

    LIFT_HOST_DEVICE reference at(size_type n)
    {
        return m_vec[offset(n)];
    }

    LIFT_HOST_DEVICE const_reference at(size_type n) const
    {
        return m_vec[offset(n)];
    }

    T *m_vec;
};

} // namespace lift

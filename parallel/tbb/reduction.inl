/*
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
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

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

namespace lift {

namespace __lift_tbb_reduction {

template <typename T>
struct parameters
{
    static constexpr uint32 sequential_threshold = 500000;
};

template <>
struct parameters<float>
{
    static constexpr uint32 sequential_threshold = 50000;
};

template <>
struct parameters<double>
{
    static constexpr uint32 sequential_threshold = 50000;
};

template <typename InputIterator, typename reduction_operator>
inline auto parallel_reduction(InputIterator first,
                               size_t len,
                               const typename std::iterator_traits<InputIterator>::value_type& initial_value,
                               reduction_operator reduction_op)
    -> typename std::iterator_traits<InputIterator>::value_type
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    // applies op sequentially across a tbb blocked_range
    // used as the range reduction operator for parallel_reduce
    auto range_reduction_operator =
            [&](const tbb::blocked_range<InputIterator>& range, const value_type& initial_value) -> value_type
            {
                value_type value = initial_value;

                for(value_type v : range)
                {
                    value = reduction_op(value, v);
                }

                return value;
            };

    return tbb::parallel_reduce(tbb::blocked_range<InputIterator>(first, first + len),
                                initial_value,
                                range_reduction_operator,
                                reduction_op);
}

template <typename InputIterator, typename reduction_operator>
inline auto sequential_reduction(InputIterator first,
                                 size_t len,
                                 const typename std::iterator_traits<InputIterator>::value_type& initial_value,
                                 reduction_operator op)
    -> typename std::iterator_traits<InputIterator>::value_type
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    value_type ret = initial_value;
    for(size_t i = 0; i < len; i++)
    {
        ret = op(ret, *(first + i));
    }

    return ret;
}

} // namespace __lift_tbb_reduction

template <>
template <typename InputIterator>
inline auto parallel<host>::sum(InputIterator first,
                                size_t len,
                                allocation<host, uint8>& temp_storage) -> typename std::iterator_traits<InputIterator>::value_type
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    auto sum_operator =
            [] (const value_type& a, const value_type& b) -> value_type
            {
                return a + b;
            };

    if (len <= __lift_tbb_reduction::parameters<value_type>::sequential_threshold)
    {
        return __lift_tbb_reduction::sequential_reduction(first, len, 0, sum_operator);
    } else {
        return __lift_tbb_reduction::parallel_reduction(first, len, 0, sum_operator);
    }
}

template <>
template <typename InputIterator>
inline auto parallel<host>::sum(InputIterator first,
                                size_t len,
                                vector<host, uint8>& temp_storage) -> typename std::iterator_traits<InputIterator>::value_type
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    auto sum_operator =
            [] (const value_type& a, const value_type& b) -> value_type
            {
                return a + b;
            };

    if (len <= __lift_tbb_reduction::parameters<value_type>::sequential_threshold)
    {
        return __lift_tbb_reduction::sequential_reduction(first, len, 0, sum_operator);
    } else {
        return __lift_tbb_reduction::parallel_reduction(first, len, 0, sum_operator);
    }
}

} // namespace lift

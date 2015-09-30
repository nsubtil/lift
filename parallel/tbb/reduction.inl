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

// below this many elements we prefer to do summation sequentially
constexpr uint32 sum_sequential_threshold = 10000;

template <typename InputIterator>
struct sum_operator
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;
    value_type value;

    sum_operator()
        : value(0)
    { }

    sum_operator(sum_operator&, tbb::split)
        : value(0)
    { }

    void operator() (const tbb::blocked_range<InputIterator>& r)
    {
        value_type temp = value;
        for(InputIterator a = r.begin(); a != r.end(); a++)
        {
            temp += *a;
        }

        value = temp;
    }

    void join(sum_operator& rhs)
    {
        value += rhs.value;
    }
};

} // namespace __lift_tbb_reduction

template <>
template <typename InputIterator>
inline auto parallel<host>::sum(InputIterator first,
                                size_t len,
                                allocation<host, uint8>& temp_storage) -> typename std::iterator_traits<InputIterator>::value_type
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    if (len < __lift_tbb_reduction::sum_sequential_threshold || true)
    {
        value_type v = 0;
        for(auto i = first; i < first + len; i++)
        {
            v += *i;
        }

        return v;
    } else {
        __lift_tbb_reduction::sum_operator<InputIterator> sum;
        tbb::parallel_reduce(tbb::blocked_range<InputIterator>(first, first + len), sum);
        return sum.value;
    }
}

template <>
template <typename InputIterator>
inline auto parallel<host>::sum(InputIterator first,
                                size_t len,
                                vector<host, uint8>& temp_storage) -> typename std::iterator_traits<InputIterator>::value_type
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    if (len < __lift_tbb_reduction::sum_sequential_threshold || true)
    {
        value_type v = 0;
        for(auto i = first; i < first + len; i++)
        {
            v += *i;
        }

        return v;
    } else {
        __lift_tbb_reduction::sum_operator<InputIterator> sum;
        tbb::parallel_reduce(tbb::blocked_range<InputIterator>(first, first + len), sum);
        return sum.value;
    }
}

} // namespace lift

/*
 * Lift
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015-2016, Nuno Subtil <subtil@gmail.com>
 * Copyright (c) 2015-2016, Roche Molecular Systems Inc.
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
#include <lift/test/test.h>
#include <lift/test/check.h>

#include <lift/memory.h>
#include <lift/backends.h>
#include <lift/parallel.h>
#include <lift/atomics.h>

using namespace lift;

constexpr uint32 vec_len = 10000;

template<target_system system, typename d_type>
struct p_add
{
    pointer<system, d_type> data;
    d_type amount;
    p_add(pointer<system, d_type> data, d_type amount)
        : data(data), amount(amount)
    { }

    LIFT_HOST_DEVICE void operator() (d_type &ind)
    {
        ind += amount;
    }
};

template<target_system system, typename d_type>
struct add
{
    pointer<system, d_type> data;
    d_type amount;

    add(pointer<system, d_type> data, d_type amount)
        : data(data), amount(amount)
    { }

    LIFT_HOST_DEVICE void operator() (const int index)
    {
       data[index] += amount;
    }
};

template <target_system system>
void for_each_end()
{
    int sub_num = 10;
    scoped_allocation<system, int> data(vec_len);
    parallel<system>::fill(data, 0);
    parallel<system>::for_each(data.size(), add<system, int>(data, sub_num));

    for (size_t i = 0; i < data.size(); i++)
    {
        LIFT_TEST_CHECK(data.peek(i) == sub_num)
    }
}
LIFT_TEST_FUNC(for_each_end_test, for_each_end);

template <target_system system>
void for_each_iter()
{
    int add_num = 10;
    scoped_allocation<system, int> data(vec_len);
    parallel<system>::fill(data, 0);
    parallel<system>::for_each(data.begin(), data.end(),
                               p_add<system, int>(data, add_num));

    for (size_t i = 0; i < data.size(); i++)
    {
        LIFT_TEST_CHECK(data.peek(i) == add_num)
    }
}
LIFT_TEST_FUNC(for_each_iter_test, for_each_iter);

template <target_system system>
void for_each_pointer()
{
    int add_num = 10;
    scoped_allocation<system, int> data(vec_len);
    parallel<system>::fill(data, 0);
    parallel<system>::for_each(data, p_add<system, int>(data, add_num));

    for (size_t i = 0; i < data.size(); i++)
    {
        LIFT_TEST_CHECK(data.peek(i) == add_num)
    }
}
LIFT_TEST_FUNC(for_each_pointer_test, for_each_pointer);

template <target_system system>
void for_each_range()
{
    int add_num = 10;
    scoped_allocation<system, int> data(vec_len);
    parallel<system>::fill(data, 0);
    parallel<system>::for_each({data.size()/2, data.size()},
                               add<system, int>(data, add_num));

    for (size_t i = 0; i < data.size()/2; i++)
    {
        LIFT_TEST_CHECK(data.peek(i) == 0)
    }
    for (size_t i = data.size()/2; i < data.size(); i++)
    {
        LIFT_TEST_CHECK(data.peek(i) == add_num)
    }
}
LIFT_TEST_FUNC(for_each_range_test, for_each_range);

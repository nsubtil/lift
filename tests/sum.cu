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
#include <lift/test/random.h>

#include <lift/memory.h>
#include <lift/parallel.h>
#include <lift/backends.h>

using namespace lift;

template <target_system system>
void sum_test(size_t size)
{
    scoped_allocation<system, int32> data(size);
    scoped_allocation<system, int32> result(size);
    scoped_allocation<system, uint8> temp;
    parallel<system>::fill(result, -1);

    test::rand_reset(1000);
    int32 exp_sum = 0;
    for (size_t i = 0; i < size; i++)
    {
        data.poke(i, test::rand_uniform<int32>(-100,100));
        exp_sum += data.peek(i);
    }

    int32 sum = parallel<system>::sum(data.begin(), data.size(), temp);
    LIFT_TEST_CHECK(sum == exp_sum);
}

#define SUM_TEST_GEN(__size__) \
    template<target_system system> \
    static void sum_test_##__size__##_run(void) { sum_test<system>(__size__); } \
    LIFT_TEST_FUNC(sum_test_##__size__, sum_test_##__size__##_run)

SUM_TEST_GEN(10);
SUM_TEST_GEN(100);
SUM_TEST_GEN(1000);
SUM_TEST_GEN(10000);


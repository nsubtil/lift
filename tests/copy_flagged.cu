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
#include <lift/test/test.h>
#include <lift/test/check.h>
#include <lift/test/random.h>

#include <lift/memory.h>
#include <lift/backends.h>
#include <lift/parallel.h>

using namespace lift;

template<target_system system>
void copy_flagged_test(size_t size)
{
    scoped_allocation<system, int32> data(size);
    scoped_allocation<system, int32> result(size);
    scoped_allocation<system, int32> expected_result(size);
    scoped_allocation<system, int8>  flags(size);
    scoped_allocation<system, uint8> temp;

    parallel<system>::fill(result, 0);
    parallel<system>::fill(expected_result, 0);

    test::rand_reset(1000);
    int iter = 0;
    for (size_t i = 0; i < size; i++)
    {
        flags.poke(i, test::rand_uniform<int8>(0,1));
        if (flags.peek(i) == 1)
        {
            expected_result.poke(iter, (i + 1));
            iter += 1;
        }
        data.poke(i, (i + 1));
    }

    parallel<system>::copy_flagged(data.begin(), data.size(), 
                                   result.begin(), flags.begin(), temp);

    for (size_t i = 0; i < size; i++)
    {
        LIFT_TEST_CHECK(result.peek(i) == expected_result.peek(i));
    }
}

#define COPY_FLAGGED_TEST_GEN(__size__) \
    template<target_system system> \
    static void copy_flagged_test_##__size__##_run(void) \
    { copy_flagged_test<system>(__size__); } \
    LIFT_TEST_FUNC(copy_flagged_test_##__size__, \
    copy_flagged_test_##__size__##_run)

COPY_FLAGGED_TEST_GEN(10);
COPY_FLAGGED_TEST_GEN(100);
COPY_FLAGGED_TEST_GEN(1000);
COPY_FLAGGED_TEST_GEN(10000);

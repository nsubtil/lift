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

#include <lift/parallel.h>
#include <lift/backends.h>
#include <lift/memory/scoped_allocation.h>

using namespace lift;

struct addition_predicate
{
    addition_predicate() {}
    LIFT_HOST_DEVICE int operator ()(const int l, const int r)
    {
        return (l + r);
    }
};

template <target_system system>
void inclusive_scan_test()
{
    constexpr uint32 size = 10000;
    scoped_allocation<system, int> data_in(size);
    scoped_allocation<system, int> data_out(size);
    parallel<system>::fill(data_in, 1);

    parallel<system>::inclusive_scan(data_in.begin(), data_in.size(), data_out.begin(), addition_predicate());

    long sum = 0, expected = 0;
    for (uint32 i = 0; i < data_out.size(); i++)
    {
        sum += data_out.peek(i);
        expected += (i + 1);
    }
    // Sum should equal size! since all inital values were 1
    LIFT_TEST_CHECK(sum == expected);
}
LIFT_TEST_FUNC(inclusive_scan_test, inclusive_scan_test);

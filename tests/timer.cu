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
#include <lift/timer.h>
#include <lift/parallel.h>

using namespace lift;

template <target_system system>
void timer_test(void)
{
    constexpr uint32 len = 100000;
    scoped_allocation<system, uint32> keys(len);
    scoped_allocation<system, uint32> temp_keys(len);
    scoped_allocation<system, uint8> temp_storage;

    timer<system> timer_oneshot;
    timer<system> timer;

    parallel<system>::sort(keys, temp_keys, temp_storage);

    timer_oneshot.start();

    for(uint32 i = 0; i < 100; i++)
    {
        timer_oneshot.data(keys);

        timer.start();
        timer.data(keys);
        parallel<system>::sort(keys, temp_keys, temp_storage);
        timer.stop();
    }

    timer_oneshot.stop();

    LIFT_TEST_CHECK(timer_oneshot.elapsed_time() > 0.0);
    LIFT_TEST_CHECK(timer_oneshot.throughput_b() > 0.0);
    LIFT_TEST_CHECK(timer_oneshot.throughput_KB() < timer_oneshot.throughput_b());
    LIFT_TEST_CHECK(timer_oneshot.throughput_GB() < timer_oneshot.throughput_KB());

    LIFT_TEST_CHECK(timer.elapsed_time() > 0.0);
    LIFT_TEST_CHECK(timer.throughput_b() > 0.0);
    LIFT_TEST_CHECK(timer.throughput_KB() < timer.throughput_b());
    LIFT_TEST_CHECK(timer.throughput_GB() < timer.throughput_KB());

    LIFT_TEST_CHECK(timer.elapsed_time() <= timer_oneshot.elapsed_time());
    // xxxnsubtil: this test might be somewhat flaky; if it starts failing, we should probably remove it
    LIFT_TEST_CHECK_FP_NEAR_TOL(timer.elapsed_time(), timer_oneshot.elapsed_time(), 0.05);
}
LIFT_TEST_FUNC(timer, timer_test);

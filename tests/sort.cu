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

#include "harness/test_harness.h"

#include <lift/memory.h>
#include <lift/backends.h>
#include <lift/parallel.h>

using namespace lift;

template <target_system system>
void sort_test_run(void)
{
    scoped_allocation<system, int> keys = {
       41, 19, 93, 79, 85, 29, 18, 95, 11, 64, 62, 27, 77, 44, 87, 31, 50,
       17,  0, 10,  9, 35, 73, 81, 47,  1, 34, 91, 32, 50, 23, 10, 65,  7,
       31, 14,  6, 10, 59, 58, 15, 77,  7, 92, 64, 21, 14, 12, 10, 37
    };

    scoped_allocation<host, int> expected_output = {
        0,  1,  6,  7,  7,  9, 10, 10, 10, 10, 11, 12, 14, 14, 15, 17, 18,
       19, 21, 23, 27, 29, 31, 31, 32, 34, 35, 37, 41, 44, 47, 50, 50, 58,
       59, 62, 64, 64, 65, 73, 77, 77, 79, 81, 85, 87, 91, 92, 93, 95
    };

    scoped_allocation<system, int> temp_keys;
    scoped_allocation<system, uint8> temp_storage;

    parallel<system>::sort(keys, temp_keys, temp_storage);
    parallel<system>::synchronize();
    parallel<system>::check_errors();

    scoped_allocation<host, int> h_keys;
    h_keys.copy(keys);

    for(size_t i = 0; i < h_keys.size(); i++)
    {
        lift_check(h_keys[i] == expected_output[i]);
    }
}
TEST_FUN_HD(sort_test, sort_test_run);

void sort_tests_register(void)
{
    TEST_REGISTER(sort_test_host);
    // disabled due to issue #24: sort_test_cuda fails
    // TEST_REGISTER(sort_test_cuda);
}

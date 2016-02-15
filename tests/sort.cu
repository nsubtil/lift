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

#include <lift/tests/harness.h>

#include <lift/memory.h>
#include <lift/backends.h>
#include <lift/parallel.h>

#include <algorithm>

using namespace lift;

// simple linear congruential random number generator
// we implement this here to avoid discrepancies in test vectors across systems
static uint32 rand_state;

static void rand_reset(void)
{
    rand_state = 0xdeadbeef;
}

static uint32 rand_next(void)
{
    rand_state = rand_state * 1103515245 + 12345;
    return rand_state;
}

// creates a test vector for sorting tests
template <target_system system>
static void generate_test_vector(allocation<system, uint32>& out, allocation<host, uint32>& out_sorted, size_t size)
{
    scoped_allocation<host, uint32> data(size);

    rand_reset();
    for(size_t i = 0; i < size; i++)
    {
        data[i] = rand_next();
    }

    out.copy(data);

    std::sort(data.begin(), data.end());
    out_sorted.copy(data);
}

template <target_system system>
void sort_test_run(size_t size)
{
    scoped_allocation<system, uint32> keys;
    scoped_allocation<host, uint32> expected_output;

    scoped_allocation<system, uint32> temp_keys;
    scoped_allocation<system, uint8> temp_storage;

    generate_test_vector(keys, expected_output, size);

    parallel<system>::sort(keys, temp_keys, temp_storage);
    parallel<system>::synchronize();
    parallel<system>::check_errors();

    scoped_allocation<host, uint32> h_keys;
    h_keys.copy(keys);

    for(size_t i = 0; i < h_keys.size(); i++)
    {
        LIFT_TEST_CHECK(h_keys[i] == expected_output[i]);
    }
}

#define SORT_TEST_GEN(__size__) \
    template <target_system system> \
    void sort_test_##__size__##_run(void) { sort_test_run<system>(__size__); } \
    LIFT_TEST_FUN_HD(sort_test_##__size__, sort_test_##__size__##_run)

#define SORT_TEST_REGISTER(__size__) \
    LIFT_TEST_REGISTER_HD(sort_test_##__size__)

SORT_TEST_GEN(100);
SORT_TEST_GEN(1000);
SORT_TEST_GEN(10000);
SORT_TEST_GEN(100000);

template <target_system system>
void sort_test_shmoo_run(size_t start_size, size_t end_size, size_t step)
{
    for(size_t size = start_size; size <= end_size; size += step)
    {
        sort_test_run<system>(size);
    }
}

#define SORT_TEST_SHMOO_GEN(__start__, __end__, __step__) \
    template <target_system system> \
    void sort_test_shmoo_##__start__##_##__end__##_##__step__##_run(void) { sort_test_shmoo_run<system>(__start__, __end__, __step__); } \
    LIFT_TEST_FUN_HD(sort_test_shmoo_##__start__##_##__end__##_##__step__, sort_test_shmoo_##__start__##_##__end__##_##__step__##_run)

#define SORT_TEST_SHMOO_REGISTER(__start__, __end__, __step__) \
    LIFT_TEST_REGISTER_HD(sort_test_shmoo_##__start__##_##__end__##_##__step__)

SORT_TEST_SHMOO_GEN(1, 100, 1);
SORT_TEST_SHMOO_GEN(100, 500, 50);
SORT_TEST_SHMOO_GEN(500, 5000, 500);
SORT_TEST_SHMOO_GEN(5000, 200000, 10000);

void sort_tests_register(void)
{
    SORT_TEST_REGISTER(100);
    SORT_TEST_REGISTER(1000);
    SORT_TEST_REGISTER(10000);
    SORT_TEST_REGISTER(100000);

    SORT_TEST_SHMOO_REGISTER(1, 100, 1);
    SORT_TEST_SHMOO_REGISTER(100, 500, 50);
    SORT_TEST_SHMOO_REGISTER(500, 5000, 500);
    SORT_TEST_SHMOO_REGISTER(5000, 200000, 10000);
}

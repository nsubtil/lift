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

#include <lift/memory/pointer.h>
#include <lift/backends.h>

using namespace lift;

// despite the fact that most of these tests check that CUDA pointers
// are handled correctly, all of this code is host-only
// all these tests can run on non-CUDA-enabled hosts, so we use
// LIFT_TEST_FUNC_HOST instead of LIFT_TEST_FUNC

template <target_system system>
void pointer_init_run(void)
{
    pointer<system, int> ptr_null;

    LIFT_TEST_CHECK(ptr_null.data() == nullptr);
    LIFT_TEST_CHECK(ptr_null.size() == 0);

    pointer<system, int> ptr((int*)0xdeadbeef, 10);

    LIFT_TEST_CHECK(ptr.data() == (int*)0xdeadbeef);
    LIFT_TEST_CHECK(ptr.size() == 10);

    pointer<system, int> ptr_2(ptr);

    LIFT_TEST_CHECK(ptr_2.data() == (int*)0xdeadbeef);
    LIFT_TEST_CHECK(ptr_2.size() == 10);
}
LIFT_TEST_FUNC_HOST(pointer_init_host, pointer_init_run<host>);
LIFT_TEST_FUNC_HOST(pointer_init_cuda, pointer_init_run<cuda>);

template <target_system system>
void pointer_assign_run(void)
{
    // test same-space assignment
    pointer<system, int> ptr((int*)0xdeadbeef, 10);
    pointer<system, int> ptr_2;

    ptr_2 = ptr;

    LIFT_TEST_CHECK(ptr_2.data() == (int*)0xdeadbeef);
    LIFT_TEST_CHECK(ptr_2.size() == 10);
}
LIFT_TEST_FUNC_HOST(pointer_assign_host, pointer_assign_run<host>);
LIFT_TEST_FUNC_HOST(pointer_assign_cuda, pointer_assign_run<cuda>);

template <target_system S1, target_system S2>
void test_cross_space_pointer_assignment(void)
{
    pointer<S1, int> ptr((int*)0xdeadbeef, 10);

    pointer<S2, int> ptr_2_ctor(ptr);

    LIFT_TEST_CHECK(ptr_2_ctor.data() == nullptr);
    LIFT_TEST_CHECK(ptr_2_ctor.size() == 0);

    pointer<S2, int> ptr_2_assign;
    ptr_2_assign = ptr;

    LIFT_TEST_CHECK(ptr_2_assign.data() == nullptr);
    LIFT_TEST_CHECK(ptr_2_assign.size() == 0);
}

void pointer_cross_space_assignment_run(void)
{
    test_cross_space_pointer_assignment<host, cuda>();
    test_cross_space_pointer_assignment<cuda, host>();
}
LIFT_TEST_FUNC_HOST(pointer_cross_space_assignment, pointer_cross_space_assignment_run);

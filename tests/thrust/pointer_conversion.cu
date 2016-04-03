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
#include <thrust/count.h>

using namespace lift;

// check that lift pointers generate correct iterators for thrust primitives
template <target_system system>
void lift_pointer_to_thrust(void)
{
    scoped_allocation<system, uint32> data = { 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0 };

    pointer<system, uint32> ptr = data;

    LIFT_TEST_CHECK(thrust::count(ptr.t_begin(), ptr.t_end(), uint32(1)) == uint32(5));
}
LIFT_TEST_FUNC(lift_pointer_to_thrust, lift_pointer_to_thrust);

// same as lift_pointer_to_thrust, but using const pointers
// covers https://github.com/nsubtil/lift/issues/77
template <target_system system>
void lift_const_pointer_to_thrust(void)
{
    scoped_allocation<system, uint32> data = { 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0 };

    const pointer<system, uint32> ptr = data;

    LIFT_TEST_CHECK(thrust::count(ptr.t_begin(), ptr.t_end(), uint32(1)) == uint32(5));
}
LIFT_TEST_FUNC(lift_const_pointer_to_thrust, lift_const_pointer_to_thrust);

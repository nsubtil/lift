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


//Unit test for vector-wide implementation of fill
template <target_system system>
void fill_vector()
{
    uint32 size = 1000;
    int fill_val = 10;

    allocation<system, int> data(size);
    parallel<system>::fill(data, fill_val);

    for (int i = 0; i < size; i++)
    {
        LIFT_TEST_CHECK(data.peek(i) == fill_val)
    }     
}
LIFT_TEST_FUNC(fill_vector_test, fill_vector);

template <target_system system>
void fill_input_iter()
{
    uint32 size = 1000;
    int fill_val = 10;

    allocation<system, int> data(size);
    parallel<system>::fill(data, fill_val);

    for (int i = 0; i < size; i++)
    {
        LIFT_TEST_CHECK(data.peek(i) == fill_val)
    }  
}
LIFT_TEST_FUNC(fill_input_iter_test, fill_vector);

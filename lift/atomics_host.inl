/*
 * Lift
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * Copyright (c) 2015, Roche Molecular Systems Inc.
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

#pragma once

#include "backends.h"

namespace lift {

template <>
inline LIFT_HOST int32 atomics<host>::add(int32 *address, int32 val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_SEQ_CST);
}

template <>
inline LIFT_HOST float atomics<host>::add(float *address, float val)
{
    volatile float *addr_v = address;
    volatile float oldval;

    for(;;)
    {
        oldval = *addr_v;
        volatile float expected = oldval;
        float desired = expected + val;

        __atomic_compare_exchange((volatile uint32 *)address,
                                  (uint32 *)&expected,
                                  (uint32 *)&desired,
                                  false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

        if (*((uint32 *)&expected) == *((uint32 *)&oldval))
            break;
    }

    return oldval;
}

template <>
inline LIFT_HOST uint32 atomics<host>::max(uint32 *address, uint32 val)
{
    volatile uint32 *addr_v = address;
    volatile uint32 oldval;

    for(;;)
    {
        oldval = *addr_v;
        if (val < oldval) break;

        volatile uint32 expected = oldval;
        uint32 desired = val;


        __atomic_compare_exchange(addr_v,
                                  (uint32 *)&expected,
                                  &desired,
                                  false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

        if (expected == oldval)
            break;
    }

    return oldval;
}

template <>
inline LIFT_HOST int32 atomics<host>::max(int32 *address, int32 val)
{
    volatile int32 *addr_v = address;
    volatile int32 oldval;

    for(;;)
    {
        oldval = *addr_v;
        if (val < oldval) break;

        volatile int32 expected = oldval;
        int32 desired = val;


        __atomic_compare_exchange((volatile uint32 *)address,
                                  (uint32 *)&expected,
                                  (uint32 *)&desired,
                                  false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

        if (*((uint32 *)&expected) == *((uint32 *)&oldval))
            break;
    }

    return oldval;
}

template <>
inline LIFT_HOST uint32 atomics<host>::min(uint32 *address, uint32 val)
{
    volatile uint32 *addr_v = address;
    volatile uint32 oldval;

    for(;;)
    {
        oldval = *addr_v;
        if (val > oldval) break;

        volatile uint32 expected = oldval;
        uint32 desired = val;


        __atomic_compare_exchange(addr_v,
                                  (uint32 *)&expected,
                                  &desired,
                                  false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

        if (expected == oldval)
            break;
    }

    return oldval;
}

template <>
inline LIFT_HOST int32 atomics<host>::min(int32 *address, int32 val)
{
    volatile int32 *addr_v = address;
    volatile int32 oldval;

    for(;;)
    {
        oldval = *addr_v;
        if (val > oldval) break;

        volatile int32 expected = oldval;
        int32 desired = val;


        __atomic_compare_exchange((volatile uint32 *)address,
                                  (uint32 *)&expected,
                                  (uint32 *)&desired,
                                  false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

        if (*((uint32 *)&expected) == *((uint32 *)&oldval))
            break;
    }

    return oldval;
}

} // namespace lift

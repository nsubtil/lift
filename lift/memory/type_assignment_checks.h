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

#include <type_traits>

#include "../types.h"
#include "../backends.h"
#include "../decorators.h"

/**
 * \file Compile-type checks for various pointer-related conditions.
 */
namespace lift {
namespace memory {

/**
 * Checks if a source and destination value types are assignment-compatible.
 * This effectively only checks that the types are the same or that the destination
 * type is a const version of the source type.
 *
 * \tparam dst_value_type   The lhs type for the assignment
 * \tparam src_value_type   The rhs type for the assignment
 *
 * \returns Nothing. The return type is declared as bool because void constexpr functions
 *          are not allowed, but the return value is not meaningful. Compilation is aborted
 *          if the test fails.
 */
template <typename dst_value_type, typename src_value_type>
static constexpr LIFT_HOST_DEVICE bool check_value_type_assignment_compatible(void)
{
    // the types must either be the same, or dst must be const src
    static_assert(std::is_same<      dst_value_type, src_value_type>::value ||
                  std::is_same<const dst_value_type, src_value_type>::value,
                  "incompatible memory_pointer data types in assignment");
    return true;
}

/**
 * Check if a source and destination pointer types are assignment-compatible. Operates the
 * same as \ref check_value_type_assignment_compatible except that the input types are Lift
 * pointer types.
 */
template <typename memptr_A, typename memptr_B>
static constexpr LIFT_HOST_DEVICE bool check_memory_pointer_assignment_compatible(void)
{
    // the types must either be the same, or our type must be a const version of their type
    static_assert(std::is_same<      typename memptr_A::value_type, typename memptr_B::value_type>::value ||
                  std::is_same<const typename memptr_A::value_type, typename memptr_B::value_type>::value,
                  "incompatible memory_pointer data types in assignment");
    return true;
}

/**
 * \deprecated Assert that a Lift pointer is mutable.
 */
template <typename memptr>
static constexpr LIFT_HOST_DEVICE bool check_memory_pointer_mutable(void)
{
    static_assert(memptr::mutable_tag == 1, "pointer is immutable");
    return true;
}

/**
 * \deprecated Assert that a Lift pointer is immutable.
 */
template <typename memptr>
static constexpr LIFT_HOST_DEVICE bool check_memory_pointer_immutable(void)
{
    static_assert(memptr::mutable_tag == 0, "pointer is mutable");
    return true;
}

/**
 * \deprecated Assert that pointer A and pointer B are either both mutable or both immutable.
 */
template <typename memptr_A, typename memptr_B>
static constexpr LIFT_HOST_DEVICE bool check_memory_pointer_mutability_matches(void)
{
    static_assert(memptr_A::mutable_tag == memptr_B::mutable_tag, "pointer mutability mismatch");
    return true;
}

/**
 * \deprecated Assert that pointer A and pointer B belong to the same memory space.
 */
template <typename memptr_A, typename memptr_B>
static constexpr LIFT_HOST_DEVICE bool check_memory_space(void)
{
    static_assert(target_system(memptr_A::system_tag) == target_system(memptr_B::system_tag), "pointer memory space mismatch");
    return true;
}

} // namespace memory
} // namespace lift

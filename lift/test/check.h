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

#pragma once

#include <cmath>

namespace lift {
namespace test {

template <typename T>
struct fp_to_integral_type
{ };

template <>
struct fp_to_integral_type<float>
{
    static_assert(sizeof(float) == sizeof(int32), "float and int32 sizes do not match");
    typedef int32 type;
    static constexpr int32 sign_bit = 0x80000000;
};

template <>
struct fp_to_integral_type<double>
{
    static_assert(sizeof(double) == sizeof(int64), "double and int64 sizes do not match");
    typedef int64 type;
    static constexpr int64 sign_bit = 0x8000000000000000;
};

// compare floating-point values
template <typename T>
static inline bool helper_check_fp_equal_ulp(T a, T b, uint32 max_ulp = 1)
{
#if !LIFT_DEVICE_COMPILATION
    int class_a = std::fpclassify(a);
    int class_b = std::fpclassify(b);

    if (class_a == FP_INFINITE ||
        class_a == FP_NAN ||
        class_b == FP_INFINITE ||
        class_b == FP_NAN)
    {
        return false;
    }

    if (class_a != class_b)
    {
        return false;
    }

    if (class_a == FP_ZERO && class_b == FP_ZERO)
    {
        // note: we assume 0.0 and -0.0 are identical here
        return true;
    }

    typedef union {
        T real;
        typename fp_to_integral_type<T>::type integer;
    } val;
    constexpr auto sign_bit = fp_to_integral_type<T>::sign_bit;

    val v_a, v_b;
    v_a.real = a;
    v_b.real = b;

    if (v_a.integer < 0)
        v_a.integer = sign_bit - v_a.integer;

    if (v_b.integer < 0)
        v_b.integer = sign_bit - v_b.integer;

    if (abs(v_a.integer - v_b.integer) <= max_ulp)
        return true;
    else
#endif
        return false;
}

template <typename T>
static inline bool helper_check_fp_equal_tol(T a, T b, double tol)
{
#if !LIFT_DEVICE_COMPILATION
    int class_a = std::fpclassify(a);
    int class_b = std::fpclassify(b);

    if (class_a == FP_INFINITE ||
        class_a == FP_NAN ||
        class_b == FP_INFINITE ||
        class_b == FP_NAN)
    {
        return false;
    }

    if (class_a != class_b)
    {
        return false;
    }

    if (class_a == FP_ZERO && class_b == FP_ZERO)
    {
        // note: we assume 0.0 and -0.0 are identical here
        return true;
    }

    // note: this won't work well for values around zero
    // if the expected value is zero, it's better to use the ULP version
    T relative_delta;
    if (fabs(a) > fabs(b))
    {
        relative_delta = fabs(b - a) / fabs(a);
    } else {
        relative_delta = fabs(a - b) / fabs(b);
    }

    return relative_delta <= tol;
#else
    return false;
#endif
}

} // namespace test
} // namespace lift

// test harness function meant as a debug aid
// set a breakpoint on this symbol to catch check failures
extern void debug_check_failure(void);

// check that expr is true, log and fail test if not
#define LIFT_TEST_CHECK(expr)   \
    if (!(expr))                \
    {                           \
        printf("\n    check failed at %s:%u: expression \"%s\"", __FILE__, __LINE__, #expr);    \
        fflush(stdout);                                                                         \
        lift::test::current_test->test_passed = false;                                          \
        debug_check_failure();  \
    }

// check that a is equal to b within 1ULP
#define LIFT_TEST_CHECK_FP_EQUAL(a, b) \
    LIFT_TEST_CHECK(lift::test::helper_check_fp_equal_ulp((a), (b), 1))

// check that a is "near" b with a relative tolerance parameter
#define LIFT_TEST_CHECK_FP_NEAR_TOL(a, b, tol) \
    LIFT_TEST_CHECK(lift::test::helper_check_fp_equal_tol((a), (b), (tol)))

// check that a is "close to" b within some ULP threshold
#define LIFT_TEST_CHECK_FP_NEAR_ULP(a, b, ulp) \
    LIFT_TEST_CHECK(lift::test::helper_check_fp_equal_ulp((a), (b), (ulp)))

// mark a test failure unconditionally
#define LIFT_TEST_FAIL()                                        \
    printf("\n    test failed at %s:%u", __FILE__, __LINE__);   \
    fflush(stdout);                                             \
    lift::test::current_test->test_passed = false;              \
    debug_check_failure()

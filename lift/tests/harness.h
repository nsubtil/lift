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

#include <string>
#include <vector>

#include <lift/types.h>

namespace lift {

/**
 * The test object interface. Defines all common bits for tests.
 */
struct test
{
    const std::string name;                 // short test name
    const std::string description;          // longer test description
    const bool need_cuda;                   // true if the test requires CUDA to run

    bool test_passed;                       // set to true after run() if test passed

    test(std::string name, bool need_cuda)
        : name(name),
          description(""),
          need_cuda(need_cuda),
          test_passed(false)
    { }

    test(std::string name, std::string description, bool need_cuda)
        : name(name),
          description(description),
          need_cuda(need_cuda),
          test_passed(false)
    { }

    // setup routine, performs pre-test initialization (optional)
    virtual void setup()
    { }

    // the actual test routine
    virtual void run() = 0;

    // teardown routine, performs post-test cleanup (optional)
    virtual void teardown()
    { }
};

/**
 * Wrapper for standalone tests, which consist of a single function with no arguments and no return value.
 */
struct standalone_test : public test
{
    typedef void (*callable) (void);
    callable entrypoint;

    standalone_test(callable entrypoint, std::string name, bool need_cuda)
        : test(name, need_cuda), entrypoint(entrypoint)
    { }

    standalone_test(callable entrypoint, std::string name, std::string description, bool need_cuda)
        : test(name, description, need_cuda), entrypoint(entrypoint)
    { }

    virtual void run(void)
    {
        entrypoint();
    }
};

// define a test object for a standalone function
#define TEST_FUN(test_name, entrypoint) \
    standalone_test test_name(entrypoint, #test_name, false)

// define a test object for a standalone function which requires CUDA to run
#define TEST_FUN_CUDA(test_name, entrypoint) \
    standalone_test test_name(entrypoint, #test_name, true)

// define a test object for a standalone function templated on the target system
// generates both host and device versions of the test
#define TEST_FUN_HD(test_name, entrypoint) \
    standalone_test test_name##_host(entrypoint<host>, #test_name "_host", false); \
    standalone_test test_name##_cuda(entrypoint<cuda>, #test_name "_cuda", true)

// register a test object
#define TEST_REGISTER(test_name) \
    test_list.push_back(&test_name)

// register a pair of host/device test objects
#define TEST_REGISTER_HD(test_name) \
    test_list.push_back(&test_name##_host); \
    test_list.push_back(&test_name##_cuda)

// the master test list
extern std::vector<test *> test_list;
// TLS pointer to the current test object being run
extern thread_local test *current_test;

// check that expr is true, log and fail test if not
#define lift_check(expr)        \
    if (!(expr))                \
    {                           \
        printf("\n    check failed at %s:%u: expression \"%s\"", __FILE__, __LINE__, #expr);    \
        fflush(stdout);                                                                         \
        current_test->test_passed = false;                                                      \
        debug_check_failure();  \
    }

template <typename T>
struct integral_type_chooser
{ };

template <>
struct integral_type_chooser<float>
{
    static_assert(sizeof(float) == sizeof(uint32), "float and uint32 sizes do not match");
    typedef int32 type;
    static constexpr int32 sign_bit = 0x80000000;
};

template <>
struct integral_type_chooser<double>
{
    static_assert(sizeof(double) == sizeof(uint64), "double and uint64 sizes do not match");
    typedef int64 type;
    static constexpr int64 sign_bit = 0x8000000000000000;
};

// compare floating-point values
template <typename T>
inline bool helper_check_fp_equal_ulp(T a, T b, uint32 max_ulp = 1)
{
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
        typename integral_type_chooser<T>::type integer;
    } val;
    constexpr auto sign_bit = integral_type_chooser<T>::sign_bit;

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
        return false;
}

template <typename T>
inline bool helper_check_fp_equal_tol(T a, T b, double tol)
{
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
}

// check that a is equal to b within 1ULP
#define lift_check_fp_equal(a, b)   lift_check(helper_check_fp_equal_ulp((a), (b), 1))
// check that a is "near" b with a relative tolerance parameter
#define lift_check_fp_near_tol(a, b, tol) lift_check(helper_check_fp_equal_tol((a), (b), (tol)))
// check that a is "close to" b within some ULP threshold
#define lift_check_fp_near_ulp(a, b, ulp) lift_check(helper_check_fp_equal_ulp((a), (b), (ulp)))

// mark a test failure
#define lift_fail()                                             \
    printf("\n    test failed at %s:%u", __FILE__, __LINE__);   \
    fflush(stdout);                                             \
    current_test->test_passed = false;                          \
    debug_check_failure()

} // namespace lift

// populate the master test list
extern void generate_test_list(void);
// debugging aid: empty function called whenever lift_check detects a failure
extern void debug_check_failure(void);

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

#include <string>
#include <vector>

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

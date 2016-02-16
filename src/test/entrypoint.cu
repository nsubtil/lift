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
#include <lift/test/random.h>

#include <lift/sys/compute_device.h>
#include <lift/sys/host/compute_device_host.h>
#include <lift/sys/cuda/compute_device_cuda.h>

using namespace lift;
using namespace lift::test;

// debugging aid: set a breakpoint here to catch check failures
void debug_check_failure(void)
{
}

int main(int argc, char **argv)
{
    compute_device_host cpu;
    bool has_cuda;

    printf("Lift test harness\n");
    printf("CPU: %s\n", cpu.get_name());

    std::vector<cuda_device_config> gpu_list;
    std::string err;

    cuda_device_config::enumerate_gpus(gpu_list, err);
    if (gpu_list.size() == 0)
    {
        has_cuda = false;
    } else {
        has_cuda = true;

        if (gpu_list.size() == 1)
        {
            printf("GPU: %s\n", gpu_list[0].device_name);
        } else {
            printf("GPU list:\n");
            for(size_t i = 0; i < gpu_list.size(); i++)
            {
                printf("%lu: %s\n", i, gpu_list[i].device_name);
            }
        }
    }
    printf("\n");

    bool success = true;

    size_t tests_run = 0;
    size_t tests_passed = 0;
    size_t tests_failed = 0;

    printf("running tests:\n");
    auto& test_list = get_test_list();

    for(size_t i = 0; i < test_list.size(); i++)
    {
        if (test_list[i]->need_cuda && !has_cuda)
        {
            continue;
        }

        printf("  %s... ", test_list[i]->name.c_str());
        fflush(stdout);

        current_test = test_list[i];

        rand_reset();

        test_list[i]->setup();

        test_list[i]->test_passed = true;
        test_list[i]->run();

        tests_run++;

        test_list[i]->teardown();

        if (!test_list[i]->test_passed)
        {
            printf("\n  FAILED!\n\n");
            success = false;
            tests_failed++;
        } else {
            printf("passed\n");
            tests_passed++;
        }

        fflush(stdout);
    }

    printf("\n%lu out of %lu tests run, %lu passed / %lu failed\n",
           tests_run, test_list.size(),
           tests_passed, tests_failed);

    return (success ? 0 : 1);
}

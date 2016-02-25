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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>

#include "command_line.h"

namespace lift {
namespace test {

struct runtime_options command_line_options;

static void show_usage(char **argv)
{
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("  -t, --test <test-name>        run only named test\n");
    printf("  -c, --cpu-only                run only CPU tests\n");
    printf("  -g, --gpu-only                run only GPU tests\n");
    printf("\n");
}

void parse_command_line(int argc, char **argv)
{
    static const char *options_short = "ht:cg";
    static struct option options_long[] = {
            { "help", no_argument, nullptr, 'h'},
            { "test", required_argument, nullptr, 't' },
            { "cpu-only", no_argument, nullptr, 'c' },
            { "gpu-only", no_argument, nullptr, 'g' },
            { 0 },
    };

    int ch;
    while((ch = getopt_long(argc, argv, options_short, options_long, NULL)) != -1)
    {
        switch(ch)
        {
        case 'h':
            // --help, -h
            // command_line_options.reference = strdup(optarg);
            show_usage(argv);
            exit(0);
            break;

        case 't':
            // --test, -t <test-name>
            command_line_options.target_test_name = std::string(optarg);
            break;

        case 'c':
            // --cpu-only, -c
            command_line_options.gpu_tests_enabled = false;
            break;

        case 'g':
            // --gpu-only, -g
            command_line_options.cpu_tests_enabled = false;
            break;

        default:
            show_usage(argv);
            exit(1);
        }
    }
}

} // namespace test
} // namespace lift

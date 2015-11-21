/*
 * Lift
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * Copyright (c) 2015, Roche Molecular Systems, Inc.
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

#include <vector>

#include <tbb/tbb_stddef.h>
#include <tbb/task_scheduler_init.h>

#include <lift/types.h>
#include <lift/sys/compute_device.h>

namespace lift {

// describes a CPU cache
struct cpu_cache
{
    typedef enum {
        null = 0,       // invalid token
        data,           // data only cache
        instruction,    // instruction cache
        unified,        // unified (data + instruction) cache
    } cache_type;

    cache_type type;
    uint32 level;
    uint32 associativity;
    uint32 total_size;    // in bytes
    uint32 line_size;     // in bytes
};

struct cpu_config
{
    std::string name;
    uint32 vector_extensions;
    std::vector<cpu_cache> caches;
};

namespace __internal {
// the appropriate arch-specific implementation is chosen at compile-time by the build system
extern bool identify_host_cpu(cpu_config& ret);
} // namespace __internal

struct compute_device_host : public compute_device
{
    cpu_config config;
    int num_threads;

    compute_device_host()
    {
        __internal::identify_host_cpu(config);
        num_threads = tbb::task_scheduler_init::default_num_threads();
    }

    virtual target_system get_system(void) override
    {
        return host;
    }

    virtual void enable(void) override
    { }

    virtual const char *get_name(void) override
    {
        return config.name.c_str();
    }

    static bool runtime_initialize(std::string& ret)
    {
        ret = std::string("Intel TBB");
        return true;
    }
};

} // namespace lift

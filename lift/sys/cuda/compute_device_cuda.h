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
#include <string>

#include <cuda_runtime.h>

#include <lift/sys/compute_device.h>

namespace lift {

struct cuda_device_config
{
    int device;
    char *device_name;
    uint64 total_memory;
    int compute_capability_major;
    int compute_capability_minor;

    cuda_device_config()
        : device(-1),
          device_name(nullptr),
          total_memory(uint64(-1)),
          compute_capability_major(-1),
          compute_capability_minor(-1)
    { }

    cuda_device_config(int dev);

    // enumerate the GPUs that match a set of minimum requirements
    // returns false if an error occurs
    static bool enumerate_gpus(std::vector<cuda_device_config>& devices,
                               std::string& error,
                               const cuda_device_config& requirements = cuda_device_config());
};

struct compute_device_cuda : public compute_device
{
    const cuda_device_config config;

    compute_device_cuda(const cuda_device_config& config)
        : config(config)
    { }

    virtual target_system get_system(void) override
    {
        return cuda;
    }

    virtual void enable(void) override
    {
        cudaSetDevice(config.device);
    }

    virtual const char *get_name(void) override
    {
        return config.device_name;
    }

    static bool runtime_initialize(std::string& ret);
};

} // namespace lift

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

#include <lift/types.h>
#include <lift/sys/cuda/compute_device_cuda.h>

#include <string>
#include <vector>

namespace lift {

cuda_device_config::cuda_device_config(int dev)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    // compose our device name
    char str[1024];
    snprintf(str, sizeof(str), "NVIDIA %s (%lu MB, CUDA device %d)",
             prop.name, prop.totalGlobalMem / (1024 * 1024), dev);

    device = dev;
    device_name = strdup(str);
    total_memory = prop.totalGlobalMem;
    compute_capability_major = prop.major;
    compute_capability_minor = prop.minor;
}

// enumerate the GPUs that match a set of minimum requirements
// returns false if an error occurs
bool cuda_device_config::enumerate_gpus(std::vector<cuda_device_config>& devices,
                                        std::string& error,
                                        const cuda_device_config& requirements)
{
    cudaError_t err;
    int gpu_count;

    err = cudaGetDeviceCount(&gpu_count);
    if (err != cudaSuccess)
    {
        error = std::string(cudaGetErrorString(err));
        return false;
    }

    for(int dev = 0; dev < gpu_count; dev++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        // match on the config requirements
        if (requirements.device != -1 &&
            dev != requirements.device)
        {
            continue;
        }

        if (requirements.device_name != nullptr &&
            strcmp(requirements.device_name, prop.name))
        {
            continue;
        }

        // the following are considered a minimum requirement and not an exact match
        if (requirements.total_memory != uint64(-1) &&
            prop.totalGlobalMem < requirements.total_memory)
        {
            continue;
        }

        if (requirements.compute_capability_major != -1)
        {
            if (prop.major < requirements.compute_capability_major)
            {
                continue;
            }

            if (prop.major == requirements.compute_capability_major)
            {
                if (requirements.compute_capability_minor != -1 &&
                    prop.minor < requirements.compute_capability_minor)
                {
                    continue;
                }
            }
        }

        devices.push_back(cuda_device_config(dev));
    }

    return true;
}

bool compute_device_cuda::runtime_initialize(std::string& ret)
{
    cudaError_t err;
    int runtime_version;

    // force explicit runtime initialization
    err = cudaFree(0);
    if (err != cudaSuccess)
    {
        ret = std::string(cudaGetErrorString(err));
        return false;
    }

    err = cudaRuntimeGetVersion(&runtime_version);
    if (err != cudaSuccess)
    {
        ret = std::string(cudaGetErrorString(err));
        return false;
    }

    char buf[256];
    snprintf(buf, sizeof(buf),
             "NVIDIA CUDA %d.%d", runtime_version / 1000, runtime_version % 100 / 10);

    ret = std::string(buf);
    return true;
}

} // namespace lift

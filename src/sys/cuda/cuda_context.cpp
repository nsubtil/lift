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

#include <lift/sys/cuda/cuda_context.h>
#include <lift/backends.h>

#include <map>
#include <mutex>
#include <thread>

#include <cuda_runtime.h>

namespace lift {
namespace __internal {

static std::map<int, cuda_context *> gpu_context_array;
static std::mutex gpu_context_array_mutex;
static thread_local cuda_context *cached_gpu_context = nullptr;

void cuda_context::set_stream(uint32 stream_id)
{
    if (stream_map.find(stream_id) == stream_map.end())
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        stream_map[stream_id] = stream;
    }

    active_cuda_stream = stream_map[stream_id];
    active_lift_stream = stream_id;
}

uint32 cuda_context::get_stream(void) const
{
    return active_lift_stream;
}

void *cuda_context::suballocate(size_t len)
{
    void *ret;
    default_suballocator.DeviceAllocate(device, &ret, len, active_cuda_stream);
    return ret;
}

void cuda_context::free_suballocation(const void *ptr)
{
    default_suballocator.DeviceFree(device, (void *) ptr);
}

cuda_context *get_cuda_context(void)
{
    int device;
    cudaGetDevice(&device);

    // if we have a cached context...
    if (cached_gpu_context != nullptr)
    {
        // .. check if the device matches our own
        if (cached_gpu_context->device == device)
        {
            return cached_gpu_context;
        }
    }

    // else, check the names array
    std::lock_guard<std::mutex> guard(gpu_context_array_mutex);

    if (gpu_context_array.find(device) == gpu_context_array.end())
    {
        struct cuda_context *ctx = new cuda_context(device);
        gpu_context_array[device] = ctx;
    }

    cached_gpu_context = gpu_context_array[device];
    return gpu_context_array[device];
}

} // namespace __internal

} // namespace lift

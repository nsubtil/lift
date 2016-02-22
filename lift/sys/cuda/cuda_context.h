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

#include <lift/backends.h>
#include <lift/types.h>
#include <lift/sys/cuda/compute_device_cuda.h>

#include <map>

#include <cuda_runtime.h>

// work-around annoying CUB warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#include <cub/util_allocator.cuh>
#pragma GCC diagnostic pop

namespace lift {
namespace __internal {

struct cuda_context
{
    int device;
    cuda_device_config config;

    // maps integers to CUDA streams
    std::map<uint32, cudaStream_t> stream_map;
    // stores the currently active Lift and CUDA stream IDs
    uint32 active_lift_stream;
    cudaStream_t active_cuda_stream;

    // suballocator for this device
    cub::CachingDeviceAllocator default_suballocator;

    cuda_context(int device)
        : device(device),
          config(device),
          stream_map({{0, 0}}),
          active_lift_stream(0),
          active_cuda_stream(0),
          default_suballocator()
    { }

    // no copy!
    cuda_context(const cuda_context&) = delete;
    cuda_context& operator=(const cuda_context&) = delete;

    /**
     * Sets the currently active CUDA stream in Lift
     * @param stream_id An integer that uniquely identifies the stream within Lift. Stream 0 is the default stream.
     */
    void set_stream(uint32 stream_id);

    /**
     * Returns the active Lift stream
     * @return  Lift stream id for the active stream.
     */
    uint32 get_stream(void) const;

    /**
     * Grabs the cudaStream_t object for an arbitrary Lift stream.
     * @param  stream_id Lift stream to convert to cudaStream_t. If -1, grabs the currently active Lift stream.
     * @return           cudaStream_t that corresponds to a Lift stream_id.
     */
    cudaStream_t cuda_stream(uint32 stream_id = uint32(-1));

    void *suballocate(size_t len);
    void free_suballocation(const void *ptr);
};

cuda_context *get_cuda_context(void);

} // namespace __internal
} // namespace lift

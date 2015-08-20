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

#include "../types.h"
#include "../backends.h"
#include "../decorators.h"

#include "default_allocator.h"
#include "memory_pointer.h"
#include "const_memory_pointer.h"
#include "type_assignment_checks.h"

namespace lift {

namespace memory {

// transfer data across devices as needed
//
// src is the source pointer that points at the data being transferred
// storage is a scoped pointer in the destination memory space
//
// returns a pointer to the data in the destination memory space
// if the source and destination memory spaces differ, storage will be used to allocate
// memory in the destination memory space and the return value will be a pointer to storage
// if the source and destination memory spaces are the same, this returns src
template <typename pointer, typename scoped_pointer>
pointer device_transfer(scoped_pointer& storage, const pointer& src)
{
    if (target_system(pointer::system_tag) == target_system(scoped_pointer::system_tag))
    {
        // on the same system, return a copy of the pointer
        return src;
    } else {
        // when the system differs, copy to scoped pointer
        storage.copy(src);
        // and return a pointer to the scoped pointer
        return (pointer) storage;
    }
}

#if 0
// create a copy of a memory pointer on a different system
template <typename target_pointer, typename source_pointer>
inline void copy(target_pointer& dst, const source_pointer src)
{
    check_memory_pointer_assignment_compatible<target_pointer, source_pointer>();

    dst.resize(src.size());

    if (target_system(target_pointer::system_tag) == cuda)
    {
        // copying to GPU...
        if (target_system(source_pointer::system_tag) == cuda)
        {
            // ... from the GPU
            cudaMemcpy((void *) dst.data(), src.data(), sizeof(typename source_pointer::value_type) * src.size(), cudaMemcpyDeviceToDevice);
        } else {
            // ... from the host
            cudaMemcpy((void *) dst.data(), src.data(), sizeof(typename source_pointer::value_type) * src.size(), cudaMemcpyHostToDevice);
        }
    } else {
        // copying to host...
        if (target_system(source_pointer::system_tag) == cuda)
        {
            // ... from the GPU
            cudaMemcpy((void *) dst.data(), src.data(), sizeof(typename source_pointer::value_type) * src.size(), cudaMemcpyDeviceToHost);
        } else {
            // ... from the host
            memcpy((void *) dst.data(), src.data(), sizeof(typename source_pointer::value_type) * src.size());
        }
    }
}

// memory transfer from another identical pointer
// this is equivalent to a copy if the target systems differ and a pointer assignment if they match
// note that unlike copy(), the pointer types here must also match in value_type constness
template <typename target_pointer, typename source_pointer>
inline void transfer(target_pointer& dst, const source_pointer src)
{
    check_memory_pointer_assignment_compatible<target_pointer, source_pointer>();

    if (target_system(target_pointer::system_tag) == target_system(source_pointer::system_tag))
    {
        dst = target_pointer(src.data(), src.size());
    } else {
        return copy(dst, src);
    }
}

#endif

} // namespace memory
} // namespace lift

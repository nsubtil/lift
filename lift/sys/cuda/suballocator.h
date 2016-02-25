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

/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Simple caching allocator for device memory allocations. The allocator is
 * thread-safe and capable of managing device allocations on multiple devices.
 ******************************************************************************/

#pragma once

#include <lift/sys/host/spinlock.h>

#include <cuda_runtime.h>

#include <set>
#include <map>

#include <math.h>

namespace lift {
namespace __internal {

/**
 * \brief A simple caching allocator for device memory allocations.
 *
 * \par Overview
 * The allocator is thread-safe and stream-safe and is capable of managing cached
 * device allocations on multiple devices.  It behaves as follows:
 *
 * \par
 * - Allocations from the allocator are associated with an \p active_stream.  Once freed,
 *   the allocation becomes available immediately for reuse within the \p active_stream
 *   with which it was associated with during allocation, and it becomes available for
 *   reuse within other streams when all prior work submitted to \p active_stream has completed.
 * - Allocations are categorized and cached by bin size.  A new allocation request of
 *   a given size will only consider cached allocations within the corresponding bin.
 * - Bin limits progress geometrically in accordance with the growth factor
 *   \p bin_growth provided during construction.  Unused device allocations within
 *   a larger bin cache are not reused for allocation requests that categorize to
 *   smaller bin sizes.
 * - Allocation requests below (\p bin_growth ^ \p min_bin) are rounded up to
 *   (\p bin_growth ^ \p min_bin).
 * - Allocations above (\p bin_growth ^ \p max_bin) are not rounded up to the nearest
 *   bin and are simply freed when they are deallocated instead of being returned
 *   to a bin-cache.
 * - %If the total storage of cached allocations on a given device will exceed
 *   \p max_cached_bytes, allocations for that device are simply freed when they are
 *   deallocated instead of being returned to their bin-cache.
 *
 * \par
 * For example, the default-constructed CachingDeviceAllocator is configured with:
 * - \p bin_growth = 8
 * - \p min_bin = 3
 * - \p max_bin = 7
 * - \p max_cached_bytes = 6MB - 1B
 *
 * \par
 * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
 * and sets a maximum of 6,291,455 cached bytes per device
 *
 */
struct CachingDeviceAllocator
{
    // \cond

    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    enum
    {
        /// Invalid device ordinal
        INVALID_DEVICE_ORDINAL  = -1,
    };

    /**
     * Descriptor for device memory allocations
     */
    struct BlockDescriptor
    {
        int             device;             // device ordinal
        void*           d_ptr;              // Device pointer
        cudaStream_t    associated_stream;  // Associated associated_stream
        cudaEvent_t     ready_event;        // Signal when associated stream has run to the point at which this block was freed
        size_t          bytes;              // Size of allocation in bytes
        unsigned int    bin;                // Bin enumeration

        // Constructor
        BlockDescriptor(void *d_ptr, int device);
        // Constructor
        BlockDescriptor(size_t bytes,
                        unsigned int bin,
                        int device,
                        cudaStream_t associated_stream);
    };

    /// BlockDescriptor comparator function interface
    typedef bool (*Compare)(const BlockDescriptor &, const BlockDescriptor &);
    /// Set type for cached blocks (ordered by size)
    typedef std::multiset<BlockDescriptor, Compare> CachedBlocks;
    /// Set type for live blocks (ordered by ptr)
    typedef std::multiset<BlockDescriptor, Compare> BusyBlocks;
    /// Map type of device ordinals to the number of cached bytes cached by each device
    typedef std::map<int, size_t> GpuCachedBytes;

    //---------------------------------------------------------------------
    // Fields
    //---------------------------------------------------------------------

    spinlock        spin_lock;          /// Spinlock for thread-safety

    unsigned int    bin_growth;         /// Geometric growth factor for bin-sizes
    unsigned int    min_bin;            /// Minimum bin enumeration
    unsigned int    max_bin;            /// Maximum bin enumeration

    size_t          min_bin_bytes;      /// Minimum bin size
    size_t          max_bin_bytes;      /// Maximum bin size
    size_t          max_cached_bytes;   /// Maximum aggregate cached bytes per device

    bool            debug;              /// Whether or not to print (de)allocation events to stdout
    bool            skip_cleanup;       /// Whether or not to skip a call to FreeAllCached() when destructor is called.  (The CUDA runtime may have already shut down for statically declared allocators)

    GpuCachedBytes  cached_bytes;       /// Map of device ordinal to aggregate cached bytes on that device
    CachedBlocks    cached_blocks;      /// Set of cached device allocations available for reuse
    BusyBlocks      live_blocks;        /// Set of live device allocations currently in use

    // \endcond

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
     * \brief Constructor.
     */
    CachingDeviceAllocator(unsigned int    bin_growth,             ///< Geometric growth factor for bin-sizes
                           unsigned int    min_bin,                ///< Minimum bin
                           unsigned int    max_bin,                ///< Maximum bin
                           size_t          max_cached_bytes,       ///< Maximum aggregate cached bytes per device
                           bool            skip_cleanup = false);  ///< Whether or not to skip a call to \p FreeAllCached() when the destructor is called.  (Useful for preventing warnings when the allocator is declared at file/static/global scope: by the time the destructor is called on program exit, the CUDA runtime may have already shut down and freed all allocations.)

    /**
     * \brief Default constructor.
     *
     * Configured with:
     * \par
     * - \p bin_growth = 8
     * - \p min_bin = 3
     * - \p max_bin = 7
     * - \p max_cached_bytes = (\p bin_growth ^ \p max_bin) * 3) - 1 = 6,291,455 bytes
     *
     * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB and
     * sets a maximum of 6,291,455 cached bytes per device
     */
    CachingDeviceAllocator(bool skip_cleanup = false);  ///< Whether or not to skip a call to \p FreeAllCached() when the destructor is called.  (Useful for preventing warnings when the allocator is declared at file/static/global scope: by the time the destructor is called on program exit, the CUDA runtime may have already shut down and freed all allocations.)

    /**
     * \brief Sets the limit on the number bytes this allocator is allowed to cache per device.
     */
    cudaError_t SetMaxCachedBytes(size_t max_cached_bytes);

    /**
     * \brief Provides a suitable allocation of device memory for the given size on the specified device.
     *
     * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
     * with which it was associated with during allocation, and it becomes available for reuse within other
     * streams when all prior work submitted to \p active_stream has completed.
     */
    cudaError_t DeviceAllocate(int             device,             ///< [in] Device on which to place the allocation
                               void            **d_ptr,            ///< [out] Reference to pointer to the allocation
                               size_t          bytes,              ///< [in] Minimum number of bytes for the allocation
                               cudaStream_t    active_stream = 0); ///< [in] The stream to be associated with this allocation

    /**
     * \brief Provides a suitable allocation of device memory for the given size on the current device.
     *
     * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
     * with which it was associated with during allocation, and it becomes available for reuse within other
     * streams when all prior work submitted to \p active_stream has completed.
     */
    cudaError_t DeviceAllocate(void            **d_ptr,            ///< [out] Reference to pointer to the allocation
                               size_t          bytes,              ///< [in] Minimum number of bytes for the allocation
                               cudaStream_t    active_stream = 0)  ///< [in] The stream to be associated with this allocation
    {
        return DeviceAllocate(INVALID_DEVICE_ORDINAL, d_ptr, bytes, active_stream);
    }

    /**
     * \brief Frees a live allocation of device memory on the specified device, returning it to the allocator.
     *
     * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
     * with which it was associated with during allocation, and it becomes available for reuse within other
     * streams when all prior work submitted to \p active_stream has completed.
     */
    cudaError_t DeviceFree(int             device,
                           void*           d_ptr);

    /**
     * \brief Frees a live allocation of device memory on the current device, returning it to the allocator.
     *
     * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
     * with which it was associated with during allocation, and it becomes available for reuse within other
     * streams when all prior work submitted to \p active_stream has completed.
     */
    cudaError_t DeviceFree(void*           d_ptr)
    {
        return DeviceFree(INVALID_DEVICE_ORDINAL, d_ptr);
    }

    /**
     * \brief Frees all cached device allocations on all devices
     */
    cudaError_t FreeAllCached();

    /**
     * \brief Destructor
     */
    virtual ~CachingDeviceAllocator()
    {
        if (!skip_cleanup)
            FreeAllCached();
    }
};

} // namespace __internal
} // namespace lift

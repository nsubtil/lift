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

#include <lift/sys/cuda/suballocator.h>

#include <stdarg.h>

namespace lift {
namespace __internal {

// #define _CubLog(a, ...)
static void _CubLog(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    fprintf(stderr, "suballocator: ");
    vfprintf(stderr, fmt, args);

    va_end(args);
}

#define CubDebug(a) (a)

/**
 * Integer pow function for unsigned base and exponent
 */
static unsigned int IntPow(unsigned int base,
                           unsigned int exp)
{
    unsigned int retval = 1;
    while (exp > 0)
    {
        if (exp & 1) {
            retval = retval * base;        // multiply the result by the current base
        }
        base = base * base;                // square the base
        exp = exp >> 1;                    // divide the exponent in half
    }
    return retval;
}

/**
 * Round up to the nearest power-of
 */
static void NearestPowerOf(unsigned int &power,
                           size_t &rounded_bytes,
                           unsigned int base,
                           size_t value)
{
    power = 0;
    rounded_bytes = 1;

    while (rounded_bytes < value)
    {
        rounded_bytes *= base;
        power++;
    }
}

// Comparison functor for comparing device pointers
static bool PtrCompare(const CachingDeviceAllocator::BlockDescriptor &a, const CachingDeviceAllocator::BlockDescriptor &b)
{
    if (a.device == b.device)
        return (a.d_ptr < b.d_ptr);
    else
        return (a.device < b.device);
}

// Comparison functor for comparing allocation sizes
static bool SizeCompare(const CachingDeviceAllocator::BlockDescriptor &a, const CachingDeviceAllocator::BlockDescriptor &b)
{
    if (a.device == b.device)
        return (a.bytes < b.bytes);
    else
        return (a.device < b.device);
}

CachingDeviceAllocator::BlockDescriptor::BlockDescriptor(void *d_ptr, int device)
    : device(device),
      d_ptr(d_ptr),
      associated_stream(0),
      ready_event(0),
      bytes(0),
      bin(0)
{ }

CachingDeviceAllocator::BlockDescriptor::BlockDescriptor(size_t bytes,
                                                         unsigned int bin,
                                                         int device,
                                                         cudaStream_t associated_stream)
    : device(device),
      d_ptr(NULL),
      associated_stream(associated_stream),
      ready_event(0),
      bytes(bytes),
      bin(bin)
{ }

/**
 * \brief Constructor.
 */
CachingDeviceAllocator::CachingDeviceAllocator(unsigned int    bin_growth,             ///< Geometric growth factor for bin-sizes
                                               unsigned int    min_bin,                ///< Minimum bin
                                               unsigned int    max_bin,                ///< Maximum bin
                                               size_t          max_cached_bytes,       ///< Maximum aggregate cached bytes per device
                                               bool            skip_cleanup)           ///< Whether or not to skip a call to \p FreeAllCached() when the destructor is called.  (Useful for preventing warnings when the allocator is declared at file/static/global scope: by the time the destructor is called on program exit, the CUDA runtime may have already shut down and freed all allocations.)
    : bin_growth(bin_growth),
      min_bin(min_bin),
      max_bin(max_bin),
      min_bin_bytes(IntPow(bin_growth, min_bin)),
      max_bin_bytes(IntPow(bin_growth, max_bin)),
      max_cached_bytes(max_cached_bytes),
      debug(false),
      skip_cleanup(skip_cleanup),
      cached_blocks(SizeCompare),
      live_blocks(PtrCompare)
{ }

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
CachingDeviceAllocator::CachingDeviceAllocator(bool skip_cleanup)  ///< Whether or not to skip a call to \p FreeAllCached() when the destructor is called.  (Useful for preventing warnings when the allocator is declared at file/static/global scope: by the time the destructor is called on program exit, the CUDA runtime may have already shut down and freed all allocations.)
    : bin_growth(8),
      min_bin(3),
      max_bin(7),
      min_bin_bytes(IntPow(bin_growth, min_bin)),
      max_bin_bytes(IntPow(bin_growth, max_bin)),
      max_cached_bytes((max_bin_bytes * 3) - 1),
      debug(false),
      skip_cleanup(skip_cleanup),
      cached_blocks(SizeCompare),
      live_blocks(PtrCompare)
{ }

/**
 * \brief Sets the limit on the number bytes this allocator is allowed to cache per device.
 */
cudaError_t CachingDeviceAllocator::SetMaxCachedBytes(size_t max_cached_bytes)
{
    // Lock
    spin_lock.lock();

    this->max_cached_bytes = max_cached_bytes;

    if (debug) _CubLog("New max_cached_bytes(%lld)\n", (long long) max_cached_bytes);

    // Unlock
    spin_lock.unlock();

    return cudaSuccess;
}

/**
 * \brief Provides a suitable allocation of device memory for the given size on the specified device.
 *
 * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
 * with which it was associated with during allocation, and it becomes available for reuse within other
 * streams when all prior work submitted to \p active_stream has completed.
 */
cudaError_t CachingDeviceAllocator::DeviceAllocate(int             device,             ///< [in] Device on which to place the allocation
                                                   void            **d_ptr,            ///< [out] Reference to pointer to the allocation
                                                   size_t          bytes,              ///< [in] Minimum number of bytes for the allocation
                                                   cudaStream_t    active_stream)      ///< [in] The stream to be associated with this allocation
{
    *d_ptr                          = NULL;
    bool locked                     = false;
    int entrypoint_device           = INVALID_DEVICE_ORDINAL;
    cudaError_t error               = cudaSuccess;

    do {

        if (CubDebug(error = cudaGetDevice(&entrypoint_device))) break;
        if (device == INVALID_DEVICE_ORDINAL)
            device = entrypoint_device;

        // Round up to nearest bin size
        unsigned int bin;
        size_t bin_bytes;
        NearestPowerOf(bin, bin_bytes, bin_growth, bytes);
        if (bin < min_bin) {
            bin = min_bin;
            bin_bytes = min_bin_bytes;
        }

        // Check if bin is greater than our maximum bin
        if (bin > max_bin)
        {
            // Allocate the request exactly and give out-of-range bin
            bin = (unsigned int) -1;
            bin_bytes = bytes;
        }

        BlockDescriptor search_key(bin_bytes, bin, device, active_stream);

        // Lock
        if (!locked) {
            spin_lock.lock();
            locked = true;
        }

        // Find the range of freed blocks big enough within the same bin on the same device
        CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);

        // Look for freed blocks from the active stream or from other idle streams
        bool found = false;
        while ((block_itr != cached_blocks.end()) &&
            (block_itr->device == device) &&
            (block_itr->bin == search_key.bin))
        {
            cudaStream_t prev_stream = block_itr->associated_stream;
            if ((active_stream == prev_stream) || (cudaEventQuery(block_itr->ready_event) != cudaErrorNotReady))
            {
                // Reuse existing cache block.  Insert into live blocks.
                found = true;
                search_key = *block_itr;
                search_key.associated_stream = active_stream;
                live_blocks.insert(search_key);

                // Remove from free blocks
                cached_blocks.erase(block_itr);
                cached_bytes[device] -= search_key.bytes;

                if (debug) _CubLog("\tdevice %d reused cached block for stream %lld (%lld bytes, previously associated with stream %lld).\n\t\t %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
                    device, (long long) active_stream, (long long) search_key.bytes, (long long) prev_stream, (long long) cached_blocks.size(), (long long) cached_bytes[device], (long long) live_blocks.size());

                break;
            }

            block_itr++;
        }

        if (!found)
        {
            // Need to allocate a new cache block. Unlock.
            if (locked) {
                spin_lock.unlock();
                locked = false;
            }

            // Set to specified device
            if (device != entrypoint_device) {
                if (CubDebug(error = cudaSetDevice(device))) break;
            }

            // Allocate
            if (CubDebug(error = cudaMalloc(&search_key.d_ptr, search_key.bytes))) break;
            if (CubDebug(error = cudaEventCreateWithFlags(&search_key.ready_event, cudaEventDisableTiming))) break;

            // Lock
            if (!locked) {
                spin_lock.lock();
                locked = true;
            }

            // Insert into live blocks
            live_blocks.insert(search_key);

            if (debug) _CubLog("\tdevice %d allocating new device block %lld bytes associated with stream %lld.\n\t\t %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
                device, (long long) search_key.bytes, (long long) search_key.associated_stream, (long long) cached_blocks.size(), (long long) cached_bytes[device], (long long) live_blocks.size());
        }

        // Copy device pointer to output parameter
        *d_ptr = search_key.d_ptr;

    } while(0);

    // Unlock
    if (locked) {
        spin_lock.unlock();
        locked = false;
    }

    // Attempt to revert back to previous device if necessary
    if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device))
    {
        if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
    }

    return error;
}

/**
 * \brief Frees a live allocation of device memory on the specified device, returning it to the allocator.
 *
 * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
 * with which it was associated with during allocation, and it becomes available for reuse within other
 * streams when all prior work submitted to \p active_stream has completed.
 */
cudaError_t CachingDeviceAllocator::DeviceFree(int             device,
                                               void*           d_ptr)
{
    bool locked                     = false;
    int entrypoint_device           = INVALID_DEVICE_ORDINAL;
    cudaError_t error               = cudaSuccess;

    do {
        if (CubDebug(error = cudaGetDevice(&entrypoint_device))) break;
        if (device == INVALID_DEVICE_ORDINAL)
            device = entrypoint_device;

        // Set to specified device
        if (device != entrypoint_device) {
            if (CubDebug(error = cudaSetDevice(device))) break;
        }

        // Lock
        if (!locked) {
            spin_lock.lock();
            locked = true;
        }

        // Find corresponding block descriptor
        BlockDescriptor search_key(d_ptr, device);
        BusyBlocks::iterator block_itr = live_blocks.find(search_key);
        if (block_itr == live_blocks.end())
        {
            // Cannot find pointer
            if (CubDebug(error = cudaErrorUnknown)) break;
        }
        else
        {
            // Remove from live blocks
            search_key = *block_itr;
            live_blocks.erase(block_itr);

            // Check if we should keep the returned allocation
            if (cached_bytes[device] + search_key.bytes <= max_cached_bytes)
            {
                // Signal the event in the associated stream
                if (CubDebug(error = cudaEventRecord(search_key.ready_event, search_key.associated_stream))) break;

                // Insert returned allocation into free blocks
                cached_blocks.insert(search_key);
                cached_bytes[device] += search_key.bytes;

                if (debug) _CubLog("\tdevice %d returned %lld bytes from associated stream %lld.\n\t\t %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
                    device, (long long) search_key.bytes, (long long) search_key.associated_stream, (long long) cached_blocks.size(), (long long) cached_bytes[device], (long long) live_blocks.size());
            }
            else
            {
                // Free the returned allocation.  Unlock.
                if (locked) {
                    spin_lock.lock();
                    locked = false;
                }

                // Free device memory
                if (CubDebug(error = cudaFree(d_ptr))) break;
                if (CubDebug(error = cudaEventDestroy(search_key.ready_event))) break;

                if (debug) _CubLog("\tdevice %d freed %lld bytes from associated stream %lld.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
                    device, (long long) search_key.bytes, (long long) search_key.associated_stream, (long long) cached_blocks.size(), (long long) cached_bytes[device], (long long) live_blocks.size());
            }
        }
    } while (0);

    // Unlock
    if (locked) {
        spin_lock.unlock();
        locked = false;
    }

    if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device))
    {
        if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
    }

    return error;
}

/**
 * \brief Frees all cached device allocations on all devices
 */
cudaError_t CachingDeviceAllocator::FreeAllCached()
{
    cudaError_t error         = cudaSuccess;
    bool locked               = false;
    int entrypoint_device     = INVALID_DEVICE_ORDINAL;
    int current_device        = INVALID_DEVICE_ORDINAL;

    // Lock
    if (!locked) {
        spin_lock.lock();
        locked = true;
    }

    while (!cached_blocks.empty())
    {
        // Get first block
        CachedBlocks::iterator begin = cached_blocks.begin();

        // Get entry-point device ordinal if necessary
        if (entrypoint_device == INVALID_DEVICE_ORDINAL)
        {
            if (CubDebug(error = cudaGetDevice(&entrypoint_device))) break;
        }

        // Set current device ordinal if necessary
        if (begin->device != current_device)
        {
            if (CubDebug(error = cudaSetDevice(begin->device))) break;
            current_device = begin->device;
        }

        // Free device memory
        if (CubDebug(error = cudaFree(begin->d_ptr))) break;
        if (CubDebug(error = cudaEventDestroy(begin->ready_event))) break;

        // Reduce balance and erase entry
        cached_bytes[current_device] -= begin->bytes;
        cached_blocks.erase(begin);

        if (debug) _CubLog("\tdevice %d freed %lld bytes.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
            current_device, (long long) begin->bytes, (long long) cached_blocks.size(), (long long) cached_bytes[current_device], (long long) live_blocks.size());
    }

    // Unlock
    if (locked) {
        spin_lock.lock();
        locked = false;
    }

    // Attempt to revert back to entry-point device if necessary
    if (entrypoint_device != INVALID_DEVICE_ORDINAL)
    {
        if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
    }

    return error;
}

} // namespace __internal
} // namespace lift

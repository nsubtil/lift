/*
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
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

#define WAR_CUB_COPY_FLAGGED 1

#include "types.h"

#include "decorators.h"
#include "backends.h"
#include "memory.h"
#include "algorithms/for_each.h"

#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_run_length_encode.cuh>
// silence warnings from debug code in cub
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <cub/device/device_radix_sort.cuh>
#pragma GCC diagnostic pop

namespace lift {

struct copy_if_flagged
{
    CUDA_HOST_DEVICE bool operator() (const uint8 val)
    {
        return bool(val);
    }
};

// thrust-based implementation of parallel primitives
template <target_system system>
struct parallel_thrust
{
    template <typename InputIterator, typename UnaryFunction>
    static inline InputIterator for_each(InputIterator first, InputIterator last, UnaryFunction f, int2 launch_parameters = { 0, 0 })
    {
        return thrust::for_each(lift::backend_policy<system>::execution_policy(),
                                first,
                                last,
                                f);
    }

    // shortcut to run for_each on a lift pointer
    template <typename T, typename UnaryFunction>
    static inline typename pointer<system, T>::iterator for_each(pointer<system, T>& vector, UnaryFunction f, int2 launch_parameters = { 0, 0 })
    {
        return thrust::for_each(lift::backend_policy<system>::execution_policy(),
                                vector.begin(),
                                vector.end(),
                                f);
    }

    // shortcut to run for_each on [range.x, range.y[
    template <typename UnaryFunction>
    static inline void for_each(uint2 range, UnaryFunction f, int2 launch_parameters = { 0, 0 })
    {
        thrust::for_each(lift::backend_policy<system>::execution_policy(),
                         thrust::make_counting_iterator(range.x),
                         thrust::make_counting_iterator(range.y),
                         f);
    }

    // shortcut to run for_each on [0, end[
    template <typename UnaryFunction>
    static inline void for_each(uint32 end, UnaryFunction f, int2 launch_parameters = { 0, 0 })
    {
        thrust::for_each(lift::backend_policy<system>::execution_policy(),
                         thrust::make_counting_iterator(0u),
                         thrust::make_counting_iterator(end),
                         f);
    }

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline void inclusive_scan(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      Predicate op)
    {
        thrust::inclusive_scan(lift::backend_policy<system>::execution_policy(),
                               first, first + len, result, op);
    }

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline size_t copy_if(InputIterator first,
                                 size_t len,
                                 OutputIterator result,
                                 Predicate op,
                                 allocation<system, uint8>& temp_storage)
    {
        // use the fallback thrust version
        OutputIterator out_last;
        out_last = thrust::copy_if(lift::backend_policy<system>::execution_policy(),
                                   first, first + len, result, op);
        return out_last - result;
    }

    template <typename InputIterator, typename FlagIterator, typename OutputIterator>
    static inline size_t copy_flagged(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      FlagIterator flags,
                                      allocation<system, uint8>& temp_storage)
    {
        OutputIterator out_last;
        out_last = thrust::copy_if(lift::backend_policy<system>::execution_policy(),
                                   first, first + len, flags, result, copy_if_flagged());
        return out_last - result;
    }

    template <typename InputIterator>
    static inline int64 sum(InputIterator first,
                            size_t len,
                            allocation<system, uint8>& temp_storage)
    {
        return thrust::reduce(lift::backend_policy<system>::execution_policy(),
                              first, first + len, int64(0));
    }

    template <typename Key, typename Value>
    static inline void sort_by_key(pointer<system, Key>& keys,
                                   pointer<system, Value>& values,
                                   pointer<system, Key>& temp_keys,
                                   pointer<system, Value>& temp_values,
                                   allocation<system, uint8>& temp_storage,
                                   int num_key_bits = sizeof(Key) * 8)
    {
        thrust::sort_by_key(lift::backend_policy<system>::execution_policy(),
                            keys.begin(), keys.end(), values.begin());
    }

    // returns the size of the output key/value
    template <typename KeyIterator, typename ValueIterator, typename ReductionOp>
    static inline size_t reduce_by_key(KeyIterator keys_begin,
                                       KeyIterator keys_end,
                                       ValueIterator values_begin,
                                       KeyIterator output_keys,
                                       ValueIterator output_values,
                                       allocation<system, uint8>& temp_storage,
                                       ReductionOp reduction_op)
    {
        auto out = thrust::reduce_by_key(lift::backend_policy<system>::execution_policy(),
                                         keys_begin,
                                         keys_end,
                                         values_begin,
                                         output_keys,
                                         output_values,
                                         thrust::equal_to<typeof(*keys_begin)>(),
                                         reduction_op);

        return out.first - output_keys.begin();
    }

    // returns the size of the output key/value vectors
    template <typename Key, typename Value, typename ReductionOp>
    static inline size_t reduce_by_key(pointer<system, Key>& keys,
                                       pointer<system, Value>& values,
                                       pointer<system, Key>& output_keys,
                                       pointer<system, Value>& output_values,
                                       allocation<system, uint8>& temp_storage,
                                       ReductionOp reduction_op)
    {
        return reduce_by_key(keys.begin(),
                             keys.end(),
                             values.begin(),
                             output_keys.begin(),
                             output_values.begin(),
                             temp_storage,
                             reduction_op);
    }

    // computes a run length encoding
    // returns the number of runs
    template <typename InputIterator, typename UniqueOutputIterator, typename LengthOutputIterator>
    static inline size_t run_length_encode(InputIterator keys_input,
                                           size_t num_keys,
                                           UniqueOutputIterator unique_keys_output,
                                           LengthOutputIterator run_lengths_output,
                                           allocation<system, uint8>& temp_storage)
    {
        return thrust::reduce_by_key(lift::backend_policy<system>::execution_policy(),
                                     keys_input, keys_input + num_keys,
                                     thrust::constant_iterator<uint32>(1),
                                     unique_keys_output,
                                     run_lengths_output).first - unique_keys_output;
    }

    static inline void synchronize()
    { }

    static inline void check_errors(void)
    { }
};

// default to thrust
template <target_system system>
struct parallel : public parallel_thrust<system>
{
    using parallel_thrust<system>::for_each;
    using parallel_thrust<system>::inclusive_scan;
    using parallel_thrust<system>::copy_if;
    using parallel_thrust<system>::copy_flagged;
    using parallel_thrust<system>::sum;
    using parallel_thrust<system>::sort_by_key;
    using parallel_thrust<system>::reduce_by_key;
    using parallel_thrust<system>::run_length_encode;

    using parallel_thrust<system>::synchronize;
    using parallel_thrust<system>::check_errors;
};

// specialization for the cuda backend based on CUB primitives
template <>
struct parallel<cuda> : public parallel_thrust<cuda>
{
    template <typename InputIterator, typename UnaryFunction>
    static inline InputIterator for_each(InputIterator first, InputIterator last, UnaryFunction f, int2 launch_parameters = { 0, 0 })
    {
        lift::for_each(first,
                       last - first,
                       f,
                       launch_parameters);
        return last;
    }

    // shortcut to run for_each on a pointer
    template <typename T, typename UnaryFunction>
    static inline typename pointer<cuda, T>::iterator_type for_each(pointer<cuda, T>& data, UnaryFunction f, int2 launch_parameters = { 0, 0 })
    {
        lift::for_each(data.begin(),
                       data.size(),
                       f,
                       launch_parameters);

        return data.end();
    }

    // shortcut to run for_each on [range.x, range.y[
    template <typename UnaryFunction>
    static inline void for_each(uint2 range, UnaryFunction f, int2 launch_parameters = { 0, 0 })
    {
        parallel::for_each(thrust::make_counting_iterator(range.x),
                           thrust::make_counting_iterator(range.y),
                           f,
                           launch_parameters);
    }

    // shortcut to run for_each on [0, end[
    template <typename UnaryFunction>
    static inline void for_each(uint32 end, UnaryFunction f, int2 launch_parameters = { 0, 0 })
    {
        parallel::for_each(thrust::make_counting_iterator(0u),
                           thrust::make_counting_iterator(end),
                           f,
                           launch_parameters);
    }

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline void inclusive_scan(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      Predicate op)
    {
        thrust::inclusive_scan(lift::backend_policy<cuda>::execution_policy(),
                               first, first + len, result, op);
    }

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline size_t copy_if(InputIterator first,
                                 size_t len,
                                 OutputIterator result,
                                 Predicate op,
                                 allocation<cuda, uint8>& temp_storage)
    {
        scoped_allocation<cuda, int32> num_selected(1);

        // determine amount of temp storage required
        size_t temp_bytes = 0;
        cub::DeviceSelect::If(nullptr,
                              temp_bytes,
                              first,
                              result,
                              num_selected.begin(),
                              len,
                              op);

        // make sure we have enough temp storage
        temp_storage.resize(temp_bytes);

        cub::DeviceSelect::If(temp_storage.data(),
                              temp_bytes,
                              first,
                              result,
                              num_selected.begin(),
                              len,
                              op);

        return size_t(num_selected[0]);
    }

    // xxxnsubtil: cub::DeviceSelect::Flagged seems problematic
#if !WAR_CUB_COPY_FLAGGED
    template <typename InputIterator, typename FlagIterator, typename OutputIterator>
    static inline size_t copy_flagged(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      FlagIterator flags,
                                      vector<cuda, uint8>& temp_storage)
    {
        vector<cuda, size_t> num_selected(1);

        // determine amount of temp storage required
        size_t temp_bytes = 0;
        cub::DeviceSelect::Flagged(nullptr,
                temp_bytes,
                first,
                flags,
                result,
                num_selected.begin(),
                len);

        // make sure we have enough temp storage
        temp_storage.resize(temp_bytes);

        cub::DeviceSelect::Flagged(thrust::raw_pointer_cast(temp_storage.data()),
                temp_bytes,
                first,
                flags,
                result,
                num_selected.begin(),
                len);

        return size_t(num_selected[0]);
    }
#else
    using parallel_thrust<cuda>::copy_flagged;
#endif

    template <typename InputIterator>
    static inline int64 sum(InputIterator first,
                            size_t len,
                            allocation<cuda, uint8>& temp_storage)
    {
        scoped_allocation<cuda, int64> result(1);

        size_t temp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr,
                               temp_bytes,
                               first,
                               result.begin(),
                               len);

        temp_storage.resize(temp_bytes);

        cub::DeviceReduce::Sum(temp_storage.data(),
                               temp_bytes,
                               first,
                               result.begin(),
                               len);

        return int64(result[0]);
    }

    template <typename Key, typename Value>
    static inline void sort_by_key(pointer<cuda, Key>& keys,
                                   pointer<cuda, Value>& values,
                                   pointer<cuda, Key>& temp_keys,
                                   pointer<cuda, Value>& temp_values,
                                   allocation<cuda, uint8>& temp_storage,
                                   int num_key_bits = sizeof(Key) * 8)
    {
        const size_t len = keys.size();
        assert(keys.size() == values.size());

        temp_keys.resize(len);
        temp_values.resize(len);

        cub::DoubleBuffer<Key> d_keys(keys.data(), temp_keys.data());
        cub::DoubleBuffer<Value> d_values(values.data(), temp_values.data());

        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr,
                                        temp_storage_bytes,
                                        d_keys,
                                        d_values,
                                        len,
                                        0,
                                        num_key_bits);

        temp_storage.resize(temp_storage_bytes);

        cub::DeviceRadixSort::SortPairs(temp_storage.data(),
                                        temp_storage_bytes,
                                        d_keys,
                                        d_values,
                                        len,
                                        0,
                                        num_key_bits);

        if (keys.data() != d_keys.Current())
        {
            cudaMemcpy(keys.data(), d_keys.Current(), sizeof(Key) * len, cudaMemcpyDeviceToDevice);
        }

        if (values.data() != d_values.Current())
        {
            cudaMemcpy(values.data(), d_values.Current(), sizeof(Value) * len, cudaMemcpyDeviceToDevice);
        }
    }

    // returns the size of the output key/value vectors
    template <typename Key, typename Value, typename ReductionOp>
    static inline size_t reduce_by_key(pointer<cuda, Key>& keys,
                                       pointer<cuda, Value>& values,
                                       pointer<cuda, Key>& output_keys,
                                       pointer<cuda, Value>& output_values,
                                       allocation<cuda, uint8>& temp_storage,
                                       ReductionOp reduction_op)
    {
        const size_t len = keys.size();
        assert(keys.size() == values.size());

        output_keys.resize(len);
        output_values.resize(len);

        return reduce_by_key(keys.begin(),
                             keys.end(),
                             values.begin(),
                             output_keys.begin(),
                             output_values.begin(),
                             temp_storage,
                             reduction_op);
    }

    template <typename KeyIterator, typename ValueIterator, typename ReductionOp>
    static inline size_t reduce_by_key(KeyIterator keys_begin,
                                       KeyIterator keys_end,
                                       ValueIterator values_begin,
                                       KeyIterator output_keys,
                                       ValueIterator output_values,
                                       allocation<cuda, uint8>& temp_storage,
                                       ReductionOp reduction_op)
    {
        const size_t len = keys_end - keys_begin;

        scoped_allocation<cuda, uint32> num_segments(1);

        size_t temp_storage_bytes = 0;

        cub::DeviceReduce::ReduceByKey(nullptr,
                                       temp_storage_bytes,
                                       keys_begin,
                                       output_keys,
                                       values_begin,
                                       output_values,
                                       num_segments.data(),
                                       reduction_op,
                                       len);

        temp_storage.resize(temp_storage_bytes);

        cub::DeviceReduce::ReduceByKey(temp_storage.data(),
                                       temp_storage_bytes,
                                       keys_begin,
                                       output_keys,
                                       values_begin,
                                       output_values,
                                       num_segments.data(),
                                       reduction_op,
                                       len);

        return num_segments[0];
    }

    template <typename InputIterator, typename UniqueOutputIterator, typename LengthOutputIterator>
    static inline size_t run_length_encode(InputIterator keys_input,
                                           size_t num_keys,
                                           UniqueOutputIterator unique_keys_output,
                                           LengthOutputIterator run_lengths_output,
                                           allocation<cuda, uint8>& temp_storage)
    {
        scoped_allocation<cuda, int64> result(1);
        size_t temp_bytes = 0;

        cub::DeviceRunLengthEncode::Encode(nullptr,
                                           temp_bytes,
                                           keys_input,
                                           unique_keys_output,
                                           run_lengths_output,
                                           result.data(),
                                           num_keys);

        temp_storage.resize(temp_bytes);

        cub::DeviceRunLengthEncode::Encode(temp_storage.data(),
                                           temp_bytes,
                                           keys_input,
                                           unique_keys_output,
                                           run_lengths_output,
                                           result.data(),
                                           num_keys);

        return size_t(result[0]);
    }

    static inline void synchronize(void)
    {
        cudaDeviceSynchronize();
    }

    static inline void check_errors(void)
    {
        synchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            int device;
            cudaGetDevice(&device);

            fprintf(stderr, "CUDA device %d: error %d (%s)\n", device, err, cudaGetErrorString(err));
            abort();
        }
    }
};

} // namespace lift

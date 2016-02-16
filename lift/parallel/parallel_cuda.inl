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

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_run_length_encode.cuh>
// silence warnings from debug code in cub
#ifdef __GNU_C__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <cub/device/device_radix_sort.cuh>

#ifdef __GNU_C__
#pragma GCC diagnostic pop
#endif

#include "../types.h"
#include "../algorithms/cuda/for_each.h"

namespace lift {

struct device_copy_if_flagged
{
    LIFT_DEVICE bool operator() (const uint8 val)
    {
        return bool(val);
    }
};

template <>
template <typename InputIterator, typename UnaryFunction>
inline void parallel<cuda>::for_each(InputIterator first,
                                     InputIterator last,
                                     UnaryFunction f,
                                     int2 launch_parameters)
{
    lift::for_each(first,
                   last - first,
                   f,
                   launch_parameters);
}

// shortcut to run for_each on a pointer
template <>
template <typename T, typename UnaryFunction>
inline void parallel<cuda>::for_each(pointer<cuda, T>& data,
                                     UnaryFunction f,
                                     int2 launch_parameters)
{
    lift::for_each(data.begin(),
                   data.size(),
                   f,
                   launch_parameters);
}

// shortcut to run for_each on [range.x, range.y[
template <>
template <typename UnaryFunction>
inline void parallel<cuda>::for_each(uint2 range,
                                     UnaryFunction f,
                                     int2 launch_parameters)
{
    lift::for_each(thrust::make_counting_iterator(range.x),
                   range.y - range.x,
                   f,
                   launch_parameters);
}

// shortcut to run for_each on [0, end[
template <>
template <typename UnaryFunction>
inline void parallel<cuda>::for_each(uint32 end,
                                     UnaryFunction f,
                                     int2 launch_parameters)
{
    lift::for_each(thrust::make_counting_iterator(0u),
                   end,
                   f,
                   launch_parameters);
}

template <target_system system, typename d_type>
struct fill_by_index
{
    pointer<system, d_type> data;
    d_type value;

    fill(pointer<system, d_type> &data, d_type value)
        : data(data), value(value)
    { }

    LIFT_HOST_DEVICE void operator() (const int index)
    {
        data[index] = value;
    }
};

template <target_system system, typename d_type>
struct fill_by_reference
{
    pointer<system, d_type> data;
    d_type value;

    fill(pointer<system, d_type> &data, d_type value)
        : data(data), value(value)
    { }

    LIFT_HOST_DEVICE void operator() (const int &ref)
    {
        ref = value;
    }
};

template <>
template <typename InputIterator, typename d_type>
inline void parallel<cuda>::fill(InputIterator begin,
                            InputIterator end,
                            d_type value,
                            pointer<system, d_type>& vector,
                            int2 launch_parameters = { 0, 0 })
{
    for_each(begin, end, fill_by_reference<cuda, d_type>(vector, value), launch_parameters);
}

template <>
template <typename d_type>
inline void parallel<cuda>::fill(pointer<system, d_type>& vector,
                            d_type value,
                            int2 launch_parameters = { 0, 0 })
{
    for_each(vector, fill_by_reference<cuda, d_type>(vector, value), launch_parameters);
}

template <>
template <typename d_type>
inline void parallel<cuda>::fill(uint2 range,
                            d_type value,
                            pointer<system, d_type>& vector,
                            int2 launch_parameters = { 0, 0 })
{
    for_each(range, fill_by_index<cuda, d_type>(vector, value), launch_parameters);
}

template <>
template <typename d_type>
inline void parallel<cuda>::fill(uint32 end,
                            d_type value,
                            pointer<system, d_type>& vector,
                            int2 launch_parameters = { 0, 0 })
{
    for_each(end, fill_by_index<cuda, d_type>(vector, value), launch_parameters);
}

template <>
template <typename InputIterator, typename OutputIterator, typename Predicate>
inline void parallel<cuda>::inclusive_scan(InputIterator first,
                                           size_t len,
                                           OutputIterator result,
                                           Predicate op)
{
    thrust::inclusive_scan(lift::backend_policy<cuda>::execution_policy(),
                           first, first + len, result, op);
}

template <>
template <typename InputIterator, typename OutputIterator, typename Predicate>
inline size_t parallel<cuda>::copy_if(InputIterator first,
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

    return size_t(num_selected.peek(0));
}

// xxxnsubtil: cub::DeviceSelect::Flagged seems problematic
template <>
template <typename InputIterator, typename FlagIterator, typename OutputIterator>
inline size_t parallel<cuda>::copy_flagged(InputIterator first,
                                           size_t len,
                                           OutputIterator result,
                                           FlagIterator flags,
                                           allocation<cuda, uint8>& temp_storage)
{
#if !WAR_CUB_COPY_FLAGGED
    scoped_allocation<cuda, size_t> num_selected(1);

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

    return size_t(num_selected.peek(0));
#else
    OutputIterator out_last;
    out_last = thrust::copy_if(lift::backend_policy<cuda>::execution_policy(),
                               first, first + len, flags, result, device_copy_if_flagged());
    return out_last - result;
#endif
}

template <>
template <typename InputIterator>
inline auto parallel<cuda>::sum(InputIterator first,
                                size_t len,
                                allocation<cuda, uint8>& temp_storage)
    -> typename std::iterator_traits<InputIterator>::value_type
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    scoped_allocation<cuda, value_type> result(1);

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

    return value_type(result.peek(0));
}

template <>
template <typename Key>
inline void parallel<cuda>::sort(allocation<cuda, Key>& keys,
                                 allocation<cuda, Key>& temp_keys,
                                 allocation<cuda, uint8>& temp_storage)
{
    const size_t len = keys.size();

    temp_keys.resize(len);

    cub::DoubleBuffer<Key> d_keys(keys.data(), temp_keys.data());

    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr,
                                   temp_bytes,
                                   d_keys,
                                   len);

    temp_storage.resize(temp_bytes);
    cub::DeviceRadixSort::SortKeys(temp_storage.data(),
                                   temp_bytes,
                                   d_keys,
                                   len);

    if (keys.data() != d_keys.Current())
    {
        cudaMemcpy(keys.data(), d_keys.Current(), sizeof(Key) * len, cudaMemcpyDeviceToDevice);
    }
}

template <>
template <typename Key, typename Value>
inline void parallel<cuda>::sort_by_key(pointer<cuda, Key>& keys,
                                        pointer<cuda, Value>& values,
                                        allocation<cuda, Key>& temp_keys,
                                        allocation<cuda, Value>& temp_values,
                                        allocation<cuda, uint8>& temp_storage,
                                        int num_key_bits)
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
template <>
template <typename Key, typename Value, typename ReductionOp>
inline size_t parallel<cuda>::reduce_by_key(pointer<cuda, Key>& keys,
                                            pointer<cuda, Value>& values,
                                            allocation<cuda, Key>& output_keys,
                                            allocation<cuda, Value>& output_values,
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

template <>
template <typename KeyIterator, typename ValueIterator, typename ReductionOp>
inline size_t parallel<cuda>::reduce_by_key(KeyIterator keys_begin,
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

    return num_segments.peek(0);
}

template <>
template <typename InputIterator, typename UniqueOutputIterator, typename LengthOutputIterator>
inline size_t parallel<cuda>::run_length_encode(InputIterator keys_input,
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

    return size_t(result.peek(0));
}

template <>
inline void parallel<cuda>::synchronize(void)
{
    cudaDeviceSynchronize();
}

template <>
inline void parallel<cuda>::check_errors(void)
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

} // namespace lift

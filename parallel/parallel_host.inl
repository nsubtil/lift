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

#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

namespace lift {

struct host_copy_if_flagged
{
    bool operator() (const uint8 val)
    {
        return bool(val);
    }
};

template <typename InputIterator, typename UnaryFunction>
InputIterator parallel<host>::for_each(InputIterator first,
                                       InputIterator last,
                                       UnaryFunction f,
                                       int2 launch_parameters)
{
    return thrust::for_each(lift::backend_policy<host>::execution_policy(),
                            first,
                            last,
                            f);
}

// shortcut to run for_each on a lift pointer
template <typename T, typename UnaryFunction>
typename pointer<host, T>::iterator parallel<host>::for_each(pointer<host, T>& vector,
                                                             UnaryFunction f,
                                                             int2 launch_parameters)
{
    return thrust::for_each(lift::backend_policy<host>::execution_policy(),
                            vector.begin(),
                            vector.end(),
                            f);
}

// shortcut to run for_each on [range.x, range.y[
template <typename UnaryFunction>
void parallel<host>::for_each(uint2 range,
                              UnaryFunction f,
                              int2 launch_parameters)
{
    thrust::for_each(lift::backend_policy<host>::execution_policy(),
                     thrust::make_counting_iterator(range.x),
                     thrust::make_counting_iterator(range.y),
                     f);
}

// shortcut to run for_each on [0, end[
template <typename UnaryFunction>
void parallel<host>::for_each(uint32 end,
                              UnaryFunction f,
                              int2 launch_parameters)
{
    thrust::for_each(lift::backend_policy<host>::execution_policy(),
                     thrust::make_counting_iterator(0u),
                     thrust::make_counting_iterator(end),
                     f);
}

template <typename InputIterator, typename OutputIterator, typename Predicate>
void parallel<host>::inclusive_scan(InputIterator first,
                                    size_t len,
                                    OutputIterator result,
                                    Predicate op)
{
    thrust::inclusive_scan(lift::backend_policy<host>::execution_policy(),
                           first, first + len, result, op);
}

template <typename InputIterator, typename OutputIterator, typename Predicate>
size_t parallel<host>::copy_if(InputIterator first,
                               size_t len,
                               OutputIterator result,
                               Predicate op,
                               allocation<host, uint8>& temp_storage)
{
    OutputIterator out_last;
    out_last = thrust::system::tbb::detail::copy_if(typename lift::backend_policy<host>::tag(),
                                                    first, first + len, first, result, op);
    return out_last - result;
}

template <typename InputIterator, typename OutputIterator, typename Predicate>
size_t parallel<host>::copy_if(InputIterator first,
                               size_t len,
                               OutputIterator result,
                               Predicate op,
                               vector<host, uint8>& temp_storage)
{
    // use the fallback thrust version
    OutputIterator out_last;
    out_last = thrust::system::tbb::detail::copy_if(typename lift::backend_policy<host>::tag(),
                                                    first, first + len, first, result, op);
    return out_last - result;
}

template <typename InputIterator, typename FlagIterator, typename OutputIterator>
size_t parallel<host>::copy_flagged(InputIterator first,
                                    size_t len,
                                    OutputIterator result,
                                    FlagIterator flags,
                                    allocation<host, uint8>& temp_storage)
{
    OutputIterator out_last;
    out_last = thrust::copy_if(lift::backend_policy<host>::execution_policy(),
                               first, first + len, flags, result, host_copy_if_flagged());
    return out_last - result;
}

template <typename Key, typename Value>
void parallel<host>::sort_by_key(pointer<host, Key>& keys,
                                 pointer<host, Value>& values,
                                 pointer<host, Key>& temp_keys,
                                 pointer<host, Value>& temp_values,
                                 allocation<host, uint8>& temp_storage,
                                 int num_key_bits)
{
    thrust::sort_by_key(lift::backend_policy<host>::execution_policy(),
                        keys.begin(), keys.end(), values.begin());
}

template <typename Key, typename Value>
void parallel<host>::sort_by_key(vector<host, Key>& keys,
                                 vector<host, Value>& values,
                                 vector<host, Key>& temp_keys,
                                 vector<host, Value>& temp_values,
                                 vector<host, uint8>& temp_storage,
                                 int num_key_bits)
{
    thrust::sort_by_key(lift::backend_policy<host>::execution_policy(),
                        keys.begin(), keys.end(), values.begin());
}

// returns the size of the output key/value
template <typename KeyIterator, typename ValueIterator, typename ReductionOp>
size_t parallel<host>::reduce_by_key(KeyIterator keys_begin,
                                     KeyIterator keys_end,
                                     ValueIterator values_begin,
                                     KeyIterator output_keys,
                                     ValueIterator output_values,
                                     allocation<host, uint8>& temp_storage,
                                     ReductionOp reduction_op)
{
    auto out = thrust::reduce_by_key(lift::backend_policy<host>::execution_policy(),
                                     keys_begin,
                                     keys_end,
                                     values_begin,
                                     output_keys,
                                     output_values,
                                     thrust::equal_to<typeof(*keys_begin)>(),
                                     reduction_op);

    return out.first - output_keys;
}

template <typename KeyIterator, typename ValueIterator, typename ReductionOp>
size_t parallel<host>::reduce_by_key(KeyIterator keys_begin,
                                     KeyIterator keys_end,
                                     ValueIterator values_begin,
                                     KeyIterator output_keys,
                                     ValueIterator output_values,
                                     vector<host, uint8>& temp_storage,
                                     ReductionOp reduction_op)
{
    auto out = thrust::reduce_by_key(lift::backend_policy<host>::execution_policy(),
                                     keys_begin,
                                     keys_end,
                                     values_begin,
                                     output_keys,
                                     output_values,
                                     thrust::equal_to<typeof(*keys_begin)>(),
                                     reduction_op);

    return out.first - output_keys;
}

// returns the size of the output key/value vectors
template <typename Key, typename Value, typename ReductionOp>
size_t parallel<host>::reduce_by_key(pointer<host, Key>& keys,
                                     pointer<host, Value>& values,
                                     pointer<host, Key>& output_keys,
                                     pointer<host, Value>& output_values,
                                     allocation<host, uint8>& temp_storage,
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

template <typename Key, typename Value, typename ReductionOp>
size_t parallel<host>::reduce_by_key(vector<host, Key>& keys,
                                     vector<host, Value>& values,
                                     vector<host, Key>& output_keys,
                                     vector<host, Value>& output_values,
                                     vector<host, uint8>& temp_storage,
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
size_t parallel<host>::run_length_encode(InputIterator keys_input,
                                         size_t num_keys,
                                         UniqueOutputIterator unique_keys_output,
                                         LengthOutputIterator run_lengths_output,
                                         allocation<host, uint8>& temp_storage)
{
    return thrust::reduce_by_key(lift::backend_policy<host>::execution_policy(),
                                 keys_input, keys_input + num_keys,
                                 thrust::constant_iterator<uint32>(1),
                                 unique_keys_output,
                                 run_lengths_output).first - unique_keys_output;
}

void parallel<host>::synchronize()
{ }

void parallel<host>::check_errors(void)
{ }

} // namespace lift

#include "tbb/reduction.inl"

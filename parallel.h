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

#include <iterator>

#include "types.h"

#include "vector.h"
#include "decorators.h"
#include "backends.h"
#include "memory.h"

namespace lift {

template <target_system system>
struct parallel
{ };

template <>
struct parallel<host>
{
    template <typename InputIterator, typename UnaryFunction>
    static inline InputIterator for_each(InputIterator first,
                                         InputIterator last,
                                         UnaryFunction f,
                                         int2 launch_parameters = { 0, 0 });

    // shortcut to run for_each on a lift pointer
    template <typename T, typename UnaryFunction>
    static inline typename pointer<host, T>::iterator for_each(pointer<host, T>& vector,
                                                               UnaryFunction f,
                                                               int2 launch_parameters = { 0, 0 });

    // shortcut to run for_each on [range.x, range.y[
    template <typename UnaryFunction>
    static inline void for_each(uint2 range,
                                UnaryFunction f,
                                int2 launch_parameters = { 0, 0 });

    // shortcut to run for_each on [0, end[
    template <typename UnaryFunction>
    static inline void for_each(uint32 end,
                                UnaryFunction f,
                                int2 launch_parameters = { 0, 0 });

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline void inclusive_scan(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      Predicate op);

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline size_t copy_if(InputIterator first,
                                 size_t len,
                                 OutputIterator result,
                                 Predicate op,
                                 allocation<host, uint8>& temp_storage);

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline size_t copy_if(InputIterator first,
                                 size_t len,
                                 OutputIterator result,
                                 Predicate op,
                                 vector<host, uint8>& temp_storage);

    template <typename InputIterator, typename FlagIterator, typename OutputIterator>
    static inline size_t copy_flagged(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      FlagIterator flags,
                                      allocation<host, uint8>& temp_storage);

    template <typename InputIterator>
    static inline auto sum(InputIterator first,
                           size_t len,
                           allocation<host, uint8>& temp_storage) -> typename std::iterator_traits<InputIterator>::value_type;

    template <typename InputIterator>
    static inline auto sum(InputIterator first,
                           size_t len,
                           vector<host, uint8>& temp_storage) -> typename std::iterator_traits<InputIterator>::value_type;

    template <typename Key, typename Value>
    static inline void sort_by_key(pointer<host, Key>& keys,
                                   pointer<host, Value>& values,
                                   pointer<host, Key>& temp_keys,
                                   pointer<host, Value>& temp_values,
                                   allocation<host, uint8>& temp_storage,
                                   int num_key_bits = sizeof(Key) * 8);

    template <typename Key, typename Value>
    static inline void sort_by_key(vector<host, Key>& keys,
                                   vector<host, Value>& values,
                                   vector<host, Key>& temp_keys,
                                   vector<host, Value>& temp_values,
                                   vector<host, uint8>& temp_storage,
                                   int num_key_bits = sizeof(Key) * 8);

    // returns the size of the output key/value
    template <typename KeyIterator, typename ValueIterator, typename ReductionOp>
    static inline size_t reduce_by_key(KeyIterator keys_begin,
                                       KeyIterator keys_end,
                                       ValueIterator values_begin,
                                       KeyIterator output_keys,
                                       ValueIterator output_values,
                                       allocation<host, uint8>& temp_storage,
                                       ReductionOp reduction_op);

    template <typename KeyIterator, typename ValueIterator, typename ReductionOp>
    static inline size_t reduce_by_key(KeyIterator keys_begin,
                                       KeyIterator keys_end,
                                       ValueIterator values_begin,
                                       KeyIterator output_keys,
                                       ValueIterator output_values,
                                       vector<host, uint8>& temp_storage,
                                       ReductionOp reduction_op);

    // returns the size of the output key/value vectors
    template <typename Key, typename Value, typename ReductionOp>
    static inline size_t reduce_by_key(pointer<host, Key>& keys,
                                       pointer<host, Value>& values,
                                       pointer<host, Key>& output_keys,
                                       pointer<host, Value>& output_values,
                                       allocation<host, uint8>& temp_storage,
                                       ReductionOp reduction_op);

    template <typename Key, typename Value, typename ReductionOp>
    static inline size_t reduce_by_key(vector<host, Key>& keys,
                                       vector<host, Value>& values,
                                       vector<host, Key>& output_keys,
                                       vector<host, Value>& output_values,
                                       vector<host, uint8>& temp_storage,
                                       ReductionOp reduction_op);

    // computes a run length encoding
    // returns the number of runs
    template <typename InputIterator, typename UniqueOutputIterator, typename LengthOutputIterator>
    static inline size_t run_length_encode(InputIterator keys_input,
                                           size_t num_keys,
                                           UniqueOutputIterator unique_keys_output,
                                           LengthOutputIterator run_lengths_output,
                                           allocation<host, uint8>& temp_storage);

    static inline void synchronize();

    static inline void check_errors(void);
};

template <>
struct parallel<cuda>
{
    template <typename InputIterator, typename UnaryFunction>
    static inline InputIterator for_each(InputIterator first,
                                         InputIterator last,
                                         UnaryFunction f,
                                         int2 launch_parameters = { 0, 0 });

    // shortcut to run for_each on a pointer
    template <typename T, typename UnaryFunction>
    static inline typename pointer<cuda, T>::iterator_type for_each(pointer<cuda, T>& data,
                                                                    UnaryFunction f,
                                                                    int2 launch_parameters = { 0, 0 });

    // shortcut to run for_each on [range.x, range.y[
    template <typename UnaryFunction>
    static inline void for_each(uint2 range, UnaryFunction f, int2 launch_parameters = { 0, 0 });

    // shortcut to run for_each on [0, end[
    template <typename UnaryFunction>
    static inline void for_each(uint32 end, UnaryFunction f, int2 launch_parameters = { 0, 0 });

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline void inclusive_scan(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      Predicate op);

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline size_t copy_if(InputIterator first,
                                 size_t len,
                                 OutputIterator result,
                                 Predicate op,
                                 allocation<cuda, uint8>& temp_storage);

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline size_t copy_if(InputIterator first,
                                 size_t len,
                                 OutputIterator result,
                                 Predicate op,
                                 vector<cuda, uint8>& temp_storage);

    template <typename InputIterator, typename FlagIterator, typename OutputIterator>
    static inline size_t copy_flagged(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      FlagIterator flags,
                                      vector<cuda, uint8>& temp_storage);

    template <typename InputIterator>
    static inline auto sum(InputIterator first,
                           size_t len,
                           allocation<cuda, uint8>& temp_storage) -> typename std::iterator_traits<InputIterator>::value_type;

    template <typename InputIterator>
    static inline auto sum(InputIterator first,
                           size_t len,
                           vector<cuda, uint8>& temp_storage) -> typename std::iterator_traits<InputIterator>::value_type;

    template <typename Key, typename Value>
    static inline void sort_by_key(pointer<cuda, Key>& keys,
                                   pointer<cuda, Value>& values,
                                   pointer<cuda, Key>& temp_keys,
                                   pointer<cuda, Value>& temp_values,
                                   allocation<cuda, uint8>& temp_storage,
                                   int num_key_bits = sizeof(Key) * 8);

    template <typename Key, typename Value>
    static inline void sort_by_key(vector<cuda, Key>& keys,
                                   vector<cuda, Value>& values,
                                   vector<cuda, Key>& temp_keys,
                                   vector<cuda, Value>& temp_values,
                                   vector<cuda, uint8>& temp_storage,
                                   int num_key_bits = sizeof(Key) * 8);

    // returns the size of the output key/value vectors
    template <typename Key, typename Value, typename ReductionOp>
    static inline size_t reduce_by_key(pointer<cuda, Key>& keys,
                                       pointer<cuda, Value>& values,
                                       pointer<cuda, Key>& output_keys,
                                       pointer<cuda, Value>& output_values,
                                       allocation<cuda, uint8>& temp_storage,
                                       ReductionOp reduction_op);

    template <typename Key, typename Value, typename ReductionOp>
    static inline size_t reduce_by_key(vector<cuda, Key>& keys,
                                       vector<cuda, Value>& values,
                                       vector<cuda, Key>& output_keys,
                                       vector<cuda, Value>& output_values,
                                       vector<cuda, uint8>& temp_storage,
                                       ReductionOp reduction_op);

    template <typename KeyIterator, typename ValueIterator, typename ReductionOp>
    static inline size_t reduce_by_key(KeyIterator keys_begin,
                                       KeyIterator keys_end,
                                       ValueIterator values_begin,
                                       KeyIterator output_keys,
                                       ValueIterator output_values,
                                       allocation<cuda, uint8>& temp_storage,
                                       ReductionOp reduction_op);

    template <typename KeyIterator, typename ValueIterator, typename ReductionOp>
    static inline size_t reduce_by_key(KeyIterator keys_begin,
                                       KeyIterator keys_end,
                                       ValueIterator values_begin,
                                       KeyIterator output_keys,
                                       ValueIterator output_values,
                                       vector<cuda, uint8>& temp_storage,
                                       ReductionOp reduction_op);

    template <typename InputIterator, typename UniqueOutputIterator, typename LengthOutputIterator>
    static inline size_t run_length_encode(InputIterator keys_input,
                                           size_t num_keys,
                                           UniqueOutputIterator unique_keys_output,
                                           LengthOutputIterator run_lengths_output,
                                           allocation<cuda, uint8>& temp_storage);

    static inline void synchronize(void);

    static inline void check_errors(void);
};

} // namespace lift

#include "parallel/parallel_cuda.inl"
#include "parallel/parallel_host.inl"

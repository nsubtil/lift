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

#include "decorators.h"
#include "backends.h"
#include "memory.h"

namespace lift {

/**
 * Dispatch structure for parallel primitives. Implementations are exposed in specializations of
 * this structure for each target system.
 */
template <target_system system>
struct parallel
{
    /**
     * Parallel for-each implementation. Applies UnaryFunction f to each element in the range [begin, end[.
     * \anchor for_each
     *
     * \tparam InputIterator Iterator type for input data.
     * \tparam UnaryFunction Any callable type taking a single argument. The argument can either be a
     *                       value of the same type obtained by dereferencing InputIterator, or else
     *                       a (const or modifiable) reference to that type. The return value is ignored.
     *
     * \param begin             Iterator pointing at the first element to be processed.
     * \param end               Iterator pointing at the end of the range to be processed.
     * \param f                 The function object to be applied to each item in the input.
     * \param launch_parameters Grid launch parameters for GPU backend.
     */
    template <typename InputIterator, typename UnaryFunction>
    static inline void for_each(InputIterator begin,
                                InputIterator end,
                                UnaryFunction f,
                                int2 launch_parameters = { 0, 0 });

    /**
     * Pointer version of for-each. Applies \c UnaryFunction \c f to each element behind \c vector.
     *
     * \tparam T                The underlying data type for \c vector.
     * \tparam UnaryFunction    Same as in \ref for_each
     *
     * \param vector            The memory region to apply \c f to
     * \param f                 UnaryFunction to apply to each element in \c vector
     * \param launch_parameters Same as in \ref for_each
     */
    template <typename T, typename UnaryFunction>
    static inline void for_each(pointer<system, T>& vector,
                                UnaryFunction f,
                                int2 launch_parameters = { 0, 0 });

    /**
     * Range-based version of for-each. Applies \c UnaryFunction \c f to each integer in the range
     * [range.x, range.y[ in parallel.
     *
     * \tparam UnaryFunction    Same as in \ref for_each
     *
     * \param range             Integer range to iterate over.
     * \param f                 Same as in \ref for_each
     * \param launch_parameters Same as in \ref for_each
     */
    template <typename UnaryFunction>
    static inline void for_each(uint2 range,
                                UnaryFunction f,
                                int2 launch_parameters = { 0, 0 });

    /**
     * 0-based range version of for-each. Applies \c UnaryFunction \c f to each integer in the range
     * [0, end[ in parallel.
     *
     * \tparam UnaryFunction    Same as in \ref for_each
     *
     * \param end               End point of the interval to iterate over.
     * \param f                 Same as in \ref for_each
     * \param launch_parameters Same as in \ref for_each
     */
    template <typename UnaryFunction>
    static inline void for_each(uint32 end,
                                UnaryFunction f,
                                int2 launch_parameters = { 0, 0 });

    /**
     * Performs a parallel inclusive scan on [first, first + len[, using \c op as the scan operator.
     *
     * \c inclusive_scan performs an inclusive prefix-scan on the input buffer. The output is a
     * buffer of the same size, with each element replaced by the result of applying \c op to every
     * element from the start of the array up to and inclusing the element itself.
     *
     *
     * \tparam InputIterator    Type of the input data iterator.
     * \tparam OutputIterator   Type of the output data iterator.
     * \tparam Predicate
     *
     * \param first             Start of the range to perform the scan over.
     * \param len               Number of data elements to scan over.
     * \param result            Iterator to the start of the output buffer. Must not be the same
     *                          as the input buffer.
     * \param op                The binary operator to be used as the scan operator. Takes two
     *                          parameters of Input type, returns the scan value corresponding to
     *                          the input pair.
     */
    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline void inclusive_scan(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      Predicate op);

    /**
     * Performs stream compaction based on a predicate.
     *
     * \c copy_if copies every element in the input buffer for which \c Predicate evaluates to true
     * into the output buffer.
     *
     * \tparam InputIterator    Type of the input data iterator.
     * \tparam OutputIterator   Type of the output data iterator.
     *
     * \param first             Iterator to the start of the input buffer.
     * \param len               Number of elements in the input buffer.
     * \param result            Iterator to the start of the output buffer.
     * \param op                Determines whether each element is copied. Called with a single
     *                          argument (a copy of or reference to the input element), returns
     *                          a boolean value that determines whether each element is copied.
     * \param temp_storage      Temporary storage for use by the implementation. Will be resized
     *                          if too small.
     */
    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline size_t copy_if(InputIterator first,
                                 size_t len,
                                 OutputIterator result,
                                 Predicate op,
                                 allocation<system, uint8>& temp_storage);

    /**
     * Performs stream compaction based on a buffer with boolean flags.
     *
     * \c copy_flagged copies every element in the input buffer for which the corresponding index
     * in the flags buffer is true.
     *
     * \tparam InputIterator    Type of the input data iterator.
     * \tparam FlagIterator     Type of the input flags iterator.
     * \tparam OutputIterator   Type of the output data iterator.
     *
     * \param first             Iterator to the start of the input buffer.
     * \param len               Number of elements in the input buffer.
     * \param result            Iterator to the start of the output buffer.
     * \param flags             Determines whether each element is copied. For a given index, the
     *                          element data at that index in the input buffer is copied if and only
     *                          if the flag at the same index evaluates to true.
     * \param temp_storage      Temporary storage for use by the implementation. Will be resized
     *                          if too small.
     */
    template <typename InputIterator, typename FlagIterator, typename OutputIterator>
    static inline size_t copy_flagged(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      FlagIterator flags,
                                      allocation<system, uint8>& temp_storage);

    /**
     * Computes the arithmetic sum of a buffer.
     *
     * \c sum performs a reduction using the '+' operator on the input data and '0' as the initial
     * value for the reduction. This works on any integral data type.
     *
     * \tparam InputIterator    Type of the input data iterator.
     *
     * \param first             Iterator to the start of the input buffer.
     * \param len               Length of the input buffer.
     * \param temp_storage      Temporary storage for use by the implementation. Will be resized
     *                          if too small.
     */
    template <typename InputIterator>
    static inline auto sum(InputIterator first,
                           size_t len,
                           allocation<system, uint8>& temp_storage) -> typename std::iterator_traits<InputIterator>::value_type;

    /**
     * Perform a sort-by-key on a key + value buffer pair.
     *
     * \c sort_by_key sorts \c keys and \c values by ascending order of keys. In other words, it
     * permutes the contents of \c keys such that they are in ascending order and applies the same
     * permutation to the contents of \c values.
     *
     * \tparam Key              Data type for the keys
     * \tparam Value            Data type for the values
     *
     * \param keys              Pointer to the buffer containing the keys to be sorted.
     * \param values            Pointer to the buffer containing the values to be sorted.
     * \param temp_keys         Pointer to a temporary buffer to hold keys during sorting. Will be
     *                          resized to match the size of \c keys .
     * \param temp_values       Pointer to a temporary buffer to hold values during sorting. Will
     *                          be resized to match the size of \c values .
     * \param temp_storage      Temporary storage for use by the implementation. Will be resized
     *                          if too small.
     */
    template <typename Key, typename Value>
    static inline void sort_by_key(pointer<system, Key>& keys,
                                   pointer<system, Value>& values,
                                   allocation<system, Key>& temp_keys,
                                   allocation<system, Value>& temp_values,
                                   allocation<system, uint8>& temp_storage,
                                   int num_key_bits = sizeof(Key) * 8);

    /**
     * Sort a buffer of keys.
     *
     * \c sort sorts the contents of \c keys into ascending order.
     *
     * \tparam Key              Data type for the keys.
     *
     * \param keys              Pointer to the buffer containing the keys to be sorted.
     * \param temp_keys         Pointer to a temporary buffer to hold keys during sorting. Will be
     *                          resized to match the size of \c keys .
     */
    template <typename Key>
    static inline void sort(allocation<system, Key>& keys,
                            allocation<system, Key>& temp_keys,
                            allocation<system, uint8>& temp_storage);

    /**
     * Perform a reduction by key on a key/value buffer pair.
     *
     * \c reduce_by_key performs per-key reduction on a key/value buffer. For each set of consecutve
     * identical keys in [\c keys_begin, \c keys_end [, the corresponding values are reduced into a
     * single output value by applying a user-specified reduction operator.
     *
     * \tparam KeyIterator      Type of the key iterator
     * \tparam ValueIterator    Type of the value iterator
     * \tparam ReductionOp      Type of the reduction operator --- any callable object that can
     *                          be called with two elements from the value buffer and returns the
     *                          result of reducing those two elements.
     *
     * \param keys_begin        Iterator pointing at the start of the input key buffer
     * \param keys_end          Iterator pointing at the end of the input key buffer
     * \param values_begin      Iterator pointing at the start of the input value buffer
     * \param output_keys       Iterator pointing at the start of the output key buffer
     * \param output_values     Iterator pointing at the start of the output value buffer
     * \param temp_storage      Temporary storage for use by the implementation. Will be resized
     *                          if too small.
     * \param reduction_op      The reduction operator.
     *
     * \return                  The number of keys/values in the output buffers
     */
    template <typename KeyIterator, typename ValueIterator, typename ReductionOp>
    static inline size_t reduce_by_key(KeyIterator keys_begin,
                                       KeyIterator keys_end,
                                       ValueIterator values_begin,
                                       KeyIterator output_keys,
                                       ValueIterator output_values,
                                       allocation<system, uint8>& temp_storage,
                                       ReductionOp reduction_op);


    /**
     * Perform a reduction by key on a key/value buffer pair.
     *
     * \c reduce_by_key performs per-key reduction on a key/value buffer. For each set of consecutve
     * identical keys in [\c keys_begin, \c keys_end [, the corresponding values are reduced into a
     * single output value by applying a user-specified reduction operator.
     *
     * Note that the output buffers are resized to match the input size, but the number of elements
     * actually output (the return value from this function) is likely going to be smaller than
     * that. This aims to prevent reallocations that may shrink buffers which are intended to be
     * reused many times on the same amount of input data.
     *
     * \tparam Key              Data type for the keys
     * \tparam Value            Data type for the values
     * \tparam ReductionOp      Type of the reduction operator --- any callable object that can
     *                          be called with two elements from the value buffer and returns the
     *                          result of reducing those two elements.
     *
     * \param keys              Pointer to the buffer containing input keys
     * \param values            Pointer to the buffer containing input values
     * \param output_keys       Buffer for output keys. Will be resized to match the size of the
     *                          input buffer.
     * \param output_values     Iterator pointing at the start of the output value buffer.
     * \param temp_storage      Temporary storage for use by the implementation. Will be resized
     *                          if too small.
     * \param reduction_op      The reduction operator.
     *
     * \return                  The number of keys/values in the output buffers
     */
    template <typename Key, typename Value, typename ReductionOp>
    static inline size_t reduce_by_key(pointer<system, Key>& keys,
                                       pointer<system, Value>& values,
                                       allocation<system, Key>& output_keys,
                                       allocation<system, Value>& output_values,
                                       allocation<system, uint8>& temp_storage,
                                       ReductionOp reduction_op);

    // computes a run length encoding
    // returns the number of runs
    /**
     * Compute a run-length encoding of the input key buffer.
     *
     * \c run_length_encode computes the lenght of runs of identical keys in \c keys_input. It
     * outputs a key/value pair for each run of consecutive, identical keys in \c keys_input, where
     * the value contains the run length.
     *
     * It is conceptually identical to calling \c reduce_by_key where all values are equal to 1.
     *
     * \tparam InputIterator        Type of the input key iterator
     * \tparam UniqueOutputIterator Type of the output key iterator
     * \tparam LengthOutputIterator Type of the output iterator for run lengths. The underlying type
     *                              for this iterator must be an integer type.
     *
     * \param keys_input            Iterator to the start of the input buffer
     * \param num_keys              Number of keys in the input buffer
     * \param unique_keys_output    Iterator to the start of the output buffer for keys. The buffer
     *                              must be the same size as the input.
     * \param run_lengths_output    Iterator to the start of the output buffer for run lengths. The
     *                              buffer must be the same size (in elements) as the input.
     *
     * \return                      The number of key/length pairs generated in the output buffers.
     */
    template <typename InputIterator, typename UniqueOutputIterator, typename LengthOutputIterator>
    static inline size_t run_length_encode(InputIterator keys_input,
                                           size_t num_keys,
                                           UniqueOutputIterator unique_keys_output,
                                           LengthOutputIterator run_lengths_output,
                                           allocation<system, uint8>& temp_storage);
    /**
     * Parallel fill implementation. Sets each element in range [begin, end[ to value.
     * \tparam InputIterator    Iterator type
     * \tparam T                Type for input data.
     *
     * \param begin             Iterator pointing at the first element to be processed.
     * \param end               Iterator pointing at the end of the range to be processed.
     * \param value             The value with which to fill the range.
     */
    template <typename InputIterator, typename T>
    static inline void fill(InputIterator begin,
                            InputIterator end,
                            T value);
    /**
     * Pointer version of fill. Fills each element of vector with value
     *
     * \tparam T                Data type for vector.
     *
     * \param vector            The memory region to fill with value
     * \param value             Value to fill each element in  vector
     */
    template <typename T>
    static inline void fill(pointer<system, T>& vector,
                            T value);

    /**
     * Synchronizes the compute device.
     *
     * This function will wait for the compute device to execute all queued work and go idle. It is
     * a no-op on the CPU and is equivalent to calling cudaDeviceSynchronize() on the GPU.
     */
    static inline void synchronize();

    /**
     * Check for compute device errors.
     *
     * This function will check for any pending errors on the compute device. It is a no-op on the
     * CPU. On the GPU, it will poll the runtime for a pending error condition; if found, it will
     * be printed to the console and the calling process will be terminated.
     */
    static inline void check_errors(void);
};

} // namespace lift

#include "parallel/parallel_cuda.inl"
#include "parallel/parallel_host.inl"

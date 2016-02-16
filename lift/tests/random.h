/*
 * Lift
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015-2016, Nuno Subtil <subtil@gmail.com>
 * Copyright (c) 2015-2016, Roche Molecular Systems Inc.
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

#include <limits>
#include <lift/types.h>

namespace lift {

/**
 * Sets the PRNG seed for the random number generator.
 * This function is meant to reset the PRNG to a known state, ensuring that a deterministic sequence
 * is generated, such that tests are predictable and repeatable.
 *
 * @param seed The random number generator seed.
 */
void lift_rand_reset(const uint32 seed = 0xdeadbeef);

// \cond

template <typename T>
struct lift_rand_uniform_default_arguments
{
    static constexpr T min = 0;
    static constexpr T max = std::numeric_limits<T>::max();
};

template <>
struct lift_rand_uniform_default_arguments<float>
{
    static constexpr float min = 0.0;
    static constexpr float max = 1.0;
};

template <>
struct lift_rand_uniform_default_arguments<double>
{
    static constexpr double min = 0.0;
    static constexpr double max = 1.0;
};

// \endcond

/**
 * Returns a random number in the range [min, max] with a uniform PDF.
 *
 * This function generates uniformly distributed random numbers in an interval.
 * For floating-point data types, the interval is open and the range defaults
 * to [0.0, 1.0[. For integer types, the range is closed and defaults to the
 * entire range of representable values.
 *
 * \tparam T Data type for the random number to be generated.
 *
 * \param min The minimum allowed value
 * \param max The maximum allowed value. (For floating point numbers, only values smaller than this will be generated.)
 *
 * \return A uniformly-distributed random number.
 */
template <typename T>
T lift_rand_uniform(T min = lift_rand_uniform_default_arguments<T>::min,
                    T max = lift_rand_uniform_default_arguments<T>::max);

/**
 * Returns a random number with a normal PDF defined by mean and standard deviation.
 *
 * This function generates random numbers with a Gaussian PDF, defined by the mean and standard deviation.
 *
 * \tparam T Data type for the random number to be generated.
 *
 * \param mean The mean of the normal distribution.
 * \param stddev The standard deviation of the normal distribution.
 *
 * \return A random number with a normal PDF.
 */
template <typename T>
T lift_rand_normal(T mean, T stddev);

} // namespace lift

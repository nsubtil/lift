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

#include <lift/tests/random.h>

#include <random>
#include <cmath>
#include <typeinfo>

namespace lift {
namespace test {

static std::minstd_rand generator(0xdeadbeef);

void rand_reset(const uint32 seed)
{
    generator.seed(seed);
}

template <typename T>
struct uniform_distribution
{
    typedef typename std::uniform_int_distribution<T> type;
};

template <>
struct uniform_distribution<float>
{
    typedef typename std::uniform_real_distribution<float> type;
};

template <>
struct uniform_distribution<double>
{
    typedef typename std::uniform_real_distribution<double> type;
};

// return a number in [min, max]
template <typename T>
T rand_uniform(T min, T max)
{
    typename uniform_distribution<T>::type d(min, max);
    return d(generator);
}

template uint8  rand_uniform<uint8> (uint8  min, uint8  max);
template uint16 rand_uniform<uint16>(uint16 min, uint16 max);
template uint32 rand_uniform<uint32>(uint32 min, uint32 max);
template uint64 rand_uniform<uint64>(uint64 min, uint64 max);
template int8   rand_uniform<int8>  (int8   min, int8   max);
template int16  rand_uniform<int16> (int16  min, int16  max);
template int32  rand_uniform<int32> (int32  min, int32  max);
template int64  rand_uniform<int64> (int64  min, int64  max);
template float  rand_uniform<float> (float  min, float  max);
template double rand_uniform<double>(double min, double max);

template <typename T>
T rand_normal(T mean, T stddev)
{
    std::normal_distribution<> d(mean, stddev);
    auto val = d(generator);

    if (typeid(T) == typeid(float) ||
        typeid(T) == typeid(double))
    {
        return T(val);
    } else {
        return T(std::round(val));
    }
}

template uint8  rand_normal<uint8> (uint8  mean, uint8  stddev);
template uint16 rand_normal<uint16>(uint16 mean, uint16 stddev);
template uint32 rand_normal<uint32>(uint32 mean, uint32 stddev);
template uint64 rand_normal<uint64>(uint64 mean, uint64 stddev);
template int8   rand_normal<int8>  (int8   mean, int8   stddev);
template int16  rand_normal<int16> (int16  mean, int16  stddev);
template int32  rand_normal<int32> (int32  mean, int32  stddev);
template int64  rand_normal<int64> (int64  mean, int64  stddev);
template float  rand_normal<float> (float  mean, float  stddev);
template double rand_normal<double>(double mean, double stddev);

} // namespace test
} // namespace lift

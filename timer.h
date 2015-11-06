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

#include "types.h"
#include <sys/time.h>

#include <stack>
#include <tuple>

namespace lift {

template <target_system system>
struct timer
{
    struct timeval start_event;
    bool started;
    float time_counter;

    timer()
        : started(false), time_counter(0.0)
    { }

    void start(void)
    {
        if (started)
        {
            fprintf(stderr, "ERROR: inconsistent timer state");
            abort();
        }

        gettimeofday(&start_event, NULL);
        started = true;
    }

    void stop(void)
    {
        if (!started)
        {
            fprintf(stderr, "ERROR: inconsistent timer state");
            abort();
        }

        struct timeval stop_event, res;

        gettimeofday(&stop_event, NULL);

        timersub(&stop_event, &start_event, &res);
        time_counter += res.tv_sec + res.tv_usec / 1000000.0;

        started = false;
    }

    float elapsed_time(void)
    {
        return time_counter;
    }
};

template <>
struct timer<cuda>
{
    typedef std::tuple<cudaEvent_t, cudaEvent_t> sample_type;
    std::stack<sample_type> retired_events;

    sample_type active_sample;
    bool started;
    float time_counter;

    timer()
        : started(false), time_counter(0.0)
    { }

    ~timer()
    {
        flush();
    }

    void start(void)
    {
        if (started)
        {
            fprintf(stderr, "ERROR: inconsistent timer state");
            abort();
        }
        
        cudaEventCreate(&std::get<0>(active_sample));
        cudaEventCreate(&std::get<1>(active_sample));

        cudaEventRecord(std::get<0>(active_sample));

        started = true;
    }

    void stop(void)
    {
        if (!started)
        {
            fprintf(stderr, "ERROR: inconsistent timer state");
            abort();
        }

        cudaEventRecord(std::get<1>(active_sample));

        retired_events.push(active_sample);
        started = false;
    }

private:
    void flush(void)
    {
        while(!retired_events.empty())
        {
            float ms;

            sample_type sample = retired_events.top();
            retired_events.pop();

            cudaEventSynchronize(std::get<1>(sample));
            cudaEventElapsedTime(&ms, std::get<0>(sample), std::get<1>(sample));
            time_counter += ms / 1000.0;

            cudaEventDestroy(std::get<0>(sample));
            cudaEventDestroy(std::get<1>(sample));
        }
    }

public:
    float elapsed_time(void)
    {
        flush();
        return time_counter;
    }
};

struct time_series
{
    float elapsed_time;

    time_series()
        : elapsed_time(0.0)
    { }

    time_series& operator+=(const time_series& other)
    {
        elapsed_time += other.elapsed_time;
        return *this;
    }

    template <typename Timer>
    void add(Timer& timer)
    {
        elapsed_time += timer.elapsed_time();
    }
};

} // namespace lift

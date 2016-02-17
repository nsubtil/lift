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

namespace lift {

/**
 * Timer structure that measures the time difference between user-defined events.
 *
 * Lift timers measure the time between start() and stop() calls. Time is measured on the target compute device.
 * Note that because GPU timers are asynchronous, calling stop() does not necessarily mean that the timer has
 * completed the measurement.
 *
 * Timers can be used multiple times and will accumulate timing information. Calling elapsed_time() will always return
 * the total amount of time measured between successive start() / stop() calls. This implies that elapsed_time() is a
 * synchronization point for GPU timers.
 *
 * In addition to measuring time, this object has the ability to keep track of the amount of data processed
 * and return a throughput value.
 */
template <target_system system>
struct timer
{
    struct timeval start_event;
    bool started;
    float time_counter;
    uint64 bytes_tracked;

    timer()
        : started(false),
          time_counter(0.0),
          bytes_tracked(0)
    { }

    /// Copy constructor is deleted to prevent GPU timers from being moved to device.
    timer(const timer&) = delete;

    /**
     * Starts the timer. Must be followed by a stop() call.
     */
    void start(void)
    {
        if (started)
        {
            fprintf(stderr, "ERROR: inconsistent timer state\n");
            abort();
        }

        gettimeofday(&start_event, NULL);
        started = true;
    }

    /**
     * Increment the byte counter for this timer. This is meant to track the
     * amount of data processed in the section of code that's being timed,
     * in order to compute throughput metrics.
     */
    void data(uint64 bytes)
    {
        bytes_tracked += bytes;
    }

    /**
     * Increment the byte counter by the number of bytes in the pointer.
     * This is meant to track the amount of data processed in the section
     * of code that's being timed, in order to compute throughput metrics.
     */
    template <typename T>
    void data(pointer<system, T> ptr)
    {
        data(ptr.size() * sizeof(T));
    }

    /**
     * Stops the timer. Must only be called after start().
     */
    void stop(void)
    {
        if (!started)
        {
            fprintf(stderr, "ERROR: inconsistent timer state\n");
            abort();
        }

        struct timeval stop_event, res;

        gettimeofday(&stop_event, NULL);

        timersub(&stop_event, &start_event, &res);
        time_counter += res.tv_sec + res.tv_usec / 1000000.0;

        started = false;
    }

    /**
     * Returns the elapsed time between start() / stop() calls in seconds.
     *
     * Note that for GPU timers, this will wait for all commands prior to the last stop() call to be executed.
     *
     * @return  Elapsed time in seconds.
     */
    float elapsed_time(void)
    {
        return time_counter;
    }

    /**
     * Return the throughput value in bytes/s. Throughput is computed
     * based on the number of bytes that were tracked by the timer.
     *
     * @return Throughput value in bytes/second.
     */
    float throughput_b(void)
    {
        return float(bytes_tracked) / elapsed_time();
    }

    /**
     * Return the throughput value in KB/s. Throughput is computed
     * based on the number of bytes that were tracked by the timer.
     *
     * @return Throughput value in KB/second.
     */
    float throughput_KB(void)
    {
        return throughput_b() / 1024.0;
    }

    /**
     * Return the throughput value in MB/s. Throughput is computed
     * based on the number of bytes that were tracked by the timer.
     *
     * @return Throughput value in MB/second.
     */
    float throughput_MB(void)
    {
        return throughput_KB() / 1024.0;
    }

    /**
     * Return the throughput value in GB/s. Throughput is computed
     * based on the number of bytes that were tracked by the timer.
     *
     * @return Throughput value in GB/second.
     */
    float throughput_GB(void)
    {
        return throughput_MB() / 1024.0;
    }
};

template <>
struct timer<cuda>
{
    typedef struct
    {
        cudaEvent_t start;
        cudaEvent_t end;
    } sample_type;

    std::stack<sample_type> retired_events;

    sample_type active_sample;
    bool started;
    float time_counter;
    uint64 bytes_tracked;

    timer()
        : retired_events(),
          started(false),
          time_counter(0.0),
          bytes_tracked(0)
    { }

    timer(const timer&) = delete;

    ~timer()
    {
        flush();
    }

    void start(void)
    {
        if (started)
        {
            fprintf(stderr, "ERROR: inconsistent timer state\n");
            abort();
        }

        cudaError_t err;

        err = cudaEventCreate(&active_sample.start);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "ERROR: cudaEventCreate failed (%d): %s\n", err, cudaGetErrorName(err));
        }


        err = cudaEventCreate(&active_sample.end);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "ERROR: cudaEventCreate failed (%d): %s\n", err, cudaGetErrorName(err));
        }


        err = cudaEventRecord(active_sample.start);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "ERROR: start: cudaEventRecord failed (%d): %s\n", err, cudaGetErrorName(err));
        }


        started = true;
    }

    void data(uint64 bytes)
    {
        bytes_tracked += bytes;
    }

    template <typename T>
    void data(pointer<cuda, T> ptr)
    {
        data(ptr.size() * sizeof(T));
    }

    void stop(void)
    {
        if (!started)
        {
            fprintf(stderr, "ERROR: inconsistent timer state\n");
            abort();
        }

        cudaError_t err;
        err = cudaEventRecord(active_sample.end);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "ERROR: stop: cudaEventRecord failed (%d): %s\n", err, cudaGetErrorName(err));
        }


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

            cudaEventSynchronize(sample.end);
            cudaEventElapsedTime(&ms, sample.start, sample.end);
            time_counter += ms / 1000.0;

            cudaEventDestroy(sample.start);
            cudaEventDestroy(sample.end);
        }
    }

public:
    float elapsed_time(void)
    {
        flush();
        return time_counter;
    }

    float throughput_b(void)
    {
        return float(bytes_tracked) / elapsed_time();
    }

    float throughput_KB(void)
    {
        return throughput_b() / 1024.0;
    }

    float throughput_MB(void)
    {
        return throughput_KB() / 1024.0;
    }

    float throughput_GB(void)
    {
        return throughput_MB() / 1024.0;
    }

};

} // namespace lift

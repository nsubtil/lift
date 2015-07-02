#pragma once

#include <algorithm>

#include "launch_parameters.h"

namespace lift {

template <typename InputIterator, typename Function, typename index_type>
__global__ void for_each_kernel(InputIterator input, size_t length, Function func)
{
    index_type index;

    for(index = blockIdx.x * blockDim.x + threadIdx.x;
        index < length;
        index += blockDim.x * gridDim.x)
    {
        func(input[index]);
    }
}

template <typename InputIterator, typename Function>
void for_each(InputIterator input, size_t length, Function func, int2 launch_params = { 0, 0 })
{
    if (launch_params.x == 0 &&
        launch_params.y == 0)
    {
        int2 params_64 = launch_parameters(for_each_kernel<InputIterator, Function, uint64>, length);
        int2 params_32 = launch_parameters(for_each_kernel<InputIterator, Function, uint32>, length);

        // figure out the type of the index required
        if (uint64(length) + params_32.x * params_32.y >= uint64(1 << 31))
        {
//            printf("computed launch params (64): %d %d\n", params_64.x, params_64.y);
            for_each_kernel<InputIterator, Function, uint64> <<<params_64.x, params_64.y>>>(input, length, func);
        } else {
//            printf("computed launch params (32): %d %d\n", params_32.x, params_32.y);
            for_each_kernel<InputIterator, Function, uint32> <<<params_32.x, params_32.y>>>(input, length, func);
        }
    } else {
        // make sure the launch parameters are not overcommitted
        int max_blocks = int((length + launch_params.y - 1) / launch_params.y);

        if (launch_params.x > int((length + launch_params.y - 1) / launch_params.y))
        {
            fprintf(stderr, "WARNING: for_each call overcommitted, reducing block size to %d\n", max_blocks);
            launch_params.x = max_blocks;
        }

        // figure out the type of the index required
        if (uint64(length) + launch_params.x * launch_params.y >= uint64(1 << 31))
        {
            for_each_kernel<InputIterator, Function, uint64> <<<launch_params.x, launch_params.y>>>(input, length, func);
        } else {

            for_each_kernel<InputIterator, Function, uint32> <<<launch_params.x, launch_params.y>>>(input, length, func);
        }
    }
}

template <typename InputIterator, typename Function>
int2 for_each_launch_parameters(InputIterator input, size_t length, Function func)
{
    int2 params_64 = launch_parameters(for_each_kernel<InputIterator, Function, uint64>, length);
    int2 params_32 = launch_parameters(for_each_kernel<InputIterator, Function, uint32>, length);

    if (uint64(length) + params_32.x * params_32.y >= uint64(1 << 31))
    {
        return params_64;
    } else {
        return params_32;
    }
}

} // namespace lift

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

#include <lift/sys/host/compute_device_host.h>

#if __x86_64__
#include <lift/sys/host/x86_64/vector_flags.h>
#else
#error "unsupported architecture"
#endif // __x86_64__

const char *cache_type_strings[] = {
    "(null)",
    "data",
    "instruction",
    "unified"
};

int main(int argc, char **argv)
{
    lift::compute_device_host cpu;

    printf("%s\n", cpu.get_name());
    printf("vector extensions:");

#if __x86_64__
#define VEC(ext) \
    if (cpu.config.vector_extensions & lift::x86_64::ext) \
        printf(" %s", "" #ext);

    VEC(SSE);
    VEC(SSE2);
    VEC(SSE3);
    VEC(SSE3_S);
    VEC(SSE4_1);
    VEC(SSE4_2);
    VEC(SSE4_a);
    VEC(SSE_XOP);
    VEC(SSE_FMA4);
    VEC(SSE_FMA3);
    VEC(AVX);
    VEC(AVX2);

#undef VEC
#endif // __x86_64__


    if (cpu.config.caches.size())
    {
        printf("\n\n");
        printf("cache topology:\n");

        for(auto cache : cpu.config.caches)
        {
            printf(" L%d: %s %d KB %u-way %u bytes/line\n",
                   cache.level,
                   cache_type_strings[cache.type],
                   cache.total_size / 1024,
                   cache.associativity,
                   cache.line_size);
        }
    }

    return 0;
}

/*
 * Lift
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * Copyright (c) 2015, Roche Molecular Systems Inc.
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

#include <lift/types.h>
#include <lift/sys/host/compute_device_host.h>
#include <lift/sys/host/x86_64/vector_flags.h>

#include "x86_64_cpuid.h"

namespace lift {

namespace x86_64 {

struct cpuid_regs
{
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;
};

static inline void cpuid(cpuid_regs& output,
                         unsigned int code,
                         unsigned int count = 0)
{
  __asm__ __volatile__ ("cpuid"
                           : "=a" (output.eax),
                             "=b" (output.ebx),
                             "=c" (output.ecx),
                             "=d" (output.edx)
                           : "0" (code),
                             "2" (count));
}

static inline unsigned int cpuid_max(unsigned int extended = 0)
{
    cpuid_regs regs;
    cpuid(regs, 0);
    return regs.eax;
}

static inline unsigned int cpuid_max_extended(void)
{
    cpuid_regs regs;
    cpuid(regs, 0x80000000);
    return regs.eax;
}

static void identify_vector_extensions(cpu_config& ret)
{
    ret.vector_extensions = 0;

    cpuid_regs regs;    
    cpuid(regs, 1);

#define XL_BIT(register, cpuid_bit, lift_bit) \
    if (regs.register & x86_64::CPUID_BIT_ ##cpuid_bit) \
        ret.vector_extensions |= x86_64::lift_bit

    XL_BIT(edx, EDX_SSE, SSE);
    XL_BIT(edx, EDX_SSE2, SSE2);
    XL_BIT(ecx, ECX_SSE3, SSE3);
    XL_BIT(ecx, ECX_SSSE3, SSE3_S);
    XL_BIT(ecx, ECX_SSE41, SSE4_1);
    XL_BIT(ecx, ECX_SSE42, SSE4_2);
    XL_BIT(ecx, ECX_FMA, SSE_FMA3);
    XL_BIT(ecx, ECX_F16C, SSE_F16C);
    XL_BIT(ecx, ECX_AVX, AVX);

    if (cpuid_max() < 7)
    {
        return;
    }

    cpuid(regs, 7);

    XL_BIT(ebx, EBX_AVX2, AVX2);

    if (cpuid_max_extended() < 1)
    {
        return;
    }

    cpuid(regs, 0x80000001);

    XL_BIT(ecx, ECX_SSE4A, SSE4_a);
    XL_BIT(ecx, ECX_FMA4, SSE_FMA4);
    XL_BIT(ecx, ECX_XOP, SSE_XOP);

#undef XL_BIT
}

static void decode_cache_descriptor(cpu_config& ret, uint8 desc)
{
    switch(desc)
    {
#define C(desc, level, type, total_size, associativity, line_size) \
    case desc: \
        ret.caches.push_back({cpu_cache::type, level, associativity, total_size * 1024, line_size}); \
        break;

        C(0x06, 1, instruction, 8, 4, 32);
        C(0x08, 1, instruction, 16, 4, 32);
        C(0x09, 1, instruction, 32, 4, 64);
        C(0x0a, 1, data, 8, 2, 32);
        C(0x0c, 1, data, 16, 4, 32);
        C(0x0d, 1, data, 16, 4, 64);
        C(0x0e, 1, data, 24, 6, 64);
        C(0x1d, 2, unified, 128, 2, 64);
        C(0x21, 2, unified, 256, 8, 64);
        C(0x22, 3, unified, 512, 4, 64);
        C(0x23, 3, unified, 1024, 8, 64);
        C(0x24, 2, unified, 1024, 16, 64);
        C(0x25, 3, unified, 2048, 8, 64);
        C(0x29, 3, unified, 4096, 8, 64);
        C(0x2c, 1, data, 32, 8, 64);
        C(0x30, 1, instruction, 32, 8, 64);
        C(0x41, 2, unified, 128, 4, 32);
        C(0x42, 2, unified, 256, 4, 32);
        C(0x43, 2, unified, 512, 4, 32);
        C(0x44, 2, unified, 1024, 4, 32);
        C(0x45, 2, unified, 2048, 4, 32);
        C(0x46, 3, unified, 4096, 4, 64);
        C(0x47, 3, unified, 8192, 8, 64);
        C(0x48, 2, unified, 3072, 12, 64);

        // thank you intel
        case 0x49:
            {
                cpuid_regs regs;
                cpuid(regs, 1);

                uint8 model = (regs.eax & 0xf0) >> 4;
                uint8 family = (regs.eax & 0xf00) >> 8;

                if (family == 0x0f && model == 0x06)
                {
                    ret.caches.push_back({cpu_cache::unified, 3, 16, 4096 * 1024, 64});
                } else {
                    ret.caches.push_back({cpu_cache::unified, 2, 16, 4096 * 1024, 64});
                }
            }

            break;

        C(0x4a, 3, unified, 6144, 12, 64);
        C(0x4b, 3, unified, 8192, 16, 64);
        C(0x4c, 3, unified, 12288, 12, 64);
        C(0x4d, 3, unified, 16 * 1024, 16, 64);
        C(0x4e, 2, unified, 6 * 1024, 24, 64);
        C(0x60, 1, data, 16, 8, 64);
        C(0x66, 1, data, 8, 4, 64);
        C(0x67, 1, data, 16, 4, 64);
        C(0x68, 1, data, 32, 4, 64);
        C(0x78, 2, unified, 1024, 4, 64);
        C(0x79, 2, unified, 128, 8, 64);
        C(0x7a, 2, unified, 256, 8, 64);
        C(0x7b, 2, unified, 512, 8, 64);
        C(0x7c, 2, unified, 1024, 8, 64);
        C(0x7d, 2, unified, 2048, 8, 64);
        C(0x7f, 2, unified, 512, 2, 64);
        C(0x80, 2, unified, 512, 8, 64);
        C(0x82, 2, unified, 256, 8, 32);
        C(0x83, 2, unified, 512, 8, 32);
        C(0x84, 2, unified, 1024, 8, 32);
        C(0x85, 2, unified, 2048, 8, 32);
        C(0x86, 2, unified, 512, 4, 64);
        C(0x87, 2, unified, 1024, 8, 64);
        C(0xd0, 3, unified, 512, 4, 64);
        C(0xd1, 3, unified, 1024, 4, 64);
        C(0xd2, 3, unified, 2048, 4, 64);
        C(0xd6, 3, unified, 1024, 8, 64);
        C(0xd7, 3, unified, 2048, 8, 64);
        C(0xd8, 3, unified, 4096, 8, 64);
        C(0xdc, 3, unified, 1536, 12, 64);
        C(0xdd, 3, unified, 3072, 12, 64);
        C(0xde, 3, unified, 6144, 12, 64);
        C(0xe2, 3, unified, 2048, 16, 64);
        C(0xe3, 3, unified, 4096, 16, 64);
        C(0xe4, 3, unified, 8192, 16, 64);
        C(0xea, 3, unified, 12288, 24, 64);
        C(0xeb, 3, unified, 18432, 24, 64);
        C(0xec, 3, unified, 24576, 24, 64);
    }
}

static void scan_leaf4_cache_info(cpu_config& ret)
{
    cpuid_regs regs;

    for(uint32 in_ecx = 0; ; in_ecx++)
    {
        cpuid(regs, 4, in_ecx);

        if ((regs.eax & 0xf) == 0)
            break;

        cpu_cache::cache_type cache_type = cpu_cache::null;

        switch(regs.eax & 0xf)
        {
            case 1:
                cache_type = cpu_cache::data;
                break;

            case 2:
                cache_type = cpu_cache::instruction;
                break;

            case 3:
                cache_type = cpu_cache::unified;
                break;    
        }

        unsigned int level = ((regs.eax >> 5) & 0x3);

        unsigned int ways = ((regs.ebx >> 22) & 0xff) + 1;
        unsigned int partitions = ((regs.ebx >> 12) & 0xff) + 1;
        unsigned int line_size = (regs.ebx & 0x3ff) + 1;
        unsigned int sets = regs.ecx + 1;

        unsigned int total_size = ways * partitions * line_size * sets;

        cpu_cache cache = { cache_type, level, ways, total_size, line_size };
        ret.caches.push_back(cache);
    }
}

static void identify_caches(cpu_config& ret)
{
    ret.caches.clear();

    cpuid_regs regs;
    cpuid(regs, 2);

    if ((regs.eax & (1u << 31)) == 0)
    {
        decode_cache_descriptor(ret, (regs.eax >> 24) & 0xff);
        decode_cache_descriptor(ret, (regs.eax >> 16) & 0xff);
        decode_cache_descriptor(ret, (regs.eax >>  8) & 0xff);
    }

    if ((regs.ebx & (1u << 31)) == 0)
    {
        decode_cache_descriptor(ret, (regs.ebx >> 24) & 0xff);
        decode_cache_descriptor(ret, (regs.ebx >> 16) & 0xff);
        decode_cache_descriptor(ret, (regs.ebx >>  8) & 0xff);
        decode_cache_descriptor(ret, (regs.ebx >>  0) & 0xff);
    }

    if ((regs.ecx & (1u << 31)) == 0)
    {
        decode_cache_descriptor(ret, (regs.ecx >> 24) & 0xff);
        decode_cache_descriptor(ret, (regs.ecx >> 16) & 0xff);
        decode_cache_descriptor(ret, (regs.ecx >>  8) & 0xff);
        decode_cache_descriptor(ret, (regs.ecx >>  0) & 0xff);
    }

    if ((regs.edx & (1u << 31)) == 0)
    {
        decode_cache_descriptor(ret, (regs.edx >> 24) & 0xff);
        decode_cache_descriptor(ret, (regs.edx >> 16) & 0xff);
        decode_cache_descriptor(ret, (regs.edx >>  8) & 0xff);
        decode_cache_descriptor(ret, (regs.edx >>  0) & 0xff);
    }

    scan_leaf4_cache_info(ret);
}

static void get_cpu_brand_string(cpu_config& ret)
{
    char brand_string[sizeof(unsigned int) * 4 * 4 + 1] = { 0 };

    cpuid_regs regs;

    cpuid(regs, 0x80000002);
    memcpy(&brand_string[ 0], &regs.eax, sizeof(regs.eax));
    memcpy(&brand_string[ 4], &regs.ebx, sizeof(regs.ebx));
    memcpy(&brand_string[ 8], &regs.ecx, sizeof(regs.ecx));
    memcpy(&brand_string[12], &regs.edx, sizeof(regs.edx));

    cpuid(regs, 0x80000003);
    memcpy(&brand_string[16], &regs.eax, sizeof(regs.eax));
    memcpy(&brand_string[20], &regs.ebx, sizeof(regs.ebx));
    memcpy(&brand_string[24], &regs.ecx, sizeof(regs.ecx));
    memcpy(&brand_string[28], &regs.edx, sizeof(regs.edx));

    cpuid(regs, 0x80000004);
    memcpy(&brand_string[32], &regs.eax, sizeof(regs.eax));
    memcpy(&brand_string[36], &regs.ebx, sizeof(regs.ebx));
    memcpy(&brand_string[40], &regs.ecx, sizeof(regs.ecx));
    memcpy(&brand_string[44], &regs.edx, sizeof(regs.edx));

    std::string name = brand_string;

    // trim leading/trailing whitespace
    auto first_non_space = name.find_first_not_of(" ");
    auto last_non_space = name.find_last_not_of(" ");

    ret.name = name.substr(first_non_space, last_non_space - first_non_space + 1);
}

} // namespace x86

namespace __internal {

bool identify_host_cpu(cpu_config& ret)
{
    unsigned int max_level;

    // get the maximum CPUID level supported
    max_level = x86_64::cpuid_max();

    if (max_level < 1)
    {
        // should never happen
        return false;
    }

    x86_64::identify_vector_extensions(ret);
    x86_64::identify_caches(ret);
    x86_64::get_cpu_brand_string(ret);

    return true;
}

} // namespace __internal

} // namespace lift

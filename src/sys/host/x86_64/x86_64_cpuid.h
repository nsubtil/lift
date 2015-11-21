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

#pragma once

namespace lift {

namespace x86_64 {

// constants for CPUID leaf 1
static constexpr uint32 CPUID_BIT_EDX_SSE       = 1 << 25;
static constexpr uint32 CPUID_BIT_EDX_SSE2      = 1 << 26;
static constexpr uint32 CPUID_BIT_ECX_SSE3      = 1 << 0;
static constexpr uint32 CPUID_BIT_ECX_SSSE3     = 1 << 9;
static constexpr uint32 CPUID_BIT_ECX_SSE41     = 1 << 19;
static constexpr uint32 CPUID_BIT_ECX_SSE42     = 1 << 20;
static constexpr uint32 CPUID_BIT_ECX_FMA       = 1 << 12;
static constexpr uint32 CPUID_BIT_ECX_F16C      = 1 << 29;
static constexpr uint32 CPUID_BIT_ECX_AVX       = 1 << 28;

// CPUID leaf 7
static constexpr uint32 CPUID_BIT_EBX_AVX2      = 1 << 5;

// CPUID leaf 0x80000001
static constexpr uint32 CPUID_BIT_ECX_SSE4A     = 1 << 6;
static constexpr uint32 CPUID_BIT_ECX_FMA4      = 1 << 16;
static constexpr uint32 CPUID_BIT_ECX_XOP       = 1 << 11;

}

}
/*
 * Lift
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * Copyright (c) 2015, Roche Molecular Systems, Inc.
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

// vector extension flags
static constexpr uint32 SSE          = 1 <<  0;
static constexpr uint32 SSE2         = 1 <<  1;
static constexpr uint32 SSE3         = 1 <<  2;
static constexpr uint32 SSE3_S       = 1 <<  3;   // SSE3 supplemental
static constexpr uint32 SSE4_1       = 1 <<  4;   // SSE 4.1 (Penryn)
static constexpr uint32 SSE4_2       = 1 <<  5;   // SSE 4.2 (Nehalem)
static constexpr uint32 SSE4_a       = 1 <<  6;
static constexpr uint32 SSE_XOP      = 1 <<  7;   // SSE extended operations
static constexpr uint32 SSE_FMA4     = 1 <<  8;
static constexpr uint32 SSE_FMA3     = 1 <<  9;
static constexpr uint32 SSE_F16C     = 1 << 10;
static constexpr uint32 AVX          = 1 << 11;
static constexpr uint32 AVX2         = 1 << 12;    

} // namespace x86_64
} // namespace lift
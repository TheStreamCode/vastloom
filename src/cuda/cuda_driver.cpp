#include "vastloom/cuda/cuda_driver.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <mutex>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "vastloom/util/string.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#define VASTLOOM_CUDA_CALL __stdcall
#elif defined(__linux__)
#include <dlfcn.h>
#define VASTLOOM_CUDA_CALL
#else
#define VASTLOOM_CUDA_CALL
#endif

namespace vastloom::cuda {
namespace {

using CuResult = int;
using CuDevice = int;
using CuContext = void*;
using CuDevicePointer = std::uint64_t;
using CuModule = void*;
using CuFunction = void*;
using CuStream = void*;

constexpr CuResult kCudaSuccess = 0;
constexpr int kAttributeMaximumThreadsPerBlock = 1;
constexpr int kAttributeMultiprocessorCount = 16;
constexpr int kAttributeUnifiedAddressing = 41;
constexpr int kAttributeComputeCapabilityMajor = 75;
constexpr int kAttributeComputeCapabilityMinor = 76;
constexpr int kJitErrorLogBuffer = 5;
constexpr int kJitErrorLogBufferSizeBytes = 6;
constexpr std::size_t kMaximumDeviceNameBytes = 256U;
constexpr std::size_t kMaximumAllocationTagBytes = 1'024U;
constexpr std::size_t kMaximumJitLogBytes = 8U * 1'024U;

constexpr std::string_view kRmsNormPtx = R"ptx(
.version 7.0
.target sm_52
.address_size 64

.visible .entry vastloom_rms_norm_f32(
    .param .u64 input_param,
    .param .u64 weight_param,
    .param .u64 output_param,
    .param .u32 rows_param,
    .param .u32 width_param,
    .param .f32 epsilon_param
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<9>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<9>;

    ld.param.u64 %rd0, [input_param];
    ld.param.u64 %rd1, [weight_param];
    ld.param.u64 %rd2, [output_param];
    ld.param.u32 %r0, [rows_param];
    ld.param.u32 %r1, [width_param];
    ld.param.f32 %f0, [epsilon_param];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %tid.x;
    setp.ge.u32 %p0, %r2, %r0;
    @%p0 bra DONE;
    setp.ne.u32 %p0, %r3, 0;
    @%p0 bra DONE;
    mul.lo.u32 %r4, %r2, %r1;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r5, 0;

SUM_LOOP:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra SUM_DONE;
    add.u32 %r6, %r4, %r5;
    mul.wide.u32 %rd3, %r6, 4;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f2, [%rd4];
    fma.rn.f32 %f1, %f2, %f2, %f1;
    add.u32 %r5, %r5, 1;
    bra SUM_LOOP;

SUM_DONE:
    cvt.rn.f32.u32 %f3, %r1;
    div.rn.f32 %f4, %f1, %f3;
    add.rn.f32 %f4, %f4, %f0;
    sqrt.rn.f32 %f5, %f4;
    mov.f32 %f6, 0f3F800000;
    div.rn.f32 %f7, %f6, %f5;
    mov.u32 %r5, 0;
OUTPUT_LOOP:
    setp.ge.u32 %p2, %r5, %r1;
    @%p2 bra DONE;
    add.u32 %r6, %r4, %r5;
    mul.wide.u32 %rd3, %r6, 4;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f2, [%rd4];
    mul.wide.u32 %rd5, %r5, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.f32 %f8, [%rd6];
    mul.rn.f32 %f2, %f2, %f7;
    mul.rn.f32 %f2, %f2, %f8;
    add.u64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f2;
    add.u32 %r5, %r5, 1;
    bra OUTPUT_LOOP;

DONE:
    ret;
}

// One thread per output row of a ggml Q8_0 matrix: 34-byte blocks of one FP16 scale followed by
// 32 int8 weights. Accumulation runs block by block in the same order as the CPU reference, but in
// FP32 rather than FP64, so results agree to a declared tolerance rather than bit for bit.
.visible .entry vastloom_matvec_q8_0_f32(
    .param .u64 matrix_param,
    .param .u64 input_param,
    .param .u64 output_param,
    .param .u32 rows_param,
    .param .u32 blocks_param
)
{
    .reg .pred %p<4>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<8>;
    .reg .f64 %fd<6>;

    ld.param.u64 %rd0, [matrix_param];
    ld.param.u64 %rd1, [input_param];
    ld.param.u64 %rd2, [output_param];
    ld.param.u32 %r0, [rows_param];
    ld.param.u32 %r1, [blocks_param];
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r3, %r2, %r4;
    setp.ge.u32 %p0, %r5, %r0;
    @%p0 bra Q8_DONE;

    // Byte offset of this row: row * blocks_per_row * 34.
    mul.lo.u32 %r6, %r5, %r1;
    mul.wide.u32 %rd3, %r6, 34;
    add.u64 %rd4, %rd0, %rd3;
    // f64 accumulator, matching the CPU reference. See the IQ1_S kernel below for why this is not
    // negotiable: the reference dequantizes to f32 and then sums in double, and a kernel that
    // sums in f32 is answering a different question.
    mov.f64 %fd0, 0d0000000000000000;
    mov.u32 %r7, 0;

Q8_BLOCK_LOOP:
    setp.ge.u32 %p1, %r7, %r1;
    @%p1 bra Q8_BLOCK_DONE;
    ld.global.u16 %rs0, [%rd4];
    cvt.f32.f16 %f1, %rs0;
    add.u64 %rd5, %rd4, 2;
    mul.lo.u32 %r8, %r7, 32;
    mul.wide.u32 %rd6, %r8, 4;
    add.u64 %rd7, %rd1, %rd6;
    mov.u32 %r9, 0;

Q8_ELEMENT_LOOP:
    setp.ge.u32 %p2, %r9, 32;
    @%p2 bra Q8_ELEMENT_DONE;
    ld.global.s8 %r10, [%rd5];
    cvt.rn.f32.s32 %f3, %r10;
    // The reference materializes scale * q as an f32 weight before it multiplies by the input,
    // so the rounding at that step has to happen here too.
    mul.rn.f32 %f5, %f1, %f3;
    ld.global.f32 %f4, [%rd7];
    cvt.f64.f32 %fd1, %f5;
    cvt.f64.f32 %fd2, %f4;
    mul.rn.f64 %fd3, %fd1, %fd2;
    add.rn.f64 %fd0, %fd0, %fd3;
    add.u64 %rd5, %rd5, 1;
    add.u64 %rd7, %rd7, 4;
    add.u32 %r9, %r9, 1;
    bra Q8_ELEMENT_LOOP;

Q8_ELEMENT_DONE:
    add.u64 %rd4, %rd4, 34;
    add.u32 %r7, %r7, 1;
    bra Q8_BLOCK_LOOP;

Q8_BLOCK_DONE:
    cvt.rn.f32.f64 %f6, %fd0;
    mul.wide.u32 %rd8, %r5, 4;
    add.u64 %rd9, %rd2, %rd8;
    st.global.f32 [%rd9], %f6;

Q8_DONE:
    ret;
}

// One thread per output row of a ggml IQ1_S matrix. Each 50-byte block covers 256 weights: one FP16
// scale, 32 grid selectors, and eight 16-bit fields carrying a per-group scale, a sign for the
// delta, and the high bits of four selectors. Each selector indexes a 2,048-entry table of eight
// packed int8 values, which the caller supplies in device memory.
//
// Accumulation follows the CPU reference's order - block, group, sub-block, lane - but in FP32
// rather than FP64, so results agree to a declared tolerance.
.visible .entry vastloom_matvec_iq1_s_f32(
    .param .u64 matrix_param,
    .param .u64 grid_param,
    .param .u64 input_param,
    .param .u64 output_param,
    .param .u32 rows_param,
    .param .u32 blocks_param
)
{
    .reg .pred %p<8>;
    .reg .b16 %rs<4>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<16>;
    .reg .f64 %fd<6>;

    ld.param.u64 %rd0, [matrix_param];
    ld.param.u64 %rd1, [grid_param];
    ld.param.u64 %rd2, [input_param];
    ld.param.u64 %rd3, [output_param];
    ld.param.u32 %r0, [rows_param];
    ld.param.u32 %r1, [blocks_param];
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r3, %r2, %r4;
    setp.ge.u32 %p0, %r5, %r0;
    @%p0 bra IQ1_DONE;

    // Byte offset of this row: row * blocks_per_row * 50.
    mul.lo.u32 %r6, %r5, %r1;
    mul.wide.u32 %rd4, %r6, 50;
    add.u64 %rd5, %rd0, %rd4;
    // The row accumulator is f64 because the CPU reference accumulates in double. Matching it is
    // the point: a 32-bit accumulator is a different algorithm, not a faster one.
    mov.f64 %fd0, 0d0000000000000000;
    mov.u32 %r8, 0;
    mov.u64 %rd6, %rd2;

IQ1_BLOCK_LOOP:
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra IQ1_BLOCK_DONE;
    ld.global.u16 %rs0, [%rd5];
    cvt.f32.f16 %f1, %rs0;
    mov.u32 %r9, 0;

IQ1_GROUP_LOOP:
    setp.ge.u32 %p2, %r9, 8;
    @%p2 bra IQ1_GROUP_DONE;
    // high = qh[group], at block + 34 + 2 * group
    mul.wide.u32 %rd7, %r9, 2;
    add.u64 %rd8, %rd5, 34;
    add.u64 %rd8, %rd8, %rd7;
    ld.global.u16 %rs1, [%rd8];
    cvt.u32.u16 %r10, %rs1;
    // local_scale = scale * (2 * ((high >> 12) & 7) + 1)
    shr.u32 %r11, %r10, 12;
    and.b32 %r11, %r11, 7;
    shl.b32 %r11, %r11, 1;
    add.u32 %r11, %r11, 1;
    cvt.rn.f32.u32 %f2, %r11;
    mul.rn.f32 %f3, %f1, %f2;
    // delta = (high & 0x8000) ? -0.125 : 0.125
    and.b32 %r12, %r10, 32768;
    setp.ne.u32 %p3, %r12, 0;
    mov.f32 %f4, 0f3E000000;
    mov.f32 %f5, 0fBE000000;
    selp.f32 %f6, %f5, %f4, %p3;
    mov.u32 %r13, 0;

IQ1_SUB_LOOP:
    setp.ge.u32 %p4, %r13, 4;
    @%p4 bra IQ1_SUB_DONE;
    // grid_index = qs[group * 4 + sub] | (((high >> (3 * sub)) & 7) << 8)
    mul.lo.u32 %r14, %r9, 4;
    add.u32 %r14, %r14, %r13;
    cvt.u64.u32 %rd9, %r14;
    add.u64 %rd10, %rd5, 2;
    add.u64 %rd10, %rd10, %rd9;
    ld.global.u8 %r15, [%rd10];
    mul.lo.u32 %r16, %r13, 3;
    shr.u32 %r17, %r10, %r16;
    and.b32 %r17, %r17, 7;
    shl.b32 %r17, %r17, 8;
    or.b32 %r18, %r15, %r17;
    mul.wide.u32 %rd11, %r18, 8;
    add.u64 %rd12, %rd1, %rd11;
    ld.global.u64 %rd13, [%rd12];
    mov.u32 %r19, 0;

IQ1_LANE_LOOP:
    setp.ge.u32 %p5, %r19, 8;
    @%p5 bra IQ1_LANE_DONE;
    cvt.u32.u64 %r20, %rd13;
    cvt.s32.s8 %r21, %r20;
    cvt.rn.f32.s32 %f7, %r21;
    add.rn.f32 %f8, %f7, %f6;
    mul.rn.f32 %f9, %f8, %f3;
    ld.global.f32 %f10, [%rd6];
    // Widen both operands and multiply in f64, exactly as the reference does. fma here would fuse
    // away an intermediate rounding the CPU performs, so the two would drift apart.
    cvt.f64.f32 %fd1, %f9;
    cvt.f64.f32 %fd2, %f10;
    mul.rn.f64 %fd3, %fd1, %fd2;
    add.rn.f64 %fd0, %fd0, %fd3;
    shr.u64 %rd13, %rd13, 8;
    add.u64 %rd6, %rd6, 4;
    add.u32 %r19, %r19, 1;
    bra IQ1_LANE_LOOP;

IQ1_LANE_DONE:
    add.u32 %r13, %r13, 1;
    bra IQ1_SUB_LOOP;

IQ1_SUB_DONE:
    add.u32 %r9, %r9, 1;
    bra IQ1_GROUP_LOOP;

IQ1_GROUP_DONE:
    add.u64 %rd5, %rd5, 50;
    add.u32 %r8, %r8, 1;
    bra IQ1_BLOCK_LOOP;

IQ1_BLOCK_DONE:
    cvt.rn.f32.f64 %f11, %fd0;
    mul.wide.u32 %rd14, %r5, 4;
    add.u64 %rd15, %rd3, %rd14;
    st.global.f32 [%rd15], %f11;

IQ1_DONE:
    ret;
}
)ptx";

// The module continues here only because MSVC caps a single string literal at 16,380 characters.
// The two halves are concatenated before the JIT ever sees them; the split point carries no
// meaning beyond that limit.
constexpr std::string_view kQuantizedMatvecPtx = R"ptx(
// One thread per output row of a ggml IQ3_XXS matrix. Each 98-byte block covers 256 weights: an
// FP16 scale, 64 grid indices, and eight 32-bit fields holding a group scale plus four 7-bit sign
// patterns. Each index selects one of 256 entries of four packed uint8 magnitudes; the eighth sign
// bit is the parity of the other seven, which is why popc appears below.
//
// The reference decodes a whole block to f32 and then walks columns 0..255 in order. This kernel
// never materializes the block, so it must visit columns in that same order to sum identically -
// hence two passes of four lanes per subblock rather than one pass handling both grids at once.
.visible .entry vastloom_matvec_iq3_xxs_f32(
    .param .u64 matrix_param,
    .param .u64 grid_param,
    .param .u64 input_param,
    .param .u64 output_param,
    .param .u32 rows_param,
    .param .u32 blocks_param
)
{
    .reg .pred %p<8>;
    .reg .b16 %rs<4>;
    .reg .b32 %r<40>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<20>;
    .reg .f64 %fd<6>;

    ld.param.u64 %rd0, [matrix_param];
    ld.param.u64 %rd1, [grid_param];
    ld.param.u64 %rd2, [input_param];
    ld.param.u64 %rd3, [output_param];
    ld.param.u32 %r0, [rows_param];
    ld.param.u32 %r1, [blocks_param];
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r3, %r2, %r4;
    setp.ge.u32 %p0, %r5, %r0;
    @%p0 bra IQ3_DONE;

    // Byte offset of this row: row * blocks_per_row * 98.
    mul.lo.u32 %r6, %r5, %r1;
    mul.wide.u32 %rd4, %r6, 98;
    add.u64 %rd5, %rd0, %rd4;
    mov.f64 %fd0, 0d0000000000000000;
    mov.u32 %r7, 0;
    // Running input pointer. Columns are visited in order, so it only ever moves forward.
    mov.u64 %rd6, %rd2;

IQ3_BLOCK_LOOP:
    setp.ge.u32 %p1, %r7, %r1;
    @%p1 bra IQ3_BLOCK_DONE;
    ld.global.u16 %rs0, [%rd5];
    cvt.f32.f16 %f1, %rs0;
    mov.u32 %r8, 0;

IQ3_GROUP_LOOP:
    setp.ge.u32 %p2, %r8, 8;
    @%p2 bra IQ3_GROUP_DONE;
    // aux = u32 at block + 2 + 64 + 4 * group, assembled from bytes: a 98-byte block stride leaves
    // this field on an odd address every other block, and a 32-bit load there faults.
    mul.wide.u32 %rd7, %r8, 4;
    add.u64 %rd8, %rd5, 66;
    add.u64 %rd8, %rd8, %rd7;
    ld.global.u8 %r9, [%rd8];
    ld.global.u8 %r27, [%rd8+1];
    shl.b32 %r27, %r27, 8;
    or.b32 %r9, %r9, %r27;
    ld.global.u8 %r27, [%rd8+2];
    shl.b32 %r27, %r27, 16;
    or.b32 %r9, %r9, %r27;
    ld.global.u8 %r27, [%rd8+3];
    shl.b32 %r27, %r27, 24;
    or.b32 %r9, %r9, %r27;
    // local_scale = scale * (0.5 + (float)(aux >> 28)) * 0.5, left to right as the reference does.
    shr.u32 %r10, %r9, 28;
    cvt.rn.f32.u32 %f2, %r10;
    mov.f32 %f3, 0f3F000000;
    add.rn.f32 %f4, %f3, %f2;
    mul.rn.f32 %f5, %f1, %f4;
    mul.rn.f32 %f6, %f5, %f3;
    mov.u32 %r11, 0;

IQ3_SUB_LOOP:
    setp.ge.u32 %p3, %r11, 4;
    @%p3 bra IQ3_SUB_DONE;
    // signs = s | ((popc(s) & 1) << 7), where s = (aux >> (7 * sub)) & 127
    mul.lo.u32 %r12, %r11, 7;
    shr.u32 %r13, %r9, %r12;
    and.b32 %r13, %r13, 127;
    popc.b32 %r14, %r13;
    and.b32 %r14, %r14, 1;
    shl.b32 %r14, %r14, 7;
    or.b32 %r15, %r13, %r14;
    // The two grid indices for this subblock: quantized[group * 8 + sub * 2 + {0,1}]
    mul.lo.u32 %r16, %r8, 8;
    mul.lo.u32 %r17, %r11, 2;
    add.u32 %r16, %r16, %r17;
    cvt.u64.u32 %rd9, %r16;
    add.u64 %rd10, %rd5, 2;
    add.u64 %rd10, %rd10, %rd9;
    ld.global.u8 %r18, [%rd10];
    ld.global.u8 %r19, [%rd10+1];
    mul.wide.u32 %rd11, %r18, 4;
    add.u64 %rd12, %rd1, %rd11;
    ld.global.u32 %r20, [%rd12];
    mul.wide.u32 %rd13, %r19, 4;
    add.u64 %rd14, %rd1, %rd13;
    ld.global.u32 %r21, [%rd14];
    // First grid: columns base+0 .. base+3.
    mov.u32 %r22, 0;

IQ3_FIRST_LOOP:
    setp.ge.u32 %p4, %r22, 4;
    @%p4 bra IQ3_FIRST_DONE;
    shl.b32 %r23, %r22, 3;
    shr.u32 %r24, %r20, %r23;
    and.b32 %r24, %r24, 255;
    cvt.rn.f32.u32 %f7, %r24;
    mov.u32 %r25, 1;
    shl.b32 %r25, %r25, %r22;
    and.b32 %r25, %r25, %r15;
    setp.ne.u32 %p5, %r25, 0;
    mov.f32 %f8, 0f3F800000;
    mov.f32 %f9, 0fBF800000;
    selp.f32 %f10, %f9, %f8, %p5;
    mul.rn.f32 %f11, %f6, %f7;
    mul.rn.f32 %f12, %f11, %f10;
    ld.global.f32 %f13, [%rd6];
    cvt.f64.f32 %fd1, %f12;
    cvt.f64.f32 %fd2, %f13;
    mul.rn.f64 %fd3, %fd1, %fd2;
    add.rn.f64 %fd0, %fd0, %fd3;
    add.u64 %rd6, %rd6, 4;
    add.u32 %r22, %r22, 1;
    bra IQ3_FIRST_LOOP;

IQ3_FIRST_DONE:
    // Second grid: columns base+4 .. base+7, signs taken from bits 4..7.
    mov.u32 %r22, 0;

IQ3_SECOND_LOOP:
    setp.ge.u32 %p6, %r22, 4;
    @%p6 bra IQ3_SECOND_DONE;
    shl.b32 %r23, %r22, 3;
    shr.u32 %r24, %r21, %r23;
    and.b32 %r24, %r24, 255;
    cvt.rn.f32.u32 %f7, %r24;
    add.u32 %r26, %r22, 4;
    mov.u32 %r25, 1;
    shl.b32 %r25, %r25, %r26;
    and.b32 %r25, %r25, %r15;
    setp.ne.u32 %p7, %r25, 0;
    mov.f32 %f8, 0f3F800000;
    mov.f32 %f9, 0fBF800000;
    selp.f32 %f10, %f9, %f8, %p7;
    mul.rn.f32 %f11, %f6, %f7;
    mul.rn.f32 %f12, %f11, %f10;
    ld.global.f32 %f13, [%rd6];
    cvt.f64.f32 %fd1, %f12;
    cvt.f64.f32 %fd2, %f13;
    mul.rn.f64 %fd3, %fd1, %fd2;
    add.rn.f64 %fd0, %fd0, %fd3;
    add.u64 %rd6, %rd6, 4;
    add.u32 %r22, %r22, 1;
    bra IQ3_SECOND_LOOP;

IQ3_SECOND_DONE:
    add.u32 %r11, %r11, 1;
    bra IQ3_SUB_LOOP;

IQ3_SUB_DONE:
    add.u32 %r8, %r8, 1;
    bra IQ3_GROUP_LOOP;

IQ3_GROUP_DONE:
    add.u64 %rd5, %rd5, 98;
    add.u32 %r7, %r7, 1;
    bra IQ3_BLOCK_LOOP;

IQ3_BLOCK_DONE:
    cvt.rn.f32.f64 %f14, %fd0;
    mul.wide.u32 %rd15, %r5, 4;
    add.u64 %rd16, %rd3, %rd15;
    st.global.f32 [%rd16], %f14;

IQ3_DONE:
    ret;
}

// One thread per output row of a dense little-endian FP32 matrix. No decoding, so this is the
// simplest kernel in the set - and by the telemetry the third most requested, because the graph's
// unquantized weights are large and expensive on the CPU precisely because nothing compresses them.
.visible .entry vastloom_matvec_f32_f32(
    .param .u64 matrix_param,
    .param .u64 input_param,
    .param .u64 output_param,
    .param .u32 rows_param,
    .param .u32 columns_param
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<8>;
    .reg .f64 %fd<6>;

    ld.param.u64 %rd0, [matrix_param];
    ld.param.u64 %rd1, [input_param];
    ld.param.u64 %rd2, [output_param];
    ld.param.u32 %r0, [rows_param];
    ld.param.u32 %r1, [columns_param];
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r3, %r2, %r4;
    setp.ge.u32 %p0, %r5, %r0;
    @%p0 bra F32_DONE;

    mul.lo.u32 %r6, %r5, %r1;
    mul.wide.u32 %rd3, %r6, 4;
    add.u64 %rd4, %rd0, %rd3;
    mov.u64 %rd5, %rd1;
    mov.f64 %fd0, 0d0000000000000000;
    mov.u32 %r7, 0;

F32_LOOP:
    setp.ge.u32 %p1, %r7, %r1;
    @%p1 bra F32_LOOP_DONE;
    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];
    cvt.f64.f32 %fd1, %f0;
    cvt.f64.f32 %fd2, %f1;
    mul.rn.f64 %fd3, %fd1, %fd2;
    add.rn.f64 %fd0, %fd0, %fd3;
    add.u64 %rd4, %rd4, 4;
    add.u64 %rd5, %rd5, 4;
    add.u32 %r7, %r7, 1;
    bra F32_LOOP;

F32_LOOP_DONE:
    cvt.rn.f32.f64 %f2, %fd0;
    mul.wide.u32 %rd6, %r5, 4;
    add.u64 %rd7, %rd2, %rd6;
    st.global.f32 [%rd7], %f2;

F32_DONE:
    ret;
}

// One thread per output row of a dense little-endian BF16 matrix. A BF16 value is the top half of
// the FP32 with the same value, so widening is a shift rather than a conversion.
.visible .entry vastloom_matvec_bf16_f32(
    .param .u64 matrix_param,
    .param .u64 input_param,
    .param .u64 output_param,
    .param .u32 rows_param,
    .param .u32 columns_param
)
{
    .reg .pred %p<4>;
    .reg .b16 %rs<4>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<8>;
    .reg .f64 %fd<6>;

    ld.param.u64 %rd0, [matrix_param];
    ld.param.u64 %rd1, [input_param];
    ld.param.u64 %rd2, [output_param];
    ld.param.u32 %r0, [rows_param];
    ld.param.u32 %r1, [columns_param];
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r3, %r2, %r4;
    setp.ge.u32 %p0, %r5, %r0;
    @%p0 bra BF16_DONE;

    mul.lo.u32 %r6, %r5, %r1;
    mul.wide.u32 %rd3, %r6, 2;
    add.u64 %rd4, %rd0, %rd3;
    mov.u64 %rd5, %rd1;
    mov.f64 %fd0, 0d0000000000000000;
    mov.u32 %r7, 0;

BF16_LOOP:
    setp.ge.u32 %p1, %r7, %r1;
    @%p1 bra BF16_LOOP_DONE;
    ld.global.u16 %rs0, [%rd4];
    cvt.u32.u16 %r8, %rs0;
    shl.b32 %r9, %r8, 16;
    mov.b32 %f0, %r9;
    ld.global.f32 %f1, [%rd5];
    cvt.f64.f32 %fd1, %f0;
    cvt.f64.f32 %fd2, %f1;
    mul.rn.f64 %fd3, %fd1, %fd2;
    add.rn.f64 %fd0, %fd0, %fd3;
    add.u64 %rd4, %rd4, 2;
    add.u64 %rd5, %rd5, 4;
    add.u32 %r7, %r7, 1;
    bra BF16_LOOP;

BF16_LOOP_DONE:
    cvt.rn.f32.f64 %f2, %fd0;
    mul.wide.u32 %rd6, %r5, 4;
    add.u64 %rd7, %rd2, %rd6;
    st.global.f32 [%rd7], %f2;

BF16_DONE:
    ret;
}
)ptx";

// The whole module, assembled once. Returned by reference so the JIT gets a stable pointer.
const std::string& module_ptx() {
    static const std::string text =
        std::string(kRmsNormPtx) + std::string(kQuantizedMatvecPtx);
    return text;
}

void quarantine_backend(std::shared_ptr<CudaDriverBackend> backend) noexcept {
    // A failed external release may leave a live device allocation/context. Keep the driver
    // module loaded until process teardown; static storage is reclaimed normally at exit.
    static std::mutex quarantine_mutex;
    static std::vector<std::shared_ptr<CudaDriverBackend>> quarantined;
    try {
        std::scoped_lock lock(quarantine_mutex);
        quarantined.push_back(std::move(backend));
    } catch (...) {
        // Accounting remains charged even if retaining the module cannot allocate bookkeeping.
    }
}

using CuInit = CuResult(VASTLOOM_CUDA_CALL*)(unsigned int);
using CuDriverGetVersion = CuResult(VASTLOOM_CUDA_CALL*)(int*);
using CuDeviceGetCount = CuResult(VASTLOOM_CUDA_CALL*)(int*);
using CuDeviceGet = CuResult(VASTLOOM_CUDA_CALL*)(CuDevice*, int);
using CuDeviceGetName = CuResult(VASTLOOM_CUDA_CALL*)(char*, int, CuDevice);
using CuDeviceTotalMem = CuResult(VASTLOOM_CUDA_CALL*)(std::size_t*, CuDevice);
using CuDeviceGetAttribute = CuResult(VASTLOOM_CUDA_CALL*)(int*, int, CuDevice);
using CuDevicePrimaryCtxRetain = CuResult(VASTLOOM_CUDA_CALL*)(CuContext*, CuDevice);
using CuDevicePrimaryCtxRelease = CuResult(VASTLOOM_CUDA_CALL*)(CuDevice);
using CuCtxPushCurrent = CuResult(VASTLOOM_CUDA_CALL*)(CuContext);
using CuCtxPopCurrent = CuResult(VASTLOOM_CUDA_CALL*)(CuContext*);
using CuMemGetInfo = CuResult(VASTLOOM_CUDA_CALL*)(std::size_t*, std::size_t*);
using CuMemAlloc = CuResult(VASTLOOM_CUDA_CALL*)(CuDevicePointer*, std::size_t);
using CuMemFree = CuResult(VASTLOOM_CUDA_CALL*)(CuDevicePointer);
using CuMemcpyHostToDevice = CuResult(VASTLOOM_CUDA_CALL*)(CuDevicePointer, const void*,
                                                          std::size_t);
using CuMemcpyDeviceToHost = CuResult(VASTLOOM_CUDA_CALL*)(void*, CuDevicePointer,
                                                          std::size_t);
using CuModuleLoadData = CuResult(VASTLOOM_CUDA_CALL*)(CuModule*, const void*, unsigned int,
                                                      int*, void**);
using CuModuleGetFunction = CuResult(VASTLOOM_CUDA_CALL*)(CuFunction*, CuModule, const char*);
using CuModuleUnload = CuResult(VASTLOOM_CUDA_CALL*)(CuModule);
using CuLaunchKernel = CuResult(VASTLOOM_CUDA_CALL*)(CuFunction, unsigned int, unsigned int,
                                                    unsigned int, unsigned int, unsigned int,
                                                    unsigned int, unsigned int, CuStream, void**,
                                                    void**);
using CuContextSynchronize = CuResult(VASTLOOM_CUDA_CALL*)();
using CuGetErrorName = CuResult(VASTLOOM_CUDA_CALL*)(CuResult, const char**);
using CuGetErrorString = CuResult(VASTLOOM_CUDA_CALL*)(CuResult, const char**);

class DynamicCudaBackend final : public CudaDriverBackend {
public:
    DynamicCudaBackend(const DynamicCudaBackend&) = delete;
    DynamicCudaBackend& operator=(const DynamicCudaBackend&) = delete;

    ~DynamicCudaBackend() override {
        // The cached module holds its own context reference, so unloading here is safe and
        // releases that reference too.
        release_cached_module_locked();
#if defined(_WIN32)
        if (library_handle_ != nullptr) FreeLibrary(library_handle_);
#elif defined(__linux__)
        if (library_handle_ != nullptr) dlclose(library_handle_);
#endif
    }

    [[nodiscard]] static Result<std::shared_ptr<CudaDriverBackend>> create() {
#if defined(_WIN32)
        auto backend = std::shared_ptr<DynamicCudaBackend>(new DynamicCudaBackend());
        backend->library_handle_ = LoadLibraryExW(
            L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        backend->library_name_ = "nvcuda.dll";
        if (backend->library_handle_ == nullptr) {
            return make_error(ErrorCode::unsupported,
                              "NVIDIA CUDA driver library is unavailable", "nvcuda.dll");
        }
#elif defined(__linux__)
        auto backend = std::shared_ptr<DynamicCudaBackend>(new DynamicCudaBackend());
        backend->library_handle_ = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        backend->library_name_ = "libcuda.so.1";
        if (backend->library_handle_ == nullptr) {
            const char* detail = dlerror();
            return make_error(ErrorCode::unsupported,
                              "NVIDIA CUDA driver library is unavailable",
                              detail == nullptr ? "libcuda.so.1" : detail);
        }
#else
        return make_error(ErrorCode::unsupported,
                          "The native CUDA driver boundary supports Windows and Linux only");
#endif

#if defined(_WIN32) || defined(__linux__)
        auto loaded = backend->load_functions();
        if (!loaded) return loaded.error();
        std::shared_ptr<CudaDriverBackend> result = std::move(backend);
        return result;
#endif
    }

    [[nodiscard]] Result<CudaDriverProbe> probe() override {
        std::scoped_lock lock(mutex_);
        auto initialized = initialize_locked();
        if (!initialized) return initialized.error();

        int driver_version = 0;
        CuResult status = driver_get_version_(&driver_version);
        if (status != kCudaSuccess) return cuda_error("cuDriverGetVersion", status);
        int device_count = 0;
        status = device_get_count_(&device_count);
        if (status != kCudaSuccess) return cuda_error("cuDeviceGetCount", status);
        if (device_count < 0 || device_count > 256) {
            return make_error(ErrorCode::invalid_format,
                              "CUDA driver returned an invalid device count",
                              std::to_string(device_count));
        }

        CudaDriverProbe result;
        result.library = library_name_;
        result.driver_version = driver_version;
        result.devices.reserve(static_cast<std::size_t>(device_count));
        for (int ordinal = 0; ordinal < device_count; ++ordinal) {
            auto device = get_device_locked(ordinal);
            if (!device) return device.error();
            CudaDeviceCapabilities capabilities;
            capabilities.ordinal = ordinal;
            std::array<char, kMaximumDeviceNameBytes> name{};
            status = device_get_name_(name.data(), static_cast<int>(name.size()), device.value());
            if (status != kCudaSuccess) return cuda_error("cuDeviceGetName", status);
            name.back() = '\0';
            capabilities.name = name.data();
            if (capabilities.name.empty()) {
                return make_error(ErrorCode::invalid_format,
                                  "CUDA driver returned an empty device name",
                                  std::to_string(ordinal));
            }
            std::size_t total_bytes = 0U;
            status = device_total_memory_(&total_bytes, device.value());
            if (status != kCudaSuccess) return cuda_error("cuDeviceTotalMem_v2", status);
            if (total_bytes == 0U) {
                return make_error(ErrorCode::invalid_format,
                                  "CUDA driver returned a zero-byte device capacity",
                                  capabilities.name);
            }
            capabilities.total_vram_bytes = static_cast<ByteCount>(total_bytes);
            auto major = attribute_locked(device.value(), kAttributeComputeCapabilityMajor,
                                          "compute capability major");
            auto minor = attribute_locked(device.value(), kAttributeComputeCapabilityMinor,
                                          "compute capability minor");
            auto multiprocessors = attribute_locked(device.value(), kAttributeMultiprocessorCount,
                                                    "multiprocessor count");
            auto maximum_threads = attribute_locked(device.value(), kAttributeMaximumThreadsPerBlock,
                                                    "maximum threads per block");
            auto unified = attribute_locked(device.value(), kAttributeUnifiedAddressing,
                                            "unified addressing");
            if (!major) return major.error();
            if (!minor) return minor.error();
            if (!multiprocessors) return multiprocessors.error();
            if (!maximum_threads) return maximum_threads.error();
            if (!unified) return unified.error();
            if (major.value() < 1 || minor.value() < 0 || multiprocessors.value() < 1 ||
                maximum_threads.value() < 1 ||
                (unified.value() != 0 && unified.value() != 1)) {
                return make_error(ErrorCode::invalid_format,
                                  "CUDA driver returned invalid device capabilities",
                                  capabilities.name);
            }
            capabilities.compute_capability_major = major.value();
            capabilities.compute_capability_minor = minor.value();
            capabilities.multiprocessor_count = multiprocessors.value();
            capabilities.maximum_threads_per_block = maximum_threads.value();
            capabilities.unified_addressing = unified.value() == 1;
            result.devices.push_back(std::move(capabilities));
        }
        return result;
    }

    [[nodiscard]] Result<CudaBackendAllocation> allocate(
        int device_ordinal,
        ByteCount bytes,
        ByteCount minimum_free_vram_after) override {
        std::scoped_lock lock(mutex_);
        auto initialized = initialize_locked();
        if (!initialized) return initialized.error();
        if (bytes > static_cast<ByteCount>(std::numeric_limits<std::size_t>::max())) {
            return make_error(ErrorCode::resource_exhausted,
                              "CUDA allocation exceeds this process address size");
        }
        auto device = get_device_locked(device_ordinal);
        if (!device) return device.error();

        CuContext context = nullptr;
        CuResult status = primary_context_retain_(&context, device.value());
        if (status != kCudaSuccess) return cuda_error("cuDevicePrimaryCtxRetain", status);
        if (context == nullptr) {
            static_cast<void>(primary_context_release_(device.value()));
            return make_error(ErrorCode::internal_error,
                              "CUDA primary context retain returned a null context");
        }
        status = context_push_(context);
        if (status != kCudaSuccess) {
            static_cast<void>(primary_context_release_(device.value()));
            return cuda_error("cuCtxPushCurrent_v2", status);
        }

        std::size_t free_bytes = 0U;
        std::size_t total_bytes = 0U;
        status = memory_info_(&free_bytes, &total_bytes);
        if (status != kCudaSuccess) {
            pop_and_release_locked(device.value());
            return cuda_error("cuMemGetInfo_v2", status);
        }
        const ByteCount free = static_cast<ByteCount>(free_bytes);
        if (bytes > free || minimum_free_vram_after > free - bytes) {
            pop_and_release_locked(device.value());
            return make_error(ErrorCode::resource_exhausted,
                              "CUDA allocation would consume the live VRAM reserve",
                              "requested=" + std::to_string(bytes) +
                                  ", free=" + std::to_string(free) +
                                  ", reserve=" + std::to_string(minimum_free_vram_after));
        }

        CuDevicePointer pointer = 0U;
        status = memory_allocate_(&pointer, static_cast<std::size_t>(bytes));
        if (status != kCudaSuccess) {
            pop_and_release_locked(device.value());
            return cuda_error("cuMemAlloc_v2", status);
        }
        CuContext popped = nullptr;
        status = context_pop_(&popped);
        if (status != kCudaSuccess) {
            static_cast<void>(memory_free_(pointer));
            static_cast<void>(primary_context_release_(device.value()));
            return cuda_error("cuCtxPopCurrent_v2", status);
        }
        if (popped != context) {
            static_cast<void>(context_push_(context));
            static_cast<void>(memory_free_(pointer));
            static_cast<void>(context_pop_(&popped));
            static_cast<void>(primary_context_release_(device.value()));
            return make_error(ErrorCode::internal_error,
                              "CUDA context stack returned a different context");
        }
        return CudaBackendAllocation{reinterpret_cast<std::uintptr_t>(context), pointer,
                                     bytes, device_ordinal};
    }

    [[nodiscard]] Result<void> release(const CudaBackendAllocation& allocation) override {
        std::scoped_lock lock(mutex_);
        auto initialized = initialize_locked();
        if (!initialized) return initialized.error();
        if (allocation.context == 0U || allocation.device_pointer == 0U ||
            allocation.bytes == 0U || allocation.device_ordinal < 0) {
            return make_error(ErrorCode::invalid_argument,
                              "Invalid CUDA allocation handle supplied for release");
        }
        auto device = get_device_locked(allocation.device_ordinal);
        if (!device) return device.error();
        auto* context = reinterpret_cast<CuContext>(allocation.context);
        CuResult status = context_push_(context);
        if (status != kCudaSuccess) return cuda_error("cuCtxPushCurrent_v2", status);
        status = memory_free_(allocation.device_pointer);
        if (status != kCudaSuccess) {
            CuContext popped = nullptr;
            static_cast<void>(context_pop_(&popped));
            return cuda_error("cuMemFree_v2", status);
        }

        // The device allocation is gone at this point. Context-stack/refcount cleanup is best
        // effort because retrying cuMemFree after either of these calls fails would double-free.
        CuContext popped = nullptr;
        static_cast<void>(context_pop_(&popped));
        static_cast<void>(primary_context_release_(device.value()));
        return {};
    }

    [[nodiscard]] Result<void> copy_from_host(
        const CudaBackendAllocation& allocation,
        ByteCount destination_offset,
        std::span<const std::byte> source) override {
        std::scoped_lock lock(mutex_);
        auto range = validate_allocation_range(allocation, destination_offset, source.size());
        if (!range) return range.error();
        auto* context = reinterpret_cast<CuContext>(allocation.context);
        CuResult status = context_push_(context);
        if (status != kCudaSuccess) return cuda_error("cuCtxPushCurrent_v2", status);
        status = copy_host_to_device_(allocation.device_pointer + destination_offset,
                                      source.data(), source.size());
        auto popped = pop_context_locked(context);
        if (status != kCudaSuccess) return cuda_error("cuMemcpyHtoD_v2", status);
        if (!popped) return popped.error();
        return {};
    }

    [[nodiscard]] Result<void> copy_to_host(
        const CudaBackendAllocation& allocation,
        ByteCount source_offset,
        std::span<std::byte> destination) override {
        std::scoped_lock lock(mutex_);
        auto range = validate_allocation_range(allocation, source_offset, destination.size());
        if (!range) return range.error();
        auto* context = reinterpret_cast<CuContext>(allocation.context);
        CuResult status = context_push_(context);
        if (status != kCudaSuccess) return cuda_error("cuCtxPushCurrent_v2", status);
        status = copy_device_to_host_(destination.data(),
                                      allocation.device_pointer + source_offset,
                                      destination.size());
        auto popped = pop_context_locked(context);
        if (status != kCudaSuccess) return cuda_error("cuMemcpyDtoH_v2", status);
        if (!popped) return popped.error();
        return {};
    }

    [[nodiscard]] Result<void> launch_rms_norm(
        const CudaBackendAllocation& allocation,
        ByteCount input_offset,
        ByteCount weight_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t width,
        float epsilon) override {
        std::scoped_lock lock(mutex_);
        if (row_count == 0U || width == 0U || !std::isfinite(epsilon) || epsilon <= 0.0F ||
            row_count > std::numeric_limits<std::uint32_t>::max() ||
            width > std::numeric_limits<std::uint32_t>::max()) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA RMSNorm launch dimensions are invalid");
        }
        if (width > std::numeric_limits<std::size_t>::max() / row_count ||
            width > std::numeric_limits<ByteCount>::max() / row_count ||
            row_count * width > std::numeric_limits<ByteCount>::max() / sizeof(float)) {
            return make_error(ErrorCode::resource_exhausted,
                              "CUDA RMSNorm launch byte count overflows");
        }
        const ByteCount element_count = static_cast<ByteCount>(row_count * width);
        const ByteCount matrix_bytes = element_count * sizeof(float);
        if (width > std::numeric_limits<ByteCount>::max() / sizeof(float)) {
            return make_error(ErrorCode::resource_exhausted,
                              "CUDA RMSNorm weight byte count overflows");
        }
        const ByteCount weight_bytes = static_cast<ByteCount>(width) * sizeof(float);
        for (const auto& [offset, bytes] :
             {std::pair{input_offset, matrix_bytes}, std::pair{weight_offset, weight_bytes},
              std::pair{output_offset, matrix_bytes}}) {
            auto range = validate_allocation_range(allocation, offset, bytes);
            if (!range) return range.error();
            if (offset % alignof(float) != 0U) {
                return make_error(ErrorCode::invalid_argument,
                                  "CUDA RMSNorm buffer offsets must be FP32-aligned");
            }
        }
        const auto overlaps = [](ByteCount first_offset, ByteCount first_bytes,
                                 ByteCount second_offset, ByteCount second_bytes) {
            return first_offset < second_offset + second_bytes &&
                   second_offset < first_offset + first_bytes;
        };
        if (overlaps(input_offset, matrix_bytes, weight_offset, weight_bytes) ||
            overlaps(input_offset, matrix_bytes, output_offset, matrix_bytes) ||
            overlaps(weight_offset, weight_bytes, output_offset, matrix_bytes)) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA RMSNorm input, weight and output ranges must not overlap");
        }

        auto* context = reinterpret_cast<CuContext>(allocation.context);
        CuResult status = context_push_(context);
        if (status != kCudaSuccess) return cuda_error("cuCtxPushCurrent_v2", status);
        CuModule module = nullptr;
        std::array<char, kMaximumJitLogBytes> jit_error_log{};
        int jit_options[] = {kJitErrorLogBufferSizeBytes, kJitErrorLogBuffer};
        void* jit_option_values[] = {
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(jit_error_log.size())),
            jit_error_log.data()};
        status = module_load_data_(&module, module_ptx().c_str(), 2U, jit_options,
                                   jit_option_values);
        if (status != kCudaSuccess) {
            static_cast<void>(pop_context_locked(context));
            auto error = cuda_error("cuModuleLoadDataEx", status);
            if (jit_error_log.front() != '\0') {
                const auto log_end = std::find(jit_error_log.begin(), jit_error_log.end(), '\0');
                error.context += ", JIT log: ";
                error.context.append(
                    jit_error_log.data(),
                    static_cast<std::size_t>(std::distance(jit_error_log.begin(), log_end)));
            }
            return error;
        }
        CuFunction function = nullptr;
        status = module_get_function_(&function, module, "vastloom_rms_norm_f32");
        if (status != kCudaSuccess) {
            static_cast<void>(module_unload_(module));
            static_cast<void>(pop_context_locked(context));
            return cuda_error("cuModuleGetFunction", status);
        }
        CuDevicePointer input_pointer = allocation.device_pointer + input_offset;
        CuDevicePointer weight_pointer = allocation.device_pointer + weight_offset;
        CuDevicePointer output_pointer = allocation.device_pointer + output_offset;
        auto rows = static_cast<std::uint32_t>(row_count);
        auto columns = static_cast<std::uint32_t>(width);
        void* parameters[] = {&input_pointer, &weight_pointer, &output_pointer,
                              &rows, &columns, &epsilon};
        status = launch_kernel_(function, rows, 1U, 1U, 1U, 1U, 1U, 0U, nullptr,
                                parameters, nullptr);
        if (status == kCudaSuccess) status = context_synchronize_();
        const CuResult unload_status = module_unload_(module);
        auto popped = pop_context_locked(context);
        if (status != kCudaSuccess) return cuda_error("CUDA RMSNorm launch/synchronize", status);
        if (unload_status != kCudaSuccess) return cuda_error("cuModuleUnload", unload_status);
        if (!popped) return popped.error();
        return {};
    }

    [[nodiscard]] Result<void> matvec_iq1_s_with_transfer(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount grid_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) override {
        std::scoped_lock lock(mutex_);
        constexpr ByteCount kBlockBytes = 50U;
        constexpr ByteCount kBlockElements = 256U;
        constexpr ByteCount kGridBytes = 2'048U * sizeof(std::uint64_t);
        if (row_count == 0U || blocks_per_row == 0U ||
            row_count > std::numeric_limits<std::uint32_t>::max() ||
            blocks_per_row > std::numeric_limits<std::uint32_t>::max() ||
            blocks_per_row > std::numeric_limits<ByteCount>::max() / row_count) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA IQ1_S matvec launch dimensions are invalid");
        }
        const ByteCount blocks = static_cast<ByteCount>(row_count) * blocks_per_row;
        if (blocks > std::numeric_limits<ByteCount>::max() / kBlockBytes ||
            blocks_per_row > std::numeric_limits<ByteCount>::max() / kBlockElements ||
            row_count > std::numeric_limits<ByteCount>::max() / sizeof(float)) {
            return make_error(ErrorCode::resource_exhausted,
                              "CUDA IQ1_S matvec byte count overflows");
        }
        const ByteCount matrix_bytes = blocks * kBlockBytes;
        const ByteCount input_bytes =
            static_cast<ByteCount>(blocks_per_row) * kBlockElements * sizeof(float);
        const ByteCount output_bytes = static_cast<ByteCount>(row_count) * sizeof(float);
        if (input.size() != input_bytes || output.size() != output_bytes) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA IQ1_S matvec host spans do not match its shape");
        }
        for (const auto& [offset, bytes] :
             {std::pair{matrix_offset, matrix_bytes}, std::pair{grid_offset, kGridBytes},
              std::pair{input_offset, input_bytes}, std::pair{output_offset, output_bytes}}) {
            auto range = validate_allocation_range(allocation, offset, bytes);
            if (!range) return range.error();
        }
        if (input_offset % alignof(float) != 0U || output_offset % alignof(float) != 0U ||
            grid_offset % alignof(std::uint64_t) != 0U) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA IQ1_S matvec ranges are misaligned");
        }

        auto* context = reinterpret_cast<CuContext>(allocation.context);
        CuResult status = context_push_(context);
        if (status != kCudaSuccess) return cuda_error("cuCtxPushCurrent_v2", status);
        auto module = cached_module_locked(context, allocation.device_ordinal);
        if (!module) {
            static_cast<void>(pop_context_locked(context));
            return module.error();
        }
        CuFunction function = nullptr;
        status = module_get_function_(&function, module.value(), "vastloom_matvec_iq1_s_f32");
        if (status != kCudaSuccess) {
            static_cast<void>(pop_context_locked(context));
            return cuda_error("cuModuleGetFunction", status);
        }
        status = copy_host_to_device_(allocation.device_pointer + input_offset, input.data(),
                                      input.size());
        if (status == kCudaSuccess) {
            CuDevicePointer matrix_pointer = allocation.device_pointer + matrix_offset;
            CuDevicePointer grid_pointer = allocation.device_pointer + grid_offset;
            CuDevicePointer input_pointer = allocation.device_pointer + input_offset;
            CuDevicePointer output_pointer = allocation.device_pointer + output_offset;
            auto rows = static_cast<std::uint32_t>(row_count);
            auto row_blocks = static_cast<std::uint32_t>(blocks_per_row);
            void* parameters[] = {&matrix_pointer, &grid_pointer, &input_pointer, &output_pointer,
                                  &rows, &row_blocks};
            constexpr std::uint32_t kThreadsPerBlock = 128U;
            const std::uint32_t grid = (rows + kThreadsPerBlock - 1U) / kThreadsPerBlock;
            status = launch_kernel_(function, grid, 1U, 1U, kThreadsPerBlock, 1U, 1U, 0U, nullptr,
                                    parameters, nullptr);
        }
        if (status == kCudaSuccess) {
            status = copy_device_to_host_(output.data(), allocation.device_pointer + output_offset,
                                          output.size());
        }
        auto popped = pop_context_locked(context);
        if (status != kCudaSuccess) return cuda_error("CUDA IQ1_S matvec transfer/launch", status);
        if (!popped) return popped.error();
        return {};
    }

    [[nodiscard]] Result<void> matvec_iq3_xxs_with_transfer(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount grid_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) override {
        std::scoped_lock lock(mutex_);
        constexpr ByteCount kBlockBytes = 98U;
        constexpr ByteCount kBlockElements = 256U;
        constexpr ByteCount kGridBytes = 256U * sizeof(std::uint32_t);
        if (row_count == 0U || blocks_per_row == 0U ||
            row_count > std::numeric_limits<std::uint32_t>::max() ||
            blocks_per_row > std::numeric_limits<std::uint32_t>::max() ||
            blocks_per_row > std::numeric_limits<ByteCount>::max() / row_count) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA IQ3_XXS matvec launch dimensions are invalid");
        }
        const ByteCount blocks = static_cast<ByteCount>(row_count) * blocks_per_row;
        if (blocks > std::numeric_limits<ByteCount>::max() / kBlockBytes ||
            blocks_per_row > std::numeric_limits<ByteCount>::max() / kBlockElements ||
            row_count > std::numeric_limits<ByteCount>::max() / sizeof(float)) {
            return make_error(ErrorCode::resource_exhausted,
                              "CUDA IQ3_XXS matvec byte count overflows");
        }
        const ByteCount matrix_bytes = blocks * kBlockBytes;
        const ByteCount input_bytes =
            static_cast<ByteCount>(blocks_per_row) * kBlockElements * sizeof(float);
        const ByteCount output_bytes = static_cast<ByteCount>(row_count) * sizeof(float);
        if (input.size() != input_bytes || output.size() != output_bytes) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA IQ3_XXS matvec host spans do not match its shape");
        }
        for (const auto& [offset, bytes] :
             {std::pair{matrix_offset, matrix_bytes}, std::pair{grid_offset, kGridBytes},
              std::pair{input_offset, input_bytes}, std::pair{output_offset, output_bytes}}) {
            auto range = validate_allocation_range(allocation, offset, bytes);
            if (!range) return range.error();
        }
        if (input_offset % alignof(float) != 0U || output_offset % alignof(float) != 0U ||
            grid_offset % alignof(std::uint32_t) != 0U) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA IQ3_XXS matvec ranges are misaligned");
        }

        auto* context = reinterpret_cast<CuContext>(allocation.context);
        CuResult status = context_push_(context);
        if (status != kCudaSuccess) return cuda_error("cuCtxPushCurrent_v2", status);
        auto module = cached_module_locked(context, allocation.device_ordinal);
        if (!module) {
            static_cast<void>(pop_context_locked(context));
            return module.error();
        }
        CuFunction function = nullptr;
        status = module_get_function_(&function, module.value(), "vastloom_matvec_iq3_xxs_f32");
        if (status != kCudaSuccess) {
            static_cast<void>(pop_context_locked(context));
            return cuda_error("cuModuleGetFunction", status);
        }
        status = copy_host_to_device_(allocation.device_pointer + input_offset, input.data(),
                                      input.size());
        if (status == kCudaSuccess) {
            CuDevicePointer matrix_pointer = allocation.device_pointer + matrix_offset;
            CuDevicePointer grid_pointer = allocation.device_pointer + grid_offset;
            CuDevicePointer input_pointer = allocation.device_pointer + input_offset;
            CuDevicePointer output_pointer = allocation.device_pointer + output_offset;
            auto rows = static_cast<std::uint32_t>(row_count);
            auto row_blocks = static_cast<std::uint32_t>(blocks_per_row);
            void* parameters[] = {&matrix_pointer, &grid_pointer, &input_pointer, &output_pointer,
                                  &rows, &row_blocks};
            constexpr std::uint32_t kThreadsPerBlock = 128U;
            const std::uint32_t grid = (rows + kThreadsPerBlock - 1U) / kThreadsPerBlock;
            status = launch_kernel_(function, grid, 1U, 1U, kThreadsPerBlock, 1U, 1U, 0U, nullptr,
                                    parameters, nullptr);
        }
        if (status == kCudaSuccess) {
            status = copy_device_to_host_(output.data(), allocation.device_pointer + output_offset,
                                          output.size());
        }
        auto popped = pop_context_locked(context);
        if (status != kCudaSuccess) {
            return cuda_error("CUDA IQ3_XXS matvec transfer/launch", status);
        }
        if (!popped) return popped.error();
        return {};
    }

    [[nodiscard]] Result<void> matvec_dense_with_transfer(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t columns,
        ByteCount element_bytes,
        std::span<const std::byte> input,
        std::span<std::byte> output) override {
        std::scoped_lock lock(mutex_);
        if (element_bytes != 2U && element_bytes != 4U) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA dense matvec supports only BF16 and FP32 elements");
        }
        if (row_count == 0U || columns == 0U ||
            row_count > std::numeric_limits<std::uint32_t>::max() ||
            columns > std::numeric_limits<std::uint32_t>::max() ||
            columns > std::numeric_limits<ByteCount>::max() / row_count) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA dense matvec launch dimensions are invalid");
        }
        const ByteCount elements = static_cast<ByteCount>(row_count) * columns;
        if (elements > std::numeric_limits<ByteCount>::max() / element_bytes ||
            columns > std::numeric_limits<ByteCount>::max() / sizeof(float) ||
            row_count > std::numeric_limits<ByteCount>::max() / sizeof(float)) {
            return make_error(ErrorCode::resource_exhausted,
                              "CUDA dense matvec byte count overflows");
        }
        const ByteCount matrix_bytes = elements * element_bytes;
        const ByteCount input_bytes = static_cast<ByteCount>(columns) * sizeof(float);
        const ByteCount output_bytes = static_cast<ByteCount>(row_count) * sizeof(float);
        if (input.size() != input_bytes || output.size() != output_bytes) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA dense matvec host spans do not match its shape");
        }
        for (const auto& [offset, bytes] :
             {std::pair{matrix_offset, matrix_bytes}, std::pair{input_offset, input_bytes},
              std::pair{output_offset, output_bytes}}) {
            auto range = validate_allocation_range(allocation, offset, bytes);
            if (!range) return range.error();
        }
        if (input_offset % alignof(float) != 0U || output_offset % alignof(float) != 0U ||
            matrix_offset % element_bytes != 0U) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA dense matvec ranges are misaligned");
        }

        auto* context = reinterpret_cast<CuContext>(allocation.context);
        CuResult status = context_push_(context);
        if (status != kCudaSuccess) return cuda_error("cuCtxPushCurrent_v2", status);
        auto module = cached_module_locked(context, allocation.device_ordinal);
        if (!module) {
            static_cast<void>(pop_context_locked(context));
            return module.error();
        }
        CuFunction function = nullptr;
        status = module_get_function_(
            &function, module.value(),
            element_bytes == 4U ? "vastloom_matvec_f32_f32" : "vastloom_matvec_bf16_f32");
        if (status != kCudaSuccess) {
            static_cast<void>(pop_context_locked(context));
            return cuda_error("cuModuleGetFunction", status);
        }
        status = copy_host_to_device_(allocation.device_pointer + input_offset, input.data(),
                                      input.size());
        if (status == kCudaSuccess) {
            CuDevicePointer matrix_pointer = allocation.device_pointer + matrix_offset;
            CuDevicePointer input_pointer = allocation.device_pointer + input_offset;
            CuDevicePointer output_pointer = allocation.device_pointer + output_offset;
            auto rows = static_cast<std::uint32_t>(row_count);
            auto width = static_cast<std::uint32_t>(columns);
            void* parameters[] = {&matrix_pointer, &input_pointer, &output_pointer, &rows, &width};
            constexpr std::uint32_t kThreadsPerBlock = 128U;
            const std::uint32_t grid = (rows + kThreadsPerBlock - 1U) / kThreadsPerBlock;
            status = launch_kernel_(function, grid, 1U, 1U, kThreadsPerBlock, 1U, 1U, 0U, nullptr,
                                    parameters, nullptr);
        }
        if (status == kCudaSuccess) {
            status = copy_device_to_host_(output.data(), allocation.device_pointer + output_offset,
                                          output.size());
        }
        auto popped = pop_context_locked(context);
        if (status != kCudaSuccess) return cuda_error("CUDA dense matvec transfer/launch", status);
        if (!popped) return popped.error();
        return {};
    }

    [[nodiscard]] Result<void> matvec_q8_0_with_transfer(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row,
        std::span<const std::byte> input,
        std::span<std::byte> output) override {
        std::scoped_lock lock(mutex_);
        auto input_range = validate_allocation_range(allocation, input_offset, input.size());
        if (!input_range) return input_range.error();
        auto output_range = validate_allocation_range(allocation, output_offset, output.size());
        if (!output_range) return output_range.error();
        auto prepared = validate_matvec_q8_0_locked(allocation, matrix_offset, input_offset,
                                                    output_offset, row_count, blocks_per_row);
        if (!prepared) return prepared.error();

        auto* context = reinterpret_cast<CuContext>(allocation.context);
        CuResult status = context_push_(context);
        if (status != kCudaSuccess) return cuda_error("cuCtxPushCurrent_v2", status);
        auto module = cached_module_locked(context, allocation.device_ordinal);
        if (!module) {
            static_cast<void>(pop_context_locked(context));
            return module.error();
        }
        CuFunction function = nullptr;
        status = module_get_function_(&function, module.value(), "vastloom_matvec_q8_0_f32");
        if (status != kCudaSuccess) {
            static_cast<void>(pop_context_locked(context));
            return cuda_error("cuModuleGetFunction", status);
        }
        status = copy_host_to_device_(allocation.device_pointer + input_offset, input.data(),
                                      input.size());
        if (status == kCudaSuccess) {
            CuDevicePointer matrix_pointer = allocation.device_pointer + matrix_offset;
            CuDevicePointer input_pointer = allocation.device_pointer + input_offset;
            CuDevicePointer output_pointer = allocation.device_pointer + output_offset;
            auto rows = static_cast<std::uint32_t>(row_count);
            auto row_blocks = static_cast<std::uint32_t>(blocks_per_row);
            void* parameters[] = {&matrix_pointer, &input_pointer, &output_pointer, &rows,
                                  &row_blocks};
            constexpr std::uint32_t kThreadsPerBlock = 128U;
            const std::uint32_t grid = (rows + kThreadsPerBlock - 1U) / kThreadsPerBlock;
            status = launch_kernel_(function, grid, 1U, 1U, kThreadsPerBlock, 1U, 1U, 0U, nullptr,
                                    parameters, nullptr);
        }
        // One synchronize covers the launch; the device-to-host copy is ordered after it on the
        // null stream, so the download needs no separate barrier.
        if (status == kCudaSuccess) {
            status = copy_device_to_host_(output.data(), allocation.device_pointer + output_offset,
                                          output.size());
        }
        auto popped = pop_context_locked(context);
        if (status != kCudaSuccess) return cuda_error("CUDA Q8_0 matvec transfer/launch", status);
        if (!popped) return popped.error();
        return {};
    }

    [[nodiscard]] Result<void> launch_matvec_q8_0(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row) override {
        std::scoped_lock lock(mutex_);
        auto prepared = validate_matvec_q8_0_locked(allocation, matrix_offset, input_offset,
                                                    output_offset, row_count, blocks_per_row);
        if (!prepared) return prepared.error();

        auto* context = reinterpret_cast<CuContext>(allocation.context);
        CuResult status = context_push_(context);
        if (status != kCudaSuccess) return cuda_error("cuCtxPushCurrent_v2", status);
        // JIT-compiling the PTX costs milliseconds, which dwarfs the kernel itself when a
        // generation issues thousands of matvecs. The module is cached per context and unloaded
        // when the backend is destroyed.
        auto module = cached_module_locked(context, allocation.device_ordinal);
        if (!module) {
            static_cast<void>(pop_context_locked(context));
            return module.error();
        }
        CuFunction function = nullptr;
        status = module_get_function_(&function, module.value(), "vastloom_matvec_q8_0_f32");
        if (status != kCudaSuccess) {
            static_cast<void>(pop_context_locked(context));
            return cuda_error("cuModuleGetFunction", status);
        }
        CuDevicePointer matrix_pointer = allocation.device_pointer + matrix_offset;
        CuDevicePointer input_pointer = allocation.device_pointer + input_offset;
        CuDevicePointer output_pointer = allocation.device_pointer + output_offset;
        auto rows = static_cast<std::uint32_t>(row_count);
        auto row_blocks = static_cast<std::uint32_t>(blocks_per_row);
        void* parameters[] = {&matrix_pointer, &input_pointer, &output_pointer, &rows,
                              &row_blocks};
        constexpr std::uint32_t kThreadsPerBlock = 128U;
        const std::uint32_t grid = (rows + kThreadsPerBlock - 1U) / kThreadsPerBlock;
        status = launch_kernel_(function, grid, 1U, 1U, kThreadsPerBlock, 1U, 1U, 0U, nullptr,
                                parameters, nullptr);
        if (status == kCudaSuccess) status = context_synchronize_();
        auto popped = pop_context_locked(context);
        if (status != kCudaSuccess) return cuda_error("CUDA Q8_0 matvec launch/synchronize", status);
        if (!popped) return popped.error();
        return {};
    }

private:
    DynamicCudaBackend() = default;

    // Shared bounds check for both Q8_0 entry points. The caller must already hold the lock.
    [[nodiscard]] Result<void> validate_matvec_q8_0_locked(
        const CudaBackendAllocation& allocation,
        ByteCount matrix_offset,
        ByteCount input_offset,
        ByteCount output_offset,
        std::size_t row_count,
        std::size_t blocks_per_row) const {
        constexpr ByteCount kBlockBytes = 34U;
        constexpr ByteCount kBlockElements = 32U;
        if (row_count == 0U || blocks_per_row == 0U ||
            row_count > std::numeric_limits<std::uint32_t>::max() ||
            blocks_per_row > std::numeric_limits<std::uint32_t>::max()) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA Q8_0 matvec launch dimensions are invalid");
        }
        if (blocks_per_row > std::numeric_limits<ByteCount>::max() / row_count) {
            return make_error(ErrorCode::resource_exhausted,
                              "CUDA Q8_0 matvec block count overflows");
        }
        const ByteCount blocks = static_cast<ByteCount>(row_count) * blocks_per_row;
        if (blocks > std::numeric_limits<ByteCount>::max() / kBlockBytes ||
            blocks_per_row > std::numeric_limits<ByteCount>::max() / kBlockElements ||
            row_count > std::numeric_limits<ByteCount>::max() / sizeof(float)) {
            return make_error(ErrorCode::resource_exhausted,
                              "CUDA Q8_0 matvec byte count overflows");
        }
        const ByteCount matrix_bytes = blocks * kBlockBytes;
        const ByteCount input_bytes =
            static_cast<ByteCount>(blocks_per_row) * kBlockElements * sizeof(float);
        const ByteCount output_bytes = static_cast<ByteCount>(row_count) * sizeof(float);
        for (const auto& [offset, bytes] :
             {std::pair{matrix_offset, matrix_bytes}, std::pair{input_offset, input_bytes},
              std::pair{output_offset, output_bytes}}) {
            auto range = validate_allocation_range(allocation, offset, bytes);
            if (!range) return range.error();
        }
        if (input_offset % alignof(float) != 0U || output_offset % alignof(float) != 0U) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA Q8_0 matvec float ranges must be FP32-aligned");
        }
        const auto overlaps = [](ByteCount first_offset, ByteCount first_bytes,
                                 ByteCount second_offset, ByteCount second_bytes) {
            return first_offset < second_offset + second_bytes &&
                   second_offset < first_offset + first_bytes;
        };
        if (overlaps(matrix_offset, matrix_bytes, input_offset, input_bytes) ||
            overlaps(matrix_offset, matrix_bytes, output_offset, output_bytes) ||
            overlaps(input_offset, input_bytes, output_offset, output_bytes)) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA Q8_0 matvec ranges must not overlap");
        }
        return {};
    }

    // Loads the kernel module once per context. The caller must already hold the lock and have
    // pushed the context.
    //
    // The primary context is reference counted, so caching a module means holding a reference of
    // our own: without it the last allocation's release would destroy the context and leave the
    // cached module dangling.
    [[nodiscard]] Result<CuModule> cached_module_locked(CuContext context, int device_ordinal) {
        if (cached_module_ != nullptr && cached_module_context_ == context) {
            return cached_module_;
        }
        release_cached_module_locked();
        auto device = get_device_locked(device_ordinal);
        if (!device) return device.error();
        CuContext retained = nullptr;
        const CuResult retain_status = primary_context_retain_(&retained, device.value());
        if (retain_status != kCudaSuccess) {
            return cuda_error("cuDevicePrimaryCtxRetain", retain_status);
        }
        if (retained != context) {
            static_cast<void>(primary_context_release_(device.value()));
            return make_error(ErrorCode::internal_error,
                              "CUDA primary context changed while caching the kernel module");
        }
        CuModule module = nullptr;
        std::array<char, kMaximumJitLogBytes> jit_error_log{};
        int jit_options[] = {kJitErrorLogBufferSizeBytes, kJitErrorLogBuffer};
        void* jit_option_values[] = {
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(jit_error_log.size())),
            jit_error_log.data()};
        const CuResult status = module_load_data_(&module, module_ptx().c_str(), 2U, jit_options,
                                                  jit_option_values);
        if (status != kCudaSuccess) {
            static_cast<void>(primary_context_release_(device.value()));
            auto error = cuda_error("cuModuleLoadDataEx", status);
            if (jit_error_log.front() != '\0') {
                const auto log_end = std::find(jit_error_log.begin(), jit_error_log.end(), '\0');
                error.context += ", JIT log: ";
                error.context.append(
                    jit_error_log.data(),
                    static_cast<std::size_t>(std::distance(jit_error_log.begin(), log_end)));
            }
            return error;
        }
        cached_module_ = module;
        cached_module_context_ = context;
        cached_module_device_ = device.value();
        cached_module_device_valid_ = true;
        return module;
    }

    void release_cached_module_locked() noexcept {
        if (cached_module_ == nullptr) return;
        if (module_unload_ != nullptr) static_cast<void>(module_unload_(cached_module_));
        if (cached_module_device_valid_ && primary_context_release_ != nullptr) {
            static_cast<void>(primary_context_release_(cached_module_device_));
        }
        cached_module_ = nullptr;
        cached_module_context_ = nullptr;
        cached_module_device_valid_ = false;
    }

    template <typename Function>
    [[nodiscard]] Result<void> load_symbol(Function& target,
                                           std::initializer_list<const char*> names) {
        for (const char* name : names) {
#if defined(_WIN32)
            const FARPROC raw = GetProcAddress(library_handle_, name);
#elif defined(__linux__)
            void* const raw = dlsym(library_handle_, name);
#endif
            if (raw != nullptr) {
                static_assert(sizeof(target) == sizeof(raw));
                std::memcpy(&target, &raw, sizeof(target));
                return {};
            }
        }
        return make_error(ErrorCode::unsupported,
                          "CUDA driver is missing a required entry point",
                          names.size() == 0U ? "unknown" : *names.begin());
    }

    [[nodiscard]] Result<void> load_functions() {
#define VASTLOOM_LOAD_CUDA(member, type, ...)                  \
    do {                                                       \
        auto loaded = load_symbol<type>(member, {__VA_ARGS__}); \
        if (!loaded) return loaded.error();                    \
    } while (false)
        VASTLOOM_LOAD_CUDA(init_, CuInit, "cuInit");
        VASTLOOM_LOAD_CUDA(driver_get_version_, CuDriverGetVersion, "cuDriverGetVersion");
        VASTLOOM_LOAD_CUDA(device_get_count_, CuDeviceGetCount, "cuDeviceGetCount");
        VASTLOOM_LOAD_CUDA(device_get_, CuDeviceGet, "cuDeviceGet");
        VASTLOOM_LOAD_CUDA(device_get_name_, CuDeviceGetName, "cuDeviceGetName");
        VASTLOOM_LOAD_CUDA(device_total_memory_, CuDeviceTotalMem, "cuDeviceTotalMem_v2");
        VASTLOOM_LOAD_CUDA(device_get_attribute_, CuDeviceGetAttribute,
                           "cuDeviceGetAttribute");
        VASTLOOM_LOAD_CUDA(primary_context_retain_, CuDevicePrimaryCtxRetain,
                           "cuDevicePrimaryCtxRetain");
        VASTLOOM_LOAD_CUDA(primary_context_release_, CuDevicePrimaryCtxRelease,
                           "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease");
        VASTLOOM_LOAD_CUDA(context_push_, CuCtxPushCurrent,
                           "cuCtxPushCurrent_v2", "cuCtxPushCurrent");
        VASTLOOM_LOAD_CUDA(context_pop_, CuCtxPopCurrent,
                           "cuCtxPopCurrent_v2", "cuCtxPopCurrent");
        VASTLOOM_LOAD_CUDA(memory_info_, CuMemGetInfo, "cuMemGetInfo_v2");
        VASTLOOM_LOAD_CUDA(memory_allocate_, CuMemAlloc, "cuMemAlloc_v2");
        VASTLOOM_LOAD_CUDA(memory_free_, CuMemFree, "cuMemFree_v2");
        VASTLOOM_LOAD_CUDA(copy_host_to_device_, CuMemcpyHostToDevice,
                           "cuMemcpyHtoD_v2");
        VASTLOOM_LOAD_CUDA(copy_device_to_host_, CuMemcpyDeviceToHost,
                           "cuMemcpyDtoH_v2");
        VASTLOOM_LOAD_CUDA(module_load_data_, CuModuleLoadData, "cuModuleLoadDataEx");
        VASTLOOM_LOAD_CUDA(module_get_function_, CuModuleGetFunction,
                           "cuModuleGetFunction");
        VASTLOOM_LOAD_CUDA(module_unload_, CuModuleUnload, "cuModuleUnload");
        VASTLOOM_LOAD_CUDA(launch_kernel_, CuLaunchKernel, "cuLaunchKernel");
        VASTLOOM_LOAD_CUDA(context_synchronize_, CuContextSynchronize,
                           "cuCtxSynchronize");
        VASTLOOM_LOAD_CUDA(error_name_, CuGetErrorName, "cuGetErrorName");
        VASTLOOM_LOAD_CUDA(error_string_, CuGetErrorString, "cuGetErrorString");
#undef VASTLOOM_LOAD_CUDA
        return {};
    }

    [[nodiscard]] Result<void> initialize_locked() {
        if (initialized_) return {};
        const CuResult status = init_(0U);
        if (status != kCudaSuccess) return cuda_error("cuInit", status);
        initialized_ = true;
        return {};
    }

    [[nodiscard]] Result<CuDevice> get_device_locked(int ordinal) const {
        if (ordinal < 0) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA device ordinal cannot be negative");
        }
        int count = 0;
        CuResult status = device_get_count_(&count);
        if (status != kCudaSuccess) return cuda_error("cuDeviceGetCount", status);
        if (ordinal >= count) {
            return make_error(ErrorCode::not_found, "CUDA device ordinal is unavailable",
                              std::to_string(ordinal));
        }
        CuDevice device = 0;
        status = device_get_(&device, ordinal);
        if (status != kCudaSuccess) return cuda_error("cuDeviceGet", status);
        return device;
    }

    [[nodiscard]] Result<int> attribute_locked(CuDevice device, int attribute,
                                               std::string_view label) const {
        int value = 0;
        const CuResult status = device_get_attribute_(&value, attribute, device);
        if (status != kCudaSuccess) {
            auto error = cuda_error("cuDeviceGetAttribute", status);
            error.context = std::string(label) + ": " + error.context;
            return error;
        }
        return value;
    }

    [[nodiscard]] Error cuda_error(std::string_view operation, CuResult status) const {
        const char* name = nullptr;
        const char* description = nullptr;
        static_cast<void>(error_name_(status, &name));
        static_cast<void>(error_string_(status, &description));
        std::string context = "status=" + std::to_string(status);
        if (name != nullptr) context += ", " + std::string(name);
        if (description != nullptr) context += ": " + std::string(description);
        return make_error(ErrorCode::io_error,
                          "CUDA driver call failed: " + std::string(operation),
                          std::move(context));
    }

    [[nodiscard]] static Result<void> validate_allocation_range(
        const CudaBackendAllocation& allocation,
        ByteCount offset,
        ByteCount bytes) {
        if (allocation.context == 0U || allocation.device_pointer == 0U ||
            allocation.bytes == 0U || bytes == 0U || offset > allocation.bytes ||
            bytes > allocation.bytes - offset ||
            offset > std::numeric_limits<CuDevicePointer>::max() -
                         allocation.device_pointer ||
            bytes > std::numeric_limits<CuDevicePointer>::max() -
                        allocation.device_pointer - offset) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA device buffer range is invalid");
        }
        return {};
    }

    [[nodiscard]] Result<void> pop_context_locked(CuContext expected) const {
        CuContext popped = nullptr;
        const CuResult status = context_pop_(&popped);
        if (status != kCudaSuccess) return cuda_error("cuCtxPopCurrent_v2", status);
        if (popped != expected) {
            return make_error(ErrorCode::internal_error,
                              "CUDA context stack returned a different context");
        }
        return {};
    }

    void pop_and_release_locked(CuDevice device) const noexcept {
        CuContext popped = nullptr;
        static_cast<void>(context_pop_(&popped));
        static_cast<void>(primary_context_release_(device));
    }

#if defined(_WIN32)
    HMODULE library_handle_{nullptr};
#elif defined(__linux__)
    void* library_handle_{nullptr};
#endif
    std::string library_name_;
    mutable std::mutex mutex_;
    CuModule cached_module_{nullptr};
    CuContext cached_module_context_{nullptr};
    CuDevice cached_module_device_{};
    bool cached_module_device_valid_{false};
    bool initialized_{false};
    CuInit init_{nullptr};
    CuDriverGetVersion driver_get_version_{nullptr};
    CuDeviceGetCount device_get_count_{nullptr};
    CuDeviceGet device_get_{nullptr};
    CuDeviceGetName device_get_name_{nullptr};
    CuDeviceTotalMem device_total_memory_{nullptr};
    CuDeviceGetAttribute device_get_attribute_{nullptr};
    CuDevicePrimaryCtxRetain primary_context_retain_{nullptr};
    CuDevicePrimaryCtxRelease primary_context_release_{nullptr};
    CuCtxPushCurrent context_push_{nullptr};
    CuCtxPopCurrent context_pop_{nullptr};
    CuMemGetInfo memory_info_{nullptr};
    CuMemAlloc memory_allocate_{nullptr};
    CuMemFree memory_free_{nullptr};
    CuMemcpyHostToDevice copy_host_to_device_{nullptr};
    CuMemcpyDeviceToHost copy_device_to_host_{nullptr};
    CuModuleLoadData module_load_data_{nullptr};
    CuModuleGetFunction module_get_function_{nullptr};
    CuModuleUnload module_unload_{nullptr};
    CuLaunchKernel launch_kernel_{nullptr};
    CuContextSynchronize context_synchronize_{nullptr};
    CuGetErrorName error_name_{nullptr};
    CuGetErrorString error_string_{nullptr};
};

}  // namespace

CudaAllocation::CudaAllocation(std::shared_ptr<CudaDriverBackend> backend,
                               CudaBackendAllocation allocation,
                               safety::Reservation reservation)
    : backend_(std::move(backend)),
      allocation_(allocation),
      reservation_(std::move(reservation)) {}

CudaAllocation::CudaAllocation(CudaAllocation&& other) noexcept
    : backend_(std::move(other.backend_)),
      allocation_(std::exchange(other.allocation_, {})),
      reservation_(std::move(other.reservation_)) {}

CudaAllocation& CudaAllocation::operator=(CudaAllocation&& other) noexcept {
    if (this != &other) {
        release_for_destruction();
        backend_ = std::move(other.backend_);
        allocation_ = std::exchange(other.allocation_, {});
        reservation_ = std::move(other.reservation_);
    }
    return *this;
}

CudaAllocation::~CudaAllocation() {
    release_for_destruction();
}

bool CudaAllocation::valid() const noexcept {
    return backend_ != nullptr && allocation_.context != 0U &&
           allocation_.device_pointer != 0U && allocation_.bytes != 0U &&
           reservation_.valid();
}

std::uint64_t CudaAllocation::device_pointer() const noexcept {
    return allocation_.device_pointer;
}

ByteCount CudaAllocation::bytes() const noexcept {
    return allocation_.bytes;
}

int CudaAllocation::device_ordinal() const noexcept {
    return allocation_.device_ordinal;
}

Result<void> CudaAllocation::release() {
    if (!valid()) return {};
    Result<void> released;
    try {
        released = backend_->release(allocation_);
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw while releasing an allocation", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw an unknown release exception");
    }
    if (!released) return released.error();
    allocation_ = {};
    reservation_.release();
    backend_.reset();
    return {};
}

void CudaAllocation::release_for_destruction() noexcept {
    if (!valid()) return;
    try {
        auto released = release();
        if (released) return;
    } catch (...) {
    }
    // The device allocation may still be live. Preserve fail-closed Governor accounting rather
    // than allowing a second allocation to consume bytes that the backend failed to free.
    reservation_.abandon_accounting_on_external_release_failure();
    allocation_ = {};
    quarantine_backend(std::move(backend_));
}

CudaDriver::CudaDriver(std::shared_ptr<CudaDriverBackend> backend)
    : backend_(std::move(backend)) {}

Result<CudaDriver> CudaDriver::load() {
    auto backend = DynamicCudaBackend::create();
    if (!backend) return backend.error();
    return CudaDriver(backend.take_value());
}

Result<CudaDriver> CudaDriver::from_backend_for_testing(
    std::shared_ptr<CudaDriverBackend> backend) {
    if (backend == nullptr) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA test backend cannot be null");
    }
    return CudaDriver(std::move(backend));
}

Result<CudaDriverProbe> CudaDriver::probe() const {
    if (backend_ == nullptr) {
        return make_error(ErrorCode::internal_error, "CUDA driver backend is unavailable");
    }
    try {
        return backend_->probe();
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw during capability probing", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw an unknown probe exception");
    }
}

Result<CudaAllocation> CudaDriver::allocate(
    int device_ordinal,
    ByteCount bytes,
    ByteCount minimum_free_vram_after,
    safety::ResourceGovernor& governor,
    std::string tag) const {
    if (backend_ == nullptr) {
        return make_error(ErrorCode::internal_error, "CUDA driver backend is unavailable");
    }
    if (device_ordinal < 0) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA device ordinal cannot be negative");
    }
    if (bytes == 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA allocation size cannot be zero", std::move(tag));
    }
    if (tag.empty() || tag.size() > kMaximumAllocationTagBytes) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA allocation tag must contain 1 to 1024 bytes");
    }
    auto reservation = governor.try_reserve(safety::ResourceKind::vram, bytes, tag);
    if (!reservation) return reservation.error();

    Result<CudaBackendAllocation> allocated = make_error(
        ErrorCode::internal_error, "CUDA backend allocation was not attempted");
    try {
        allocated = backend_->allocate(device_ordinal, bytes, minimum_free_vram_after);
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw while allocating VRAM", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw an unknown allocation exception");
    }
    if (!allocated) return allocated.error();
    const auto& handle = allocated.value();
    if (handle.context == 0U || handle.device_pointer == 0U || handle.bytes != bytes ||
        handle.device_ordinal != device_ordinal) {
        bool cleanup_succeeded = handle.context == 0U || handle.device_pointer == 0U;
        if (handle.context != 0U && handle.device_pointer != 0U) {
            try {
                cleanup_succeeded = static_cast<bool>(backend_->release(handle));
            } catch (...) {
                cleanup_succeeded = false;
            }
        }
        if (!cleanup_succeeded) {
            reservation.value().abandon_accounting_on_external_release_failure();
            quarantine_backend(backend_);
            return make_error(
                ErrorCode::integrity_error,
                "CUDA backend returned an invalid handle and cleanup failed; VRAM accounting remains charged");
        }
        return make_error(ErrorCode::integrity_error,
                          "CUDA backend returned an invalid allocation handle");
    }
    return CudaAllocation(backend_, allocated.take_value(), reservation.take_value());
}

Result<void> CudaDriver::copy_from_host(
    CudaAllocation& allocation,
    ByteCount destination_offset,
    std::span<const std::byte> source) const {
    if (!allocation.valid() || allocation.backend_.get() != backend_.get()) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA allocation does not belong to this driver");
    }
    if (source.empty() || destination_offset > allocation.bytes() ||
        source.size() > allocation.bytes() - destination_offset) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA host-to-device copy range is invalid");
    }
    try {
        return backend_->copy_from_host(allocation.allocation_, destination_offset, source);
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw during host-to-device copy", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw an unknown host-to-device copy exception");
    }
}

Result<void> CudaDriver::copy_to_host(
    CudaAllocation& allocation,
    ByteCount source_offset,
    std::span<std::byte> destination) const {
    if (!allocation.valid() || allocation.backend_.get() != backend_.get()) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA allocation does not belong to this driver");
    }
    if (destination.empty() || source_offset > allocation.bytes() ||
        destination.size() > allocation.bytes() - source_offset) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA device-to-host copy range is invalid");
    }
    try {
        return backend_->copy_to_host(allocation.allocation_, source_offset, destination);
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw during device-to-host copy", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw an unknown device-to-host copy exception");
    }
}

Result<void> CudaDriver::launch_rms_norm(
    CudaAllocation& allocation,
    ByteCount input_offset,
    ByteCount weight_offset,
    ByteCount output_offset,
    std::size_t row_count,
    std::size_t width,
    float epsilon) const {
    if (!allocation.valid() || allocation.backend_.get() != backend_.get()) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA allocation does not belong to this driver");
    }
    if (row_count == 0U || width == 0U || !std::isfinite(epsilon) || epsilon <= 0.0F) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA RMSNorm launch parameters are invalid");
    }
    if (row_count > std::numeric_limits<std::uint32_t>::max() ||
        width > std::numeric_limits<std::uint32_t>::max() ||
        width > std::numeric_limits<std::size_t>::max() / row_count ||
        width > std::numeric_limits<ByteCount>::max() / row_count ||
        row_count * width > std::numeric_limits<ByteCount>::max() / sizeof(float) ||
        width > std::numeric_limits<ByteCount>::max() / sizeof(float)) {
        return make_error(ErrorCode::resource_exhausted,
                          "CUDA RMSNorm launch byte count overflows");
    }
    const ByteCount matrix_bytes =
        static_cast<ByteCount>(row_count * width) * sizeof(float);
    const ByteCount weight_bytes = static_cast<ByteCount>(width) * sizeof(float);
    const auto valid_range = [&](ByteCount offset, ByteCount bytes) {
        return offset <= allocation.bytes() && bytes <= allocation.bytes() - offset;
    };
    if (!valid_range(input_offset, matrix_bytes) ||
        !valid_range(weight_offset, weight_bytes) ||
        !valid_range(output_offset, matrix_bytes) ||
        input_offset % alignof(float) != 0U ||
        weight_offset % alignof(float) != 0U ||
        output_offset % alignof(float) != 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA RMSNorm buffer ranges are invalid");
    }
    const auto overlaps = [](ByteCount first_offset, ByteCount first_bytes,
                             ByteCount second_offset, ByteCount second_bytes) {
        return first_offset < second_offset + second_bytes &&
               second_offset < first_offset + first_bytes;
    };
    if (overlaps(input_offset, matrix_bytes, weight_offset, weight_bytes) ||
        overlaps(input_offset, matrix_bytes, output_offset, matrix_bytes) ||
        overlaps(weight_offset, weight_bytes, output_offset, matrix_bytes)) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA RMSNorm input, weight and output ranges must not overlap");
    }
    try {
        return backend_->launch_rms_norm(allocation.allocation_, input_offset, weight_offset,
                                         output_offset, row_count, width, epsilon);
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw during RMSNorm launch", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw an unknown RMSNorm exception");
    }
}

Result<void> CudaDriver::matvec_iq1_s_with_transfer(
    CudaAllocation& allocation,
    ByteCount matrix_offset,
    ByteCount grid_offset,
    ByteCount input_offset,
    ByteCount output_offset,
    std::size_t row_count,
    std::size_t blocks_per_row,
    std::span<const std::byte> input,
    std::span<std::byte> output) const {
    if (!allocation.valid() || allocation.backend_.get() != backend_.get()) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA allocation does not belong to this driver");
    }
    if (row_count == 0U || blocks_per_row == 0U || input.empty() || output.empty()) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA IQ1_S matvec transfer parameters are invalid");
    }
    try {
        return backend_->matvec_iq1_s_with_transfer(allocation.allocation_, matrix_offset,
                                                    grid_offset, input_offset, output_offset,
                                                    row_count, blocks_per_row, input, output);
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw during IQ1_S matvec", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw an unknown IQ1_S matvec exception");
    }
}

Result<void> CudaDriver::matvec_iq3_xxs_with_transfer(
    CudaAllocation& allocation,
    ByteCount matrix_offset,
    ByteCount grid_offset,
    ByteCount input_offset,
    ByteCount output_offset,
    std::size_t row_count,
    std::size_t blocks_per_row,
    std::span<const std::byte> input,
    std::span<std::byte> output) const {
    if (!allocation.valid() || allocation.backend_.get() != backend_.get()) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA allocation does not belong to this driver");
    }
    if (row_count == 0U || blocks_per_row == 0U || input.empty() || output.empty()) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA IQ3_XXS matvec transfer parameters are invalid");
    }
    try {
        return backend_->matvec_iq3_xxs_with_transfer(allocation.allocation_, matrix_offset,
                                                      grid_offset, input_offset, output_offset,
                                                      row_count, blocks_per_row, input, output);
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw during IQ3_XXS matvec", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw an unknown IQ3_XXS matvec exception");
    }
}

Result<void> CudaDriver::matvec_dense_with_transfer(
    CudaAllocation& allocation,
    ByteCount matrix_offset,
    ByteCount input_offset,
    ByteCount output_offset,
    std::size_t row_count,
    std::size_t columns,
    ByteCount element_bytes,
    std::span<const std::byte> input,
    std::span<std::byte> output) const {
    if (!allocation.valid() || allocation.backend_.get() != backend_.get()) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA allocation does not belong to this driver");
    }
    if (row_count == 0U || columns == 0U || input.empty() || output.empty()) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA dense matvec transfer parameters are invalid");
    }
    try {
        return backend_->matvec_dense_with_transfer(allocation.allocation_, matrix_offset,
                                                    input_offset, output_offset, row_count,
                                                    columns, element_bytes, input, output);
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw during dense matvec", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw an unknown dense matvec exception");
    }
}

Result<void> CudaDriver::matvec_q8_0_with_transfer(
    CudaAllocation& allocation,
    ByteCount matrix_offset,
    ByteCount input_offset,
    ByteCount output_offset,
    std::size_t row_count,
    std::size_t blocks_per_row,
    std::span<const std::byte> input,
    std::span<std::byte> output) const {
    if (!allocation.valid() || allocation.backend_.get() != backend_.get()) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA allocation does not belong to this driver");
    }
    if (row_count == 0U || blocks_per_row == 0U || input.empty() || output.empty()) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA Q8_0 matvec transfer parameters are invalid");
    }
    try {
        return backend_->matvec_q8_0_with_transfer(allocation.allocation_, matrix_offset,
                                                   input_offset, output_offset, row_count,
                                                   blocks_per_row, input, output);
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw during Q8_0 matvec transfer", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw an unknown Q8_0 matvec transfer exception");
    }
}

Result<void> CudaDriver::launch_matvec_q8_0(
    CudaAllocation& allocation,
    ByteCount matrix_offset,
    ByteCount input_offset,
    ByteCount output_offset,
    std::size_t row_count,
    std::size_t blocks_per_row) const {
    if (!allocation.valid() || allocation.backend_.get() != backend_.get()) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA allocation does not belong to this driver");
    }
    if (row_count == 0U || blocks_per_row == 0U ||
        row_count > std::numeric_limits<std::uint32_t>::max() ||
        blocks_per_row > std::numeric_limits<std::uint32_t>::max()) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA Q8_0 matvec launch parameters are invalid");
    }
    try {
        return backend_->launch_matvec_q8_0(allocation.allocation_, matrix_offset, input_offset,
                                            output_offset, row_count, blocks_per_row);
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw during Q8_0 matvec launch", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw an unknown Q8_0 matvec exception");
    }
}

std::string cuda_driver_version_text(int encoded_version) {
    if (encoded_version <= 0) return "unknown";
    const int major = encoded_version / 1'000;
    const int minor = (encoded_version % 1'000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

std::string to_json(const CudaDriverProbe& probe) {
    std::ostringstream output;
    output << "{\"ok\":true,\"library\":\"" << util::json_escape(probe.library)
           << "\",\"driver_version\":" << probe.driver_version
           << ",\"driver_version_text\":\""
           << util::json_escape(cuda_driver_version_text(probe.driver_version))
           << "\",\"devices\":[";
    for (std::size_t index = 0U; index < probe.devices.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& device = probe.devices[index];
        output << "{\"ordinal\":" << device.ordinal << ",\"name\":\""
               << util::json_escape(device.name) << "\",\"total_vram_bytes\":"
               << device.total_vram_bytes << ",\"compute_capability_major\":"
               << device.compute_capability_major << ",\"compute_capability_minor\":"
               << device.compute_capability_minor << ",\"multiprocessor_count\":"
               << device.multiprocessor_count << ",\"maximum_threads_per_block\":"
               << device.maximum_threads_per_block << ",\"unified_addressing\":"
               << (device.unified_addressing ? "true" : "false") << '}';
    }
    output << "]}";
    return output.str();
}

std::string to_text(const CudaDriverProbe& probe) {
    std::ostringstream output;
    output << "Vastloom CUDA driver report\n"
           << "Library:        " << probe.library << '\n'
           << "Driver API:     " << cuda_driver_version_text(probe.driver_version) << '\n'
           << "Devices:        " << probe.devices.size() << '\n';
    for (const auto& device : probe.devices) {
        output << "GPU " << device.ordinal << ":          " << device.name << '\n'
               << "  Capability:   " << device.compute_capability_major << '.'
               << device.compute_capability_minor << '\n'
               << "  Total VRAM:   " << format_bytes(device.total_vram_bytes) << '\n'
               << "  SMs/threads:  " << device.multiprocessor_count << '/'
               << device.maximum_threads_per_block << '\n'
               << "  UVA:          " << (device.unified_addressing ? "yes" : "no") << '\n';
    }
    return output.str();
}

}  // namespace vastloom::cuda

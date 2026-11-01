#ifndef TQ1_0R_1024_COMMON_GLSL
#define TQ1_0R_1024_COMMON_GLSL

// Shared helpers of the TQ1_0R_1024 shaders. CPU reference: ggml-quants.c (ggml_tq1_0r_1024_*).

// Fixed sign vector S of the rotation, packed 32 per word (bit set = -1); same values as
// ggml_tq1_0r_1024_sign in ggml-quants.c. A switch on literals instead of a const array: glslang
// materializes a dynamically indexed const array as per-invocation scratch memory.
uint tq1_0r_1024_sign_word(const uint w) {
    switch (w) {
        case 0u: return 0x4ee9d1aau;
        case 1u: return 0xffb3e643u;
        case 2u: return 0x8444b23bu;
        case 3u: return 0x18df7f70u;
        case 4u: return 0xfcd787ddu;
        case 5u: return 0x24065ad5u;
        case 6u: return 0xff6bf71eu;
        case 7u: return 0x9f5b8522u;
        case 8u: return 0xc998dff5u;
        case 9u: return 0x6ee3d92du;
        case 10u: return 0x9c5cf507u;
        case 11u: return 0x15b1cef6u;
        case 12u: return 0x97a9a694u;
        case 13u: return 0x8eb6ac06u;
        case 14u: return 0x650fe151u;
        case 15u: return 0xca48d555u;
        case 16u: return 0x2309815du;
        case 17u: return 0x01721db6u;
        case 18u: return 0x90b14a21u;
        case 19u: return 0x00ccd1f9u;
        case 20u: return 0x85554b92u;
        case 21u: return 0xa7ad0a28u;
        case 22u: return 0x737b15c6u;
        case 23u: return 0x9196684fu;
        case 24u: return 0xbc3e4dd3u;
        case 25u: return 0xc4c90d73u;
        case 26u: return 0x980b0a9eu;
        case 27u: return 0x2171c76au;
        case 28u: return 0x4c4101b8u;
        case 29u: return 0x186554d3u;
        case 30u: return 0x7e2a4c8fu;
        default: return 0x7d28e321u; // w == 31
    }
}

float tq1_0r_1024_sign(const uint e) {
    return (((tq1_0r_1024_sign_word(e >> 5u) >> (e & 31u)) & 1u) != 0u) ? -1.0 : 1.0;
}

// Element k (0..1023) of a block: quarter q = k >> 8, packed like block_tq1_0. Returns the byte index
// (flat in qs[192] or qh[16]) and the base-3 digit t to extract.
void tq1_0r_1024_locate(const uint k, out uint byte_idx, out bool in_qh, out uint t) {
    const uint q = k >> 8;
    const uint e = k & 255u;
    if (e < 160u) {
        t = e / 32u;  byte_idx = q * 48u + (e % 32u);         in_qh = false;
    } else if (e < 240u) {
        const uint e2 = e - 160u;
        t = e2 / 16u; byte_idx = q * 48u + 32u + (e2 % 16u);  in_qh = false;
    } else {
        const uint e3 = e - 240u;
        t = e3 / 4u;  byte_idx = q * 4u + (e3 % 4u);          in_qh = true;
    }
}

// trit in {-1, 0, +1}: same integer formula as the CPU reference ((b * 3^t) & 255) * 3 >> 8
float tq1_0r_1024_trit(const uint qbyte, const uint t) {
    const uint pow3[5] = uint[5](1u, 3u, 9u, 27u, 81u);
    return float((((qbyte * pow3[t]) & 255u) * 3u) >> 8) - 1.0;
}

// Sylvester-Hadamard entry H[a, b] = (-1)^popcount(a & b) (unnormalized)
float tq1_0r_1024_hsign(const uint a, const uint b) {
    return ((bitCount(a & b) & 1) != 0) ? -1.0 : 1.0;
}

// Decode of the five base-3 digits of the four bytes of one 32-bit word, digits in {0, 1, 2}.
// With c_0 = b and c_(t+1) = 3 * c_t mod 256, c_t = b * 3^t mod 256 and digit_t = (3 * c_t) >> 8, i.e. the
// same integers as the reference formula. Bytes 0,2 and 1,3 are kept as two 16-bit fields of one word:
// 3 * 255 < 65536, so no carry crosses a field and the digit is bits 8..9 of its field.
vec4 tq1_swar_step(inout uint w02, inout uint w13) {
    const uint m02 = (w02 << 1) + w02;
    const uint m13 = (w13 << 1) + w13;
    w02 = m02 & 0x00FF00FFu;
    w13 = m13 & 0x00FF00FFu;
    return vec4(uvec4((m02 >> 8) & 255u, (m13 >> 8) & 255u, m02 >> 24, m13 >> 24));
}

void tq1_swar_digits(const uint w, out vec4 x0, out vec4 x1, out vec4 x2, out vec4 x3, out vec4 x4) {
    uint w02 = w & 0x00FF00FFu;
    uint w13 = (w >> 8) & 0x00FF00FFu;
    x0 = tq1_swar_step(w02, w13);
    x1 = tq1_swar_step(w02, w13);
    x2 = tq1_swar_step(w02, w13);
    x3 = tq1_swar_step(w02, w13);
    x4 = tq1_swar_step(w02, w13);
}

#endif // TQ1_0R_1024_COMMON_GLSL

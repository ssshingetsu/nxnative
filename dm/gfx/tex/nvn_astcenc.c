// See nvn_astcenc.h. A small real-time ASTC 4x4 LDR encoder for re-encoding decoded BC7 blocks on GPUs without BC.
// Only two block layouts are produced, both chosen so that no trit / quint integer sequence encoding is needed:
//   opaque blocks:   1 partition, CEM 8  (LDR RGB direct),  8-bit endpoints (6 values), 4x4 weights of 3 bits
//   blocks w/ alpha: 1 partition, CEM 12 (LDR RGBA direct), 8-bit endpoints (8 values), 4x4 weights of 2 bits
// Bit budget: 11 (block mode) + 2 (partitions) + 4 (CEM) = 17; endpoints start at bit 17 and the weights are stored
// bit-reversed from bit 127 down. Opaque: 48 + 48 = 96 bits, leaving 63 for colour, of which the 8-bit range uses 48
// (the largest range that fits, so the decoder infers 8 bits). Alpha: 32 weight bits, 79 left, 8 x 8 = 64 used.
// Endpoints: principal axis (power iteration on the covariance), projection extent, then one least-squares refinement
// given the weights. Endpoint order is fixed so that the RGB sum of endpoint 1 >= endpoint 0 (no blue contraction).
// Checked against ARM astcenc's decoder on Chocobo GP's BC7 textures: about 35 dB PSNR.
#include <math.h>
#include <string.h>
#include "nvn_astcenc.h"

static const int k_w3[8] = {0, 9, 18, 27, 37, 46, 55, 64}, k_w2[4] = {0, 21, 43, 64};

static void put_bits(uint8_t *b, int pos, int n, unsigned v) { for (int i = 0; i < n; i++) if (v >> i & 1) b[(pos + i) >> 3] |= (uint8_t)(1u << ((pos + i) & 7)); }
static int clamp255(float v) { int i = (int)lrintf(v); return i < 0 ? 0 : i > 255 ? 255 : i; }

void astc_enc4x4(const uint8_t *px, uint8_t out[16]) {
    int opaque = 1; for (int i = 0; i < 16; i++) if (px[i * 4 + 3] != 255) { opaque = 0; break; }
    const int nc = opaque ? 3 : 4, nl = opaque ? 8 : 4; const int *wt = opaque ? k_w3 : k_w2;
    float m[4] = {0}, x[16][4];
    for (int i = 0; i < 16; i++) for (int c = 0; c < 4; c++) { x[i][c] = px[i * 4 + c]; m[c] += x[i][c] / 16.0f; }
    float cv[4][4] = {{0}};
    for (int i = 0; i < 16; i++) for (int a = 0; a < nc; a++) for (int b = 0; b < nc; b++) cv[a][b] += (x[i][a] - m[a]) * (x[i][b] - m[b]);
    float ax[4] = {0}; { float lo[4] = {255, 255, 255, 255}, hi[4] = {0};   // start: bounding-box diagonal
        for (int i = 0; i < 16; i++) for (int c = 0; c < nc; c++) { if (x[i][c] < lo[c]) lo[c] = x[i][c]; if (x[i][c] > hi[c]) hi[c] = x[i][c]; }
        for (int c = 0; c < nc; c++) ax[c] = hi[c] - lo[c]; }
    for (int it = 0; it < 6; it++) {
        float n[4] = {0}, l = 0; for (int a = 0; a < nc; a++) for (int b = 0; b < nc; b++) n[a] += cv[a][b] * ax[b];
        for (int a = 0; a < nc; a++) l += n[a] * n[a];
        if (l < 1e-12f) break;
        l = 1.0f / sqrtf(l); for (int a = 0; a < nc; a++) ax[a] = n[a] * l;
    }
    { float l = 0; for (int c = 0; c < nc; c++) l += ax[c] * ax[c]; if (l < 1e-12f) { ax[0] = ax[1] = ax[2] = 0.57735f; ax[3] = 0; } else { l = 1.0f / sqrtf(l); for (int c = 0; c < nc; c++) ax[c] *= l; } }
    float tmin = 1e9f, tmax = -1e9f;
    for (int i = 0; i < 16; i++) { float t = 0; for (int c = 0; c < nc; c++) t += (x[i][c] - m[c]) * ax[c]; if (t < tmin) tmin = t; if (t > tmax) tmax = t; }
    float e[2][4]; for (int c = 0; c < 4; c++) { e[0][c] = m[c] + ax[c] * tmin; e[1][c] = m[c] + ax[c] * tmax; }
    if (opaque) e[0][3] = e[1][3] = 255;
    int q[16], ei[2][4];
    for (int pass = 0; pass < 2; pass++) {
        for (int k = 0; k < 2; k++) for (int c = 0; c < 4; c++) ei[k][c] = clamp255(e[k][c]);
        float d[4], dd = 0; for (int c = 0; c < nc; c++) { d[c] = (float)(ei[1][c] - ei[0][c]); dd += d[c] * d[c]; }
        for (int i = 0; i < 16; i++) {
            float t = 0; if (dd > 0) { for (int c = 0; c < nc; c++) t += (x[i][c] - ei[0][c]) * d[c]; t = t / dd * 64.0f; }
            int best = 0; float be = 1e9f; for (int k = 0; k < nl; k++) { float df = fabsf(t - (float)wt[k]); if (df < be) { be = df; best = k; } }
            q[i] = best;
        }
        if (pass) break;
        float s00 = 0, s01 = 0, s11 = 0, r0[4] = {0}, r1[4] = {0};   // least squares on the endpoints given the weights
        for (int i = 0; i < 16; i++) { float a = wt[q[i]] / 64.0f, b = 1 - a; s00 += b * b; s01 += a * b; s11 += a * a; for (int c = 0; c < nc; c++) { r0[c] += b * x[i][c]; r1[c] += a * x[i][c]; } }
        float det = s00 * s11 - s01 * s01;
        if (fabsf(det) > 1e-6f) for (int c = 0; c < nc; c++) { e[0][c] = (r0[c] * s11 - r1[c] * s01) / det; e[1][c] = (r1[c] * s00 - r0[c] * s01) / det; }
    }
    if (ei[1][0] + ei[1][1] + ei[1][2] < ei[0][0] + ei[0][1] + ei[0][2]) {   // keep sum(e1) >= sum(e0): swap and mirror the weights (tables are symmetric)
        for (int c = 0; c < 4; c++) { int t = ei[0][c]; ei[0][c] = ei[1][c]; ei[1][c] = t; }
        for (int i = 0; i < 16; i++) q[i] = nl - 1 - q[i];
    }
    memset(out, 0, 16);
    put_bits(out, 0, 11, opaque ? 0x53 : 0x42);   // block mode: 4x4 weight grid, 3-bit (R=7) / 2-bit (R=4) weights, single plane
    put_bits(out, 13, 4, opaque ? 8 : 12);        // bits 11-12 = 0: one partition; CEM
    for (int c = 0; c < nc; c++) { put_bits(out, 17 + c * 16, 8, (unsigned)ei[0][c]); put_bits(out, 17 + c * 16 + 8, 8, (unsigned)ei[1][c]); }
    const int nb = opaque ? 3 : 2;
    for (int i = 0; i < 16; i++) for (int j = 0; j < nb; j++) if (q[i] >> j & 1) { int pos = 127 - (i * nb + j); out[pos >> 3] |= (uint8_t)(1u << (pos & 7)); }
}

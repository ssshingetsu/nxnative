// GPUs without BC compressed format support (A15 / A16 etc., Apple GPU family Apple8 and below): BC textures are decompressed on the CPU into uncompressed formats.
// On a GPU without BC, BC7 is re-encoded to ASTC 4x4 here (nvn_bc_to_astc); the other BC formats are stored decompressed: BC1-5 are decoded on the GPU at upload (nvn_mtl.m, bcd), BC6H and 3D textures go through here.
#pragma once
#include <stddef.h>
#include <stdint.h>

// NVN format is BC → returns 1, *mtl_pf = Metal pixel format after decompression, *bpp = bytes per pixel after decompression; returns 0 if not BC
int nvn_bc_unpacked(uint32_t nvn_fmt, uint32_t *mtl_pf, int *bpp);
// Decode a w×h pixel region: src is laid out by block rows (pitch bytes per block row), dst has dpitch bytes per pixel row. Returns 0 on success
int nvn_bc_decode(uint32_t nvn_fmt, const uint8_t *src, size_t pitch, uint32_t w, uint32_t h, uint8_t *dst, size_t dpitch);
// BC7, the bulk of most games' textures, can instead be re-encoded to ASTC 4x4, which these GPUs sample natively at the same 16 bytes per 4x4 block, instead of
// 4x the size as RGBA8. NVN format is BC7 -> returns 1 and *mtl_pf = ASTC_4x4_LDR / ASTC_4x4_sRGB.
int nvn_bc_astc(uint32_t nvn_fmt, uint32_t *mtl_pf);
// Re-encode a w x h region of BC7 blocks (src: pitch bytes per block row) into ASTC 4x4 blocks (dst: dpitch bytes per block row; src == dst works in place).
// Rows of blocks are spread over the CPU cores. Returns 0 on success.
int nvn_bc_to_astc(uint32_t nvn_fmt, const uint8_t *src, size_t pitch, uint32_t w, uint32_t h, uint8_t *dst, size_t dpitch);

// Real-time ASTC 4x4 LDR encoder (nvn_astcenc.c): 16 RGBA8 texels (row-major) -> one 16-byte ASTC block.
#pragma once
#include <stdint.h>
void astc_enc4x4(const uint8_t *rgba64, uint8_t out[16]);

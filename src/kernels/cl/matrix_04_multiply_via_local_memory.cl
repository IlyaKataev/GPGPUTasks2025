#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define TILE_SIZE GROUP_SIZE_X

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);
    const uint global_x = get_global_id(0);
    const uint global_y = get_global_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0f;

    for (uint tile_k = 0; tile_k < k; tile_k += TILE_SIZE) {
        const uint a_col = tile_k + local_x;
        if (global_y < h && a_col < k) {
            tile_a[local_y][local_x] = a[global_y * k + a_col];
        } else {
            tile_a[local_y][local_x] = 0.0f;
        }

        const uint b_row = tile_k + local_y;
        if (b_row < k && global_x < w) {
            tile_b[local_y][local_x] = b[b_row * w + global_x];
        } else {
            tile_b[local_y][local_x] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint j = 0; j < TILE_SIZE; ++j) {
            sum += tile_a[local_y][j] * tile_b[j][local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_x < w && global_y < h) {
        c[global_y * w + global_x] = sum;
    }
}

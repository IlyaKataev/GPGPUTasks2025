#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

#define TILE_SIZE GROUP_SIZE_X

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);
    const uint group_x = get_group_id(0);
    const uint group_y = get_group_id(1);

    const uint read_x = group_x * TILE_SIZE + local_x;
    const uint read_y = group_y * TILE_SIZE + local_y;

    if (read_x < w && read_y < h) {
        tile[local_y][local_x] = matrix[read_y * w + read_x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint write_x = group_y * TILE_SIZE + local_x;
    const uint write_y = group_x * TILE_SIZE + local_y;

    if (write_x < h && write_y < w) {
        transposed_matrix[write_y * h + write_x] = tile[local_x][local_y];
    }
}

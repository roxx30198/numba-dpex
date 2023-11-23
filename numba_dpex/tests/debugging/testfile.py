import dpnp

import numba_dpex


@numba_dpex.kernel
def _pathfinder_kernel(prev, deviceWall, cols, cur_row, result):
    current_element = numba_dpex.get_global_id(0)
    left_ind = current_element - 1 if current_element >= 1 else current_element
    up_ind = current_element

    index = cur_row * cols + current_element
    left = prev[left_ind]
    up = prev[up_ind]

    shortest = left if left <= up else up

    numba_dpex.barrier(numba_dpex.GLOBAL_MEM_FENCE)
    prev[current_element] = deviceWall[index] + shortest
    numba_dpex.barrier(numba_dpex.GLOBAL_MEM_FENCE)
    result[current_element] = prev[current_element]


def pathfinder(data, cols, result):
    # create a temp list that hold first row of data as first element and empty numpy array as second element
    device_dest = dpnp.array(data[:cols], dtype=dpnp.int64)  # first row
    device_wall = dpnp.array(data[cols:], dtype=dpnp.int64)

    _pathfinder_kernel[numba_dpex.Range(cols)](
        device_dest, device_wall, cols, 0, result
    )


data = dpnp.array([3, 0, 7, 5, 6, 5, 4, 2], dtype=dpnp.int64)

res = dpnp.zeros(shape=(4), dtype=dpnp.int64)

pathfinder(data, 4, res)
print(res)

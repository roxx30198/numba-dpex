import dpctl.tensor as dpt
import numpy as np

import numba_dpex as dpex

dtype = np.float32


@dpex.kernel(debug=True)
def kernel(result):
    local_col_idx = dpex.get_local_id(0)
    local_values = dpex.local.array((1,), dtype=dtype)

    dpex.barrier(dpex.LOCAL_MEM_FENCE)

    a = local_col_idx + 12
    if local_col_idx < 1:
        local_values[0] = 1

    dpex.barrier(dpex.LOCAL_MEM_FENCE)
    b = local_col_idx + a
    if b > 1:
        result[0] = 10


result = dpt.zeros(shape=(1), dtype=dtype)
kernel[dpex.Range(4)](result)
print(result)

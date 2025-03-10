# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
from numba.core.extending import overload

import numba_dpex as dpex
import numba_dpex.experimental as dpex_exp
from numba_dpex.core.descriptor import dpex_kernel_target
from numba_dpex.experimental.target import (
    DPEX_KERNEL_EXP_TARGET_NAME,
    dpex_exp_kernel_target,
)


def scalar_add(a, b):
    return a + b


@overload(scalar_add, target=DPEX_KERNEL_EXP_TARGET_NAME)
def _ol_scalar_add(a, b):
    def ol_scalar_add_impl(a, b):
        return a + b

    return ol_scalar_add_impl


@dpex_exp.kernel
def kernel_calling_overload(a, b, c):
    i = dpex.get_global_id(0)
    c[i] = scalar_add(a[i], b[i])


a = dpnp.ones(10, dtype=dpnp.int64)
b = dpnp.ones(10, dtype=dpnp.int64)
c = dpnp.zeros(10, dtype=dpnp.int64)

dpex_exp.call_kernel(kernel_calling_overload, dpex.Range(10), a, b, c)


def test_end_to_end_overload_execution():
    """Tests that an overload function can be called from an experimental.kernel
    decorated function and works end to end.
    """
    for i in range(c.shape[0]):
        assert c[i] == scalar_add(a[i], b[i])


def test_overload_registration():
    """Tests that the overload _ol_scalar_add is registered only in the
    "dpex_kernel_exp" target and not in the "dpex_kernel" target.
    """

    def check_for_overload_registration(targetctx, key):
        found_key = False
        for fn_key in targetctx._defns.keys():
            if isinstance(fn_key, str) and fn_key.startswith(key):
                found_key = True
                break
        return found_key

    assert check_for_overload_registration(
        dpex_exp_kernel_target.target_context, "_ol_scalar_add"
    )
    assert not check_for_overload_registration(
        dpex_kernel_target.target_context, "_ol_scalar_add"
    )

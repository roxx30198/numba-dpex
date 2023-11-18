# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
from llvmlite.ir import IRBuilder
from numba import types
from numba.core import cgutils, imputils
from numba.core.datamodel import default_manager
from numba.extending import intrinsic, overload, overload_method, type_callable

import numba_dpex.dpctl_iface.libsyclinterface_bindings as sycl
from numba_dpex.core import types as dpex_types
from numba_dpex.core.runtime import context as dpexrt


@intrinsic
def sycl_event_create(
    ty_context,
):
    """A numba "intrinsic" function to inject dpctl.SyclEvent constructor code.

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """
    ty_event = dpex_types.DpctlSyclEvent()

    sig = ty_event(types.void)

    def codegen(context, builder: IRBuilder, sig, args: list):
        pyapi = context.get_python_api(builder)

        event_struct_proxy = cgutils.create_struct_proxy(ty_event)(
            context, builder
        )

        event = sycl.dpctl_event_create(builder)
        dpexrtCtx = dpexrt.DpexRTContext(context)

        # Ref count after the call is equal to 1.
        dpexrtCtx.eventstruct_init(
            pyapi, event, event_struct_proxy._getpointer()
        )

        event_value = event_struct_proxy._getvalue()

        return event_value

    return sig, codegen


@intrinsic
def sycl_event_wait(typingctx, ty_event: dpex_types.DpctlSyclEvent):
    sig = types.void(dpex_types.DpctlSyclEvent())

    # defines the custom code generation
    def codegen(context, builder, signature, args):
        sycl_event_dm = default_manager.lookup(ty_event)
        event_ref = builder.extract_value(
            args[0],
            sycl_event_dm.get_field_position("event_ref"),
        )

        sycl.dpctl_event_wait(builder, event_ref)

    return sig, codegen


@overload(dpctl.SyclEvent)
def ol_dpctl_sycl_event_create():
    """Implementation of an overload to support dpctl.SyclEvent() inside
    a dpjit function.
    """
    return lambda: sycl_event_create()


@overload_method(dpex_types.DpctlSyclEvent, "wait")
def ol_dpctl_sycl_event_wait(
    event,
):
    """Implementation of an overload to support dpctl.SyclEvent.wait() inside
    a dpjit function.
    """
    return lambda event: sycl_event_wait(event)


# We don't want user to call sycl_event_wait(event), instead it must be called
# with event.wait(). In that way we guarantee the argument type by the
# @overload_method.
__all__ = []

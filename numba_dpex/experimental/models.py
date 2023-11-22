# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Provides the Numba data models for the numba_dpex types introduced in the
numba_dpex.experimental module.
"""

from numba.core.datamodel import DataModelManager, models
from numba.core.extending import register_model

import numba_dpex.core.datamodel.models as dpex_core_models

from .types import KernelDispatcherType


def _init_exp_data_model_manager() -> DataModelManager:
    """Initializes a DpexExpKernelTarget-specific data model manager.

    Extends the DpexKernelTargetContext's datamodel manager with all
    experimental types that are getting added to the kernel API.

    Returns:
        DataModelManager: A numba-dpex DpexExpKernelTarget-specific data model
        manager
    """

    dmm = dpex_core_models.dpex_data_model_manager.copy()

    # Register the types and data model in the DpexExpTargetContext
    # Add here...

    return dmm


exp_dmm = _init_exp_data_model_manager()

# Register any new type that should go into numba.core.datamodel.default_manager
register_model(KernelDispatcherType)(models.OpaqueModel)

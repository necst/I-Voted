# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import sys

import aie.iron as iron
import aie.utils.test as test_utils


def main(opts):
    opts = test_utils.create_npu_kernel(opts)

    INOUT0_VOLUME = 4096
    INOUT1_VOLUME = 1
    INOUT2_VOLUME = 4096

    INOUT0_DATATYPE = np.int32
    INOUT1_DATATYPE = np.int32
    INOUT2_DATATYPE = np.int32

    # Create XRT-backed tensors on the NPU device
    inout0 = iron.tensor(
        np.arange(1, INOUT0_VOLUME + 1, dtype=INOUT0_DATATYPE),
        dtype=INOUT0_DATATYPE,
        device="npu",
    )
    scale_factor = iron.tensor(
        np.array([3], dtype=INOUT1_DATATYPE),
        dtype=INOUT1_DATATYPE,
        device="npu",
    )
    inout2 = iron.zeros(
        (INOUT2_VOLUME,),
        dtype=INOUT2_DATATYPE,
        device="npu",
    )

    if opts.verbosity >= 1:
        print("Running Kernel.")

    opts.npu_kernel(inout0, scale_factor, inout2)

    errors = 0
    if opts.verify:
        if opts.verbosity >= 1:
            print("Verifying results ...")
        ref = np.arange(1, INOUT0_VOLUME + 1, dtype=INOUT0_DATATYPE) * 3
        output_np = np.asarray(inout2)
        e = np.equal(output_np, ref)
        errors = np.size(e) - np.count_nonzero(e)

    if not errors:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        sys.exit(1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
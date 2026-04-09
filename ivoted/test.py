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

    INOUT0_VOLUME = 16384
    INOUT1_VOLUME = 16384
    OUT_VOLUME = 1

    INOUT0_DATATYPE = np.uint8
    INOUT1_DATATYPE = np.uint8
    OUT_DATATYPE = np.float32

    # Use a deterministic input where the expected MSE is exact and easy to verify.
    img0_np = np.full((INOUT0_VOLUME,), 10, dtype=INOUT0_DATATYPE)
    img1_np = np.full((INOUT1_VOLUME,), 13, dtype=INOUT1_DATATYPE)

    img0 = iron.tensor(img0_np, dtype=INOUT0_DATATYPE, device="npu")
    img1 = iron.tensor(img1_np, dtype=INOUT1_DATATYPE, device="npu")
    out = iron.zeros((OUT_VOLUME,), dtype=OUT_DATATYPE, device="npu")

    if opts.verbosity >= 1:
        print("Running Kernel.")

    opts.npu_kernel(img0, img1, out)

    errors = 0
    if opts.verify:
        if opts.verbosity >= 1:
            print("Verifying results ...")

        ref = np.array([9.0], dtype=OUT_DATATYPE)
        out_np = np.asarray(out)

        if not np.allclose(out_np, ref, rtol=1e-5, atol=1e-5):
            errors = 1
            print("Expected:", ref)
            print("Got     :", out_np)

    if not errors:
        print("\nPASS!\n")
        print(f"mse value is {out_np}")
        sys.exit(0)
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        sys.exit(1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
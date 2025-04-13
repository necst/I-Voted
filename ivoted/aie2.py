# section-3/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_


def my_mse(dev, in1_size, in2_size, out_size, trace_size):
    image_dtype = np.uint8
    out_dtype = np.float32
    tile_size = 1024

    tensor_size = in1_size // image_dtype(0).nbytes
    num_sub_vectors = tensor_size // tile_size

    assert in2_size == in1_size, "Different resolution images not supported yet"
    
    # Define tensor types
    image = np.ndarray[(tensor_size,), np.dtype[image_dtype]]


    image_tile  = np.ndarray[(tile_size,), np.dtype[image_dtype]]
    image_tile  = np.ndarray[(tile_size,), np.dtype[image_dtype]]
    output_val = np.ndarray[(1,), np.dtype[out_dtype]]
    # External, binary kernel definition
    my_fn = Kernel(
        "mse",
        "scale.o",
        [image_tile, image_tile, output_val],
    )

    # Input data movement
    of_image_flt = ObjectFifo(image_tile, name="float")
    of_image_ref = ObjectFifo(image_tile, name="ref")

    # Output data movement
    of_out = ObjectFifo(output_val, name="out")

    # Task for the core to perform

    def core_fn(of_image_flt, of_image_ref, of_out, my_function):
        out_val = of_out.acquire(1)
        for i in range_(num_sub_vectors):
            tile_flt_in = of_image_flt.acquire(1)
            tile_ref_in = of_image_ref.acquire(1)
            my_function(tile_flt_in, tile_ref_in, out_val)
            of_image_flt.release(1)
            of_image_ref.release(1)
        out_val[0] = out_val[0] / tensor_size
        of_out.release(1)
        
    # Create a worker to perform the task
    my_worker = Worker(
        core_fn, [of_image_flt.cons(), of_image_ref.cons(), of_out.prod(), my_fn]
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(image, image, output_val) as (flt_in, ref_in, mi_out):
        rt.enable_trace(trace_size, workers=[my_worker])
        rt.start(my_worker)
        rt.fill(of_image_flt.prod(), flt_in)
        rt.fill(of_image_ref.prod(), ref_in)
        rt.drain(of_out.cons(), mi_out, wait=True)

    # Create the program from the device type and runtime
    my_program = Program(dev, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    return my_program.resolve_program(SequentialPlacer())

# Check if there are at least 4 arguments (dev, in1_size, in2_size, out_size)
if len(sys.argv) < 5:
    raise ValueError(
        "[ERROR] Need at least 4 arguments (dev, in1_size, in2_size, out_size)"
    )

# Create an argument parser
p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument("-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size")
p.add_argument("-i2s", "--in2_size", required=True, dest="in2_size", help="Input 2 size")
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size")
p.add_argument(
    "-t",
    "--trace_size",
    required=False,
    dest="trace_size",
    default=0,
    help="Trace buffer size",
)
opts = p.parse_args(sys.argv[1:])

# Determine the device based on the provided argument
if opts.device == "npu":
    dev = NPU1Col1()
elif opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

# Validate the size of the first input
in1_size = int(opts.in1_size)
if in1_size % 128 != 0 or in1_size < 1024:
    print("In1 buffer size must be a multiple of 128 (so length is a multiple of 64) and greater than or equal to 1024 (so length >= 512)")
    raise ValueError

# Convert the remaining arguments to integers
in2_size = int(opts.in2_size)
out_size = int(opts.out_size)
trace_size = int(opts.trace_size)
module = my_mse(dev, in1_size, in2_size, out_size, trace_size)

# Print the generated MLIR module
print(module)
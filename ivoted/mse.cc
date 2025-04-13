//===- vector_scaler_mul.cc -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>

#define vector_size 128 // Size of the vector unit, used for reading from input stream
#define read_type uint8 // type of the input stream
#define func_type int // type used for doing the computations
#define write_type float // type of the output stream
#define SIZE 1024

inline aie::vector<func_type,vector_size> convert_to_func_type(aie::vector<read_type,vector_size> vec) {
    aie::accum<acc32, vector_size> acc;
    acc.from_vector(vec, 0);
    return acc.to_vector<func_type>();
}

extern "C" {

void mse(uint8_t* img1, uint8_t* img2, float* out) {
  event0();
    uint8_t *__restrict p_img1 = img1;
    uint8_t *__restrict p_img2 = img2;
    float *__restrict p_out = out;

    aie::vector<uint8_t,vector_size> vect1;
    aie::vector<uint8_t,vector_size> vect2;
    aie::vector<func_type,vector_size> diff;
    float partial_sum = 0;

    for(int i = 0; i< SIZE/vector_size; i++){
        vect1 = aie::load_v<vector_size>(p_img1);
        p_img1+=vector_size;
        vect2 = aie::load_v<vector_size>(p_img2);
        p_img2+=vector_size;
        diff = aie::sub(convert_to_func_type(vect1), convert_to_func_type(vect2));
        auto square = aie::mul(diff, diff).to_vector<int>();
        partial_sum += aie::reduce_add(square);
    }
    p_out[0] += partial_sum;
  
  event1();
}
} // extern "C"

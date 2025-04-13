//===- test_no_templates.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for further information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Main file containing the main function and routines for kernel execution,
// using the utilities defined in utils.hpp/ utils.cpp, without templates.
//===----------------------------------------------------------------------===//

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cassert>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <cfloat>
#include <cstdint>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

// Include XRT libraries
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Include the utilities header
#include "utils.hpp"

//--------------------------------------------------------------------------
// Buffer size macros (modify if necessary)
//--------------------------------------------------------------------------
#ifndef IN1_SIZE
#define IN1_SIZE 128*128     // Example: change as needed
#endif

#ifndef IN2_SIZE
#define IN2_SIZE 128*128     // For scalar multiplication, one element is sufficient
#endif

#ifndef OUT_SIZE
#define OUT_SIZE (1*sizeof(float))     // Example: change as needed
#endif

//--------------------------------------------------------------------------
// XRT Wrapper struct and functions (only those used)
//--------------------------------------------------------------------------

/**
 * @brief Structure to hold command-line arguments.
 *
 * This struct stores all the necessary parameters read from the command line.
 */
struct args {
  int verbosity;
  int do_verify;
  int n_iterations;
  int n_warmup_iterations;
  int trace_size;
  std::string instr;
  std::string xclbin;
  std::string kernel;
  std::string trace_file;
};

/**
 * @brief Parses the command-line arguments.
 *
 * This function defines the allowed options, parses the arguments using boost::program_options,
 * and fills in the args structure.
 *
 * @param argc The count of command-line arguments.
 * @param argv The array of command-line argument strings.
 * @return A populated args structure.
 */
args parse_args(int argc, const char *argv[]) {
  po::options_description desc("Allowed options");
  po::variables_map vm;
  utils::add_default_options(desc);
  args myargs;
  utils::parse_options(argc, argv, desc, vm);
  myargs.verbosity = vm["verbosity"].as<int>();
  myargs.do_verify = vm["verify"].as<bool>();
  myargs.n_iterations = vm["iters"].as<int>();
  myargs.n_warmup_iterations = vm["warmup"].as<int>();
  myargs.trace_size = vm["trace_sz"].as<int>();
  myargs.instr = vm["instr"].as<std::string>();
  myargs.xclbin = vm["xclbin"].as<std::string>();
  myargs.kernel = vm["kernel"].as<std::string>();
  myargs.trace_file = vm["trace_file"].as<std::string>();
  return myargs;
}

/**
 * @brief Initializes the input buffer.
 *
 * Fills bufIn with the given value.
 *
 * @param bufIn Pointer to the input buffer.
 * @param SIZE Number of elements in the buffer.
 * @param val The value to fill.
 */
void initialize_bufIn_random(std::uint8_t *bufIn, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    bufIn[i] = std::rand() % 256;
}

void initialize_bufIn(std::uint8_t *bufIn, int SIZE, std::uint8_t val) {
  for (int i = 0; i < SIZE; i++)
    bufIn[i] = val;
}


/**
 * @brief Initializes the output buffer.
 *
 * Clears the output buffer by setting all its bytes to zero.
 *
 * @param bufOut Pointer to the output buffer.
 * @param SIZE Number of elements in the buffer.
 */
void initialize_bufOut(std::float_t *bufOut, int SIZE) {
  // Assicurati di usare SIZE * sizeof(std::float_t) se inizializzi un buffer di float
  memset(bufOut, 0, SIZE * sizeof(std::float_t));
}

/**
 * @brief Verifies the Mean Squared Error (MSE) between two input buffers.
 *
 * Computes the MSE between the two buffers (assumed to be of type std::uint8_t)
 * and prints per-index differences if verbosity >= 1. In this esempio, la funzione
 * restituisce 0 se il MSE è quasi zero, oppure 1 se c'è una discrepanza.
 *
 * @param flt Pointer to the first input buffer.
 * @param ref Pointer to the second input buffer.
 * @param SIZE Number of elements in the buffers.
 * @param verbosity Verbosity level.
 * @return 0 if verification passes, 1 otherwise.
 */
double software_mse(const std::uint8_t *flt, const std::uint8_t *ref, int SIZE, int verbosity) {
    double sum = 0.0;
    for (int i = 0; i < SIZE; i++) {
        float diff = static_cast<double>(flt[i]) - static_cast<double>(ref[i]);
        double sq = diff * diff;
        sum += sq;
        if (verbosity >= 1) {
            std::cout << "Index " << i << ": (flt - ref)^2 = " << sq << std::endl;
        }
    }
    double mse = sum / (double)SIZE;
    if (verbosity >= 1) {
        std::cout << "Computed Mean Squared Error: " << mse << std::endl;
    }
    return mse;
}

/**
 * @brief XRT-based test wrapper for two inputs and one output.
 *
 * Sets up the device, buffers, and kernel; runs the kernel for the specified number
 * of iterations (including warmup iterations); and verifies the result.
 *
 * This version non-templated utilizza:
 *   - std::uint8_t per i buffer di input,
 *   - std::float_t per il buffer di output.
 *
 * @param IN1_VOLUME Volume (numero di elementi) del primo input.
 * @param IN2_VOLUME Volume (numero di elementi) del secondo input.
 * @param OUT_VOLUME Volume (numero di elementi) dell'output.
 * @param myargs Command-line arguments.
 * @return 0 se il test passa, 1 altrimenti.
 */
int setup_and_run_aie(int IN1_VOLUME, int IN2_VOLUME, int OUT_VOLUME, args myargs) {
  srand(time(NULL));
  std::vector<uint32_t> instr_v = utils::load_instr_binary(myargs.instr);
  if (myargs.verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Inizializza dispositivo e kernel con XRT
  xrt::device device;
  xrt::kernel kernel;
  utils::init_xrt_load_kernel(device, kernel, myargs.verbosity,
                              myargs.xclbin, myargs.kernel);

  // Crea buffer per istruzioni e dati
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in1 = xrt::bo(device, IN1_VOLUME * sizeof(std::uint8_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_in2 = xrt::bo(device, IN2_VOLUME * sizeof(std::uint8_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT_VOLUME * sizeof(std::float_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  auto bo_tmp1 = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  int tmp_trace_size = (myargs.trace_size > 0) ? myargs.trace_size : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  if (myargs.verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Carica le istruzioni nel buffer
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Mappa i buffer per i dati
  std::uint8_t *bufIn1 = bo_in1.map<std::uint8_t *>();
  std::uint8_t *bufIn2 = bo_in2.map<std::uint8_t *>();
  std::float_t *bufOut = bo_out.map<std::float_t *>();
  char *bufTrace = bo_trace.map<char *>();

  // Inizializza i buffer: per questo esempio, entrambi gli input vengono
  // inizializzati chiamando la stessa funzione con valori diversi.
  initialize_bufIn_random(bufIn1, IN1_VOLUME);
  initialize_bufIn_random(bufIn2, IN2_VOLUME);
  initialize_bufOut(bufOut, OUT_VOLUME);

  std::cout << std::endl;
  // Sincronizza i buffer verso il device
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  // Se si usa il tracing, sincronizza anche bo_trace (se necessario)
  // if (myargs.trace_size > 0)
  //   bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Esegui il kernel per il numero richiesto di iterazioni
  unsigned num_iter = myargs.n_iterations + myargs.n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;
  int ret_val = 0;

  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (myargs.verbosity >= 1)
      std::cout << "Running Kernel.\n";

    std::cout<< "Going to start the execution" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in1, bo_in2, bo_out, bo_tmp1, bo_trace);
    run.wait();
    std::cout<< "Execution finished" << std::endl;
    auto stop = std::chrono::high_resolution_clock::now();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (myargs.trace_size > 0)
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < myargs.n_warmup_iterations)
      continue;

    std::cout << "Verifying results ..." << std::endl;
    auto vstart = std::chrono::system_clock::now();
    double sw_mi = software_mse(bufIn1, bufIn2, IN1_VOLUME, myargs.verbosity);
    float hw_mi = static_cast<float>(bufOut[0]);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Software MSE: " << sw_mi << std::endl;
    std::cout << "Hardware MSE: " << hw_mi << std::endl;
    std::cout << "Error: " << std::abs(sw_mi - hw_mi) << std::endl;
    std::cout << std::defaultfloat;
    auto vstop = std::chrono::system_clock::now();
    float vtime = std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart).count();
    if (myargs.verbosity >= 1)
      std::cout << "Verify time: " << vtime << " secs." << std::endl;


    float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
    // compare software and hardware results
    if (myargs.do_verify) {
      if (std::abs(sw_mi - hw_mi) > 0.01) {
        std::cout << "Verification failed: MSE mismatch!" << std::endl;
        ret_val = 1;
      } else {
        std::cout << "Verification passed: MSE match!" << std::endl;
        ret_val = 0;
      }
    }
  }

  float macs = 0;
  std::cout << std::endl << "Avg NPU time: " << npu_time_total / myargs.n_iterations << " us." << std::endl;
  if (macs > 0)
    std::cout << "Avg NPU gflops: " << macs / (1000 * npu_time_total / myargs.n_iterations) << std::endl;
  std::cout << std::endl << "Min NPU time: " << npu_time_min << " us." << std::endl;
  if (macs > 0)
    std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min) << std::endl;
  std::cout << std::endl << "Max NPU time: " << npu_time_max << " us." << std::endl;
  if (macs > 0)
    std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max) << std::endl;

  std::cout << std::endl << "FINISHED - Cleaning up." << std::endl;

  return ret_val;
}

/**
 * @brief Main function.
 *
 * Parses command-line arguments, prints the parameters, and launches the kernel execution.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @return 0 on success, 1 on failure.
 */
int main(int argc, const char *argv[]) {
  constexpr int IN1_VOLUME = IN1_SIZE / sizeof(std::uint8_t);
  constexpr int IN2_VOLUME = IN2_SIZE / sizeof(std::uint8_t);
  constexpr int OUT_VOLUME = OUT_SIZE / sizeof(std::float_t);

  std::cout << "Going to run the kernel with the following parameters:\n";
  std::cout << "IN1_VOLUME: " << IN1_VOLUME << "\n";
  std::cout << "IN2_VOLUME: " << IN2_VOLUME << "\n";
  std::cout << "OUT_VOLUME: " << OUT_VOLUME << "\n";
  std::cout << "IN1_SIZE: " << IN1_SIZE << "\n";
  std::cout << "IN2_SIZE: " << IN2_SIZE << "\n";
  std::cout << "OUT_SIZE: " << OUT_SIZE << "\n";

  args myargs = parse_args(argc, argv);
  int res = setup_and_run_aie(IN1_VOLUME, IN2_VOLUME, OUT_VOLUME, myargs);
  return 0;
}

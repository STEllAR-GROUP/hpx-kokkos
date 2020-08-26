//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Benchmarks the creation and synchronization of an execution space-specific
/// future.

#include <Kokkos_Core.hpp>
#include <hpx/chrono.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/kokkos.hpp>

#if defined(HPX_HAVE_CUDA)
#include <hpx/modules/async_cuda.hpp>
#endif

void print_header() {
  std::cout << "test_name,execution_space,subtest_name,time" << std::endl;
}

template <typename ExecutionSpace>
void print_result(std::string const &label, ExecutionSpace const &inst,
                  double time) {
  std::cout << "future_overhead," << inst.name() << "," << label << "," << time
            << std::endl;
}

template <typename ExecutionSpace>
void test_future_get(ExecutionSpace const &inst) {
  hpx::util::high_resolution_timer timer;
  hpx::kokkos::get_future<>(inst).get();
  print_result("future_get", inst, timer.elapsed());
}

template <typename ExecutionSpace>
void test_future_then_sync(ExecutionSpace const &inst) {
  hpx::util::high_resolution_timer timer;
  auto f = hpx::kokkos::get_future<>(inst);
  f.then(hpx::launch::sync, [&](auto &&f) {
     print_result("future_then_sync", inst, timer.elapsed());
   }).get();
}

template <typename ExecutionSpace>
void test_future_then_async(ExecutionSpace const &inst) {
  hpx::util::high_resolution_timer timer;
  auto f = hpx::kokkos::get_future<>(inst);
  f.then(hpx::launch::async, [&](auto &&f) {
     print_result("future_then_async", inst, timer.elapsed());
   }).get();
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);

  {
#if defined(HPX_HAVE_CUDA)
    hpx::cuda::experimental::enable_user_polling p;
#endif
    print_header();
    for (int r = 1; r <= 100; ++r) {
      test_future_get(Kokkos::DefaultExecutionSpace());
      test_future_then_sync(Kokkos::DefaultExecutionSpace());
      test_future_then_async(Kokkos::DefaultExecutionSpace());
    }
  }

  Kokkos::finalize();

  return 0;
}
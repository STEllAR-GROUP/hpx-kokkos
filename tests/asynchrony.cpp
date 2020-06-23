//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Tests that parallel algorithms actually execute asynchronously. If a
/// parallel algorithm is asynchronous more time will be spent waiting for the
/// future than spawning the parallel algorithm. If the parallel algorithm does
/// not actually execution asynchronously more time will be spent spawning the
/// parallel algorithm.

#include "test.hpp"

#include <hpx/kokkos.hpp>

#include <boost/range/irange.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/util.hpp>

template <typename ExecutionSpace>
void test_kokkos_basic(ExecutionSpace &&inst) {
  int const n = 1000000;

  Kokkos::View<int *, typename std::decay<ExecutionSpace>::type> a("a", n);
  Kokkos::View<int *, typename std::decay<ExecutionSpace>::type> b("b", n);
  Kokkos::parallel_for(
      Kokkos::Experimental::require(
          Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(inst,
                                                                         0, n),
          Kokkos::Experimental::WorkItemProperty::HintLightWeight),
      KOKKOS_LAMBDA(int i) { a(i) = i; });
  inst.fence();

  hpx::util::high_resolution_timer timer;
  auto f = hpx::kokkos::parallel_for_async(
      Kokkos::Experimental::require(
          Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(inst,
                                                                         0, n),
          Kokkos::Experimental::WorkItemProperty::HintLightWeight),
      KOKKOS_LAMBDA(int i) { b(i) = a(i); });
  double t_spawn = timer.elapsed();
  timer.restart();
  f.get();
  double t_get = timer.elapsed();

  HPX_KOKKOS_DETAIL_TEST(t_spawn < t_get);
}

template <typename Executor> void test_hpx_basic(Executor &&exec) {
  int const n = 1000000;

  Kokkos::View<int *, typename std::decay<Executor>::type::execution_space> a(
      "a", n);
  Kokkos::View<int *, typename std::decay<Executor>::type::execution_space> b(
      "b", n);
  hpx::parallel::for_loop(
      hpx::parallel::execution::par.on(exec), 0, n,
      KOKKOS_LAMBDA(int i) { a(i) = i; });

  hpx::util::high_resolution_timer timer;
  auto f = hpx::parallel::for_loop(
      hpx::parallel::execution::par(hpx::parallel::execution::task).on(exec), 0,
      n, KOKKOS_LAMBDA(int i) { b(i) = a(i); });
  double t_spawn = timer.elapsed();
  timer.restart();
  f.get();
  double t_get = timer.elapsed() - t_spawn;

  HPX_KOKKOS_DETAIL_TEST(t_spawn < t_get);
}

template <typename ExecutionSpace> void test(ExecutionSpace &&inst) {
  test_hpx_basic(
      hpx::kokkos::executor<typename std::decay<ExecutionSpace>::type>(inst));
  test_kokkos_basic(inst);
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  test(hpx::kokkos::make_execution_space<Kokkos::DefaultExecutionSpace>());
  if (!std::is_same<Kokkos::DefaultExecutionSpace,
                    Kokkos::DefaultHostExecutionSpace>::value) {
    test(
        hpx::kokkos::make_execution_space<Kokkos::DefaultHostExecutionSpace>());
  }
  Kokkos::finalize();

  return hpx::kokkos::detail::report_errors();
}

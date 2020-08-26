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

#include <hpx/algorithm.hpp>
#include <hpx/chrono.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/kokkos.hpp>

#if defined(HPX_HAVE_CUDA)
#include <hpx/modules/async_cuda.hpp>
#endif

template <typename ExecutionSpace>
void test_kokkos_basic(ExecutionSpace &&inst, int const n,
                       int const repetitions) {
  for (int r = 0; r < repetitions; ++r) {
    Kokkos::View<int *, typename std::decay<ExecutionSpace>::type> a("a", n);
    Kokkos::View<int *, typename std::decay<ExecutionSpace>::type> b("b", n);
    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(
                inst, 0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int i) { a(i) = i; });
    inst.fence();

    hpx::util::high_resolution_timer timer;
    auto f = hpx::kokkos::parallel_for_async(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(
                inst, 0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int i) { b(i) = a(i); });
    double t_spawn = timer.elapsed();
    timer.restart();
    f.get();
    double t_get = timer.elapsed();
    std::cout << "kokkos_basic: t_spawn = " << t_spawn << ", t_get = " << t_get
              << std::endl;

    HPX_KOKKOS_DETAIL_TEST(t_spawn < t_get);
  }
}

template <typename Executor>
void test_hpx_basic(Executor &&exec, int const n, int const repetitions) {
  Kokkos::View<int *, typename std::decay<Executor>::type::execution_space> a(
      "a", n);

  for (int r = 0; r < repetitions; ++r) {
    hpx::util::high_resolution_timer timer;
    auto f = hpx::for_each(
        hpx::kokkos::kok(hpx::execution::task).on(exec), a.data(),
        a.data() + a.size(), KOKKOS_LAMBDA(int &x) { x = std::sqrt(x) / 3; });
    double t_spawn = timer.elapsed();
    timer.restart();
    f.get();
    double t_get = timer.elapsed();
    std::cout << "hpx_basic: t_spawn = " << t_spawn << ", t_get = " << t_get
              << std::endl;

    HPX_KOKKOS_DETAIL_TEST(t_spawn < t_get);
  }

  for (int r = 0; r < repetitions; ++r) {
    hpx::util::high_resolution_timer timer;
    auto f = hpx::reduce(
        hpx::kokkos::kok(hpx::execution::task).on(exec), a.data(),
        a.data() + a.size(), 0, KOKKOS_LAMBDA(int x, int y) { return x + y; });
    double t_spawn = timer.elapsed();
    timer.restart();
    f.get();
    double t_get = timer.elapsed();
    std::cout << "hpx_basic: t_spawn = " << t_spawn << ", t_get = " << t_get
              << std::endl;

    HPX_KOKKOS_DETAIL_TEST(t_spawn < t_get);
  }
}

void test_kokkos_basic_default(int const n, int const repetitions) {
  for (int r = 0; r < repetitions; ++r) {
    Kokkos::View<int *> a("a", n);
    Kokkos::View<int *> b("b", n);
    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int i) { a(i) = i; });
    Kokkos::fence();

    hpx::util::high_resolution_timer timer;
    auto f = hpx::kokkos::parallel_for_async(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int i) { b(i) = a(i); });
    double t_spawn = timer.elapsed();
    timer.restart();
    f.get();
    double t_get = timer.elapsed();
    std::cout << "kokkos_basic: t_spawn = " << t_spawn << ", t_get = " << t_get
              << std::endl;

    HPX_KOKKOS_DETAIL_TEST(t_spawn < t_get);
  }
}

template <typename ExecutionSpace>
void test(ExecutionSpace &&inst, int const n, int const repetitions) {
  std::cout << "testing execution space \"" << inst.name() << "\"" << std::endl;

  test_kokkos_basic(inst, n, repetitions);
  test_hpx_basic(
      hpx::kokkos::executor<typename std::decay<ExecutionSpace>::type>(inst), n,
      repetitions);
}

void test_default(int const n, int const repetitions) {
  std::cout << "testing default execution space \""
            << Kokkos::DefaultExecutionSpace().name() << "\"" << std::endl;

  test_kokkos_basic_default(n, repetitions);
}

int main(int argc, char *argv[]) {
  int const n = 1000000;
  int const repetitions = 10;

  Kokkos::initialize(argc, argv);

  {
#if defined(HPX_HAVE_CUDA)
    hpx::cuda::experimental::enable_user_polling p;
#endif

    test(Kokkos::DefaultExecutionSpace(), n, repetitions);
    if (!std::is_same<Kokkos::DefaultExecutionSpace,
                      Kokkos::DefaultHostExecutionSpace>::value) {
      test(Kokkos::DefaultHostExecutionSpace(), n, repetitions);
    }
    test_default(n, repetitions);
  }

  Kokkos::finalize();

  return hpx::kokkos::detail::report_errors();
}

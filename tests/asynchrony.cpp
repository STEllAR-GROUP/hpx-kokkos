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
#include <hpx/hpx_init.hpp>
#include <hpx/kokkos.hpp>
#include <hpx/kokkos/detail/polling_helper.hpp>

template <typename ExecutionSpace>
void test_kokkos_plain(ExecutionSpace &&inst, int const n,
                       int const repetitions) {
  for (int r = -1; r < repetitions; ++r) {
    Kokkos::View<int *, typename std::decay<ExecutionSpace>::type> a("a", n);
    Kokkos::View<int *, typename std::decay<ExecutionSpace>::type> b("b", n);
    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(
                inst, 0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int i) { a(i) = i; });
    inst.fence();

    hpx::chrono::high_resolution_timer timer;
    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(
                inst, 0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int i) { b(i) = a(i); });
    double t_spawn = timer.elapsed();
    timer.restart();
    Kokkos::fence();
    double t_get = timer.elapsed();
    std::cout << "kokkos_plain: t_spawn = " << t_spawn << ", t_get = " << t_get
              << std::endl;

    if (r >= 0) {
      HPX_KOKKOS_DETAIL_TEST(t_spawn < t_get);
    }
  }
}

template <typename ExecutionSpace>
void test_kokkos_async(ExecutionSpace &&inst, int const n,
                       int const repetitions) {
  for (int r = -1; r < repetitions; ++r) {
    Kokkos::View<int *, typename std::decay<ExecutionSpace>::type> a("a", n);
    Kokkos::View<int *, typename std::decay<ExecutionSpace>::type> b("b", n);
    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(
                inst, 0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int i) { a(i) = i; });
    inst.fence();

    hpx::chrono::high_resolution_timer timer;
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
    std::cout << "kokkos_async: t_spawn = " << t_spawn << ", t_get = " << t_get
              << std::endl;

    if (r >= 0) {
      HPX_KOKKOS_DETAIL_TEST(t_spawn < t_get);
    }
  }
}

template <typename Executor>
void test_hpx_async(Executor &&exec, int const n, int const repetitions) {
  Kokkos::View<int *, typename std::decay<Executor>::type::execution_space> a(
      "a", n);

  for (int r = -1; r < repetitions; ++r) {
    hpx::chrono::high_resolution_timer timer;
    auto f = hpx::for_each(
        hpx::kokkos::kok(hpx::execution::task).on(exec), a.data(),
        a.data() + a.size(), KOKKOS_LAMBDA(int &x) { x = std::sqrt(x) / 3; });
    double t_spawn = timer.elapsed();
    timer.restart();
    f.get();
    double t_get = timer.elapsed();
    std::cout << "hpx_async: t_spawn = " << t_spawn << ", t_get = " << t_get
              << std::endl;

    if (r >= 0) {
      HPX_KOKKOS_DETAIL_TEST(t_spawn < t_get);
    }
  }

  for (int r = -1; r < repetitions; ++r) {
    hpx::chrono::high_resolution_timer timer;
    auto f = hpx::reduce(
        hpx::kokkos::kok(hpx::execution::task).on(exec), a.data(),
        a.data() + a.size(), 0, KOKKOS_LAMBDA(int x, int y) { return x + y; });
    double t_spawn = timer.elapsed();
    timer.restart();
    f.get();
    double t_get = timer.elapsed();
    std::cout << "hpx_async: t_spawn = " << t_spawn << ", t_get = " << t_get
              << std::endl;

    if (r >= 0) {
      HPX_KOKKOS_DETAIL_TEST(t_spawn < t_get);
    }
  }
}

void test_kokkos_plain_default(int const n, int const repetitions) {
  for (int r = -1; r < repetitions; ++r) {
    Kokkos::View<int *> a("a", n);
    Kokkos::View<int *> b("b", n);
    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int i) { a(i) = i; });
    Kokkos::fence();

    hpx::chrono::high_resolution_timer timer;
    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int i) { b(i) = a(i); });
    double t_spawn = timer.elapsed();
    timer.restart();
    Kokkos::fence();
    double t_get = timer.elapsed();
    std::cout << "kokkos_plain: t_spawn = " << t_spawn << ", t_get = " << t_get
              << std::endl;

    if (r >= 0) {
      HPX_KOKKOS_DETAIL_TEST(t_spawn < t_get);
    }
  }
}

void test_kokkos_async_default(int const n, int const repetitions) {
  for (int r = -1; r < repetitions; ++r) {
    Kokkos::View<int *> a("a", n);
    Kokkos::View<int *> b("b", n);
    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int i) { a(i) = i; });
    Kokkos::fence();

    hpx::chrono::high_resolution_timer timer;
    auto f = hpx::kokkos::parallel_for_async(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int i) { b(i) = a(i); });
    double t_spawn = timer.elapsed();
    timer.restart();
    f.get();
    double t_get = timer.elapsed();
    std::cout << "kokkos_async: t_spawn = " << t_spawn << ", t_get = " << t_get
              << std::endl;

    if (r >= 0) {
      HPX_KOKKOS_DETAIL_TEST(t_spawn < t_get);
    }
  }
}

template <typename ExecutionSpace>
void test(ExecutionSpace &&inst, int const n, int const repetitions) {
  std::cout << "testing execution space \"" << inst.name() << "\"" << std::endl;

  test_kokkos_plain(inst, n, repetitions);
  test_kokkos_async(inst, n, repetitions);
  test_hpx_async(
      hpx::kokkos::executor<typename std::decay<ExecutionSpace>::type>(inst), n,
      repetitions);
}

void test_default(int const n, int const repetitions) {
  std::cout << "testing default execution space \""
            << Kokkos::DefaultExecutionSpace().name() << "\"" << std::endl;

  test_kokkos_plain_default(n, repetitions);
  test_kokkos_async_default(n, repetitions);
  test_hpx_async(hpx::kokkos::executor<>(), n, repetitions);
}

int test_main(int argc, char *argv[]) {
  int const n = 10000000;
  int const repetitions = 10;

  Kokkos::initialize(argc, argv);

  {
    hpx::kokkos::detail::polling_helper p;
    (void)p;

    test(Kokkos::DefaultExecutionSpace(), n, repetitions);
    if (!std::is_same<Kokkos::DefaultExecutionSpace,
                      Kokkos::DefaultHostExecutionSpace>::value) {
      test(Kokkos::DefaultHostExecutionSpace(), n, repetitions);
    }
    test_default(n, repetitions);
  }

  Kokkos::finalize();
  hpx::finalize();

  return hpx::kokkos::detail::report_errors();
}

int main(int argc, char *argv[]) {
  return hpx::init(test_main, argc, argv);
}

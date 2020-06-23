//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Tests HPX parallel algorithms using Kokkos executors.

#include "test.hpp"

#include <hpx/kokkos.hpp>

#include <boost/range/irange.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_transform.hpp>

template <typename Executor> void test_for_loop(Executor &&exec) {
  int const n = 43;

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> for_loop_result_host(
      "for_loop_result_host", n);
  Kokkos::View<int *, typename std::decay<Executor>::type::execution_space>
      for_loop_result("for_loop_result", n);
  for (std::size_t i = 0; i < n; ++i) {
    for_loop_result_host(i) = 0;
  }
  Kokkos::deep_copy(for_loop_result, for_loop_result_host);

  hpx::parallel::for_loop(
      hpx::parallel::execution::par.on(exec), 0, n,
      KOKKOS_LAMBDA(int i) { for_loop_result(i) = i; });
  Kokkos::deep_copy(for_loop_result_host, for_loop_result);
  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(for_loop_result_host(i) == i);
  }

  for (std::size_t i = 0; i < n; ++i) {
    for_loop_result_host(i) = 0;
  }
  Kokkos::deep_copy(for_loop_result, for_loop_result_host);

  auto f = hpx::parallel::for_loop(
      hpx::parallel::execution::par(hpx::parallel::execution::task).on(exec), 0,
      n, KOKKOS_LAMBDA(int i) { for_loop_result(i) = i; });
  f.get();
  Kokkos::deep_copy(for_loop_result_host, for_loop_result);
  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(for_loop_result_host(i) == i);
  }
}

template <typename Executor> void test_for_each(Executor &&exec) {
  int const n = 43;

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> for_each_index_host(
      "for_each_index_host", n);
  Kokkos::View<int *, typename std::decay<Executor>::type::execution_space>
      for_each_index("for_each_index", n);
  for (std::size_t i = 0; i < n; ++i) {
    for_each_index_host(i) = i;
  }
  Kokkos::deep_copy(for_each_index, for_each_index_host);

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> for_each_result_host(
      "for_each_result_host", n);
  Kokkos::View<int *, typename std::decay<Executor>::type::execution_space>
      for_each_result("for_each_result", n);
  for (std::size_t i = 0; i < n; ++i) {
    for_each_result_host(i) = 0;
  }
  Kokkos::deep_copy(for_each_result, for_each_result_host);

  hpx::parallel::for_each(
      hpx::parallel::execution::par.on(exec), for_each_index.data(),
      for_each_index.data() + for_each_result.size(),
      KOKKOS_LAMBDA(int i) { for_each_result(i) = i; });
  Kokkos::deep_copy(for_each_result_host, for_each_result);
  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(for_each_result_host(i) == i);
  }

  for (std::size_t i = 0; i < n; ++i) {
    for_each_result_host(i) = 0;
  }
  Kokkos::deep_copy(for_each_result, for_each_result_host);

  auto f = hpx::parallel::for_each(
      hpx::parallel::execution::par(hpx::parallel::execution::task).on(exec),
      for_each_index.data(), for_each_index.data() + for_each_result.size(),
      KOKKOS_LAMBDA(int i) { for_each_result(i) = i; });
  f.get();
  Kokkos::deep_copy(for_each_result_host, for_each_result);
  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(for_each_result_host(i) == i);
  }
}

template <typename Executor> void test_transform(Executor &&exec) {
  int const n = 43;

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> transform_result_host(
      "transform_result_host", n);
  Kokkos::View<int *, typename std::decay<Executor>::type::execution_space>
      transform_result("transform_result", n);
  for (std::size_t i = 0; i < n; ++i) {
    transform_result_host(i) = i;
  }
  Kokkos::deep_copy(transform_result, transform_result_host);

  hpx::parallel::transform(
      hpx::parallel::execution::par.on(exec), transform_result.data(),
      transform_result.data() + transform_result.size(),
      transform_result.data(), KOKKOS_LAMBDA(int i) { return i + 1; });
  Kokkos::deep_copy(transform_result_host, transform_result);
  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(transform_result_host(i) == (i + 1));
  }

  auto f = hpx::parallel::transform(
      hpx::parallel::execution::par(hpx::parallel::execution::task).on(exec),
      transform_result.data(),
      transform_result.data() + transform_result.size(),
      transform_result.data(), KOKKOS_LAMBDA(int i) { return i + 1; });
  f.get();
  Kokkos::deep_copy(transform_result_host, transform_result);
  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(transform_result_host(i) == (i + 2));
  }
}

template <typename Executor> void test(Executor &&exec) {
  static_assert(hpx::kokkos::is_kokkos_executor<Executor>::value,
                "Executor is not a Kokkos executor");
  static_assert(
      Kokkos::is_execution_space<typename Executor::execution_space>::value,
      "Executor::execution_space is not a Kokkos execution space");

  test_for_loop(exec);
  test_for_each(exec);
  test_transform(exec);
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  test(hpx::kokkos::default_executor{});
  if (!std::is_same<hpx::kokkos::default_executor,
                    hpx::kokkos::default_host_executor>::value) {
    test(hpx::kokkos::default_host_executor{});
  }
  Kokkos::finalize();

  return hpx::kokkos::detail::report_errors();
}

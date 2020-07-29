//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Tests using an iterator based on a Kokkos view.

#include "test.hpp"

#include <hpx/kokkos.hpp>

#include <hpx/algorithm.hpp>
#include <hpx/hpx_main.hpp>

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

  auto b = hpx::kokkos::view_begin(for_each_index);
  auto e = hpx::kokkos::view_end(for_each_index);
  hpx::for_each(
      hpx::kokkos::kokkos_policy{}.on(exec), b, e,
      KOKKOS_LAMBDA(int i) { for_each_result(i) = i; });
  Kokkos::deep_copy(for_each_result_host, for_each_result);
  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(for_each_result_host(i) == i);
  }

  for (std::size_t i = 0; i < n; ++i) {
    for_each_result_host(i) = 0;
  }
  Kokkos::deep_copy(for_each_result, for_each_result_host);

  auto f = hpx::for_each(
      hpx::kokkos::kokkos_policy{}(hpx::parallel::execution::task).on(exec),
      for_each_index.data(), for_each_index.data() + for_each_result.size(),
      KOKKOS_LAMBDA(int i) { for_each_result(i) = i; });
  f.get();
  Kokkos::deep_copy(for_each_result_host, for_each_result);
  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(for_each_result_host(i) == i);
  }
}

template <typename Executor> void test(Executor &&exec) { test_for_each(exec); }

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

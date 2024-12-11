//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Tests basic functionality of Kokkos executors.

#include "test.hpp"

#include <hpx/execution.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/kokkos.hpp>
#include <hpx/kokkos/detail/polling_helper.hpp>

#include <atomic>
#include <cassert>
#include <vector>

template <typename Executor> void test(Executor &&exec) {
  // Check static properties
  static_assert(hpx::kokkos::is_kokkos_executor<Executor>::value,
                "Executor is not a Kokkos executor");
  static_assert(
      Kokkos::is_execution_space<typename Executor::execution_space>::value,
      "Executor::execution_space is not a Kokkos execution space");

  // Check single execution
  Kokkos::View<std::size_t, Kokkos::DefaultHostExecutionSpace>
      executed_count_host("executed_count_host");
  Kokkos::View<std::size_t, typename Executor::execution_space> executed_count(
      "executed_count");
  executed_count_host() = 0;
  Kokkos::deep_copy(executed_count, executed_count_host);

  auto f_single = KOKKOS_LAMBDA() { executed_count() += 1; };

  hpx::parallel::execution::post(exec, f_single);
  hpx::parallel::execution::sync_execute(exec, f_single);
  hpx::parallel::execution::async_execute(exec, f_single).get();

  Kokkos::deep_copy(executed_count_host, executed_count);

  HPX_KOKKOS_DETAIL_TEST(executed_count_host() == 3);

  // Check bulk execution; all indices should be handled
  std::size_t const n = 43;
  Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace> index_handled_host(
      "index_handled_host", n);
  Kokkos::View<bool *, typename Executor::execution_space> index_handled(
      "index_handled", n);
  for (std::size_t i = 0; i < n; ++i) {
    index_handled_host(i) = false;
  }
  Kokkos::deep_copy(index_handled, index_handled_host);

  hpx::wait_all(hpx::parallel::execution::bulk_async_execute(
      exec, KOKKOS_LAMBDA(std::size_t i) { index_handled(i) = true; }, n));
  Kokkos::deep_copy(index_handled_host, index_handled);
  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(index_handled_host(i));
  }

  // Check bulk execution; arguments should be passed through
  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
      argument_passthrough_host("argument_passthrough_host", n);
  Kokkos::View<int *, typename Executor::execution_space> argument_passthrough(
      "argument_passthrough", n);
  for (std::size_t i = 0; i < n; ++i) {
    argument_passthrough_host(i) = 0;
  }
  Kokkos::deep_copy(argument_passthrough, argument_passthrough_host);
  hpx::wait_all(hpx::parallel::execution::bulk_async_execute(
      exec,
      KOKKOS_LAMBDA(std::size_t i, int passthrough) {
        argument_passthrough(i) = passthrough;
      },
      n, 42));
  Kokkos::deep_copy(argument_passthrough_host, argument_passthrough);
  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(argument_passthrough_host(i) == 42);
  }
}

int hpx_main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);

  {
    hpx::kokkos::detail::polling_helper p;
    (void)p;

    test(hpx::kokkos::default_executor{});
    if (!std::is_same<hpx::kokkos::default_executor,
                      hpx::kokkos::default_host_executor>::value) {
      test(hpx::kokkos::default_host_executor{});
    }
  }

  Kokkos::finalize();
  hpx::finalize();

  return hpx::kokkos::detail::report_errors();
}

int main(int argc, char *argv[]) {
  return hpx::init(hpx_main, argc, argv);
}

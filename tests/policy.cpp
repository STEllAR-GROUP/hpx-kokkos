//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Tests construction of Kokkos execution policies.

#include "test.hpp"

#include <hpx/kokkos.hpp>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);

  auto p1 = hpx::kokkos::kok;
  static_assert(std::is_same<std::decay<decltype(p1.executor())>::type,
                             hpx::kokkos::default_executor>::value,
                "kok executor should be default_executor");
  static_assert(std::is_same<std::decay<decltype(p1)>::type::executor_type,
                             hpx::kokkos::default_executor>::value,
                "kok executor_type should be default_executor");
  static_assert(
      hpx::detail::is_execution_policy<std::decay<decltype(p1)>::type>::value,
      "kok should be an execution policy");
  static_assert(hpx::detail::is_parallel_execution_policy<
                    std::decay<decltype(p1)>::type>::value,
                "kok should be a parallel execution policy");
  static_assert(!hpx::detail::is_async_execution_policy<
                    std::decay<decltype(p1)>::type>::value,
                "kok should be an async execution policy");

  auto p2 = hpx::kokkos::kok(hpx::execution::task);
  static_assert(std::is_same<std::decay<decltype(p2.executor())>::type,
                             hpx::kokkos::default_executor>::value,
                "kok(task) executor should be default_executor");
  static_assert(std::is_same<std::decay<decltype(p2)>::type::executor_type,
                             hpx::kokkos::default_executor>::value,
                "kok(task) executor_type should be default_executor");
  static_assert(
      hpx::detail::is_execution_policy<std::decay<decltype(p2)>::type>::value,
      "kok(task) should be an execution policy");
  static_assert(hpx::detail::is_parallel_execution_policy<
                    std::decay<decltype(p2)>::type>::value,
                "kok(task) should be a parallel execution policy");
  static_assert(hpx::detail::is_async_execution_policy<
                    std::decay<decltype(p2)>::type>::value,
                "kok(task) should be an async execution policy");

  auto p3 = hpx::kokkos::kok.on(hpx::kokkos::default_executor{});
  static_assert(std::is_same<std::decay<decltype(p3.executor())>::type,
                             hpx::kokkos::default_executor>::value,
                "kok.on(default_executor) executor should be default_executor");
  static_assert(
      std::is_same<std::decay<decltype(p3)>::type::executor_type,
                   hpx::kokkos::default_executor>::value,
      "kok.on(default_executor) executor_type should be default_executor");
  static_assert(
      hpx::detail::is_execution_policy<std::decay<decltype(p3)>::type>::value,
      "kok.on(default_executor) should be an execution policy");
  static_assert(
      hpx::detail::is_parallel_execution_policy<
          std::decay<decltype(p3)>::type>::value,
      "kok.on(default_executor) should be a parallel execution policy");
  static_assert(!hpx::detail::is_async_execution_policy<
                    std::decay<decltype(p3)>::type>::value,
                "kok.on(default_executor) should be an async execution policy");

  auto p4 = hpx::kokkos::kok(hpx::execution::task)
                .on(hpx::kokkos::default_executor{});
  static_assert(
      std::is_same<std::decay<decltype(p4.executor())>::type,
                   hpx::kokkos::default_executor>::value,
      "kok(task).on(default_executor) executor should be default_executor");
  static_assert(std::is_same<std::decay<decltype(p4)>::type::executor_type,
                             hpx::kokkos::default_executor>::value,
                "kok(task).on(default_executor) executor_type should be "
                "default_executor");
  static_assert(
      hpx::detail::is_execution_policy<std::decay<decltype(p4)>::type>::value,
      "kok(task).on(default_executor) should be an execution policy");
  static_assert(
      hpx::detail::is_parallel_execution_policy<
          std::decay<decltype(p4)>::type>::value,
      "kok(task).on(default_executor) should be a parallel execution policy");
  static_assert(
      hpx::detail::is_async_execution_policy<
          std::decay<decltype(p4)>::type>::value,
      "kok(task).on(default_executor) should be an async execution policy");

  auto p5 = hpx::kokkos::kok.on(hpx::kokkos::default_host_executor{});
  static_assert(
      std::is_same<std::decay<decltype(p4.executor())>::type,
                   hpx::kokkos::default_executor>::value,
      "kok.on(default_host_executor) executor should be default_host_executor");
  static_assert(
      std::is_same<std::decay<decltype(p5.executor())>::type,
                   hpx::kokkos::default_host_executor>::value,
      "kok.on(default_host_executor) executor should be default_host_executor");
  static_assert(std::is_same<std::decay<decltype(p5)>::type::executor_type,
                             hpx::kokkos::default_host_executor>::value,
                "kok.on(default_host_executor) executor_type should be "
                "default_host_executor");
  static_assert(
      hpx::detail::is_execution_policy<std::decay<decltype(p5)>::type>::value,
      "kok.on(default_host_executor) should be an execution policy");
  static_assert(
      hpx::detail::is_parallel_execution_policy<
          std::decay<decltype(p5)>::type>::value,
      "kok.on(default_host_executor) should be a parallel execution policy");
  static_assert(
      !hpx::detail::is_async_execution_policy<
          std::decay<decltype(p5)>::type>::value,
      "kok.on(default_host_executor) should be an async execution policy");

  auto p6 = hpx::kokkos::kok(hpx::execution::task)
                .on(hpx::kokkos::default_host_executor{});
  static_assert(std::is_same<std::decay<decltype(p6.executor())>::type,
                             hpx::kokkos::default_host_executor>::value,
                "kok(task).on(default_host_executor) executor should be "
                "default_host_executor");
  static_assert(std::is_same<std::decay<decltype(p6)>::type::executor_type,
                             hpx::kokkos::default_host_executor>::value,
                "kok(task).on(default_host_executor) executor_type should be "
                "default_host_executor");
  static_assert(
      hpx::detail::is_execution_policy<std::decay<decltype(p6)>::type>::value,
      "kok(task).on(default_host_executor) should be an execution policy");
  static_assert(hpx::detail::is_parallel_execution_policy<
                    std::decay<decltype(p6)>::type>::value,
                "kok(task).on(default_host_executor) should be a parallel "
                "execution policy");
  static_assert(hpx::detail::is_async_execution_policy<
                    std::decay<decltype(p6)>::type>::value,
                "kok(task).on(default_host_executor) should be an async "
                "execution policy");

  Kokkos::finalize();

  HPX_KOKKOS_DETAIL_TEST(true);

  return hpx::kokkos::detail::report_errors();
}

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
                             hpx::kokkos::default_executor>::value);
  static_assert(std::is_same<std::decay<decltype(p1)>::type::executor_type,
                             hpx::kokkos::default_executor>::value);
  static_assert(hpx::parallel::execution::detail::is_execution_policy<
                std::decay<decltype(p1)>::type>::value);
  static_assert(hpx::parallel::execution::detail::is_parallel_execution_policy<
                std::decay<decltype(p1)>::type>::value);
  static_assert(!hpx::parallel::execution::detail::is_async_execution_policy<
                std::decay<decltype(p1)>::type>::value);

  auto p2 = hpx::kokkos::kok(hpx::execution::task);
  static_assert(std::is_same<std::decay<decltype(p2.executor())>::type,
                             hpx::kokkos::default_executor>::value);
  static_assert(std::is_same<std::decay<decltype(p2)>::type::executor_type,
                             hpx::kokkos::default_executor>::value);
  static_assert(hpx::parallel::execution::detail::is_execution_policy<
                std::decay<decltype(p2)>::type>::value);
  static_assert(hpx::parallel::execution::detail::is_parallel_execution_policy<
                std::decay<decltype(p2)>::type>::value);
  static_assert(hpx::parallel::execution::detail::is_async_execution_policy<
                std::decay<decltype(p2)>::type>::value);

  auto p3 = hpx::kokkos::kok.on(hpx::kokkos::default_executor{});
  static_assert(std::is_same<std::decay<decltype(p3.executor())>::type,
                             hpx::kokkos::default_executor>::value);
  static_assert(std::is_same<std::decay<decltype(p3)>::type::executor_type,
                             hpx::kokkos::default_executor>::value);
  static_assert(hpx::parallel::execution::detail::is_execution_policy<
                std::decay<decltype(p3)>::type>::value);
  static_assert(hpx::parallel::execution::detail::is_parallel_execution_policy<
                std::decay<decltype(p3)>::type>::value);
  static_assert(!hpx::parallel::execution::detail::is_async_execution_policy<
                std::decay<decltype(p3)>::type>::value);

  auto p4 = hpx::kokkos::kok(hpx::execution::task)
                .on(hpx::kokkos::default_executor{});
  static_assert(std::is_same<std::decay<decltype(p4.executor())>::type,
                             hpx::kokkos::default_executor>::value);
  static_assert(std::is_same<std::decay<decltype(p4)>::type::executor_type,
                             hpx::kokkos::default_executor>::value);
  static_assert(hpx::parallel::execution::detail::is_execution_policy<
                std::decay<decltype(p4)>::type>::value);
  static_assert(hpx::parallel::execution::detail::is_parallel_execution_policy<
                std::decay<decltype(p4)>::type>::value);
  static_assert(hpx::parallel::execution::detail::is_async_execution_policy<
                std::decay<decltype(p4)>::type>::value);

  auto p5 = hpx::kokkos::kok.on(hpx::kokkos::default_host_executor{});
  static_assert(std::is_same<std::decay<decltype(p5.executor())>::type,
                             hpx::kokkos::default_host_executor>::value);
  static_assert(std::is_same<std::decay<decltype(p5)>::type::executor_type,
                             hpx::kokkos::default_host_executor>::value);
  static_assert(hpx::parallel::execution::detail::is_execution_policy<
                std::decay<decltype(p5)>::type>::value);
  static_assert(hpx::parallel::execution::detail::is_parallel_execution_policy<
                std::decay<decltype(p5)>::type>::value);
  static_assert(!hpx::parallel::execution::detail::is_async_execution_policy<
                std::decay<decltype(p5)>::type>::value);

  auto p6 = hpx::kokkos::kok(hpx::execution::task)
                .on(hpx::kokkos::default_host_executor{});
  static_assert(std::is_same<std::decay<decltype(p6.executor())>::type,
                             hpx::kokkos::default_host_executor>::value);
  static_assert(std::is_same<std::decay<decltype(p6)>::type::executor_type,
                             hpx::kokkos::default_host_executor>::value);
  static_assert(hpx::parallel::execution::detail::is_execution_policy<
                std::decay<decltype(p6)>::type>::value);
  static_assert(hpx::parallel::execution::detail::is_parallel_execution_policy<
                std::decay<decltype(p6)>::type>::value);
  static_assert(hpx::parallel::execution::detail::is_async_execution_policy<
                std::decay<decltype(p6)>::type>::value);

  Kokkos::finalize();

  HPX_KOKKOS_DETAIL_TEST(true);

  return hpx::kokkos::detail::report_errors();
}

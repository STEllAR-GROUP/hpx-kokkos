//  Copyright (c) 2019-2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file Contains wrappers for Kokkos parallel algorithms that return futures.

#pragma once

#include <hpx/kokkos/detail/logging.hpp>
#include <hpx/kokkos/future.hpp>

#include <hpx/local/future.hpp>

#include <Kokkos_Core.hpp>

namespace hpx {
namespace kokkos {
// Asynchronous versions of Kokkos algorithms
template <typename ExecutionPolicy, typename... Args,
          typename Enable = typename std::enable_if<Kokkos::is_execution_policy<
              typename std::decay<ExecutionPolicy>::type>::value>::type>
hpx::shared_future<void> parallel_for_async(ExecutionPolicy &&policy,
                                            Args &&...args) {
  HPX_KOKKOS_DETAIL_LOG("calling parallel_for_async with execution policy");
  Kokkos::parallel_for(policy, std::forward<Args>(args)...);
  return detail::get_future<typename std::decay<decltype(
      policy.space())>::type>::call(policy.space());
}

template <typename... Args>
hpx::shared_future<void> parallel_for_async(std::size_t const work_count,
                                            Args &&...args) {
  HPX_KOKKOS_DETAIL_LOG("calling parallel_for_async without execution policy");
  Kokkos::parallel_for(work_count, std::forward<Args>(args)...);
  return detail::get_future<Kokkos::DefaultExecutionSpace>::call(
      Kokkos::DefaultExecutionSpace{});
}

template <typename ExecutionPolicy, typename... Args>
hpx::shared_future<void> parallel_for_async(std::string const &label,
                                            ExecutionPolicy &&policy,
                                            Args &&...args) {
  HPX_KOKKOS_DETAIL_LOG(
      "calling parallel_for_async with label and execution policy");
  Kokkos::parallel_for(label, policy, std::forward<Args>(args)...);
  return detail::get_future<typename std::decay<decltype(
      policy.space())>::type>::call(policy.space());
}

template <typename ExecutionPolicy, typename... Args,
          typename Enable = typename std::enable_if<
              Kokkos::is_execution_policy<ExecutionPolicy>::value>::type>
hpx::shared_future<void> parallel_reduce_async(ExecutionPolicy &&policy,
                                               Args &&...args) {
  HPX_KOKKOS_DETAIL_LOG("calling parallel_reduce_async with execution policy");
  Kokkos::parallel_reduce(policy, std::forward<Args>(args)...);
  return detail::get_future<typename std::decay<decltype(
      policy.space())>::type>::call(policy.space());
}

template <typename... Args>
hpx::shared_future<void> parallel_reduce_async(std::size_t const work_count,
                                               Args &&...args) {
  HPX_KOKKOS_DETAIL_LOG(
      "calling parallel_reduce_async without execution policy");
  Kokkos::parallel_reduce(work_count, std::forward<Args>(args)...);
  return detail::get_future<Kokkos::DefaultExecutionSpace>::call(
      Kokkos::DefaultExecutionSpace{});
}

template <typename ExecutionPolicy, typename... Args>
hpx::shared_future<void> parallel_reduce_async(std::string const &label,
                                               ExecutionPolicy &&policy,
                                               Args &&...args) {
  HPX_KOKKOS_DETAIL_LOG(
      "calling parallel_reduce_async with label and execution policy");
  Kokkos::parallel_reduce(label, policy, std::forward<Args>(args)...);
  return detail::get_future<typename std::decay<decltype(
      policy.space())>::type>::call(policy.space());
}

template <typename ExecutionPolicy, typename... Args,
          typename Enable = typename std::enable_if<
              Kokkos::is_execution_policy<ExecutionPolicy>::value>::type>
hpx::shared_future<void> parallel_scan_async(ExecutionPolicy &&policy,
                                             Args &&...args) {
  HPX_KOKKOS_DETAIL_LOG("calling parallel_scan_async with execution policy");
  Kokkos::parallel_scan(policy, std::forward<Args>(args)...);
  return detail::get_future<typename std::decay<decltype(
      policy.space())>::type>::call(policy.space());
}

template <typename... Args>
hpx::shared_future<void> parallel_scan_async(std::size_t const work_count,
                                             Args &&...args) {
  HPX_KOKKOS_DETAIL_LOG("calling parallel_scan_async without execution policy");
  Kokkos::parallel_scan(work_count, std::forward<Args>(args)...);
  return detail::get_future<Kokkos::DefaultExecutionSpace>::call(
      Kokkos::DefaultExecutionSpace{});
}

template <typename ExecutionPolicy, typename... Args>
hpx::shared_future<void> parallel_scan_async(std::string const &label,
                                             ExecutionPolicy &&policy,
                                             Args &&...args) {
  HPX_KOKKOS_DETAIL_LOG(
      "calling parallel_scan_async with label and execution policy");
  Kokkos::parallel_scan(label, policy, std::forward<Args>(args)...);
  return detail::get_future<typename std::decay<decltype(
      policy.space())>::type>::call(policy.space());
}
} // namespace kokkos
} // namespace hpx

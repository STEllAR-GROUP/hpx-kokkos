///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2019-2020 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/// \file Contains wrappers for Kokkos parallel dispatch functions that return
/// futures.

#ifndef HPX_KOKKOS_PARALLEL_HPP
#define HPX_KOKKOS_PARALLEL_HPP

#include <hpx/include/compute.hpp>

#include <Kokkos_Concepts.hpp>
#include <Kokkos_Core.hpp>

namespace hpx {
namespace kokkos {
template <typename ExecutionPolicy, typename... Args,
          typename Enable = typename std::enable_if<
              Kokkos::Impl::is_execution_policy<ExecutionPolicy>::value>::type>
hpx::future<void> parallel_for_async(ExecutionPolicy &&policy,
                                     Args &&... args) {
  printf("calling parallel_for_async with execution policy\n");
  Kokkos::parallel_for(policy, std::forward<Args>(args)...);
  return detail::get_future<typename std::decay<decltype(
      policy.space())>::type>::call(policy.space());
}

template <typename... Args>
hpx::future<void> parallel_for_async(std::size_t const work_count,
                                     Args &&... args) {
  printf("calling parallel_for_async without execution policy\n");
  Kokkos::parallel_for(work_count, std::forward<Args>(args)...);
  return detail::get_future<Kokkos::DefaultExecutionSpace>::call(
      Kokkos::DefaultExecutionSpace{});
}

template <typename ExecutionPolicy, typename... Args>
hpx::future<void> parallel_for_async(std::string const &label,
                                     ExecutionPolicy &&policy,
                                     Args &&... args) {
  printf("calling parallel_for_async with label and execution policy\n");
  Kokkos::parallel_for(label, policy, std::forward<Args>(args)...);
  return detail::get_future<typename std::decay<decltype(
      policy.space())>::type>::call(policy.space());
}
} // namespace kokkos
} // namespace hpx

#endif

//  Copyright (c) 2020-2022 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file Contains specializations of HPX algorithms for the Kokkos execution
/// policy.

#pragma once

#include <hpx/kokkos/detail/logging.hpp>
#include <hpx/kokkos/policy.hpp>

#include <hpx/algorithm.hpp>

#include <Kokkos_Core.hpp>

#include <utility>

namespace hpx {
namespace kokkos {
namespace detail {
template <typename ExecutionSpace, typename I, typename F>
hpx::shared_future<void>
for_loop_helper(char const *label, ExecutionSpace &&instance,
                typename std::decay<I>::type first, I last, F &&f) {
  return parallel_for_async(
      label,
      Kokkos::Experimental::require(
          Kokkos::RangePolicy<ExecutionSpace>(instance, first, last),
          Kokkos::Experimental::WorkItemProperty::HintLightWeight),
      std::forward<F>(f));
}

template <typename ExecutionSpace, typename I, std::size_t N, typename F>
hpx::shared_future<void> for_loop_helper(char const *label,
                                         ExecutionSpace &&instance,
                                         Kokkos::Array<I, N> const &first,
                                         Kokkos::Array<I, N> last, F &&f) {
  return parallel_for_async(
      Kokkos::Experimental::require(
          Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<N>,
                                Kokkos::IndexType<I>>(instance, first, last),
          Kokkos::Experimental::WorkItemProperty::HintLightWeight),
      std::forward<F>(f));
}
} // namespace detail

template <typename ExecutionPolicy, typename I, typename F,
          typename Enable = std::enable_if_t<
              is_kokkos_execution_policy<std::decay_t<ExecutionPolicy>>::value>>
auto tag_invoke(hpx::experimental::for_loop_t, ExecutionPolicy &&policy,
                typename std::decay<I>::type first, I last, F &&f) {
  return detail::get_policy_result<ExecutionPolicy>::call(
      detail::for_loop_helper(policy.label(), policy.executor().instance(),
                              first, last, std::forward<F>(f)));
}

template <typename ExecutionPolicy, typename I, std::size_t N, typename F,
          typename Enable = std::enable_if_t<
              is_kokkos_execution_policy<std::decay_t<ExecutionPolicy>>::value>>
auto tag_invoke(hpx::experimental::for_loop_t, ExecutionPolicy &&policy,
                Kokkos::Array<I, N> const &first,
                Kokkos::Array<I, N> const &last, F &&f) {
  return detail::get_policy_result<ExecutionPolicy>::call(
      detail::for_loop_helper(policy.label(), policy.executor().instance(),
                              first, last, std::forward<F>(f)));
}

template <typename ExecutionPolicy, typename I, std::size_t N, typename F,
          typename Enable = std::enable_if_t<
              is_kokkos_execution_policy<std::decay_t<ExecutionPolicy>>::value>>
auto tag_invoke(hpx::experimental::for_loop_t, ExecutionPolicy &&policy,
                Kokkos::Array<I, N> const &first, std::initializer_list<I> last,
                F &&f) {
  return detail::get_policy_result<ExecutionPolicy>::call(
      detail::for_loop_helper(policy.label(), policy.executor().instance(),
                              first, last, f));
}
} // namespace kokkos
} // namespace hpx

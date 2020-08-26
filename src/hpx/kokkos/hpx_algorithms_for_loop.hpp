//  Copyright (c) 2020 ETH Zurich
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
hpx::shared_future<void> for_loop_helper(ExecutionSpace &&instance,
                                         typename std::decay<I>::type first,
                                         I last, F &&f) {
  return parallel_for_async(
      Kokkos::Experimental::require(
          Kokkos::RangePolicy<ExecutionSpace>(instance, first, last),
          Kokkos::Experimental::WorkItemProperty::HintLightWeight),
      KOKKOS_LAMBDA(int const i) {
        HPX_KOKKOS_DETAIL_LOG("for_loop i = %d", i);
        hpx::invoke(f, i);
      });
}
} // namespace detail

template <typename I, typename F>
void tag_invoke(hpx::for_loop_t, hpx::kokkos::kokkos_policy policy,
                typename std::decay<I>::type first, I last, F &&f) {

  detail::for_loop_helper(policy.executor().instance(), first, last,
                          std::forward<F>(f))
      .get();
}

template <typename I, typename F>
hpx::shared_future<void>
tag_invoke(hpx::for_loop_t, hpx::kokkos::kokkos_task_policy policy,
           typename std::decay<I>::type first, I last, F &&f) {
  return detail::for_loop_helper(policy.executor().instance(), first, last, f);
}

template <typename Executor, typename Parameters, typename I, typename F>
void tag_invoke(hpx::for_loop_t,
                hpx::kokkos::kokkos_policy_shim<Executor, Parameters> policy,
                typename std::decay<I>::type first, I last, F &&f) {

  detail::for_loop_helper(policy.executor().instance(), first, last,
                          std::forward<F>(f))
      .get();
}

template <typename Executor, typename Parameters, typename I, typename F>
hpx::shared_future<void>
tag_invoke(hpx::for_loop_t,
           hpx::kokkos::kokkos_task_policy_shim<Executor, Parameters> policy,
           I first, I last, F &&f) {
  return detail::for_loop_helper(policy.executor().instance(), first, last, f);
}
} // namespace kokkos
} // namespace hpx

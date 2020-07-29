//  Copyright (c) 2019-2020 ETH Zurich
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
template <typename ExecutionSpace, typename IterB, typename IterE, typename F>
hpx::shared_future<void> for_each_helper(ExecutionSpace &&instance, IterB first,
                                         IterE last, F &&f) {
  return parallel_for_async(
      Kokkos::Experimental::require(
          Kokkos::RangePolicy<ExecutionSpace>(instance, 0,
                                              std::distance(first, last)),
          Kokkos::Experimental::WorkItemProperty::HintLightWeight),
      KOKKOS_LAMBDA(int const i) {
        HPX_KOKKOS_DETAIL_LOG("for_each i = %d", i);
        hpx::invoke(f, *(first + i));
      });
}

// TODO: This is overfenced at the moment (meaning it may block the worker
// thread).
template <typename ExecutionSpace, typename IterB, typename IterE, typename T,
          typename F>
T reduce_helper(ExecutionSpace &&instance, IterB first, IterE last, T init,
                F &&f) {
  T result;
  parallel_reduce_async(
      Kokkos::Experimental::require(
          Kokkos::RangePolicy<ExecutionSpace>(instance, 0,
                                              std::distance(first, last)),
          Kokkos::Experimental::WorkItemProperty::HintLightWeight),
      KOKKOS_LAMBDA(int const i, T &update) {
        HPX_KOKKOS_DETAIL_LOG("reduce i = %d", i);
        update = hpx::invoke(f, update, *(first + i));
      },
      result)
      .get();

  return hpx::invoke(f, init, result);
}
} // namespace detail

template <typename Iter, typename F>
void tag_invoke(hpx::for_each_t, hpx::kokkos::kokkos_policy policy, Iter first,
                Iter last, F &&f) {

  detail::for_each_helper(policy.executor().instance(), first, last,
                          std::forward<F>(f))
      .get();
}

template <typename Iter, typename F>
hpx::shared_future<void> tag_invoke(hpx::for_each_t,
                                    hpx::kokkos::kokkos_task_policy policy,
                                    Iter first, Iter last, F &&f) {
  return detail::for_each_helper(policy.executor().instance(), first, last, f);
}

template <typename Executor, typename Parameters, typename Iter, typename F>
void tag_invoke(hpx::for_each_t,
                hpx::kokkos::kokkos_policy_shim<Executor, Parameters> policy,
                Iter first, Iter last, F &&f) {

  detail::for_each_helper(policy.executor().instance(), first, last,
                          std::forward<F>(f))
      .get();
}

template <typename Executor, typename Parameters, typename Iter, typename F>
hpx::shared_future<void>
tag_invoke(hpx::for_each_t,
           hpx::kokkos::kokkos_task_policy_shim<Executor, Parameters> policy,
           Iter first, Iter last, F &&f) {
  return detail::for_each_helper(policy.executor().instance(), first, last, f);
}

template <typename Iter, typename T, typename F>
T tag_invoke(hpx::reduce_t, hpx::kokkos::kokkos_policy policy, Iter first,
             Iter last, T init, F &&f) {

  return detail::reduce_helper(
      typename hpx::kokkos::kokkos_policy::executor_type::execution_space{},
      first, last, init, std::forward<F>(f));
}

template <typename Iter, typename T, typename F>
hpx::future<T> tag_invoke(hpx::reduce_t, hpx::kokkos::kokkos_task_policy policy,
                          Iter first, Iter last, T init, F &&f) {
  return hpx::async([=]() {
    return detail::reduce_helper(
        typename hpx::kokkos::kokkos_policy::executor_type::execution_space{},
        first, last, init, f);
  });
}

template <typename Executor, typename Parameters, typename Iter, typename T,
          typename F>
T tag_invoke(hpx::reduce_t,
             hpx::kokkos::kokkos_policy_shim<Executor, Parameters> policy,
             Iter first, Iter last, T init, F &&f) {

  return detail::reduce_helper(policy.executor().instance(), first, last, init,
                               std::forward<F>(f));
}

template <typename Executor, typename Parameters, typename Iter, typename T,
          typename F>
hpx::future<T>
tag_invoke(hpx::reduce_t,
           hpx::kokkos::kokkos_task_policy_shim<Executor, Parameters> policy,
           Iter first, Iter last, T init, F &&f) {
  return hpx::async([=]() {
    return detail::reduce_helper(policy.executor().instance(), first, last,
                                 init, f);
  });
}
} // namespace kokkos
} // namespace hpx

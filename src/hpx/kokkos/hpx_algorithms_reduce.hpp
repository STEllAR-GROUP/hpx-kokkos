//  Copyright (c) 2019-2022 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file Contains specializations of HPX algorithms for the Kokkos execution
/// policy.

#pragma once

#include <hpx/kokkos/detail/logging.hpp>
#include <hpx/kokkos/policy.hpp>

#include <hpx/local/algorithm.hpp>
#include <hpx/local/functional.hpp>

#include <Kokkos_Core.hpp>

#include <utility>

namespace hpx {
namespace kokkos {
namespace detail {
template <typename ExecutionSpace>
// This helper trait specifies what memory space to use for the reduction
// result in order to ensure that the final copy from the device scratch space
// is asynchronous to host memory.
struct reduce_result_space {
  using type = Kokkos::HostSpace;
};

#if defined(KOKKOS_ENABLE_CUDA)
// CUDA memory copies from device to host are asynchronous only to pinned host
// memory.
template <> struct reduce_result_space<Kokkos::Cuda> {
  using type = Kokkos::CudaHostPinnedSpace;
};
#endif

#if defined(KOKKOS_ENABLE_HIP)
// HIP memory copies from device to host are asynchronous only to pinned host
// memory.
template <> struct reduce_result_space<Kokkos::Experimental::HIP> {
  using type = Kokkos::Experimental::HIPHostPinnedSpace;
};
#endif

#if defined(KOKKOS_ENABLE_SYCL)
// Needs to be a SYCL host space for async deep copies
template <> struct reduce_result_space<Kokkos::Experimental::SYCL> {
  using type = Kokkos::Experimental::SYCLHostUSMSpace;
};
#endif

template <typename ExecutionSpace>
using reduce_result_space_t =
    typename reduce_result_space<ExecutionSpace>::type;

template <typename ExecutionSpace, typename IterB, typename IterE, typename T,
          typename F>
hpx::shared_future<T> reduce_helper(char const *label,
                                    ExecutionSpace &&instance, IterB first,
                                    IterE last, T init, F &&f) {
  Kokkos::View<T,
               reduce_result_space_t<typename std::decay<ExecutionSpace>::type>>
      result(Kokkos::view_alloc(Kokkos::WithoutInitializing, "reduce_result"));

  return parallel_reduce_async(
             label,
             Kokkos::Experimental::require(
                 Kokkos::RangePolicy<ExecutionSpace>(
                     instance, 0, std::distance(first, last)),
                 Kokkos::Experimental::WorkItemProperty::HintLightWeight),
             KOKKOS_LAMBDA(int const i, T &update) {
               HPX_KOKKOS_DETAIL_LOG("reduce i = %d", i);
               update = hpx::invoke(f, update, *(first + i));
             },
             result)
      .then(hpx::launch::sync, [f, init, result](hpx::shared_future<void> &&) {
        return hpx::invoke(f, init, result());
      });
}
} // namespace detail

// Reduce non-range overloads
template <typename ExecutionPolicy, typename Iter, typename T, typename F,
          typename Enable = std::enable_if_t<
              is_kokkos_execution_policy<std::decay_t<ExecutionPolicy>>::value>>
auto tag_invoke(hpx::reduce_t, ExecutionPolicy &&policy, Iter first, Iter last,
                T init, F &&f) {
  return detail::get_policy_result<ExecutionPolicy>::call(
      detail::reduce_helper(policy.label(), policy.executor().instance(), first,
                            last, init, std::forward<F>(f)));
}
} // namespace kokkos
} // namespace hpx

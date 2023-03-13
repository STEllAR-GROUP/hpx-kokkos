///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2019-2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/// \file Contains wrappers for Kokkos deep copy functions that return futures.

#pragma once

#include <impl/Kokkos_Error.hpp>

#include <hpx/kokkos/future.hpp>

namespace hpx {
namespace kokkos {
// TODO: Do we need more overloads here?
template <typename ExecutionSpace, typename... Args,
          typename Enable = typename std::enable_if<Kokkos::is_execution_space<
              typename std::decay<ExecutionSpace>::type>::value>::type>
hpx::shared_future<void> deep_copy_async(ExecutionSpace &&space,
                                         Args &&...args) {
  Kokkos::deep_copy(space, std::forward<Args>(args)...);
  return detail::get_future<typename std::decay<ExecutionSpace>::type>::call(
      std::forward<ExecutionSpace>(space));
}
#if defined(KOKKOS_ENABLE_SYCL)
#if !defined(HPX_KOKKOS_SYCL_FUTURE_TYPE)
// polling is default (0) as it is simply faster)
// 1 would be using host_tasks which is slower but useful for comparisons
#define HPX_KOKKOS_SYCL_FUTURE_TYPE 0
#warning "HPX_KOKKOS_SYCL_FUTURE_TYPE was not defined! Defining it to 0 (event)
#endif
/// deep_copy_async specialization for SYCL spaces. It comes with the advantage
/// of not having to create our own sycl::event in get_future - instead it uses
/// the copy event directly by circumventing kokkos::deep_copy and running
/// sycl:memcpy itself. This reduces the overhead. 
template <typename TargetSpace, typename SourceSpace>
hpx::shared_future<void> deep_copy_async(Kokkos::Experimental::SYCL &&instance,
                                         TargetSpace &&t, SourceSpace &&s) {
  // Usually, Kokkos does a bunch of safety checks before deep copies. Here, we
  // have to do those ourselves unfortunately since we want to use a normal
  // memcpy. For the deep_copy view requirements (implemented in those checks),
  // see https://github.com/kokkos/kokkos/wiki/Kokkos::deep_copy#requirements

  // Safety checks 1: Statically check for same data types, non-const target and
  // same ranks/layout
  static_assert(
      std::is_same<typename std::decay<TargetSpace>::type::data_type,
                   typename std::decay<SourceSpace>::type::data_type>::value,
      "deep_copy_async requires the same datatypes for src and target");
  static_assert(
      std::is_same<
          typename std::decay<TargetSpace>::type::value_type,
          typename std::decay<TargetSpace>::type::non_const_value_type>::value,
      "deep_copy_async requires non-const destination type");
  static_assert((unsigned(std::decay<TargetSpace>::type::rank) ==
                 unsigned(std::decay<SourceSpace>::type::rank)),
                "deep_copy_async requires SYCL views of equal rank");
  static_assert(
      std::is_same<
          typename std::decay<TargetSpace>::type::array_layout,
          typename std::decay<SourceSpace>::type::array_layout>::value,
      "deep_copy_requires same array_layout in both source and target SYCL views");

  // Safety checks 2: Check that there's no dimension mismatch and that the memory is contiguous
  if (!t.span_is_contiguous() || !s.span_is_contiguous()) {
    Kokkos::Impl::throw_runtime_exception(
        "deep_copy_async: Both source and target SYCL views must be contiguos");
  }
  if ((s.extent(0) != t.extent(0)) || (s.extent(1) != t.extent(1)) ||
      (s.extent(2) != t.extent(2)) || (s.extent(3) != t.extent(3)) ||
      (s.extent(4) != t.extent(4)) || (s.extent(5) != t.extent(5)) ||
      (s.extent(6) != t.extent(6)) || (s.extent(7) != t.extent(7))) {
    Kokkos::Impl::throw_runtime_exception(
        "deep_copy_async: Error, dimension/size mismatch between source and target SYCL views");
  }

  // Safety checks 3: Check for overlapping views and in_order semantics
  typename std::decay<TargetSpace>::type::value_type* dst_start = t.data();
  typename std::decay<TargetSpace>::type::value_type* dst_end   = t.data() + t.span();
  typename std::decay<SourceSpace>::type::value_type* src_start = s.data();
  typename std::decay<SourceSpace>::type::value_type* src_end   = s.data() + s.span();
  if (((std::ptrdiff_t)dst_start < (std::ptrdiff_t)src_end) &&
      ((std::ptrdiff_t)dst_end > (std::ptrdiff_t)src_start)) {
    Kokkos::Impl::throw_runtime_exception(
        "deep_copy_async: Error, SYCL views are overlapping");
  }
  auto& q = *instance.impl_internal_space_instance()->m_queue;
  if (!(q.is_in_order())) {
    Kokkos::Impl::throw_runtime_exception(
        "deep_copy_async: Error, underlying SYCL queue is not in-order");
  }

  // Start to actually copy with a normal SYCL memcpy
  auto event = q.memcpy(t.data(), s.data(), t.size() *
      sizeof(typename std::decay<TargetSpace>::type::data_type));
  // Use event from memcpy to get a future
#if HPX_KOKKOS_SYCL_FUTURE_TYPE == 0 
  return hpx::sycl::experimental::detail::get_future(event);
#elif HPX_KOKKOS_SYCL_FUTURE_TYPE == 1
  return hpx::sycl::experimental::detail::get_future_using_host_task(event, q);
#else
#error "HPX_KOKKOS_SYCL_FUTURE_TYPE is invalid (must be host_task or event)"
#endif
}
#endif
} // namespace kokkos
} // namespace hpx

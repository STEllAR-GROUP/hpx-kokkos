///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2019-2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/// \file Contains wrappers for Kokkos deep copy functions that return futures.

#pragma once

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
/// deep_copy_async specialization for SYCL spaces. It comes with the advantage of not having
/// to create our own sycl::event in get_future - instead it uses the copy event directly by
/// circumventing kokkos::deep_copy and running sycl:memcpy itself. This reduces the 
/// overhead.
template<typename TargetSpace, typename SourceSpace>
hpx::shared_future<void> deep_copy_async(Kokkos::Experimental::SYCL &&instance,
    TargetSpace &&t, SourceSpace &&s) {
  assert(t.size() == s.size());
  assert(q.is_in_order());
  static_assert(std::is_same<typename std::decay<TargetSpace>::type::data_type,
      typename std::decay<SourceSpace>::type::data_type>::value);
  auto& q = *instance.impl_internal_space_instance()->m_queue;
  auto event = q.memcpy(t.data(), s.data(), t.size() * sizeof(typename std::decay<TargetSpace>::type::data_type));
  return hpx::sycl::experimental::detail::get_future(event);
}
#endif
} // namespace kokkos
} // namespace hpx

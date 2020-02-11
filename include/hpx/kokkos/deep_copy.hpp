///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2019-2020 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/// \file Contains wrappers for Kokkos deep copy functions that return futures.

#ifndef HPX_KOKKOS_DEEP_COPY_HPP
#define HPX_KOKKOS_DEEP_COPY_HPP

#include <hpx/kokkos/future.hpp>

namespace hpx {
namespace kokkos {
// TODO: Do we need more overloads here?
template <typename ExecutionSpace, typename... Args,
          typename Enable = typename std::enable_if<
              Kokkos::is_execution_space<ExecutionSpace>::value>::type>
hpx::future<void> deep_copy_async(ExecutionSpace &&space, Args &&... args) {
  deep_copy(space, std::forward<Args>(args)...);
  return detail::get_future<ExecutionSpace>::call(
      std::forward<ExecutionSpace>(space));
}
} // namespace kokkos
} // namespace hpx

#endif

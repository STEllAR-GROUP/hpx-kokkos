///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/// \file Contains utilities for dealing with Kokkos execution spaces and HPX
/// futures.

#pragma once

#include <hpx/kokkos/detail/logging.hpp>

#include <hpx/include/compute.hpp>
#include <hpx/include/future.hpp>

#include <Kokkos_Core.hpp>

namespace hpx {
namespace kokkos {
namespace detail {
template <typename ExecutionSpace> struct get_future {
  template <typename E> static hpx::shared_future<void> call(E &&inst) {
    // The best we can do generically at the moment is to fence on the
    // instance and return a ready future. It would be nice to be able to
    // attach a callback to any execution space instance to trigger future
    // completion.
    inst.fence();
    HPX_KOKKOS_DETAIL_LOG("getting generic ready future after fencing");
    return hpx::make_ready_future();
  }
};

#if defined(KOKKOS_ENABLE_CUDA)
template <> struct get_future<Kokkos::Cuda> {
  template <typename E> static hpx::shared_future<void> call(E &&inst) {
    HPX_KOKKOS_DETAIL_LOG("getting future from stream %p", inst.cuda_stream());
    return hpx::compute::cuda::get_future(inst.cuda_stream());
  }
};
#endif

#if defined(KOKKOS_ENABLE_HPX) && KOKKOS_VERSION >= 30000 &&                   \
    defined(KOKKOS_ENABLE_HPX_ASYNC_DISPATCH)
template <> struct get_future<Kokkos::Experimental::HPX> {
  template <typename E> static hpx::shared_future<void> call(E &&inst) {
    HPX_KOKKOS_DETAIL_LOG("getting future from HPX instance %x",
                          inst.impl_instance_id());
    return inst.impl_get_future();
  }
};
#endif
} // namespace detail

/// Make a future for a particular execution space instance. This might be
/// useful for functions that don't have *_async overloads yet but take an
/// execution space instance for asynchronous execution.
template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
hpx::shared_future<void> make_execution_space_future(ExecutionSpace &&inst) {
  return detail::get_future<typename std::decay<ExecutionSpace>::type>::call(
      std::forward<ExecutionSpace>(inst));
}
} // namespace kokkos
} // namespace hpx

///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2020 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/// \file Contains utilities for dealing with Kokkos execution spaces and HPX
/// futures.

#ifndef HPX_KOKKOS_UTIL_HPP
#define HPX_KOKKOS_UTIL_HPP

#include <hpx/compute/cuda/target.hpp>
#include <hpx/include/future.hpp>

#include <Kokkos_Core.hpp>

namespace hpx {
namespace kokkos {
namespace detail {
template <typename ExecutionSpace> struct get_future {
  template <typename E> static hpx::future<void> call(E &&inst) {
    // The best we can do generically at the moment is to fence on the
    // instance and return a ready future. It would be nice to be able to
    // attach a callback to any execution space instance to trigger future
    // completion.
    inst.fence();
    printf("getting generic ready future after fencing\n");
    return hpx::make_ready_future();
  }
};

template <> struct get_future<Kokkos::Cuda> {
  template <typename E> static hpx::future<void> call(E &&inst) {
    // NOTE: 0 is the device. This is probably not correct on multi-GPU
    // systems.
    printf("getting future from stream %x\n", inst.cuda_stream());
    return hpx::compute::cuda::get_future(inst.cuda_stream());
  }
};

// TODO: We *can* specialize for Kokkos::HPX. However, the correct
// functionality is not there yet in the HPX backend (it only stores a
// single (non-shared) future, which can't be accessed).
} // namespace detail
} // namespace kokkos
} // namespace hpx

#endif

//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Contains a helper for creating thread-local execution space instances and
/// executors.

#pragma once

#include <hpx/kokkos/execution_spaces.hpp>

#include <Kokkos_Core.hpp>

namespace hpx {
namespace kokkos {
namespace detail {
template <typename ExecutionSpace>
ExecutionSpace make_independent_execution_space_instance() {
  return {};
}

#if defined(KOKKOS_ENABLE_CUDA)
template <>
inline Kokkos::Cuda make_independent_execution_space_instance<Kokkos::Cuda>() {
  cudaStream_t s;
  cudaError_t error = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    HPX_THROW_EXCEPTION(
        kernel_error, "hpx::kokkos::detail::initialize_instances",
        std::string("cudaStreamCreate failed: ") + cudaGetErrorString(error));
  }
  return {s};
}
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <>
inline Kokkos::Experimental::HIP
make_independent_execution_space_instance<Kokkos::Experimental::HIP>() {
  cudaStream_t s;
  cudaError_t error = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    HPX_THROW_EXCEPTION(
        kernel_error, "hpx::kokkos::detail::initialize_instances",
        std::string("cudaStreamCreate failed: ") + cudaGetErrorString(error));
  }
  return {s};
}
#endif

#if defined(KOKKOS_ENABLE_HPX)
template <>
inline Kokkos::Experimental::HPX
make_independent_execution_space_instance<Kokkos::Experimental::HPX>() {
  return {Kokkos::Experimental::HPX::instance_mode::independent};
}
#endif
} // namespace detail
} // namespace kokkos
} // namespace hpx

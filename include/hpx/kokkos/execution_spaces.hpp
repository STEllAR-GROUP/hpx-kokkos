///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2020 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/// \file Contains wrappers to create Kokkos execution space instances. Some may
/// be specialized for a particular execution space.

#ifndef HPX_KOKKOS_EXECUTION_SPACES_HPP
#define HPX_KOKKOS_EXECUTION_SPACES_HPP

#include <hpx/include/compute.hpp>

#include <Kokkos_Core.hpp>

namespace hpx {
namespace kokkos {
template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
ExecutionSpace make_execution_space() {
  return {};
}

#if defined(KOKKOS_ENABLE_CUDA)
namespace detail {
// TODO: The streams are never destroyed.
std::vector<Kokkos::Cuda>
initialize_instances(std::size_t const num_instances) {
  std::vector<Kokkos::Cuda> instances;
  instances.reserve(num_instances);
  for (std::size_t i = 0; i < num_instances; ++i) {
    cudaStream_t s;
    cudaError_t error = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (error != cudaSuccess) {
      HPX_THROW_EXCEPTION(
          kernel_error, "hpx::kokkos::detail::initialize_instances",
          std::string("cudaStreamCreate failed: ") + cudaGetErrorString(error));
    }
    instances.emplace_back(s);
  }
  return instances;
}
} // namespace detail

template <> Kokkos::Cuda make_execution_space<Kokkos::Cuda>() {
  static constexpr std::size_t num_instances = 10;
  static thread_local std::size_t current_instance = 0;
  static thread_local std::vector<Kokkos::Cuda> instances =
      detail::initialize_instances(num_instances);
  return instances[current_instance++ % num_instances];
}
#endif
} // namespace kokkos
} // namespace hpx

#endif

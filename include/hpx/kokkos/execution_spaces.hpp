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

namespace detail {
// TODO: These streams are never destroyed.
std::vector<cudaStream_t> initialize_streams(std::size_t const num_streams) {
  std::vector<cudaStream_t> streams(num_streams);
  for (auto &s : streams) {
    cudaError_t error =
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (error != cudaSuccess) {
      HPX_THROW_EXCEPTION(
          kernel_error, "hpx::kokkos::detail::initialize_streams",
          std::string("cudaStreamCreate failed: ") + cudaGetErrorString(error));
    }
  }
  return streams;
}
} // namespace detail

template <> Kokkos::Cuda make_execution_space<Kokkos::Cuda>() {
  static constexpr std::size_t num_streams = 3;
  static thread_local std::size_t current_stream = 0;
  static thread_local std::vector<cudaStream_t> streams =
      detail::initialize_streams(num_streams);
  return {streams[current_stream++ % num_streams]};
}
} // namespace kokkos
} // namespace hpx

#endif

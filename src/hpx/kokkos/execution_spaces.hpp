///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/// \file Contains wrappers to create Kokkos execution space instances. Some may
/// be specialized for a particular execution space.

#pragma once

#include <hpx/include/compute.hpp>

#include <Kokkos_Core.hpp>

namespace hpx {
namespace kokkos {
template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
ExecutionSpace make_execution_space() {
  return {};
}

template <typename ExecutionSpace>
struct is_execution_space_independent : std::false_type {};

#if defined(KOKKOS_ENABLE_CUDA)
namespace detail {
std::vector<Kokkos::Cuda>
initialize_instances(std::size_t const num_instances);
} // namespace detail

template<> Kokkos::Cuda make_execution_space<Kokkos::Cuda>();
extern template Kokkos::Cuda make_execution_space<Kokkos::Cuda>();

template <>
struct is_execution_space_independent<Kokkos::Cuda> : std::true_type {};
#endif

#if defined(KOKKOS_ENABLE_HPX) && KOKKOS_VERSION >= 30000
template <>
inline Kokkos::Experimental::HPX make_execution_space<Kokkos::Experimental::HPX>() {
  return Kokkos::Experimental::HPX(
      Kokkos::Experimental::HPX::instance_mode::independent);
}

template <>
struct is_execution_space_independent<Kokkos::Experimental::HPX>
    : std::true_type {};
#endif
} // namespace kokkos
} // namespace hpx

///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/// \file Contains wrappers to create Kokkos execution space instances. Some may
/// be specialized for a particular execution space.

#pragma once

#include <Kokkos_Core.hpp>

namespace hpx {
namespace kokkos {
template <typename ExecutionSpace>
struct is_execution_space_independent : std::false_type {};

#if defined(KOKKOS_ENABLE_CUDA)
template <>
struct is_execution_space_independent<Kokkos::Cuda> : std::true_type {};
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <>
struct is_execution_space_independent<Kokkos::Experimental::HIP>
    : std::true_type {};
#endif

#if defined(KOKKOS_ENABLE_HPX)
template <>
struct is_execution_space_independent<Kokkos::Experimental::HPX>
    : std::true_type {};
#endif
} // namespace kokkos
} // namespace hpx

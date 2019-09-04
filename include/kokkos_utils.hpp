///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2019 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_KOKKOS_UTILS_HPP
#define HPX_COMPUTE_KOKKOS_UTILS_HPP

#include <Kokkos_Core.hpp>

namespace hpx {
namespace compute {
namespace kokkos {
/// RAII guard for initializing and finalizing the Kokkos runtime.
struct runtime_guard {
  template <typename... Args> runtime_guard(Args &&... args) {
    Kokkos::initialize(std::forward<Args>(args)...);
  }

  ~runtime_guard() { Kokkos::finalize(); }
};
} // namespace kokkos
} // namespace compute
} // namespace hpx

#endif

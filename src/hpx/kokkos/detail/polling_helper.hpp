///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2021 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/// \file Contains a RAII helper utility for enabling polling if HIP or CUDA is
/// enabled, and hpx-kokkos is configured to use polling.

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_HAVE_CUDA) || defined(HPX_HAVE_HIP)
#include <hpx/modules/async_cuda.hpp>
#endif
#if defined(HPX_HAVE_SYCL)
#include <hpx/modules/async_sycl.hpp>
#endif

namespace hpx {
namespace kokkos {
namespace detail {
struct polling_helper {
#if (defined(HPX_HAVE_CUDA) || defined(HPX_HAVE_HIP)) && (HPX_KOKKOS_CUDA_FUTURE_TYPE == 0)
  hpx::cuda::experimental::enable_user_polling p;
#endif
#if defined(HPX_HAVE_SYCL)
  hpx::sycl::experimental::enable_user_polling p_sycl;
#endif
};
} // namespace detail
} // namespace kokkos
} // namespace hpx

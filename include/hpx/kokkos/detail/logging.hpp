///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/// \file Logging functionality. Provides HPX_KOKKOS_DETAIL_LOG for internal
/// logging.

#pragma once

#include <cstdio>

#if defined(HPX_KOKKOS_ENABLE_LOGGING)
#if defined(__CUDA_ARCH__)
#define HPX_KOKKOS_DETAIL_LOG_HELPER(FMT, ...)                                 \
  printf("[cuda] " FMT "%s\n", __VA_ARGS__)
#elif defined(__HIP_DEVICE_COMPILE__)
#define HPX_KOKKOS_DETAIL_LOG_HELPER(FMT, ...)                                 \
  printf("[hip ] " FMT "%s\n", __VA_ARGS__)
#else
#define HPX_KOKKOS_DETAIL_LOG_HELPER(FMT, ...)                                 \
  printf("[host] " FMT "%s\n", __VA_ARGS__)
#endif
#define HPX_KOKKOS_DETAIL_LOG(...) HPX_KOKKOS_DETAIL_LOG_HELPER(__VA_ARGS__, "")
#else
#define HPX_KOKKOS_DETAIL_LOG(user_fmt, ...)
#endif

///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2019-2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config/version.hpp>

#if HPX_VERSION_FULL >= 0x011100
#define HPXKOKKOS_HPX_EXECUTOR_NS hpx::execution::experimental
#else
#define HPXKOKKOS_HPX_EXECUTOR_NS hpx::parallel::execution
#endif

//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Tests that there are no multiple definitions when including the headers in
/// multiple files. dummy.cpp is also built with this test.

#include "test.hpp"

#include <hpx/kokkos.hpp>

#include <hpx/hpx_main.hpp>

int main(int argc, char *argv[]) {
  HPX_KOKKOS_DETAIL_TEST(true);
  return hpx::kokkos::detail::report_errors();
}


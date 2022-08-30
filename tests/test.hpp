//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Contains testing utilities. Is not safe to include in multiple files.

#pragma once

#include <atomic>
#include <cstddef>
#include <iostream>
#include <string>

namespace hpx {
namespace kokkos {
namespace detail {
static std::atomic<std::size_t> tests_failed{0};
static std::atomic<std::size_t> tests_done{0};

void check_test(std::string filename, std::size_t linenumber, bool success) {
  ++tests_done;
  if (!success) {
    ++tests_failed;
    std::cerr << "Test failed in " << filename << " on line " << linenumber
              << '\n';
  }
}

int report_errors() {
  if (tests_done == 0) {
    std::cerr << "No tests run. Did you forget to add tests?\n";
    return 1;
  } else if (tests_failed == 0) {
    std::cerr << "All tests passed (" << tests_done << " tests run).\n";
    return 0;
  } else {
    std::cerr << tests_failed << "/" << tests_done << " test"
              << (tests_done == 1 ? "" : "s") << " failed.\n";
    return 1;
  }
}
} // namespace detail
} // namespace kokkos
} // namespace hpx

#define HPX_KOKKOS_DETAIL_TEST(expr)                                           \
  hpx::kokkos::detail::check_test(__FILE__, __LINE__, expr)

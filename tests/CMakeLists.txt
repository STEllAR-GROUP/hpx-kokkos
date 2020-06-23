# Copyright (c) 2019-2020 ETH Zurich
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_custom_target(tests)

set(_tests
  asynchrony
  executors
  kokkos_async_parallel
  parallel_algorithms
  view_iterator)
foreach(_test ${_tests})
  set(_test_name ${_test}_test)
  add_executable(${_test_name} EXCLUDE_FROM_ALL ${_test}.cpp)
  target_link_libraries(${_test_name} PRIVATE hpx_kokkos)
  add_dependencies(tests ${_test_name})
  add_test(NAME ${_test} COMMAND ${_test_name})
endforeach()

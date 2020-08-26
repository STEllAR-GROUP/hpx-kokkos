# Copyright (c) 2019-2020 ETH Zurich
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_custom_target(benchmarks)

set(_benchmarks future_overheads overheads overheads_multi_instance stream)

foreach(_benchmark ${_benchmarks})
  set(_benchmark_name ${_benchmark}_benchmark)
  add_executable(${_benchmark_name}
    EXCLUDE_FROM_ALL
    ${_benchmark}.cpp ${${_benchmark}_extra_sources})
  target_link_libraries(${_benchmark_name} PRIVATE hpx_kokkos HPX::hpx HPX::wrap_main)
  add_dependencies(benchmarks ${_benchmark_name})
  add_test(NAME ${_benchmark} COMMAND ${_benchmark_name})
endforeach()

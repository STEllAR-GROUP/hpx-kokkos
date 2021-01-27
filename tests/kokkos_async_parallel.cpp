//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Tests asynchronous versions of Kokkos parallel algorithms.

#include "test.hpp"

#include <hpx/chrono.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/kokkos.hpp>
#include <hpx/kokkos/detail/polling_helper.hpp>

#include <string>

template <typename ExecutionSpace> struct scan_kernel {
  Kokkos::View<int *, ExecutionSpace> a;

  scan_kernel(Kokkos::View<int *> a) : a(a) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, int &update, const bool final_pass) const {
    update += a(i);
  }

  KOKKOS_INLINE_FUNCTION
  void init(int &update) const { update = 0; }

  KOKKOS_INLINE_FUNCTION
  void join(int &update, int const &input) const { update += input; }
};

template <typename ExecutionSpace>
void test_parallel_for(ExecutionSpace &&inst) {
  int const n = 43;

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
      parallel_for_result_host("parallel_for_result_host", n);
  Kokkos::View<int *, typename std::decay<ExecutionSpace>::type>
      parallel_for_result("parallel_for_result", n);
  for (std::size_t i = 0; i < n; ++i) {
    parallel_for_result_host(i) = 0;
  }
  hpx::kokkos::deep_copy_async(inst, parallel_for_result,
                               parallel_for_result_host);
  hpx::kokkos::parallel_for_async(
      Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(inst, 0,
                                                                     n),
      KOKKOS_LAMBDA(int i) { parallel_for_result(i) = i; });
  hpx::kokkos::deep_copy_async(inst, parallel_for_result_host,
                               parallel_for_result);
  inst.fence();
  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(parallel_for_result_host(i) == i);
  }
}

template <typename ExecutionSpace>
void test_parallel_reduce(ExecutionSpace &&inst) {
  int const n = 43;

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
      parallel_reduce_result_host("parallel_reduce_result_host", n);
  Kokkos::View<int *, typename std::decay<ExecutionSpace>::type>
      parallel_reduce_result("parallel_reduce_result", n);
  for (std::size_t i = 0; i < n; ++i) {
    parallel_reduce_result_host(i) = i;
  }
  hpx::kokkos::deep_copy_async(inst, parallel_reduce_result,
                               parallel_reduce_result_host);
  int sum = 0;
  hpx::kokkos::parallel_reduce_async(
      Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(inst, 0,
                                                                     n),
      KOKKOS_LAMBDA(int const &i, int &acc) {
        acc += parallel_reduce_result(i);
      },
      Kokkos::Sum<int>(sum));
  hpx::kokkos::deep_copy_async(inst, parallel_reduce_result,
                               parallel_reduce_result_host);
  inst.fence();
  HPX_KOKKOS_DETAIL_TEST(sum == n * (n - 1) / 2);
}

template <typename ExecutionSpace>
void test_parallel_scan(ExecutionSpace &&inst) {
  int const n = 43;

  int sum = 0;
  hpx::kokkos::parallel_scan_async(
      Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(inst, 0,
                                                                     n),
      KOKKOS_LAMBDA(int const i, int &sum, bool const) { sum += i; }, sum);
  inst.fence();
  HPX_KOKKOS_DETAIL_TEST(sum == (n - 1) * n / 2);
}

template <typename ExecutionSpace> void test(ExecutionSpace &&inst) {
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value,
                "ExecutionSpace is not a Kokkos execution space");
  test_parallel_for(inst);
  test_parallel_reduce(inst);
  test_parallel_scan(inst);
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);

  {
    hpx::kokkos::detail::polling_helper p;

    test(Kokkos::DefaultExecutionSpace{});
    if (!std::is_same<Kokkos::DefaultExecutionSpace,
                      Kokkos::DefaultHostExecutionSpace>::value) {
      test(Kokkos::DefaultHostExecutionSpace{});
    }
  }

  Kokkos::finalize();

  return hpx::kokkos::detail::report_errors();
}

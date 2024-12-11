//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Tests that executors created with the independent flag are different, and
/// those created with the global flag are the same.

#include "test.hpp"

#include <hpx/execution.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/kokkos.hpp>
#include <hpx/kokkos/detail/polling_helper.hpp>

#include <atomic>
#include <cassert>
#include <vector>

template <typename ExecutionSpace> void test() {
  // NOTE: Most execution spaces only have a global instance and no way to
  // compare instances. We just check that they can be constructed with the
  // global and independent flags.

  hpx::kokkos::executor<ExecutionSpace> exec_global1{};
  hpx::kokkos::executor<ExecutionSpace> exec_global2{
      hpx::kokkos::execution_space_mode::global};

  hpx::kokkos::executor<ExecutionSpace> exec_independent1{
      hpx::kokkos::execution_space_mode::independent};
  hpx::kokkos::executor<ExecutionSpace> exec_independent2{
      hpx::kokkos::execution_space_mode::independent};
}

#if defined(KOKKOS_ENABLE_HPX)
template <> void test<Kokkos::Experimental::HPX>() {
  hpx::kokkos::executor<Kokkos::Experimental::HPX> exec_global1{};
  hpx::kokkos::executor<Kokkos::Experimental::HPX> exec_global2{
      hpx::kokkos::execution_space_mode::global};

  hpx::kokkos::executor<Kokkos::Experimental::HPX> exec_independent1{
      hpx::kokkos::execution_space_mode::independent};
  hpx::kokkos::executor<Kokkos::Experimental::HPX> exec_independent2{
      hpx::kokkos::execution_space_mode::independent};

  HPX_KOKKOS_DETAIL_TEST(exec_global1.instance().impl_instance_id() ==
                         exec_global2.instance().impl_instance_id());
  HPX_KOKKOS_DETAIL_TEST(exec_independent1.instance().impl_instance_id() !=
                         exec_independent2.instance().impl_instance_id());
}
#endif

#if defined(KOKKOS_ENABLE_CUDA)
template <> void test<Kokkos::Cuda>() {
  hpx::kokkos::executor<Kokkos::Cuda> exec_global1{};
  hpx::kokkos::executor<Kokkos::Cuda> exec_global2{
      hpx::kokkos::execution_space_mode::global};

  hpx::kokkos::executor<Kokkos::Cuda> exec_independent1{
      hpx::kokkos::execution_space_mode::independent};
  hpx::kokkos::executor<Kokkos::Cuda> exec_independent2{
      hpx::kokkos::execution_space_mode::independent};

  HPX_KOKKOS_DETAIL_TEST(exec_global1.instance().cuda_stream() ==
                         exec_global2.instance().cuda_stream());
  HPX_KOKKOS_DETAIL_TEST(exec_independent1.instance().cuda_stream() !=
                         exec_independent2.instance().cuda_stream());
}
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <> void test<Kokkos::Experimental::HIP>() {
  hpx::kokkos::executor<Kokkos::Experimental::HIP> exec_global1{};
  hpx::kokkos::executor<Kokkos::Experimental::HIP> exec_global2{
      hpx::kokkos::execution_space_mode::global};

  hpx::kokkos::executor<Kokkos::Experimental::HIP> exec_independent1{
      hpx::kokkos::execution_space_mode::independent};
  hpx::kokkos::executor<Kokkos::Experimental::HIP> exec_independent2{
      hpx::kokkos::execution_space_mode::independent};

  HPX_KOKKOS_DETAIL_TEST(exec_global1.instance().hip_stream() ==
                         exec_global2.instance().hip_stream());
  HPX_KOKKOS_DETAIL_TEST(exec_independent1.instance().hip_stream() !=
                         exec_independent2.instance().hip_stream());
}
#endif

int hpx_main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);

  {
    test<Kokkos::DefaultExecutionSpace>();
    if (!std::is_same<Kokkos::DefaultExecutionSpace,
                      Kokkos::DefaultHostExecutionSpace>::value) {
      test<Kokkos::DefaultHostExecutionSpace>();
    }
  }

  Kokkos::finalize();
  hpx::finalize();

  return hpx::kokkos::detail::report_errors();
}

int main(int argc, char *argv[]) {
  return hpx::init(hpx_main, argc, argv);
}

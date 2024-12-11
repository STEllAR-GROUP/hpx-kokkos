//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// This test is like the overheads test but instead of using a single instance
/// for all launches it uses the instances provided by kokkos_instance_helper.

#include <Kokkos_Core.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/chrono.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/kokkos.hpp>
#include <hpx/kokkos/detail/polling_helper.hpp>

void print_header() {
  std::cout << "test_name,execution_space,subtest_name,vector_size,launches_"
               "per_test,time"
            << std::endl;
}

template <typename ExecutionSpace, typename F, typename Views, typename... Args>
void time_test(std::string const &label, F const &f,
               hpx::kokkos::kokkos_instance_helper<ExecutionSpace> &h,
               Views const &views, int const n, int const launches_per_test,
               Args... args) {
  hpx::chrono::high_resolution_timer timer;
  f(h, views, n, launches_per_test, std::forward<Args>(args)...);
  std::cout << "overhead_multi_instance," << ExecutionSpace().name() << ","
            << label << "," << n << "," << launches_per_test << ","
            << timer.elapsed() << std::endl;
}

// Plain Kokkos::parallel_for, with a fence at the end.
template <typename ExecutionSpace, typename Views>
void test_for_loop_kokkos(
    hpx::kokkos::kokkos_instance_helper<ExecutionSpace> &h, Views const &views,
    int const n, int const launches_per_test) {
  for (int l = 0; l < launches_per_test; ++l) {
    // Init-capture not allowed by nvcc, so we initialize a here.
    auto a = views[l];
    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(
                h.get_execution_space(), 0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        [a] KOKKOS_IMPL_FUNCTION(int i) { a(i) = i; });
  }

  Kokkos::fence();
}

enum class sync_type { fence, future };

// Futurized Kokkos::parallel_for, synchronized either with a fence or the
// returned futures.
template <typename ExecutionSpace, typename Views>
void test_for_loop_kokkos_async(
    hpx::kokkos::kokkos_instance_helper<ExecutionSpace> &h, Views const &views,
    int const n, int const launches_per_test, sync_type s) {
  std::vector<hpx::shared_future<void>> futures;
  futures.reserve(launches_per_test);

  for (int l = 0; l < launches_per_test; ++l) {
    // Init-capture not allowed by nvcc, so we initialize a here.
    auto a = views[l];
    futures.push_back(hpx::kokkos::parallel_for_async(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(
                h.get_execution_space(), 0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        [a] KOKKOS_IMPL_FUNCTION(int i) { a(i) = i; }));
  }

  switch (s) {
  case sync_type::fence:
    Kokkos::fence();
    break;
  case sync_type::future:
    hpx::wait_all(futures);
    break;
  default:
    std::cerr << "Unknown sync_type" << std::endl;
    std::terminate();
  }
}

// hpx::for_each with a Kokkos execution policy, synchronized either with a
// fence or the returned futures.
template <typename ExecutionSpace, typename Views>
void test_for_loop_hpx_async(
    hpx::kokkos::kokkos_instance_helper<ExecutionSpace> &h, Views const &views,
    int const n, int const launches_per_test, sync_type s) {
  std::vector<hpx::shared_future<void>> futures;
  futures.reserve(launches_per_test);

  for (int l = 0; l < launches_per_test; ++l) {
    futures.push_back(
        // TODO: This should use hpx::for_loop
        hpx::for_each(
            hpx::kokkos::kok(hpx::execution::task).on(h.get_executor()),
            views[l].data(), views[l].data() + views[l].size(),
            [] KOKKOS_IMPL_FUNCTION(int &x) { x = 42; }));
  }

  switch (s) {
  case sync_type::fence:
    Kokkos::fence();
    break;
  case sync_type::future:
    hpx::wait_all(futures);
    break;
  default:
    std::cerr << "Unknown sync_type" << std::endl;
    std::terminate();
  }
}

template <typename ExecutionSpace>
void test_for_loop(hpx::kokkos::kokkos_instance_helper<ExecutionSpace> &h,
                   int const n, int const launches_per_test,
                   int const repetitions) {
  std::vector<Kokkos::View<int *, ExecutionSpace>> views;
  views.reserve(launches_per_test);
  for (int l = 0; l < launches_per_test; ++l) {
    views.emplace_back("a", n);
  }

  for (int r = 0; r < repetitions; ++r) {
    time_test<ExecutionSpace>(
        "kokkos", &test_for_loop_kokkos<ExecutionSpace, decltype(views)>, h,
        views, n, launches_per_test);
    time_test<ExecutionSpace>(
        "kokkos_async_fence",
        &test_for_loop_kokkos_async<ExecutionSpace, decltype(views)>, h, views,
        n, launches_per_test, sync_type::fence);
    time_test<ExecutionSpace>(
        "kokkos_async_future",
        &test_for_loop_kokkos_async<ExecutionSpace, decltype(views)>, h, views,
        n, launches_per_test, sync_type::future);
    time_test<ExecutionSpace>(
        "hpx_async_fence",
        &test_for_loop_hpx_async<ExecutionSpace, decltype(views)>, h, views, n,
        launches_per_test, sync_type::fence);
    time_test<ExecutionSpace>(
        "hpx_async_future",
        &test_for_loop_hpx_async<ExecutionSpace, decltype(views)>, h, views, n,
        launches_per_test, sync_type::future);
  }
}

int hpx_main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);

  {
    hpx::kokkos::detail::polling_helper p;

    print_header();
    hpx::kokkos::kokkos_instance_helper<Kokkos::DefaultExecutionSpace> h;
    test_for_loop(h, 1, 10, 10);
    for (int n = 1; n <= 100000; n *= 10) {
      for (int l = 1; l <= (1 << 10); l *= 2) {
        test_for_loop(h, n, l, 3);
      }
    }
  }

  Kokkos::finalize();
  hpx::finalize();

  return 0;
}

int main(int argc, char *argv[]) {
  return hpx::init(hpx_main, argc, argv);
}

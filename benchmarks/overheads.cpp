//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Compares kernel launches using plain Kokkos to the asynchronous Kokkos
/// parallel algorithms and HPX parallel algorithms using Kokkos executors.
///
/// The test consists of launching a parallel for-loop which simply sets a
/// value in a Kokkos::View. The launches are synchronized using a Kokkos
/// fence, or futures if available. Each test can also launch multiple kernels
/// on the same execution space instance to see the effects of hiding latencies
/// of multiple launches.

#include <Kokkos_Core.hpp>
#include <hpx/config.hpp>
#include <hpx/kokkos.hpp>
#include <hpx/kokkos/detail/polling_helper.hpp>
#include <hpx/local/algorithm.hpp>
#include <hpx/local/chrono.hpp>
#include <hpx/local/init.hpp>

void print_header() {
  std::cout << "test_name,execution_space,subtest_name,vector_size,launches_"
               "per_test,time"
            << std::endl;
}

template <typename ExecutionSpace, typename F, typename Views, typename... Args>
void time_test(std::string const &label, F const &f, ExecutionSpace const &inst,
               Views const &views, int const n, int const launches_per_test,
               Args... args) {
  hpx::chrono::high_resolution_timer timer;
  f(inst, views, n, launches_per_test, std::forward<Args>(args)...);
  std::cout << "overhead," << inst.name() << "," << label << "," << n << ","
            << launches_per_test << "," << timer.elapsed() << std::endl;
}

// Plain Kokkos::parallel_for, with a fence at the end.
template <typename ExecutionSpace, typename Views>
void test_for_loop_kokkos(ExecutionSpace const &inst, Views const &views,
                          int const n, int const launches_per_test) {
  for (int l = 0; l < launches_per_test; ++l) {
    // Init-capture not allowed by nvcc, so we initialize a here.
    auto a = views[l];
    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(
                inst, 0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        [a] KOKKOS_IMPL_FUNCTION(int i) { a(i) = i; });
  }

  inst.fence();
}

// Plain Kokkos::parallel_for, synchronized with a future.
template <typename ExecutionSpace, typename Views>
void test_for_loop_kokkos_future(ExecutionSpace const &inst, Views const &views,
                                 int const n, int const launches_per_test) {
  for (int l = 0; l < launches_per_test; ++l) {
    // Init-capture not allowed by nvcc, so we initialize a here.
    auto a = views[l];
    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(
                inst, 0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        [a] KOKKOS_IMPL_FUNCTION(int i) { a(i) = i; });
  }

  hpx::kokkos::get_future<typename std::decay<ExecutionSpace>::type>().get();
}

enum class sync_type { fence, future };

// Futurized Kokkos::parallel_for, synchronized either with a fence or the
// returned futures.
template <typename ExecutionSpace, typename Views>
void test_for_loop_kokkos_async(ExecutionSpace const &inst, Views const &views,
                                int const n, int const launches_per_test,
                                sync_type s) {
  std::vector<hpx::shared_future<void>> futures;
  futures.reserve(launches_per_test);

  for (int l = 0; l < launches_per_test; ++l) {
    // Init-capture not allowed by nvcc, so we initialize a here.
    auto a = views[l];
    futures.push_back(hpx::kokkos::parallel_for_async(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(
                inst, 0, n),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        [a] KOKKOS_IMPL_FUNCTION(int i) { a(i) = i; }));
  }

  switch (s) {
  case sync_type::fence:
    inst.fence();
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
void test_for_loop_hpx_async(ExecutionSpace const &inst, Views const &views,
                             int const n, int const launches_per_test,
                             sync_type s) {
  std::vector<hpx::shared_future<void>> futures;
  futures.reserve(launches_per_test);

  hpx::kokkos::executor<typename std::decay<ExecutionSpace>::type> exec(inst);
  auto policy = hpx::kokkos::kok(hpx::execution::task).on(exec);

  for (int l = 0; l < launches_per_test; ++l) {
    futures.push_back(
        // TODO: This should use hpx::for_loop
        hpx::for_each(policy, views[l].data(),
                      views[l].data() + views[l].size(),
                      [] KOKKOS_IMPL_FUNCTION(int &x) { x = 42; }));
  }

  switch (s) {
  case sync_type::fence:
    inst.fence();
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
void test_for_loop(ExecutionSpace const &inst, int const n,
                   int const launches_per_test, int const repetitions) {
  std::vector<Kokkos::View<int *, typename std::decay<ExecutionSpace>::type>>
      views;
  views.reserve(launches_per_test);
  for (int l = 0; l < launches_per_test; ++l) {
    views.emplace_back("a", n);
  }

  for (int r = 0; r < repetitions; ++r) {
    time_test("kokkos", &test_for_loop_kokkos<decltype(inst), decltype(views)>,
              inst, views, n, launches_per_test);
    time_test("kokkos_future",
              &test_for_loop_kokkos_future<decltype(inst), decltype(views)>,
              inst, views, n, launches_per_test);
    time_test("kokkos_async_fence",
              &test_for_loop_kokkos_async<decltype(inst), decltype(views)>,
              inst, views, n, launches_per_test, sync_type::fence);
    time_test("kokkos_async_future",
              &test_for_loop_kokkos_async<decltype(inst), decltype(views)>,
              inst, views, n, launches_per_test, sync_type::future);
    time_test("hpx_async_fence",
              &test_for_loop_hpx_async<decltype(inst), decltype(views)>, inst,
              views, n, launches_per_test, sync_type::fence);
    time_test("hpx_async_future",
              &test_for_loop_hpx_async<decltype(inst), decltype(views)>, inst,
              views, n, launches_per_test, sync_type::future);
  }
}

int hpx_main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);

  {
    hpx::kokkos::detail::polling_helper p;

    print_header();
    hpx::kokkos::kokkos_instance_helper<> h;
    for (int n = 1; n <= 100000; n *= 10) {
      for (int l = 1; l <= (1 << 10); l *= 2) {
        test_for_loop(h.get_execution_space(), n, l, 3);
      }
    }
  }

  Kokkos::finalize();
  hpx::local::finalize();

  return 0;
}

int main(int argc, char *argv[]) {
  return hpx::local::init(hpx_main, argc, argv);
}

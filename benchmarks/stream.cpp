//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2015 Thomas Heller
//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This code is based on the STREAM benchmark:
// https://www.cs.virginia.edu/stream/ref.html

#include <Kokkos_Core.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/chrono.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/kokkos.hpp>
#include <hpx/kokkos/detail/polling_helper.hpp>

using elem_type = double;
using view_type = Kokkos::View<elem_type *>;
using host_view_type = Kokkos::View<elem_type *>::HostMirror;

void print_header() {
  std::cout << "test_name,execution_space,subtest_name,step_name,vector_size,"
               "element_size,num_stores_loads,time"
            << std::endl;
}

template <typename F, typename Step>
void time_test(std::string const &label, F const &f, Step const &step) {
  hpx::chrono::high_resolution_timer timer;
  f(step);
  std::cout << "stream," << Kokkos::DefaultExecutionSpace().name() << ","
            << label << "," << step.name << "," << step.a.extent(0) << ","
            << sizeof(elem_type) << "," << step.num_stores_loads << ","
            << timer.elapsed() << std::endl;
}

void init(view_type a, view_type b, view_type c, host_view_type ah,
          host_view_type bh, host_view_type ch) {
  Kokkos::parallel_for(
      "init",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, a.extent(0)),
      KOKKOS_LAMBDA(int i) {
        ah[i] = 1.0;
        bh[i] = 2.0;
        ch[i] = 0.0;
      });

  Kokkos::deep_copy(a, ah);
  Kokkos::deep_copy(b, bh);
  Kokkos::deep_copy(c, ch);
}

void check_results(view_type a, view_type b, view_type c, host_view_type ah,
                   host_view_type bh, host_view_type ch) {
  Kokkos::deep_copy(ah, a);
  Kokkos::deep_copy(bh, b);
  Kokkos::deep_copy(ch, c);

  elem_type aj, bj, cj, scalar;
  elem_type aSumErr, bSumErr, cSumErr;
  elem_type aAvgErr, bAvgErr, cAvgErr;

  double epsilon;
  int ierr, err;

  aj = 1.0;
  bj = 2.0;
  cj = 0.0;
  scalar = 3.0;

  cj = aj;
  bj = scalar * cj;
  cj = aj + bj;
  aj = bj + scalar * cj;

  aSumErr = 0.0;
  bSumErr = 0.0;
  cSumErr = 0.0;

  for (std::size_t j = 0; j < a.size(); j++) {
    aSumErr += std::abs(ah[j] - aj);
    bSumErr += std::abs(bh[j] - bj);
    cSumErr += std::abs(ch[j] - cj);
  }

  aAvgErr = aSumErr / (elem_type)a.size();
  bAvgErr = bSumErr / (elem_type)a.size();
  cAvgErr = cSumErr / (elem_type)a.size();

  if (sizeof(elem_type) == 4) {
    epsilon = 1.e-6;
  } else if (sizeof(elem_type) == 8) {
    epsilon = 1.e-13;
  } else {
    std::cerr << "unusual element size " << sizeof(elem_type) << std::endl;
    epsilon = 1.e-6;
  }

  err = 0;

  if (std::abs(aAvgErr / aj) > epsilon) {
    err++;
    std::cerr << "Failed Validation on array a, AvgRelAbsErr > epsilon ("
              << epsilon << ")" << std::endl;
    std::cerr << "Expected value: " << aj << ", AvgAbsErr: " << aAvgErr
              << ", AvgRelAbsErr: " << (std::abs(aAvgErr) / aj) << std::endl,
        ierr = 0;
    for (std::size_t j = 0; j < a.size(); j++) {
      if (std::abs(ah[j] / aj - 1.0) > epsilon) {
        ierr++;
        std::cerr << "a[" << j << "] = " << ah[j] << ", expected " << aj
                  << std::endl;
      }
    }
    std::cerr << "     For array a, " << ierr << " errors were found."
              << std::endl;
  }

  if (std::abs(bAvgErr / bj) > epsilon) {
    err++;
    std::cerr << "Failed Validation on array b, AvgRelAbsErr > epsilon ("
              << epsilon << ")" << std::endl;
    std::cerr << "Expected value: " << aj << ", AvgAbsErr: " << bAvgErr
              << ", AvgRelAbsErr: " << (std::abs(bAvgErr) / bj) << std::endl,
        ierr = 0;
    for (std::size_t j = 0; j < a.size(); j++) {
      if (std::abs(bh[j] / bj - 1.0) > epsilon) {
        ierr++;
        std::cerr << "b[" << j << "] = " << bh[j] << ", expected " << bj
                  << std::endl;
      }
    }
    std::cerr << "     For array b, " << ierr << " errors were found."
              << std::endl;
  }

  if (std::abs(cAvgErr / cj) > epsilon) {
    err++;
    std::cerr << "Failed Validation on array c, AvgRelAbsErr > epsilon ("
              << epsilon << ")" << std::endl;
    std::cerr << "Expected value: " << aj << ", AvgAbsErr: " << cAvgErr
              << ", AvgRelAbsErr: " << (std::abs(cAvgErr) / cj) << std::endl,
        ierr = 0;
    for (std::size_t j = 0; j < a.size(); j++) {
      if (std::abs(ch[j] / cj - 1.0) > epsilon) {
        ierr++;
        std::cerr << "c[" << j << "] = " << ch[j] << ", expected " << cj
                  << std::endl;
      }
    }
    std::cerr << "     For array c, " << ierr << " errors were found."
              << std::endl;
  }

  if (err != 0) {
    std::cerr << "Errors found" << std::endl;
    throw std::runtime_error("Solution does not validate");
  }
}

struct copy_step {
  view_type a;
  view_type b;
  view_type c;

  static constexpr int num_stores_loads = 2;
  static constexpr char const *name = "copy";

  KOKKOS_INLINE_FUNCTION void operator()(int i) const { c[i] = a[i]; }
};

struct scale_step {
  view_type a;
  view_type b;
  view_type c;

  static constexpr int num_stores_loads = 2;
  static constexpr char const *name = "scale";

  elem_type scalar = 3.0;

  KOKKOS_INLINE_FUNCTION void operator()(int i) const { b[i] = scalar * c[i]; }
};

struct add_step {
  view_type a;
  view_type b;
  view_type c;

  static constexpr int num_stores_loads = 3;
  static constexpr char const *name = "add";

  KOKKOS_INLINE_FUNCTION void operator()(int i) const { c[i] = a[i] + b[i]; }
};

struct triad_step {
  view_type a;
  view_type b;
  view_type c;

  static constexpr int num_stores_loads = 3;
  static constexpr char const *name = "triad";

  elem_type scalar = 3.0;

  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    a[i] = b[i] + scalar * c[i];
  }
};

// Plain Kokkos::parallel_for.
template <typename Step> void test_stream_kokkos_fence_impl(Step step) {
  Kokkos::parallel_for(Kokkos::RangePolicy<>(0, step.a.extent(0)), step);
  Kokkos::fence();
}

void test_stream_kokkos_fence(std::string const &label, view_type a,
                              view_type b, view_type c) {
  time_test(label, &test_stream_kokkos_fence_impl<copy_step>,
            copy_step{a, b, c});
  time_test(label, &test_stream_kokkos_fence_impl<scale_step>,
            scale_step{a, b, c});
  time_test(label, &test_stream_kokkos_fence_impl<add_step>, add_step{a, b, c});
  time_test(label, &test_stream_kokkos_fence_impl<triad_step>,
            triad_step{a, b, c});
}

// Plain Kokkos::parallel_for using a future for synchronization.
template <typename Step> void test_stream_kokkos_future_impl(Step step) {
  Kokkos::parallel_for(Kokkos::RangePolicy<>(0, step.a.extent(0)), step);
  hpx::kokkos::get_future<>().get();
}

void test_stream_kokkos_future(std::string const &label, view_type a,
                               view_type b, view_type c) {
  time_test(label, &test_stream_kokkos_future_impl<copy_step>,
            copy_step{a, b, c});
  time_test(label, &test_stream_kokkos_future_impl<scale_step>,
            scale_step{a, b, c});
  time_test(label, &test_stream_kokkos_future_impl<add_step>,
            add_step{a, b, c});
  time_test(label, &test_stream_kokkos_future_impl<triad_step>,
            triad_step{a, b, c});
}

// Futurized Kokkos::parallel_for using fence for synchronization.
template <typename Step> void test_stream_kokkos_async_fence_impl(Step step) {
  hpx::kokkos::parallel_for_async(Kokkos::RangePolicy<>(0, step.a.extent(0)),
                                  step);
  Kokkos::fence();
}

void test_stream_kokkos_async_fence(std::string const &label, view_type a,
                                    view_type b, view_type c) {
  time_test(label, &test_stream_kokkos_async_fence_impl<copy_step>,
            copy_step{a, b, c});
  time_test(label, &test_stream_kokkos_async_fence_impl<scale_step>,
            scale_step{a, b, c});
  time_test(label, &test_stream_kokkos_async_fence_impl<add_step>,
            add_step{a, b, c});
  time_test(label, &test_stream_kokkos_async_fence_impl<triad_step>,
            triad_step{a, b, c});
}

// Futurized Kokkos::parallel_for.
template <typename Step> void test_stream_kokkos_async_future_impl(Step step) {
  hpx::kokkos::parallel_for_async(Kokkos::RangePolicy<>(0, step.a.extent(0)),
                                  step)
      .get();
}

void test_stream_kokkos_async_future(std::string const &label, view_type a,
                                     view_type b, view_type c) {
  time_test(label, &test_stream_kokkos_async_future_impl<copy_step>,
            copy_step{a, b, c});
  time_test(label, &test_stream_kokkos_async_future_impl<scale_step>,
            scale_step{a, b, c});
  time_test(label, &test_stream_kokkos_async_future_impl<add_step>,
            add_step{a, b, c});
  time_test(label, &test_stream_kokkos_async_future_impl<triad_step>,
            triad_step{a, b, c});
}

// Synchronous HPX for_loop.
template <typename Step> void test_stream_hpx_impl(Step step) {
  hpx::for_loop(hpx::kokkos::kok, 0, step.a.extent(0), step);
}

void test_stream_hpx(std::string const &label, view_type a, view_type b,
                     view_type c) {
  time_test(label, &test_stream_hpx_impl<copy_step>, copy_step{a, b, c});
  time_test(label, &test_stream_hpx_impl<scale_step>, scale_step{a, b, c});
  time_test(label, &test_stream_hpx_impl<add_step>, add_step{a, b, c});
  time_test(label, &test_stream_hpx_impl<triad_step>, triad_step{a, b, c});
}

// Asynchronous HPX for_loop.
template <typename Step> void test_stream_hpx_future_impl(Step step) {
  hpx::for_loop(hpx::kokkos::kok(hpx::execution::task), 0, step.a.extent(0),
                step)
      .get();
}

void test_stream_hpx_future(std::string const &label, view_type a, view_type b,
                            view_type c) {
  time_test(label, &test_stream_hpx_future_impl<copy_step>, copy_step{a, b, c});
  time_test(label, &test_stream_hpx_future_impl<scale_step>,
            scale_step{a, b, c});
  time_test(label, &test_stream_hpx_future_impl<add_step>, add_step{a, b, c});
  time_test(label, &test_stream_hpx_future_impl<triad_step>,
            triad_step{a, b, c});
}

void test_stream(int repetitions, int size) {
  view_type a("a", size);
  view_type b("b", size);
  view_type c("c", size);

  host_view_type ah = Kokkos::create_mirror_view(a);
  host_view_type bh = Kokkos::create_mirror_view(b);
  host_view_type ch = Kokkos::create_mirror_view(c);

  double scalar = 3.0;

  for (int i = 0; i < repetitions; ++i) {
    init(a, b, c, ah, bh, ch);
    test_stream_kokkos_fence("kokkos_fence", a, b, c);
    check_results(a, b, c, ah, bh, ch);

    init(a, b, c, ah, bh, ch);
    test_stream_kokkos_future("kokkos_future", a, b, c);
    check_results(a, b, c, ah, bh, ch);

    init(a, b, c, ah, bh, ch);
    test_stream_kokkos_async_fence("kokkos_async_fence", a, b, c);
    check_results(a, b, c, ah, bh, ch);

    init(a, b, c, ah, bh, ch);
    test_stream_kokkos_async_future("kokkos_async_future", a, b, c);
    check_results(a, b, c, ah, bh, ch);

    init(a, b, c, ah, bh, ch);
    test_stream_hpx("hpx", a, b, c);
    check_results(a, b, c, ah, bh, ch);

    init(a, b, c, ah, bh, ch);
    test_stream_hpx_future("hpx_future", a, b, c);
    check_results(a, b, c, ah, bh, ch);
  }
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);

  {
    hpx::kokkos::detail::polling_helper p;

    print_header();
    for (int size = 1024; size <= (1024 << 17); size *= 2) {
      test_stream(10, size);
    }
  }

  Kokkos::finalize();

  return 0;
}

//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Tests HPX parallel algorithms using Kokkos executors.

#include "test.hpp"

#include <hpx/kokkos.hpp>

#include <hpx/algorithm.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/numeric.hpp>

template <typename Executor> void test_for_each(Executor &&exec) {
  int const n = 43;

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> for_each_data_host(
      "for_each_data_host", n);
  Kokkos::View<int *, typename std::decay<Executor>::type::execution_space>
      for_each_data("for_each_data", n);
  for (std::size_t i = 0; i < n; ++i) {
    for_each_data_host(i) = i;
  }
  Kokkos::deep_copy(for_each_data, for_each_data_host);

  hpx::for_each(
      hpx::kokkos::kokkos_policy{}.on(exec), for_each_data.data(),
      for_each_data.data() + for_each_data.size(),
      KOKKOS_LAMBDA(int &x) { x *= 2; });

  Kokkos::deep_copy(for_each_data_host, for_each_data);

  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(for_each_data_host(i) == 2 * i);
  }

  auto f = hpx::for_each(
      hpx::kokkos::kokkos_policy{}(hpx::execution::task).on(exec),
      for_each_data.data(), for_each_data.data() + for_each_data.size(),
      KOKKOS_LAMBDA(int &x) { x *= 3; });

  f.get();

  Kokkos::deep_copy(for_each_data_host, for_each_data);

  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(for_each_data_host(i) == 3 * 2 * i);
  }

  auto f2 = hpx::for_each(
      hpx::kokkos::kokkos_policy{}(hpx::execution::task).on(exec),
      hpx::util::make_counting_iterator<>(0),
      hpx::util::make_counting_iterator<>(n),
      KOKKOS_LAMBDA(int i) { for_each_data(i) = i; });

  f2.get();

  Kokkos::deep_copy(for_each_data_host, for_each_data);

  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(for_each_data_host(i) == i);
  }
}

template <typename Executor> void test_for_each_range(Executor &&exec) {
  int const n = 43;
  Kokkos::RangePolicy<> const p(0, n);

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> for_each_data_host(
      "for_each_data_host", n);
  Kokkos::View<int *, typename std::decay<Executor>::type::execution_space>
      for_each_data("for_each_data", n);
  for (std::size_t i = 0; i < n; ++i) {
    for_each_data_host(i) = 0;
  }
  Kokkos::deep_copy(for_each_data, for_each_data_host);

  auto f = hpx::ranges::for_each(
      hpx::kokkos::kokkos_policy{}(hpx::execution::task).on(exec), p,
      KOKKOS_LAMBDA(int i) { for_each_data(i) = i; });

  f.get();

  Kokkos::deep_copy(for_each_data_host, for_each_data);

  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(for_each_data_host(i) == i);
  }
}

template <typename Executor> void test_for_each_mdrange(Executor &&exec) {
  int const n = 43;
  int const m = 17;
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> const p({0, 0}, {n, m});

  Kokkos::View<
      int **,
      typename std::decay<Executor>::type::execution_space::array_layout,
      Kokkos::DefaultHostExecutionSpace>
      for_each_data_host("for_each_data_host", n, m);
  Kokkos::View<int **, typename std::decay<Executor>::type::execution_space>
      for_each_data("for_each_data", n, m);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      for_each_data_host(i, j) = 0;
    }
  }
  Kokkos::deep_copy(for_each_data, for_each_data_host);

  auto f = hpx::ranges::for_each(
      hpx::kokkos::kokkos_policy{}(hpx::execution::task).on(exec), p,
      KOKKOS_LAMBDA(int i, int j) { for_each_data(i, j) = i + j; });

  f.get();

  Kokkos::deep_copy(for_each_data_host, for_each_data);

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      HPX_KOKKOS_DETAIL_TEST(for_each_data_host(i, j) == i + j);
    }
  }
}

void test_for_each_default() {
  int const n = 43;

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> for_each_data_host(
      "for_each_data_host", n);
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> for_each_data(
      "for_each_data", n);
  for (std::size_t i = 0; i < n; ++i) {
    for_each_data_host(i) = i;
  }
  Kokkos::deep_copy(for_each_data, for_each_data_host);

  hpx::for_each(
      hpx::kokkos::kokkos_policy{}, for_each_data.data(),
      for_each_data.data() + for_each_data.size(),
      KOKKOS_LAMBDA(int &x) { x *= 2; });

  Kokkos::deep_copy(for_each_data_host, for_each_data);

  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(for_each_data_host(i) == 2 * i);
  }

  auto f = hpx::for_each(
      hpx::kokkos::kokkos_policy{}(hpx::execution::task), for_each_data.data(),
      for_each_data.data() + for_each_data.size(),
      KOKKOS_LAMBDA(int &x) { x *= 3; });

  f.get();

  Kokkos::deep_copy(for_each_data_host, for_each_data);

  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(for_each_data_host(i) == 3 * 2 * i);
  }

  auto f2 = hpx::for_each(
      hpx::kokkos::kokkos_policy{}(hpx::execution::task),
      hpx::util::make_counting_iterator<>(0),
      hpx::util::make_counting_iterator<>(n),
      KOKKOS_LAMBDA(int i) { for_each_data(i) = i; });

  f2.get();

  Kokkos::deep_copy(for_each_data_host, for_each_data);

  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(for_each_data_host(i) == i);
  }
}

void test_for_each_default_range() {
  int const n = 43;
  Kokkos::RangePolicy<> const p(0, n);

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> for_each_data_host(
      "for_each_data_host", n);
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> for_each_data(
      "for_each_data", n);
  for (std::size_t i = 0; i < n; ++i) {
    for_each_data_host(i) = 0;
  }
  Kokkos::deep_copy(for_each_data, for_each_data_host);

  auto f = hpx::ranges::for_each(
      hpx::kokkos::kokkos_policy{}(hpx::execution::task), p,
      KOKKOS_LAMBDA(int i) { for_each_data(i) = i; });

  f.get();

  Kokkos::deep_copy(for_each_data_host, for_each_data);

  for (std::size_t i = 0; i < n; ++i) {
    HPX_KOKKOS_DETAIL_TEST(for_each_data_host(i) == i);
  }
}

void test_for_each_default_mdrange() {
  int const n = 43;
  int const m = 17;
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> const p({0, 0}, {n, m});

  Kokkos::View<int **, typename Kokkos::DefaultExecutionSpace::array_layout,
               Kokkos::DefaultHostExecutionSpace>
      for_each_data_host("for_each_data_host", n, m);
  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> for_each_data(
      "for_each_data", n, m);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      for_each_data_host(i, j) = 0;
    }
  }
  Kokkos::deep_copy(for_each_data, for_each_data_host);

  auto f = hpx::ranges::for_each(
      hpx::kokkos::kokkos_policy{}(hpx::execution::task), p,
      KOKKOS_LAMBDA(int i, int j) { for_each_data(i, j) = i + j; });

  f.get();

  Kokkos::deep_copy(for_each_data_host, for_each_data);

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      HPX_KOKKOS_DETAIL_TEST(for_each_data_host(i, j) == i + j);
    }
  }
}

template <typename Executor> void test_reduce(Executor &&exec) {
  int const n = 43;

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> reduce_data_host(
      "reduce_data_host", n);
  Kokkos::View<int *, typename std::decay<Executor>::type::execution_space>
      reduce_data("reduce_data", n);
  for (std::size_t i = 0; i < n; ++i) {
    reduce_data_host(i) = i;
  }
  Kokkos::deep_copy(reduce_data, reduce_data_host);

  int offset = -3;
  int result = hpx::reduce(
      hpx::kokkos::kokkos_policy{}.on(exec), reduce_data.data(),
      reduce_data.data() + reduce_data.size(), offset,
      KOKKOS_LAMBDA(int x, int y) { return x + y; });

  HPX_KOKKOS_DETAIL_TEST(result == (offset + (n * (n - 1)) / 2));

  hpx::future<int> f_result = hpx::reduce(
      hpx::kokkos::kokkos_policy{}(hpx::execution::task).on(exec),
      reduce_data.data(), reduce_data.data() + reduce_data.size(), offset,
      KOKKOS_LAMBDA(int x, int y) { return x + y; });

  HPX_KOKKOS_DETAIL_TEST(f_result.get() == (offset + (n * (n - 1)) / 2));
}

void test_reduce_default() {
  int const n = 43;

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> reduce_data_host(
      "reduce_data_host", n);
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> reduce_data("reduce_data",
                                                                 n);
  for (std::size_t i = 0; i < n; ++i) {
    reduce_data_host(i) = i;
  }
  Kokkos::deep_copy(reduce_data, reduce_data_host);

  int offset = -3;
  int result = hpx::reduce(
      hpx::kokkos::kokkos_policy{}, reduce_data.data(),
      reduce_data.data() + reduce_data.size(), offset,
      KOKKOS_LAMBDA(int x, int y) { return x + y; });

  HPX_KOKKOS_DETAIL_TEST(result == (offset + (n * (n - 1)) / 2));

  hpx::future<int> f_result = hpx::reduce(
      hpx::kokkos::kokkos_policy{}(hpx::execution::task), reduce_data.data(),
      reduce_data.data() + reduce_data.size(), offset,
      KOKKOS_LAMBDA(int x, int y) { return x + y; });

  HPX_KOKKOS_DETAIL_TEST(f_result.get() == (offset + (n * (n - 1)) / 2));
}

template <typename Executor> void test(Executor &&exec) {
  static_assert(hpx::kokkos::is_kokkos_executor<Executor>::value,
                "Executor is not a Kokkos executor");
  static_assert(
      Kokkos::is_execution_space<typename Executor::execution_space>::value,
      "Executor::execution_space is not a Kokkos execution space");

  std::cout << "testing executor with execution space \""
            << exec.instance().name() << "\"" << std::endl;

  test_for_each(exec);
  test_for_each_range(exec);
  test_for_each_mdrange(exec);
  test_reduce(exec);
}

void test_default() {
  std::cout << "testing default execution space \""
            << Kokkos::DefaultExecutionSpace().name() << "\"" << std::endl;

  test_for_each_default();
  test_for_each_default_range();
  test_for_each_default_mdrange();
  test_reduce_default();
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  test(hpx::kokkos::default_executor{});
  if (!std::is_same<hpx::kokkos::default_executor,
                    hpx::kokkos::default_host_executor>::value) {
    test(hpx::kokkos::default_host_executor{});
  }
  test_default();
  Kokkos::finalize();

  return hpx::kokkos::detail::report_errors();
}

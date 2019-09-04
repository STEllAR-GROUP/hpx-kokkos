//  Copyright (c) 2019 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Contains HPX executors that forward to a Kokkos backend.

#ifndef HPX_COMPUTE_KOKKOS_EXECUTORS_HPP
#define HPX_COMPUTE_KOKKOS_EXECUTORS_HPP

#include <hpx/include/iostreams.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace hpx {
namespace compute {
namespace kokkos {

// TODO: Specialize for each ExecutionSpace?
template <typename ExecutionSpace> struct kokkos_executor;

template <typename ExecutionSpace> struct kokkos_executor {
  using execution_space = ExecutionSpace;

  using executor_parameters_type = hpx::parallel::execution::static_chunk_size;

  std::size_t processing_units_count() const {
    return execution_space::concurrency();
  }

  // This is one-way, single-task execution. This cheats and blocks.
  template <typename F, typename... Ts> void post(F &&f, Ts &&... ts) const {
    std::cout << "kokkos_executor::post\n";
    sync_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
  }

  template <typename F, typename... Ts>
  hpx::future<
      typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
  async_execute(F &&f, Ts &&... ts) const {
    std::cout << "kokkos_executor::async_execute\n";
    return hpx::make_ready_future<>(
        sync_execute(std::forward<F>(f), std::forward<Ts>(ts)...));
  }

  template <typename F, typename... Ts>
  typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
  sync_execute(F &&f, Ts &&... ts) const {
    std::cout << "kokkos_executor::sync_execute\n";
    return hpx::util::invoke(std::forward<F>(f), std::forward<Ts>(ts)...);
  }

  template <typename F, typename Shape, typename... Ts>
  std::vector<hpx::future<typename parallel::execution::detail::
                              bulk_function_result<F, Shape, Ts...>::type>>
  bulk_async_execute(F &&f, Shape const &shape, Ts &&... ts) const {
    std::cout << "kokkos_executor::bulk_async_execute\n";

    using result_type = std::vector<
        hpx::future<typename hpx::parallel::execution::detail::
                        bulk_function_result<F, Shape, Ts...>::type>>;

    result_type results;
    std::size_t size = hpx::util::size(shape);
    results.resize(size);

    using value_type = typename hpx::traits::range_traits<Shape>::value_type;

    std::vector<value_type> const chunks(hpx::util::begin(shape),
                                         hpx::util::end(shape));

    // TODO: Make this work on GPU as well.
    //Kokkos::parallel_for(
    //    Kokkos::RangePolicy<execution_space>(0, size),
    //    KOKKOS_LAMBDA(std::size_t i) {
    //      std::cout << "calling chunk function with "
    //                   "argument i = "
    //                << i << std::endl;
    //      hpx::util::invoke(std::forward<F>(f), chunks[i],
    //                        std::forward<Ts>(ts)...);
    //    });

    // TODO: Return a future.

    return results;
  }

  template <typename F, typename Shape, typename... Ts>
  typename parallel::execution::detail::bulk_execute_result<F, Shape,
                                                            Ts...>::type
  bulk_sync_execute(F &&f, Shape const &shape, Ts &&... ts) const {
    std::cout << "kokkos_executor::bulk_sync_execute\n";
    typename parallel::execution::detail::bulk_execute_result<
        F, Shape, Ts...>::type results;
    std::size_t size = hpx::util::size(shape);
    results.reserve(size);

    using value_type = typename hpx::traits::range_traits<Shape>::value_type;

    std::vector<value_type> const chunks(hpx::util::begin(shape),
                                         hpx::util::end(shape));

    // TODO: Make this work on GPU as well.
    //Kokkos::parallel_for(
    //    Kokkos::RangePolicy<execution_space>(0, size),
    //    KOKKOS_LAMBDA(std::size_t i) {
    //      std::cout << "calling chunk function with "
    //                   "argument i = "
    //                << i << std::endl;
    //      hpx::util::invoke(std::forward<F>(f), chunks[i],
    //                        std::forward<Ts>(ts)...);
    //    });

    execution_space().fence();

    return {};
  }
};

using default_executor = kokkos_executor<Kokkos::DefaultExecutionSpace>;
using host_executor = kokkos_executor<Kokkos::DefaultHostExecutionSpace>;

// The following execution spaces are only conditionally enabled in Kokkos.
#if defined(KOKKOS_ENABLE_SERIAL)
using serial_executor = kokkos_executor<Kokkos::Serial>;
#endif

#if defined(KOKKOS_ENABLE_HPX)
using hpx_executor = kokkos_executor<Kokkos::Experimental::HPX>;
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
using openmp_executor = kokkos_executor<Kokkos::OpenMP>;
#endif

#if defined(KOKKOS_ENABLE_CUDA)
using cuda_executor = kokkos_executor<Kokkos::Cuda>;
#endif

#if defined(KOKKOS_ENABLE_ROCM)
using rocm_executor = kokkos_executor<Kokkos::ROCm>;
#endif

template <typename Executor>
struct is_kokkos_executor : std::false_type {};

template <typename ExecutionSpace>
struct is_kokkos_executor<kokkos_executor<ExecutionSpace>>
    : std::true_type {};
} // namespace kokkos
} // namespace compute
} // namespace hpx

namespace hpx {
namespace parallel {
namespace execution {
template <>
struct executor_execution_category<compute::kokkos::serial_executor> {
  typedef parallel::execution::sequenced_execution_tag type;
};

template <> struct executor_execution_category<compute::kokkos::hpx_executor> {
  typedef parallel::execution::parallel_execution_tag type;
};

#if defined(KOKKOS_ENABLE_CUDA)
template <>
struct executor_execution_category<compute::kokkos::cuda_executor> {
  typedef parallel::execution::parallel_execution_tag type;
};

template <>
struct is_one_way_executor<compute::kokkos::cuda_executor> : std::true_type {};
#endif

template <>
struct is_one_way_executor<compute::kokkos::host_executor> : std::true_type {};

// NOTE: Kokkos executors don't return futures, thus no two-way. But,
// hpx::compute::kokkos::hpx_executor could be two-way. We could also block and
// return ready futures for other backends.
// template <>
// struct is_two_way_executor<compute::kokkos::serial_executor>
//   : std::true_type
// {
// };
template <>
struct is_bulk_one_way_executor<compute::kokkos::serial_executor>
    : std::true_type {};

// template <>
// struct is_bulk_two_way_executor<compute::kokkos::serial_executor>
//   : std::true_type
// {
// };
} // namespace execution
} // namespace parallel
} // namespace hpx

#endif

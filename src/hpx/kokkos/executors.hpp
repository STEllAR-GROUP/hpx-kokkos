//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Contains HPX executors that forward to a Kokkos backend.

#pragma once

#include <hpx/kokkos/config.hpp>
#include <hpx/kokkos/deep_copy.hpp>
#include <hpx/kokkos/detail/logging.hpp>
#include <hpx/kokkos/kokkos_algorithms.hpp>
#include <hpx/kokkos/make_instance.hpp>

#include <hpx/local/algorithm.hpp>
#include <hpx/local/numeric.hpp>
#include <hpx/local/tuple.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace hpx {
namespace kokkos {
namespace detail {
template <std::size_t... Is, typename F, typename A, typename Tuple>
HPX_HOST_DEVICE void invoke_helper(hpx::util::index_pack<Is...>, F &&f, A &&a,
                                   Tuple &&t) {
#if HPX_VERSION_FULL > 0x010801
  hpx::invoke_r<void>(std::forward<F>(f), std::forward<A>(a),
                      hpx::get<Is>(std::forward<Tuple>(t))...);
#else
  hpx::util::invoke_r<void>(std::forward<F>(f), std::forward<A>(a),
                      hpx::get<Is>(std::forward<Tuple>(t))...);
#endif
}
} // namespace detail

/// \brief The mode of an executor. Determines whether an executor should be
/// constructed with the global/default Kokkos execution space instance, or if
/// it should be independent (when possible).
enum class execution_space_mode { global, independent };

/// \brief HPX executor wrapping a Kokkos execution space.
template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class executor {
public:
  using execution_space = ExecutionSpace;
  using execution_category = hpx::execution::parallel_execution_tag;

  explicit executor(execution_space_mode mode = execution_space_mode::global)
      : inst(mode == execution_space_mode::global
                 ? ExecutionSpace{}
                 : detail::make_independent_execution_space_instance<
                       ExecutionSpace>()) {}
  explicit executor(execution_space const &instance) : inst(instance) {}

  execution_space instance() const { return inst; }

  template <typename F, typename... Ts> void post(F &&f, Ts &&...ts) {
    auto ts_pack = hpx::make_tuple(std::forward<Ts>(ts)...);
    parallel_for_async(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<execution_space>(inst, 0, 1),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
#if HPX_VERSION_FULL > 0x010801
        KOKKOS_LAMBDA(int) { hpx::invoke_fused_r<void>(f, ts_pack); });
#else
        KOKKOS_LAMBDA(int) { hpx::util::invoke_fused_r<void>(f, ts_pack); });
#endif
  }

  template <typename F, typename... Ts>
  hpx::shared_future<void> async_execute(F &&f, Ts &&...ts) {
    auto ts_pack = hpx::make_tuple(std::forward<Ts>(ts)...);
    return parallel_for_async(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<execution_space>(inst, 0, 1),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
#if HPX_VERSION_FULL > 0x010801
        KOKKOS_LAMBDA(int) { hpx::invoke_fused_r<void>(f, ts_pack); });
#else
        KOKKOS_LAMBDA(int) { hpx::util::invoke_fused_r<void>(f, ts_pack); });
#endif
  }

  template <typename F, typename S, typename... Ts>
  std::vector<hpx::shared_future<void>> bulk_async_execute(F &&f, S const &s,
                                                           Ts &&...ts) {
    HPX_KOKKOS_DETAIL_LOG("bulk_async_execute");
    auto ts_pack = hpx::make_tuple(std::forward<Ts>(ts)...);
    auto size = hpx::util::size(s);
    auto b = hpx::util::begin(s);

    return {parallel_for_async(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<ExecutionSpace>(inst, 0, size),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int i) {
          HPX_KOKKOS_DETAIL_LOG("bulk_async_execute i = %d", i);
          using index_pack_type =
#if HPX_VERSION_FULL > 0x010801
            typename hpx::detail::fused_index_pack<decltype(ts_pack)>::type;
#else
            typename hpx::util::detail::fused_index_pack<decltype(ts_pack)>::type;
#endif
          detail::invoke_helper(index_pack_type{}, f, *(b + i), ts_pack);
        })};
  }

  hpx::shared_future<void> get_future() {
    return detail::get_future<typename std::decay<ExecutionSpace>::type>::call(
        std::forward<ExecutionSpace>(inst));
  }

  template <typename Parameters, typename F>
  constexpr std::size_t get_chunk_size(Parameters &&params, F &&f,
                                       std::size_t cores,
                                       std::size_t count) const {
    return std::size_t(-1);
  }

private:
  execution_space inst{};
};

// Define type aliases
using default_executor = executor<Kokkos::DefaultExecutionSpace>;
using default_host_executor = executor<Kokkos::DefaultHostExecutionSpace>;

#if defined(KOKKOS_ENABLE_CUDA)
using cuda_executor = executor<Kokkos::Cuda>;
#endif

#if defined(KOKKOS_ENABLE_HIP)
using hip_executor = executor<Kokkos::Experimental::HIP>;
#endif

#if defined(KOKKOS_ENABLE_SYCL)
using sycl_executor = executor<Kokkos::Experimental::SYCL>;
#endif

#if defined(KOKKOS_ENABLE_HPX)
using hpx_executor = executor<Kokkos::Experimental::HPX>;
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
using openmp_executor = executor<Kokkos::OpenMP>;
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
using serial_executor = executor<Kokkos::Serial>;
#endif

template <typename Executor> struct is_kokkos_executor : std::false_type {};

template <typename ExecutionSpace>
struct is_kokkos_executor<executor<ExecutionSpace>> : std::true_type {};
} // namespace kokkos
} // namespace hpx

namespace HPXKOKKOS_HPX_EXECUTOR_NS {
template <typename ExecutionSpace>
struct is_one_way_executor<hpx::kokkos::executor<ExecutionSpace>>
    : std::true_type {};

template <typename ExecutionSpace>
struct is_two_way_executor<hpx::kokkos::executor<ExecutionSpace>>
    : std::true_type {};

template <typename ExecutionSpace>
struct is_bulk_two_way_executor<hpx::kokkos::executor<ExecutionSpace>>
    : std::true_type {};
} // namespace HPXKOKKOS_HPX_EXECUTOR_NS

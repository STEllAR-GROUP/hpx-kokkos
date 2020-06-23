//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Contains HPX executors that forward to a Kokkos backend.

#pragma once

#include <hpx/kokkos/deep_copy.hpp>
#include <hpx/kokkos/detail/logging.hpp>
#include <hpx/kokkos/parallel.hpp>

#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace hpx {
namespace kokkos {
namespace detail {
template <std::size_t... Is, typename F, typename A, typename Tuple>
HPX_HOST_DEVICE void invoke_helper(hpx::util::index_pack<Is...>, F &&f, A &&a,
                                   Tuple &&t) {
  hpx::util::invoke_r<void>(std::forward<F>(f), std::forward<A>(a),
                            hpx::util::get<Is>(std::forward<Tuple>(t))...);
}

template <typename ShapeValue, typename Enable = void> struct bulk_helper {
  template <typename ExecutionSpace, typename F, typename S, typename... Ts>
  static std::vector<hpx::future<void>> call(ExecutionSpace inst, F &&f,
                                             S const &s, Ts &&... ts) {
    HPX_KOKKOS_DETAIL_LOG("bulk_async_execute copy-shape-variant");
    using value_type = typename hpx::traits::range_traits<S>::value_type;
    auto size = hpx::util::size(s);
    Kokkos::View<value_type *, Kokkos::DefaultHostExecutionSpace> host_shape(
        "kokkos_executor::bulk_async_execute host_shape", size);
    Kokkos::View<value_type *, ExecutionSpace> device_shape(
        "kokkos_executor::bulk_async_execute device_shape", size);

    auto b = hpx::util::begin(s);
    for (int i = 0; i < size; ++i) {
      host_shape(i) = *(b++);
    }

    HPX_KOKKOS_DETAIL_LOG("size = %zu", size);

    auto ts_pack = hpx::util::make_tuple(std::forward<Ts>(ts)...);

    HPX_KOKKOS_DETAIL_LOG("bulk_async_execute deep_copy");
    deep_copy(inst, device_shape, host_shape);

    HPX_KOKKOS_DETAIL_LOG("bulk_async_execute parallel_for_async");
    auto fut = parallel_for_async(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<ExecutionSpace>(inst, 0, size),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int i) {
          HPX_KOKKOS_DETAIL_LOG("bulk_async_execute i = %d", i);
          using index_pack_type =
              typename hpx::util::detail::fused_index_pack<decltype(
                  ts_pack)>::type;

          detail::invoke_helper(index_pack_type{}, f, device_shape(i), ts_pack);
        });
    fut.wait();

    std::vector<hpx::future<void>> result;
    // TODO: The empty continuation is used to extend the lifetime of
    // device_shape and to return a future from a shared_future. However, for
    // certain memory spaces it will still block the whole execution space on
    // destruction. Ways around this?
    result.push_back(fut.then(hpx::launch::sync, [device_shape](auto &&) {}));
    return result;
  }
};

template <typename IntegralOrIterator>
struct bulk_helper<
    hpx::util::tuple<IntegralOrIterator, std::size_t, std::size_t>,
    typename std::enable_if<
        std::is_integral<IntegralOrIterator>::value ||
        hpx::traits::is_iterator<IntegralOrIterator>::value>::type> {
  template <typename ExecutionSpace, typename F, typename S, typename... Ts>
  static std::vector<hpx::future<void>> call(ExecutionSpace inst, F &&f,
                                             S const &s, Ts &&... ts) {
    HPX_KOKKOS_DETAIL_LOG(
        "bulk_async_execute custom-algorithms-variant integral/iterator");
    using value_type = typename hpx::traits::range_traits<S>::value_type;

    // We override get_chunk_size for parallel algorithms. This lets Kokkos do
    // the chunking itself.
    auto size = hpx::util::size(s);
    HPX_ASSERT(size == 1);

    auto ts_pack = hpx::util::make_tuple(std::forward<Ts>(ts)...);

    auto chunk = *hpx::util::begin(s);
    auto begin = hpx::util::get<0>(chunk);
    std::size_t const chunk_size = hpx::util::get<1>(chunk);

    auto fut = parallel_for_async(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<ExecutionSpace>(inst, 0, chunk_size),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int i) {
          HPX_KOKKOS_DETAIL_LOG("bulk_async_execute i = %d", i);
          using index_pack_type =
              typename hpx::util::detail::fused_index_pack<decltype(
                  ts_pack)>::type;
          detail::invoke_helper(index_pack_type{}, f,
                                value_type(begin + i, 1, i), ts_pack);
        });

    std::vector<hpx::future<void>> result;
    // TODO: The empty continuation is only used to get a future from a shared
    // future.
    result.push_back(fut.then(hpx::launch::sync, [](auto &&) {}));
    return result;
  }
};
} // namespace detail

/// \brief HPX executor wrapping a Kokkos execution space.
template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class executor {
public:
  using execution_space = ExecutionSpace;
  using execution_category = hpx::parallel::execution::parallel_execution_tag;

  explicit executor(execution_space const &instance = {}) : inst(instance) {}

  execution_space instance() { return inst; }

  template <typename F, typename... Ts> void post(F &&f, Ts &&... ts) {
    auto ts_pack = hpx::util::make_tuple(std::forward<Ts>(ts)...);
    parallel_for_async(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<execution_space>(inst, 0, 1),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int) { hpx::util::invoke_fused_r<void>(f, ts_pack); });
  }

  template <typename F, typename... Ts>
  hpx::shared_future<void> async_execute(F &&f, Ts &&... ts) {
    auto ts_pack = hpx::util::make_tuple(std::forward<Ts>(ts)...);
    return parallel_for_async(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<execution_space>(inst, 0, 1),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int) { hpx::util::invoke_fused_r<void>(f, ts_pack); });
  }

  template <typename F, typename S, typename... Ts>
  std::vector<hpx::future<void>> bulk_async_execute(F &&f, S const &s,
                                                    Ts &&... ts) {
    return detail::
        bulk_helper<typename hpx::traits::range_traits<S>::value_type>::call(
            inst, std::forward<F>(f), s, std::forward<Ts>(ts)...);
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

#if defined(KOKKOS_ENABLE_HPX)
using hpx_executor = executor<Kokkos::Experimental::HPX>;
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
using openmp_executor = executor<Kokkos::OpenMP>;
#endif

#if defined(KOKKOS_ENABLE_ROCM)
using rocm_executor = executor<Kokkos::ROCm>;
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
using serial_executor = executor<Kokkos::Serial>;
#endif

template <typename Executor> struct is_kokkos_executor : std::false_type {};

template <typename ExecutionSpace>
struct is_kokkos_executor<executor<ExecutionSpace>> : std::true_type {};

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
hpx::kokkos::executor<typename std::decay<ExecutionSpace>::type>
make_executor() {
  return {hpx::kokkos::make_execution_space<
      typename std::decay<ExecutionSpace>::type>};
}
} // namespace kokkos
} // namespace hpx

namespace hpx {
namespace parallel {
namespace execution {
template <typename ExecutionSpace>
struct is_one_way_executor<hpx::kokkos::executor<ExecutionSpace>>
    : std::true_type {};

template <typename ExecutionSpace>
struct is_two_way_executor<hpx::kokkos::executor<ExecutionSpace>>
    : std::true_type {};

template <typename ExecutionSpace>
struct is_bulk_two_way_executor<hpx::kokkos::executor<ExecutionSpace>>
    : std::true_type {};
} // namespace execution
} // namespace parallel
} // namespace hpx

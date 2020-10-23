//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Contains a helper for creating thread-local execution space instances and
/// executors.

#pragma once

#include <hpx/kokkos/execution_spaces.hpp>
#include <hpx/kokkos/executors.hpp>

#include <Kokkos_Core.hpp>

namespace hpx {
namespace kokkos {
namespace detail {
template <typename ExecutionSpace> ExecutionSpace make_kokkos_instance() {
  return {};
}

#if defined(KOKKOS_ENABLE_CUDA)
template <> inline Kokkos::Cuda make_kokkos_instance<Kokkos::Cuda>() {
  cudaStream_t s;
  cudaError_t error = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    HPX_THROW_EXCEPTION(
        kernel_error, "hpx::kokkos::detail::initialize_instances",
        std::string("cudaStreamCreate failed: ") + cudaGetErrorString(error));
  }
  return {s};
}
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <> inline Kokkos::Experimental::HIP make_kokkos_instance<Kokkos::Experimental::HIP>() {
  cudaStream_t s;
  cudaError_t error = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    HPX_THROW_EXCEPTION(
        kernel_error, "hpx::kokkos::detail::initialize_instances",
        std::string("cudaStreamCreate failed: ") + cudaGetErrorString(error));
  }
  return {s};
}
#endif

#if defined(KOKKOS_ENABLE_HPX) && KOKKOS_VERSION >= 30000
template <>
inline Kokkos::Experimental::HPX
make_kokkos_instance<Kokkos::Experimental::HPX>() {
  return {Kokkos::Experimental::HPX::instance_mode::independent};
}
#endif
} // namespace detail

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class kokkos_instance_helper {
public:
  using execution_space = ExecutionSpace;

  explicit kokkos_instance_helper(
      std::size_t const num_instances_per_thread = 10,
      std::size_t const num_threads = hpx::get_num_worker_threads())
      : num_instances_per_thread(num_instances_per_thread),
        num_threads(num_threads), instances(num_threads),
        instance_counters(num_threads, 0) {
    for (std::size_t t = 0; t < num_threads; ++t) {
      instances[t].reserve(num_instances_per_thread);
      for (std::size_t i = 0; i < num_instances_per_thread; ++i) {
        instances[t].push_back(detail::make_kokkos_instance<execution_space>());
      }
    }
  }

  execution_space const &get_execution_space(
      std::size_t const thread_num = hpx::get_worker_thread_num()) {
    return instances[thread_num][++instance_counters[thread_num] %
                                 num_instances_per_thread];
  }

  executor<execution_space>
  get_executor(std::size_t const thread_num = hpx::get_worker_thread_num()) {
    return executor<execution_space>(get_execution_space(thread_num));
  }

private:
  std::size_t const num_instances_per_thread = 10;
  std::size_t const num_threads = hpx::get_num_worker_threads();
  std::vector<std::vector<execution_space>> instances;
  std::vector<std::size_t> instance_counters;
};
} // namespace kokkos
} // namespace hpx

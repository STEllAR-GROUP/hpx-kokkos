//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Contains the execution policy used for dispatching parallel algorithms to
/// Kokkos.

#pragma once

#include <hpx/kokkos/detail/logging.hpp>
#include <hpx/kokkos/executors.hpp>

#include <hpx/execution.hpp>
#include <hpx/future.hpp>

namespace hpx {
namespace kokkos {
struct kokkos_task_policy;
template <typename Executor, typename Parameters>
struct kokkos_task_policy_shim;

struct kokkos_policy;
template <typename Executor, typename Parameters> struct kokkos_policy_shim;

struct kokkos_task_policy {
  using executor_type = default_executor;
  using executor_parameters_type =
      hpx::parallel::execution::extract_executor_parameters<
          executor_type>::type;
  using execution_category = hpx::execution::parallel_execution_tag;

  template <typename Executor_, typename Parameters_> struct rebind {
    using type = kokkos_task_policy_shim<Executor_, Parameters_>;
  };

  constexpr kokkos_task_policy() {}

  kokkos_task_policy operator()(hpx::execution::task_policy_tag) const {
    return *this;
  }

  template <typename Executor>
  typename hpx::parallel::execution::rebind_executor<
      kokkos_task_policy, Executor, executor_parameters_type>::type
  on(Executor &&exec) const {
    using executor_type = typename std::decay<Executor>::type;

    static_assert(is_kokkos_executor<executor_type>::value,
                  "hpx::kokkos::is_kokkos_executor<Executor>::value");

    using rebound_type = typename hpx::parallel::execution::rebind_executor<
        kokkos_task_policy, Executor, executor_parameters_type>::type;
    return rebound_type(std::forward<Executor>(exec), parameters());
  }

  template <typename... Parameters,
            typename ParametersType = typename hpx::parallel::execution::
                executor_parameters_join<Parameters...>::type>
  typename hpx::parallel::execution::rebind_executor<
      kokkos_task_policy, executor_type, ParametersType>::type
  with(Parameters &&...params) const {
    using rebound_type = typename hpx::parallel::execution::rebind_executor<
        kokkos_task_policy, executor_type, ParametersType>::type;
    return rebound_type(executor(), join_executor_parameters(
                                        std::forward<Parameters>(params)...));
  }

  kokkos_task_policy label(char const *l) const {
    auto p = *this;
    p.label_ = l;
    return p;
  }
  char const *label() { return label_; }

  executor_type executor() const { return executor_type{}; }

  executor_parameters_type &parameters() { return params_; }
  constexpr executor_parameters_type const &parameters() const {
    return params_;
  }

private:
  executor_parameters_type params_{};
  char const *label_ = "unnamed kernel";
};

template <typename Executor, typename Parameters>
struct kokkos_task_policy_shim : kokkos_task_policy {
  using executor_type = Executor;
  using executor_parameters_type = Parameters;
  using execution_category =
      typename hpx::traits::executor_execution_category<executor_type>::type;

  template <typename Executor_, typename Parameters_> struct rebind {
    typedef kokkos_task_policy_shim<Executor_, Parameters_> type;
  };

  kokkos_task_policy_shim
  operator()(hpx::execution::task_policy_tag tag) const {
    return *this;
  }

  template <typename Executor_>
  typename hpx::parallel::execution::rebind_executor<
      kokkos_task_policy_shim, Executor_, executor_parameters_type>::type
  on(Executor_ &&exec) const {
    using executor_type = typename std::decay<Executor>::type;

    static_assert(is_kokkos_executor<executor_type>::value,
                  "hpx::kokkos::is_kokkos_executor<Executor>::value");

    using rebound_type = typename hpx::parallel::execution::rebind_executor<
        kokkos_task_policy_shim, Executor_, executor_parameters_type>::type;
    return rebound_type(std::forward<Executor_>(exec), params_);
  }

  template <typename... Parameters_,
            typename ParametersType = typename hpx::parallel::execution::
                executor_parameters_join<Parameters_...>::type>
  typename hpx::parallel::execution::rebind_executor<
      kokkos_task_policy_shim, executor_type, ParametersType>::type
  with(Parameters_ &&...params) const {
    using rebound_type = typename hpx::parallel::execution::rebind_executor<
        kokkos_task_policy_shim, executor_type, ParametersType>::type;
    return rebound_type(
        exec_, join_executor_parameters(std::forward<Parameters_>(params)...));
  }

  kokkos_task_policy_shim &label(char const *l) {
    label_ = l;
    return *this;
  }
  char const *label() { return label_; }

  Executor &executor() { return exec_; }

  Executor const &executor() const { return exec_; }

  Parameters &parameters() { return params_; }

  constexpr Parameters const &parameters() const { return params_; }

  template <typename Dependent = void,
            typename Enable = typename std::enable_if<
                std::is_constructible<Executor>::value &&
                    std::is_constructible<Parameters>::value,
                Dependent>::type>
  constexpr kokkos_task_policy_shim() {}

  template <typename Executor_, typename Parameters_>
  constexpr kokkos_task_policy_shim(Executor_ &&exec, Parameters_ &&params)
      : exec_(std::forward<Executor_>(exec)),
        params_(std::forward<Parameters_>(params)) {}

private:
  Executor exec_;
  Parameters params_;
  char const *label_ = "unnamed kernel";
};

struct kokkos_policy {
  using executor_type = default_executor;
  using executor_parameters_type =
      hpx::parallel::execution::extract_executor_parameters<
          executor_type>::type;
  using execution_category = hpx::execution::parallel_execution_tag;

  template <typename Executor_, typename Parameters_> struct rebind {
    using type = kokkos_policy_shim<Executor_, Parameters_>;
  };

  constexpr kokkos_policy() {}

  kokkos_task_policy operator()(hpx::execution::task_policy_tag) const {
    return kokkos_task_policy();
  }

  template <typename Executor>
  typename hpx::parallel::execution::rebind_executor<
      kokkos_policy, Executor, executor_parameters_type>::type
  on(Executor &&exec) const {
    typedef typename std::decay<Executor>::type executor_type;

    static_assert(is_kokkos_executor<executor_type>::value,
                  "hpx::kokkos::is_kokkos_executor<Executor>::value");

    using rebound_type = typename hpx::parallel::execution::rebind_executor<
        kokkos_policy, Executor, executor_parameters_type>::type;
    return rebound_type(std::forward<Executor>(exec), parameters());
  }

  template <typename... Parameters,
            typename ParametersType = typename hpx::parallel::execution::
                executor_parameters_join<Parameters...>::type>
  typename hpx::parallel::execution::rebind_executor<
      kokkos_policy, executor_type, ParametersType>::type
  with(Parameters &&...params) const {
    using rebound_type = typename hpx::parallel::execution::rebind_executor<
        kokkos_policy, executor_type, ParametersType>::type;
    return rebound_type(executor(), join_executor_parameters(
                                        std::forward<Parameters>(params)...));
  }

  kokkos_policy label(char const *l) const {
    auto p = *this;
    p.label_ = l;
    return p;
  }
  char const *label() { return label_; }

public:
  executor_type executor() const { return executor_type{}; }

  executor_parameters_type &parameters() { return params_; }
  constexpr executor_parameters_type const &parameters() const {
    return params_;
  }

private:
  executor_parameters_type params_{};
  char const *label_ = "unnamed kernel";
};

template <typename Executor, typename Parameters>
struct kokkos_policy_shim : kokkos_policy {
  using executor_type = Executor;
  using executor_parameters_type = Parameters;
  using exeucution_category =
      typename hpx::traits::executor_execution_category<executor_type>::type;

  template <typename Executor_, typename Parameters_> struct rebind {
    typedef kokkos_policy_shim<Executor_, Parameters_> type;
  };

  kokkos_task_policy_shim<Executor, Parameters>
  operator()(hpx::execution::task_policy_tag) const {
    return kokkos_task_policy_shim<Executor, Parameters>(exec_, params_);
  }

  template <typename Executor_>
  typename hpx::parallel::execution::rebind_executor<
      kokkos_policy_shim, Executor_, executor_parameters_type>::type
  on(Executor_ &&exec) const {
    using executor_type = typename std::decay<Executor>::type;

    static_assert(is_kokkos_executor<executor_type>::value,
                  "hpx::kokkos::is_kokkos_executor<Executor>::value");

    using rebound_type = typename hpx::parallel::execution::rebind_executor<
        kokkos_policy_shim, Executor_, executor_parameters_type>::type;
    return rebound_type(std::forward<Executor_>(exec), params_);
  }

  template <typename... Parameters_,
            typename ParametersType = typename hpx::parallel::execution::
                executor_parameters_join<Parameters_...>::type>
  typename hpx::parallel::execution::rebind_executor<
      kokkos_policy_shim, executor_type, ParametersType>::type
  with(Parameters_ &&...params) const {
    using rebound_type = typename hpx::parallel::execution::rebind_executor<
        kokkos_policy_shim, executor_type, ParametersType>::type;
    return rebound_type(
        exec_, join_executor_parameters(std::forward<Parameters_>(params)...));
  }

  kokkos_policy_shim &label(char const *l) {
    label_ = l;
    return *this;
  }
  char const *label() { return label_; }

  Executor &executor() { return exec_; }

  Executor const &executor() const { return exec_; }

  Parameters &parameters() { return params_; }

  Parameters const &parameters() const { return params_; }

  template <typename Dependent = void,
            typename Enable = typename std::enable_if<
                std::is_constructible<Executor>::value &&
                    std::is_constructible<Parameters>::value,
                Dependent>::type>
  constexpr kokkos_policy_shim() {}

  template <typename Executor_, typename Parameters_>
  constexpr kokkos_policy_shim(Executor_ &&exec, Parameters_ &&params)
      : exec_(std::forward<Executor_>(exec)),
        params_(std::forward<Parameters_>(params)) {}

private:
  Executor exec_{};
  Parameters params_{};
  char const *label_ = "unnamed kernel";
  /// \endcond
};

static constexpr kokkos_policy kok;

template <typename ExecutionPolicy>
struct is_kokkos_execution_policy : std::false_type {};

template <>
struct is_kokkos_execution_policy<hpx::kokkos::kokkos_policy> : std::true_type {
};

template <typename Executor, typename Parameters>
struct is_kokkos_execution_policy<
    hpx::kokkos::kokkos_policy_shim<Executor, Parameters>> : std::true_type {};

template <>
struct is_kokkos_execution_policy<hpx::kokkos::kokkos_task_policy>
    : std::true_type {};

template <typename Executor, typename Parameters>
struct is_kokkos_execution_policy<
    hpx::kokkos::kokkos_task_policy_shim<Executor, Parameters>>
    : std::true_type {};
} // namespace kokkos
} // namespace hpx

namespace hpx {
namespace detail {

template <typename Executor, typename Parameters>
struct is_rebound_execution_policy<
    hpx::kokkos::kokkos_policy_shim<Executor, Parameters>> : std::true_type {};

template <typename Executor, typename Parameters>
struct is_rebound_execution_policy<
    hpx::kokkos::kokkos_task_policy_shim<Executor, Parameters>>
    : std::true_type {};

template <>
struct is_execution_policy<hpx::kokkos::kokkos_policy> : std::true_type {};

template <typename Executor, typename Parameters>
struct is_execution_policy<
    hpx::kokkos::kokkos_policy_shim<Executor, Parameters>> : std::true_type {};

template <>
struct is_execution_policy<hpx::kokkos::kokkos_task_policy> : std::true_type {};

template <typename Executor, typename Parameters>
struct is_execution_policy<
    hpx::kokkos::kokkos_task_policy_shim<Executor, Parameters>>
    : std::true_type {};

template <>
struct is_parallel_execution_policy<hpx::kokkos::kokkos_policy>
    : std::true_type {};

template <typename Executor, typename Parameters>
struct is_parallel_execution_policy<
    hpx::kokkos::kokkos_policy_shim<Executor, Parameters>> : std::true_type {};

template <>
struct is_parallel_execution_policy<hpx::kokkos::kokkos_task_policy>
    : std::true_type {};

template <typename Executor, typename Parameters>
struct is_parallel_execution_policy<
    hpx::kokkos::kokkos_task_policy_shim<Executor, Parameters>>
    : std::true_type {};

template <>
struct is_async_execution_policy<hpx::kokkos::kokkos_task_policy>
    : std::true_type {};

template <typename Executor, typename Parameters>
struct is_async_execution_policy<
    hpx::kokkos::kokkos_task_policy_shim<Executor, Parameters>>
    : std::true_type {};
} // namespace detail
} // namespace hpx

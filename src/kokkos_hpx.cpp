#include <kokkos.hpp>

#include <hpx/hpx_main.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/parallel_reduce.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/irange.hpp>

// This example shows examples of calling Kokkos functions within HPX, either
// through HPX executors or directly.

// TODO: Make parallel algorithm fallback work for Kokkos executors (with all
// backends and policies).

int hpx_main(int argc, char **argv) {
  using namespace hpx::compute::kokkos;

  {
    kokkos::runtime_guard k(argc, argv);

    std::size_t n = 10;

    // Don't initialize views as they require kernel launches. These are
    // allocated on memory space of the default device (i.e. GPU if available,
    // otherwise host).
    std::cout << "constructing views" << std::endl;
    kokkos::View<double *> a(kokkos::ViewAllocateWithoutInitializing("a"), n);
    kokkos::View<double *> b(kokkos::ViewAllocateWithoutInitializing("b"), n);
    kokkos::View<double *> c(kokkos::ViewAllocateWithoutInitializing("c"), n);
    std::cout << "done constructing views" << std::endl;

    // Create a Kokkos executor which forwards to Kokkos' DefaultExecutionSpace.
    auto exec = kokkos::default_executor();
    auto pol = hpx::parallel::execution::par.on(exec);
    auto pol_task = hpx::parallel::execution::par(hpx::parallel::execution::task).on(exec);
    hpx::compute::cuda::target t(0);

    // Can call parallel for_loop that forwards to Kokkos parallel_for. The
    // parallel for_loop has been specialized for Kokkos executors.
    std::cout << "initializing views" << std::endl;
    std::cout << "hpx for_loop with kokkos executor" << std::endl;
    hpx::parallel::for_loop(pol, 0, n, KOKKOS_LAMBDA(std::size_t i) {
      a[i] = double(i) * 1.0;
      b[i] = double(i) * 2.0;
    });
    std::cout << "for_loop done" << std::endl;

    // Can call parallel for loop with task policy. Returns a future<void>.
    std::cout << "hpx for_loop kokkos task executor" << std::endl;
    auto f1 = hpx::parallel::for_loop(pol_task, 0, n, KOKKOS_LAMBDA(std::size_t i) {
      a[i] = double(i) * 1.0;
      b[i] = double(i) * 2.0;
    });
    std::cout << "for_loop spawned" << std::endl;

    // Can interleave normal HPX functionality with Kokkos functionality.
    kokkos::View<double *, kokkos::HostSpace> d(kokkos::ViewAllocateWithoutInitializing("d"), n);
    std::cout << "hpx for_loop on host" << std::endl;
    auto g1 = hpx::parallel::for_loop(
        hpx::parallel::execution::par(hpx::parallel::execution::task), 0, n,
        [] HPX_HOST_DEVICE (std::size_t i) { d[i] = std::sin(double(i)); });
    std::cout << "for_loop spawned" << std::endl;

    kokkos::View<double *, kokkos::DeviceSpace> e(kokkos::ViewAllocateWithoutInitializing("e"), n);
    std::cout << "hpx for_loop on device" << std::endl;
    auto g2 = hpx::parallel::for_loop(
        hpx::parallel::execution::par(hpx::parallel::execution::task).on(hpx::compute::cuda::default_executor(t)), 0, n,
        [] HPX_HOST_DEVICE (std::size_t i) { e[i] = cos::sin(double(i)); });
    std::cout << "for_loop spawned" << std::endl;

    std::cout << "hpx for_loop with kokkos executor" << std::endl;
    hpx::parallel::for_loop(
        pol, 0, n, KOKKOS_LAMBDA(std::size_t i) { c[i] = a[i] + b[i]; });
    std::cout << "for_loop done" << std::endl;

    // Can call Kokkos functionality directly, but then we can't control the
    // blocking behaviour automatically. Can still get a future from e.g. the
    // CUDA device.
    double result = 0.0;
    std::cout << "kokkos reduce" << std::endl;
    kokkos::parallel_reduce(
        kokkos::RangePolicy<kokkos::DefaultExecutionSpace>(0, n),
        KOKKOS_LAMBDA(std::size_t i, double &update) { update += c[i]; },
        result);

    // We can get a future for the asynchronous CUDA launch through Kokkos.
    auto f2 = t.get_future();

    // Can also call HPX algorithms that haven't been specialized for Kokkos
    // executors.
    // TODO: This requires some massaging between iterators and Kokkos'
    // index-based loops.
    //std::cout << "hpx kokkos for_each (fall back to executor)" << std::endl;
    //auto range = boost::irange(std::size_t(0), n);
    //hpx::parallel::for_each(
    //    pol, std::begin(range), std::end(range),
    //    KOKKOS_LAMBDA(std::size_t i) { d[i] = std::cos(double(i)); });

    hpx::wait_all(f1, f2, g1, g2);

    std::cout << result << std::endl;
  }

  return hpx::finalize();
}

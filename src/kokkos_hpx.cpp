#include <hpx/hpx_main.hpp>
#include <hpx/kokkos.hpp>

int main(int argc, char **argv) {
  {
    hpx::kokkos::ScopeGuard k(argc, argv);

    std::size_t n = 10;

    // Don't initialize views as they require kernel launches. These are
    // allocated on memory space of the default device (i.e. GPU if available,
    // otherwise host). These are blocking calls.
    std::cout << "constructing views... " << std::flush;
    hpx::kokkos::View<double *> a(
        hpx::kokkos::ViewAllocateWithoutInitializing("a"), n);
    hpx::kokkos::View<double *> b(
        hpx::kokkos::ViewAllocateWithoutInitializing("b"), n);
    hpx::kokkos::View<double *> c(
        hpx::kokkos::ViewAllocateWithoutInitializing("c"), n);
    std::cout << "done" << std::endl;

    // Plain Kokkos parallel_for. This is asynchronous on most GPUs, but does
    // not return a future. The only way of waiting is to fence the whole
    // execution space which blocks the OS thread. This is *not* recommended.
    std::cout << "parallel_for... " << std::flush;
    hpx::kokkos::parallel_for(
        n, KOKKOS_LAMBDA(std::size_t i) {
          printf("parallel_for with index %u\n", i);
          a[i] = double(i) * 1.0;
          b[i] = double(i) * 2.0;
        });
    hpx::kokkos::DefaultExecutionSpace().fence();
    std::cout << "done" << std::endl;

    // Kokkos parallel_for with a single execution space instance. The current
    // semantics are to serialize kernel launches on the same instance.
    // Different instances may have work launched concurrently, but must not.
    std::cout
        << "launching two asynchronous parallel_fors on the same instance... "
        << std::flush;
    auto inst = hpx::kokkos::make_execution_space<>();
    // Same as:
    // auto inst =
    // hpx::kokkos::make_execution_space_instance<hpx::kokkos::DefaultExecutionSpace>();
    // Can also get a host instance:
    // auto inst =
    // hpx::kokkos::make_execution_space_instance<hpx::kokkos::DefaultHostExecutionSpace>();
    // Or a specific instance:
    // auto inst =
    // hpx::kokkos::make_execution_space_instance<hpx::kokkos::Cuda>();
    hpx::future<void> f1 = hpx::kokkos::parallel_for_async(
        hpx::kokkos::RangePolicy<>(inst, 0, n), KOKKOS_LAMBDA(std::size_t i) {
          printf("asynchronous parallel_for with index %u\n", i);
          a[i] = double(i) * 1.0;
          b[i] = double(i) * 2.0;
        });

    hpx::future<void> f2 = hpx::kokkos::parallel_for_async(
        hpx::kokkos::RangePolicy<>(inst, 0, n), KOKKOS_LAMBDA(std::size_t i) {
          printf("asynchronous parallel_for with index %u\n", i);
          a[i] = double(i) * 1.0;
          b[i] = double(i) * 2.0;
        });

    std::cout << "returned... " << std::flush;
    hpx::wait_all(f1, f2);
    std::cout << "done" << std::endl;

    // Kokkos parallel_for with two different execution space instances. The two
    // kernel launches may be concurrent, but must not.
    std::cout
        << "launching two asynchronous parallel_fors on different instances... "
        << std::flush;
    hpx::future<void> f3 = hpx::kokkos::parallel_for_async(
        hpx::kokkos::RangePolicy<>(hpx::kokkos::make_execution_space<>(), 0, n),
        KOKKOS_LAMBDA(std::size_t i) {
          printf("asynchronous parallel_for with index %u\n", i);
          a[i] = double(i) * 1.0;
          b[i] = double(i) * 2.0;
        });

    hpx::future<void> f4 = hpx::kokkos::parallel_for_async(
        hpx::kokkos::RangePolicy<>(hpx::kokkos::make_execution_space<>(), 0, n),
        KOKKOS_LAMBDA(std::size_t i) {
          printf("asynchronous parallel_for with index %u\n", i);
          a[i] = double(i) * 1.0;
          b[i] = double(i) * 2.0;
        });

    std::cout << "returned... " << std::flush;
    hpx::wait_all(f3, f4);
    std::cout << "done" << std::endl;
  }

  return hpx::finalize();
}

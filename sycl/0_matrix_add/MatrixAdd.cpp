#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

// Size of the matrices
constexpr size_t M = 8;     // row
constexpr size_t N = 32;    // colomn

int main() {
    // Create a queue to work on default device
    queue q;

    // M * N, row(M), colomn(N) in device memory
    buffer<float, 2> a { range<2> { M, N } };
    buffer<float, 2> b { range<2> { M, N } };
    buffer<float, 2> c { range<2> { M, N } };

    { // tasks must complete before exiting the block
        // Launch a kernel to initialize A
        q.submit([&] (handler &cgh) {
            // Get a write accessor to update A matrix
            auto A = a.get_access<access::mode::write>(cgh);

            cgh.parallel_for<class init_a>(range<2> { M, N },
                    [=] (id<2> index) {
                    A[index] = index[0] * N + index[1];
                    });
            });

        // Launch a kernel to initialize B
        // From the access pattern above, the SYCL runtime detects this
        // command group is independent from the first one and can be
        // scheduled independently
        q.submit([&] (handler &cgh) {
            // Get a write accessor to update B matrix
            auto B = b.get_access<access::mode::write>(cgh);

            cgh.parallel_for<class init_b>(range<2> { M, N },
                    [=] (id<2> index) {
                    B[index] = 1;
                    });
            });

        // c = a + b
        // From these accessors, the SYCL runtime will ensure that when
        // this kernel is run, the kernels computing a and b completed
        q.submit([&] (handler &cgh) {
            // In the kernel A and B are read, but C is written
            auto A = a.get_access<access::mode::read>(cgh);
            auto B = b.get_access<access::mode::read>(cgh);
            auto C = c.get_access<access::mode::write>(cgh);

            cgh.parallel_for<class matrix_add>(range<2> { M, N },
                    [=] (id<2> index) {
                    C[index] = A[index] + B[index];
                    });
            });

    } // End scope, so wait for the queue to complete.

#ifdef _DEBUG
    auto A = a.get_access<access::mode::read>();
    std::cout << std::endl << "A:" << std::endl;
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++)
            std::cout << A[i][j] << ' ';
        std::cout << "\n";
    }
    std::cout << "\n";

    auto B = b.get_access<access::mode::read>();
    std::cout << std::endl << "B:" << std::endl;
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++)
            std::cout << B[i][j] << ' ';
        std::cout << "\n";
    }
    std::cout << "\n";

    auto C = c.get_access<access::mode::read>();
    std::cout << std::endl << "Result:" << std::endl;
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++)
            std::cout << C[i][j] << ' ';
        std::cout << "\n";
    }
    std::cout << "\n";
#endif

    // Request an accessor to read c from the host-side. The SYCL runtime
    // ensures that c is ready when the accessor is returned
#ifndef _DEBUG
    auto C = c.get_access<access::mode::read>();
#endif
    std::cout << std::endl << "Result:" << std::endl;
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            if (C[i][j] != i * N + j + 1){
                std::cout << "Wrong value " << C[i][j] << " on itme: "
                    << i << ' ' << j << std::endl;
                exit(-1);
            }

    std::cout << "Accurate computation!" << std::endl;
    return 0;
}

#include "mkl.h"
#include <iostream>
#include <chrono>

int main(int argc, char *argv[])
{
    std::cout << "MKL Long OC GEMM" << std::endl;
    float *a, *b, *c = nullptr;
    MKL_INT m, n, k;
    m = 128;
    n = 51864;
    k = 384;
    a = (float *)mkl_calloc(m * k, sizeof( float ), 64);
    b = (float *)mkl_calloc(k * n, sizeof( float ), 64);
    c = (float *)mkl_calloc(m * n, sizeof( float ), 64);

    CBLAS_IDENTIFIER identifier = CblasBMatrix;
    CBLAS_LAYOUT layout = CblasRowMajor;
    size_t destsize = cblas_sgemm_pack_get_size(identifier, m, n, k);
    std::cout << "destsize packed size " << std::endl;
    float* dest = static_cast<float*>(mkl_malloc(destsize, 64));

    if( dest == NULL ) {
        std::cout << "\n Can't allocate memory for buffer\n" << std::endl;
        mkl_free(a);
        mkl_free(b);
        mkl_free(c);
        return 1;
    }
    //prepack B weight
    cblas_sgemm_pack(layout, identifier, CblasNoTrans, m, n, k, 1.0, b, n, dest);

    //prepare gemm args
    CBLAS_STORAGE storage = CblasPacked;
    size_t trials = 500;
    auto begin = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < trials; i++) {
        cblas_sgemm_compute(layout, CblasNoTrans, storage, m, n, k,
                            a, k, dest, n, 1.0, c, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << total_time / trials << std::endl;

    // Free allocated memory
    mkl_free(dest);
    mkl_free(a);
    mkl_free(b);
    mkl_free(c);
    return 0;
}


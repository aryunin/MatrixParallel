#ifndef PARALLEL_LIB_H
#define PARALLEL_LIB_H

#include <omp.h>
#include <random>


std::random_device SEED_SRC;
std::mt19937 RANDOM(SEED_SRC());

template <typename T>
using DARR = T**;

template <typename T>
DARR<T> allocate(const size_t size) {
    auto arr = new T*[size];
    for (size_t i = 0; i < size; i++) {
        arr[i] = new T[size];
        for (size_t j = 0; j < size; j++) {
            arr[i][j] = 0;
        }
    }
    return arr;
}

template <typename T>
void deallocate(const DARR<T> &matrix, size_t size) {
    for (size_t i = 0; i < size; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

template <typename T>
void generateRandom(DARR<T> &matrix, const size_t size, const T &min, const T &max) {
    if constexpr(std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> distr(min, max);

        for (size_t i = 0; i < size; i++)
            for (size_t j = 0; j < size; j++)
                matrix[i][j] = distr(RANDOM);
    } else {
        std::uniform_int_distribution<T> distr(min, max);

        for (size_t i = 0; i < size; i++)
            for (size_t j = 0; j < size; j++)
                matrix[i][j] = distr(RANDOM);
    }
}

template <typename T>
T max(const DARR<T> &matrix, const size_t size, const bool absolute) {
    T res = absolute ? std::abs(matrix[0][0]) : matrix[0][0];

    for (size_t i = 0; i < size; i++)
        for (size_t j = 0; j < size; j++)
            res = absolute ? std::max(std::abs(matrix[i][j]), res) : std::max(matrix[i][j], res);

    return res;
}

template <typename T>
void multiplyLinear(const DARR<T> &first, const DARR<T> &second, DARR<T> &res, size_t size) {
    for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
            res[i][j] = 0;
            for (size_t k = 0; k < size; k++){
                res[i][j] += first[i][k] * second[k][j];
            }
        }
    }
}

template <typename T>
void multiplyLinearParallel(const DARR<T> &first, const DARR<T> &second, DARR<T> &res, const size_t size, const int threads) {
    omp_set_num_threads(threads);

#pragma omp parallel for default(none) shared(size, res, first, second)
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            res[i][j] = 0;
            for (size_t k = 0; k < size; k++){
                res[i][j] += first[i][k] * second[k][j];
            }
        }
    }
}

template <typename T>
void multiplyBlockParallel(const DARR<T> &first, const DARR<T> &second, DARR<T> &res, const size_t size, const int threads) {
    DARR<T> A = first;
    DARR<T> B = second;
    DARR<T> R = res;
    size_t newSize = size;

    size_t gridSize = int(sqrt(double(threads)));

    bool realloc = size % gridSize != 0;
    if (realloc) {
        newSize = size + (gridSize - size % gridSize);

        A = allocate<T>(newSize);
        copy(first, size, A, newSize);
        B = allocate<T>(newSize);
        copy(second, size, B, newSize);
        R = allocate<T>(newSize);
    }

    size_t blockSize = newSize/gridSize;

    int numThreads = (int)(gridSize * gridSize);
    omp_set_num_threads(numThreads);

#pragma omp parallel default(none) shared(gridSize, blockSize, R, A, B)
    {
        int id = omp_get_thread_num();
        size_t row = id / gridSize;
        size_t col = id % gridSize;

        for (size_t iter = 0; iter < gridSize; iter++)
            for (size_t i = row * blockSize; i < (row + 1) * blockSize; i++)
                for (size_t j = col * blockSize; j < (col + 1) * blockSize; j++)
                    for (size_t k = iter * blockSize; k < (iter + 1) * blockSize; k++)
                        R[i][j] += A[i][k] * B[k][j];
    }

    if (realloc) {
        copy(R, newSize, res, size);

        deallocate(A, newSize);
        deallocate(B, newSize);
        deallocate(R, newSize);
    }
}

template <typename T>
bool equals(const DARR<T> &first, const DARR<T> &second, const size_t size, const int precision = 2) {
    auto rounder = pow(10, precision);

    bool res = true;
    for (size_t i = 0; i < size; i++)
        for (size_t j = 0; j < size; j++)
            res &= (int64_t) (first[i][j] * rounder) == (int64_t) (second[i][j] * rounder);
    return res;
}

template <typename T>
void copy(const DARR<T> &src, size_t srcSize, DARR<T> &res, const size_t resSize) {
    if (srcSize < resSize) {
        for (size_t i = 0; i < srcSize; i++)
            std::copy_n(src[i], srcSize, res[i]);
    }
    else {
        for (size_t i = 0; i < resSize; i++)
            std::copy_n(src[i], resSize, res[i]);
    }
}



#endif //PARALLEL_LIB_H

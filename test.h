#ifndef PARALLEL_TEST_H
#define PARALLEL_TEST_H

#include "lib.h"
#include <iomanip>
#include <sstream>
#include <iostream>


struct TestParams {
    const int threads;
    const int startSize;
    const int endSize;
    const int increment;
    const int minValue;
    const int maxValue;
    const int cellWidth;
};

template <typename T>
void show(const DARR<T> &matrix, const size_t size, const int precision = 2) {
    int CELL_SIZE = std::to_string((int)max(matrix, size, true)).length();

    std::stringstream stream;

    if constexpr(std::is_floating_point_v<T>) {
        CELL_SIZE += precision + 1;
        stream << std::fixed << std::setprecision(precision);
    }
    CELL_SIZE += 2;

    // header
    for (size_t i = 0; i < size * CELL_SIZE; i++) {
        stream << "~";
    }
    stream << "\n";

    // matrix
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            stream << std::setw(CELL_SIZE) << matrix[i][j];
        }
        stream << "\n";
    }

    // header
    for (size_t i = 0; i < size* CELL_SIZE; i++) {
        stream << "~";
    }
    stream << "\n";

    std::cout << stream.str();
}

template <typename T, typename F, typename ... Params>
void testFunc(F f, int64_t &time, Params&& ... params) {
    auto t1 = std::chrono::steady_clock::now();
    f(std::forward<decltype(params)>(params)...);
    auto t2 = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<T>(t2 - t1).count();
}

template <typename TYPE, typename TIME>
void test(const TestParams &params) {
    std::cout << "THREADS COUNT: " << params.threads << std::endl;

    std::cout << std::setw(params.cellWidth) << "size " << std::setw(params.cellWidth) << "L " << std::setw(params.cellWidth) << "LP " << std::setw(params.cellWidth) << "B " << std::setw(params.cellWidth) << "EQ ";
    std::cout << std::endl;

    for (int size = params.startSize; size < params.endSize; size += params.increment) {
        std::cout << std::setw(params.cellWidth) << std::to_string(size) + " ";

        auto first = allocate<TYPE>(size);
        generateRandom<TYPE>(first, size, params.minValue, params.maxValue);

        auto second = allocate<TYPE>(size);
        generateRandom<TYPE>(second, size, params.minValue, params.maxValue);

        int64_t elapsedTime;

        // linear
        auto res1 = allocate<TYPE>(size);
        testFunc<TIME>(multiplyLinear<TYPE>, elapsedTime, first, second, res1, size);
        std::cout << std::setw(params.cellWidth) << std::to_string(elapsedTime) + " ";

        // linear parallel
        auto res2 = allocate<TYPE>(size);
        testFunc<TIME>(multiplyLinearParallel<TYPE>, elapsedTime, first, second, res2, size, params.threads);
        std::cout << std::setw(params.cellWidth) << std::to_string(elapsedTime) + " ";

        // block
        auto res3 = allocate<TYPE>(size);
        testFunc<TIME>(multiplyBlockParallel<TYPE>, elapsedTime, first, second, res3, size, params.threads);
        std::cout << std::setw(params.cellWidth) << std::to_string(elapsedTime) + " ";

        // equality
        std::string equalsStr;
        equalsStr += equals(res1, res2, size) ? "T" : "F";
        equalsStr += equals(res2, res3, size) ? "T" : "F";
        equalsStr += equals(res1, res3, size) ? "T" : "F";

        std::cout << std::setw(params.cellWidth) << equalsStr + " ";

        std::cout << std::endl;

        // dealloc
        deallocate(first, size);
        deallocate(second, size);
        deallocate(res1, size);
        deallocate(res2, size);
        deallocate(res3, size);
    }
}


#endif //PARALLEL_TEST_H

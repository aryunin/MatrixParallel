#include <chrono>
#include <thread>
#include "test.h"

const char* const THREADS_ARG = "threads";

int getThreadsCount(int argc, const char* argv[]);

int main(int argc, const char* argv[]) {
    TestParams params {
            getThreadsCount(argc, argv),
            100,
            10000,
            25,
            -100,
            100,
            10
    };

    test<int, std::chrono::microseconds>(params);
    return 0;
}

int getThreadsCount(int argc, const char* argv[]) {
    int threadsCount = (int)std::thread::hardware_concurrency();
//    for (int i = 0; i < argc; i++) {
//        if (strcmp(argv[i], THREADS_ARG) != 0) {
//            threadsCount = std::stoi(argv[i + 1]);
//        }
//    }
    return threadsCount;
}
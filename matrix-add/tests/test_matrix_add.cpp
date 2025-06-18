#include <gtest/gtest.h>
#include "../src/matrix_add.hpp"
#include <vector>
#include <cmath>

const float EPSILON = 1e-5;

bool isEqual(const std::vector<float>& a, const std::vector<float>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (fabs(a[i] - b[i]) > EPSILON)
            return false;
    }
    return true;
}

TEST(MatrixAddTest, SmallMatrixCompareCPUvsGPU) {
    const int N = 4;
    std::vector<float> A = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    std::vector<float> B(N * N, 1.0f);
    std::vector<float> C_cpu(N * N, 0.0f);
    std::vector<float> C_gpu(N * N, 0.0f);

    cpuMatrixAdd(A.data(), B.data(), C_cpu.data(), N);
    launchMatrixAddGPU(A.data(), B.data(), C_gpu.data(), N);

    ASSERT_TRUE(isEqual(C_cpu, C_gpu));
}

#pragma once

void cpuMatrixAdd(const float* A, const float* B, float* C, int width);
void launchMatrixAddGPU(const float* A, const float* B, float* C, int width);

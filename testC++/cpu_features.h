#pragma once
// CPU feature engine: same algorithm as the CUDA engine (ooc/feature_math.h), OpenMP over the points of a tile.
#include <memory>
#include "ooc/feature_engine.h"

// numThreads <= 0: use all the CPUs the process may use (cgroup and affinity aware).
std::unique_ptr<FeatureEngine> makeCpuEngine(const FeatureConfig& cfg, int numSlots, int numThreads);
// Host bytes per tile point and slot (staging + workspace).
double cpuBytesPerPoint(const FeatureConfig& cfg);

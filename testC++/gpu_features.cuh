#pragma once

// GPU feature computation interface (host side, no CUDA types: it can be included by plain C++).
//
// One GpuEngine owns `numSlots` independent slots (stream + pinned staging buffers + device workspace). A tile is
// processed in a slot with submit() (asynchronous: upload, grid, kernel, download of the finished rows) and wait();
// while slot k computes, the host can prepare slot k+1 and consume slot k-1 (see feature_extraction_viewer.cpp).
//
// Algorithm (see ooc/feature_math.h): float arithmetic on tile-centred coordinates, uniform grid with cell = largest
// radius / 2 sorted by cell (x fastest, so the cells of a row are contiguous in memory), ONE pass over the neighbours of
// the largest radius with per-scale shell accumulators, closed formulas per scale. Only the core points are computed and
// only the requested (feature, scale) columns are written.

#include <cstddef>
#include <cstdint>

#include "ooc/feature_engine.h"

using GpuConfig = FeatureConfig;

struct GpuDeviceInfo {
    bool   ok = false;
    char   name[256] = {};
    int    ccMajor = 0, ccMinor = 0, multiprocessors = 0, asyncEngines = 0;
    size_t freeBytes = 0, totalBytes = 0;
    char   error[256] = {};
};
// Queries device 0 (never throws; ok == false and `error` set when there is no usable CUDA device).
GpuDeviceInfo gpuQueryDevice();

class GpuEngine : public FeatureEngine {
public:
    // Throws std::runtime_error when CUDA is not usable.
    GpuEngine(const GpuConfig& cfg, int numSlots);
    ~GpuEngine() override;
    GpuEngine(const GpuEngine&) = delete;
    GpuEngine& operator=(const GpuEngine&) = delete;

    // Device bytes needed per tile point and slot (workspace + output rows).
    static double deviceBytesPerPoint(const GpuConfig& cfg);
    // Floats per output row (numSlots rounded up to a multiple of 4).
    const char* name() const override { return "GPU (CUDA)"; }
    int rowStride() const override;

    // Makes slot `slot` able to process tiles of up to `capacity` points (pinned staging + device workspace). false on failure.
    bool reserve(int slot, size_t capacity) override;

    // Staging buffers of the slot (valid after reserve): integer coordinates (3 per point) in ticks of `unit` metres counted
    // from the minimum of the tile (all >= 0), 1 byte per point (1 = core). Integers make the radius test exact.
    int32_t*       hostXyz(int slot) override;
    uint8_t*       hostCore(int slot) override;
    // Rows of the core points, in the order of the core points of the tile; row stride = rowStride() floats.
    // Valid after a successful wait() until the next submit() on the slot.
    const float*   hostRows(int slot) override;

    // extent: maximum coordinate of the points in ticks, per axis. Returns 0 when queued, non-zero on error (lastError()).
    int submit(int slot, int n, int nCore, double unit, const int64_t extent[3]) override;
    // Waits for the slot; 0 on success.
    int wait(int slot) override;

    const char* lastError() const override;
    StageTimes times() const override;

private:
    struct Impl;
    Impl* p_;
};

bool gpuProfileEnabled();

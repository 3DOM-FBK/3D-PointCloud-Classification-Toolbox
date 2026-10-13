// CUDA feature extraction, see gpu_features.cuh for the design.
#include "gpu_features.cuh"
#include "ooc/feature_math.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

bool gpuProfileEnabled() { static const bool on = std::getenv("FEAT_PROFILE") != nullptr; return on; }

namespace {

// ------------------------------------------------------------------ constants and grid

struct DevConst {
    int   orig[fm::kMaxScales];         // ascending position -> index of the scale in the requested order
    int   slotOf[FE_F_COUNT][fm::kMaxScales];   // by REQUESTED scale index
};
__constant__ DevConst c_cfg;

// Coordinates are integers (ticks of `unit` metres, origin = minimum of the tile): the distance test is exact.
struct GridP {
    int minX, minY, minZ, cell;        // cell: edge in ticks
    int dimX, dimY, dimZ;
};
struct KernelScales {
    long long r2t[fm::kMaxScales];      // squared radii in ticks^2 (ascending), inclusive
    float unit;                         // metres per tick
};

__device__ __forceinline__ int cell_coord(int v, int mn, int cell, int dim) {
    int c = (v - mn) / cell;
    return c < 0 ? 0 : (c >= dim ? dim - 1 : c);
}

// key = cell id (x fastest); idx = identity; hist = points per cell
__global__ void cellKeyKernel(const int* __restrict__ xyz, int n, GridP g, int* keys, int* idx, int* hist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cx = cell_coord(xyz[3 * i + 0], g.minX, g.cell, g.dimX);
    int cy = cell_coord(xyz[3 * i + 1], g.minY, g.cell, g.dimY);
    int cz = cell_coord(xyz[3 * i + 2], g.minZ, g.cell, g.dimZ);
    int key = cx + g.dimX * (cy + g.dimY * cz);
    keys[i] = key;
    idx[i] = i;
    atomicAdd(&hist[key], 1);
}

__global__ void reorderKernel(const int* __restrict__ xyz, const uint8_t* __restrict__ core, const int* __restrict__ idxSorted,
                              int n, int4* __restrict__ pts, uint8_t* __restrict__ sortedCore) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int s = idxSorted[i];
    pts[i] = make_int4(xyz[3 * s + 0], xyz[3 * s + 1], xyz[3 * s + 2], 0);
    sortedCore[i] = core[s];
}

struct ByteToInt {
    __host__ __device__ __forceinline__ int operator()(const uint8_t& b) const { return b ? 1 : 0; }
};

// ------------------------------------------------------------------ feature kernel

// Distance from p to the interval [lo, hi] along one axis (0 inside), in ticks.
__device__ __forceinline__ long long axis_gap(int p, int lo, int hi) {
    return p < lo ? (long long)(lo - p) : (p > hi ? (long long)(p - hi) : 0LL);
}

// One thread per CORE point. The neighbours of the largest radius are visited once, row by row: the cells of a row
// (same y, z; x range) are consecutive in the sorted array, so a row is ONE contiguous range of points.
template <int NSC>
__global__ void __launch_bounds__(128)
featureKernel(const int4* __restrict__ pts, const int* __restrict__ idxSorted, const int* __restrict__ coreList,
              int nCore, const int* __restrict__ coreRank, const int* __restrict__ cellOff, GridP g, KernelScales ks,
              float* __restrict__ rows, int stride) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nCore) return;
    const int pos = coreList[t];
    const int4 p = pts[pos];

    long long r2s[NSC];
#pragma unroll
    for (int s = 0; s < NSC; s++) r2s[s] = ks.r2t[s];
    const long long rmax2 = r2s[NSC - 1];
    const int rt = (int)ceil(sqrt((double)rmax2)) + 1;      // radius in ticks (a tick more: never misses a neighbour)

    fm::Acc acc[NSC];
#pragma unroll
    for (int s = 0; s < NSC; s++) fm::acc_init(acc[s]);

    const int ix0 = cell_coord(p.x - rt, g.minX, g.cell, g.dimX), ix1 = cell_coord(p.x + rt, g.minX, g.cell, g.dimX);
    const int iy0 = cell_coord(p.y - rt, g.minY, g.cell, g.dimY), iy1 = cell_coord(p.y + rt, g.minY, g.cell, g.dimY);
    const int iz0 = cell_coord(p.z - rt, g.minZ, g.cell, g.dimZ), iz1 = cell_coord(p.z + rt, g.minZ, g.cell, g.dimZ);

    for (int iz = iz0; iz <= iz1; iz++) {
        const int z0 = g.minZ + iz * g.cell;
        const long long dz = axis_gap(p.z, z0, z0 + g.cell - 1);
        const long long dz2 = dz * dz;
        if (dz2 > rmax2) continue;
        for (int iy = iy0; iy <= iy1; iy++) {
            const int y0 = g.minY + iy * g.cell;
            const long long dy = axis_gap(p.y, y0, y0 + g.cell - 1);
            if (dz2 + dy * dy > rmax2) continue;
            const int row = g.dimX * (iy + g.dimY * iz);
            const int b = cellOff[row + ix0], e = cellOff[row + ix1 + 1];
            for (int j = b; j < e; j++) {
                const int4 q = pts[j];
                const int dx = q.x - p.x, dyy = q.y - p.y, dzz = q.z - p.z;
                const long long d2 = (long long)dx * dx + (long long)dyy * dyy + (long long)dzz * dzz;
#pragma unroll
                for (int s = 0; s < NSC; s++) {
                    if (d2 <= r2s[s]) { fm::acc_add(acc[s], dx * ks.unit, dyy * ks.unit, dzz * ks.unit); break; }
                }
            }
        }
    }
#pragma unroll
    for (int s = 1; s < NSC; s++) fm::acc_merge(acc[s], acc[s - 1]);

    float* out = rows + (size_t)coreRank[idxSorted[pos]] * stride;
#pragma unroll
    for (int s = 0; s < NSC; s++) {
        float f[fm::kFeatureCount];
        fm::finalize(acc[s], f);
        const int o = c_cfg.orig[s];
#pragma unroll
        for (int k = 0; k < fm::kFeatureCount; k++) {
            const int slot = c_cfg.slotOf[k][o];
            if (slot >= 0) out[slot] = f[k];
        }
    }
}

struct LaunchArgs {
    cudaStream_t st; const int4* pts; const int* idxSorted; const int* coreList; int nCore; const int* coreRank;
    const int* cellOff; GridP g; KernelScales ks; float* rows; int stride;
};

// kernel instance for exactly `nsc` scales (the shell accumulators live in registers, so the count is a template parameter)
template <int N>
void launchFeature(int nsc, const LaunchArgs& a) {
    if (nsc != N) { if constexpr (N > 1) launchFeature<N - 1>(nsc, a); return; }
    const int block = 128;
    featureKernel<N><<<(a.nCore + block - 1) / block, block, 0, a.st>>>(a.pts, a.idxSorted, a.coreList, a.nCore, a.coreRank, a.cellOff, a.g, a.ks, a.rows, a.stride);
}

}  // namespace

// ------------------------------------------------------------------ engine

namespace {
struct Slot {
    cudaStream_t st = nullptr;
    cudaEvent_t ev[6] = {};              // start, h2d, grid, sort, kernel, d2h (profiling); ev[5] doubles as "done"
    size_t cap = 0;
    int32_t* hXyz = nullptr; uint8_t* hCore = nullptr; float* hRows = nullptr;       // pinned
    int32_t* dXyz = nullptr; uint8_t* dCore = nullptr;
    int *dKeys = nullptr, *dKeysS = nullptr, *dIdx = nullptr, *dIdxS = nullptr;
    int4* dPts = nullptr; uint8_t* dSortedCore = nullptr;
    int *dRank = nullptr, *dCoreList = nullptr, *dNumSel = nullptr;
    float* dRows = nullptr;
    size_t cellsCap = 0; int* dHist = nullptr;
    size_t tempCap = 0; void* dTemp = nullptr;
    bool pending = false;
};
}  // namespace

struct GpuEngine::Impl {
    int nsc = 0;
    int stride = 0;
    double rAsc[fm::kMaxScales] = {};
    std::vector<Slot> slots;
    std::string error;
    StageTimes times;
    bool profile = false;
};

namespace {
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { P.error = std::string("CUDA error: ") + cudaGetErrorString(e_) + " (" #call ")"; return -1; } } while (0)
// after the first asynchronous call: on failure wait for what was queued, so the slot is never reused while the stream uses it
#define CKQ(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { P.error = std::string("CUDA error: ") + cudaGetErrorString(e_) + " (" #call ")"; cudaStreamSynchronize(s.st); cudaGetLastError(); return -1; } } while (0)

void freeSlot(Slot& s) {
    cudaFreeHost(s.hXyz); cudaFreeHost(s.hCore); cudaFreeHost(s.hRows);
    cudaFree(s.dXyz); cudaFree(s.dCore); cudaFree(s.dKeys); cudaFree(s.dKeysS); cudaFree(s.dIdx); cudaFree(s.dIdxS);
    cudaFree(s.dPts); cudaFree(s.dSortedCore); cudaFree(s.dRank); cudaFree(s.dCoreList); cudaFree(s.dNumSel);
    cudaFree(s.dRows); cudaFree(s.dHist); cudaFree(s.dTemp);
    s.hXyz = nullptr; s.hCore = nullptr; s.hRows = nullptr; s.dXyz = nullptr; s.dCore = nullptr; s.dKeys = s.dKeysS = s.dIdx = s.dIdxS = nullptr;
    s.dPts = nullptr; s.dSortedCore = nullptr; s.dRank = s.dCoreList = s.dNumSel = nullptr; s.dRows = nullptr;
    s.dHist = nullptr; s.dTemp = nullptr; s.cap = 0; s.cellsCap = 0; s.tempCap = 0;
}
}  // namespace

namespace { __global__ void probeKernel() {} }

GpuDeviceInfo gpuQueryDevice() {
    GpuDeviceInfo d;
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count == 0) {
        std::snprintf(d.error, sizeof d.error, "%s", e != cudaSuccess ? cudaGetErrorString(e) : "no CUDA device");
        cudaGetLastError();
        return d;
    }
    cudaDeviceProp prop;
    if ((e = cudaGetDeviceProperties(&prop, 0)) != cudaSuccess || (e = cudaMemGetInfo(&d.freeBytes, &d.totalBytes)) != cudaSuccess) {
        std::snprintf(d.error, sizeof d.error, "%s", cudaGetErrorString(e));
        cudaGetLastError();
        return d;
    }
    std::snprintf(d.name, sizeof d.name, "%s", prop.name);
    d.ccMajor = prop.major; d.ccMinor = prop.minor;
    d.multiprocessors = prop.multiProcessorCount;
    d.asyncEngines = prop.asyncEngineCount;
    // A binary without code for this architecture fails at the first launch: detect it now.
    probeKernel<<<1, 1>>>();
    if ((e = cudaGetLastError()) != cudaSuccess || (e = cudaDeviceSynchronize()) != cudaSuccess) {
        std::snprintf(d.error, sizeof d.error, "%s (this binary has no code for compute capability %d.%d)", cudaGetErrorString(e), d.ccMajor, d.ccMinor);
        cudaGetLastError();
        return d;
    }
    d.ok = true;
    return d;
}

GpuEngine::GpuEngine(const GpuConfig& cfg, int numSlots) : p_(new Impl) {
    Impl& P = *p_;
    P.nsc = cfg.numScales;
    if (P.nsc < 1 || P.nsc > fm::kMaxScales) throw std::runtime_error("GpuEngine: bad number of scales");
    P.stride = (cfg.numSlots + 3) / 4 * 4;
    P.profile = gpuProfileEnabled();

    // ascending radii
    int order[fm::kMaxScales];
    for (int i = 0; i < P.nsc; i++) order[i] = i;
    std::sort(order, order + P.nsc, [&](int a, int b) { return cfg.scales[a] < cfg.scales[b]; });
    DevConst dc;
    std::memset(&dc, 0, sizeof dc);
    for (int i = 0; i < P.nsc; i++) {
        P.rAsc[i] = cfg.scales[order[i]];
        dc.orig[i] = order[i];
    }
    for (int f = 0; f < FE_F_COUNT; f++)
        for (int s = 0; s < fm::kMaxScales; s++) dc.slotOf[f][s] = s < P.nsc ? cfg.slotOf[f][s] : -1;

    cudaError_t e = cudaMemcpyToSymbol(c_cfg, &dc, sizeof dc);
    if (e != cudaSuccess) throw std::runtime_error(std::string("GpuEngine: ") + cudaGetErrorString(e));
    P.slots.resize((size_t)std::max(1, numSlots));
    for (Slot& s : P.slots) {
        if ((e = cudaStreamCreateWithFlags(&s.st, cudaStreamNonBlocking)) != cudaSuccess)
            throw std::runtime_error(std::string("GpuEngine: ") + cudaGetErrorString(e));
        for (cudaEvent_t& ev : s.ev) cudaEventCreate(&ev);
    }
}

GpuEngine::~GpuEngine() {
    for (Slot& s : p_->slots) {
        if (s.pending) cudaStreamSynchronize(s.st);
        freeSlot(s);
        for (cudaEvent_t ev : s.ev) cudaEventDestroy(ev);
        cudaStreamDestroy(s.st);
    }
    delete p_;
}

double GpuEngine::deviceBytesPerPoint(const GpuConfig& cfg) {
    const int stride = (cfg.numSlots + 3) / 4 * 4;
    // xyz 12, core 1, keys 8, idx 8, sorted pts 16, sorted core 1, rank 4, core list 4, rows 4*stride
    // + cell offsets (up to 8 cells per point, 4 B) and the cub temporary storage
    return 110.0 + 4.0 * stride;
}
int GpuEngine::rowStride() const { return p_->stride; }
const char* GpuEngine::lastError() const { return p_->error.c_str(); }
StageTimes GpuEngine::times() const { return p_->times; }

bool GpuEngine::reserve(int slot, size_t capacity) {
    Impl& P = *p_;
    Slot& s = P.slots[(size_t)slot];
    if (capacity <= s.cap) return true;
    if (s.pending) cudaStreamSynchronize(s.st);
    freeSlot(s);
    const size_t n = capacity, st = (size_t)P.stride;
    auto fail = [&](cudaError_t e) { P.error = std::string("CUDA error: ") + cudaGetErrorString(e); cudaGetLastError(); freeSlot(s); return false; };
    cudaError_t e;
    if ((e = cudaMallocHost(&s.hXyz, n * 12)) != cudaSuccess) return fail(e);
    if ((e = cudaMallocHost(&s.hCore, n)) != cudaSuccess) return fail(e);
    if ((e = cudaMallocHost(&s.hRows, n * st * 4)) != cudaSuccess) return fail(e);
    if ((e = cudaMalloc(&s.dXyz, n * 12)) != cudaSuccess) return fail(e);
    if ((e = cudaMalloc(&s.dCore, n)) != cudaSuccess) return fail(e);
    if ((e = cudaMalloc(&s.dKeys, n * 4)) != cudaSuccess) return fail(e);
    if ((e = cudaMalloc(&s.dKeysS, n * 4)) != cudaSuccess) return fail(e);
    if ((e = cudaMalloc(&s.dIdx, n * 4)) != cudaSuccess) return fail(e);
    if ((e = cudaMalloc(&s.dIdxS, n * 4)) != cudaSuccess) return fail(e);
    if ((e = cudaMalloc(&s.dPts, n * 16)) != cudaSuccess) return fail(e);
    if ((e = cudaMalloc(&s.dSortedCore, n)) != cudaSuccess) return fail(e);
    if ((e = cudaMalloc(&s.dRank, n * 4)) != cudaSuccess) return fail(e);
    if ((e = cudaMalloc(&s.dCoreList, n * 4)) != cudaSuccess) return fail(e);
    if ((e = cudaMalloc(&s.dNumSel, 4)) != cudaSuccess) return fail(e);
    if ((e = cudaMalloc(&s.dRows, n * st * 4)) != cudaSuccess) return fail(e);
    s.cap = capacity;
    return true;
}

int32_t* GpuEngine::hostXyz(int slot) { return p_->slots[(size_t)slot].hXyz; }
uint8_t* GpuEngine::hostCore(int slot) { return p_->slots[(size_t)slot].hCore; }
const float* GpuEngine::hostRows(int slot) { return p_->slots[(size_t)slot].hRows; }

int GpuEngine::submit(int slot, int n, int nCore, double unit, const int64_t extent[3]) {
    Impl& P = *p_;
    Slot& s = P.slots[(size_t)slot];
    (void)cudaGetLastError();
    if (s.pending) { P.error = "slot still in use"; return -1; }
    if ((size_t)n > s.cap) { P.error = "tile larger than the reserved capacity"; return -1; }
    if (n == 0 || nCore == 0) return 0;

    // --- grid in ticks: cell = largest radius / 2, enlarged when the grid would have far more cells than points
    KernelScales ks;
    std::memset(&ks, 0, sizeof ks);
    ks.unit = (float)unit;
    for (int i = 0; i < P.nsc; i++) ks.r2t[i] = (long long)std::floor((P.rAsc[i] * P.rAsc[i] + 1e-7) / (unit * unit));
    for (int i = P.nsc; i < fm::kMaxScales; i++) ks.r2t[i] = ks.r2t[P.nsc - 1];
    double cell = std::max(std::ceil(P.rAsc[P.nsc - 1] / unit * 0.5), 1.0);
    GridP g;
    const size_t maxCells = (size_t)std::min<double>(100e6, std::max<double>(8.0 * n, 4e6));
    size_t cells;
    for (;;) {
        const double dx = std::floor((double)extent[0] / cell) + 4, dy = std::floor((double)extent[1] / cell) + 4, dz = std::floor((double)extent[2] / cell) + 4;
        const double total = dx * dy * dz;
        if (total <= (double)maxCells) { g.dimX = (int)dx; g.dimY = (int)dy; g.dimZ = (int)dz; cells = (size_t)total; break; }
        cell = std::ceil(cell * 1.25);
    }
    g.cell = (int)cell;
    g.minX = g.minY = g.minZ = -g.cell;       // one empty cell of margin: the tile origin is the minimum of the points

    if (cells + 1 > s.cellsCap) {
        cudaFree(s.dHist); s.dHist = nullptr;
        const size_t want = cells + 1 + (cells >> 3);
        CK(cudaMalloc(&s.dHist, want * 4));
        s.cellsCap = want;
    }
    int bits = 1;
    while (((size_t)1 << bits) < cells) bits++;

    // --- temporary storage of the cub primitives (largest of the four)
    size_t b1 = 0, b2 = 0, b3 = 0, b4 = 0;
    cub::TransformInputIterator<int, ByteToInt, const uint8_t*> coreIt(s.dCore, ByteToInt());
    cub::CountingInputIterator<int> counting(0);
    CK(cub::DeviceRadixSort::SortPairs(nullptr, b1, s.dKeys, s.dKeysS, s.dIdx, s.dIdxS, n, 0, bits, s.st));
    CK(cub::DeviceScan::ExclusiveSum(nullptr, b2, s.dHist, s.dHist, (int)(cells + 1), s.st));
    CK(cub::DeviceScan::ExclusiveSum(nullptr, b3, coreIt, s.dRank, n, s.st));
    CK(cub::DeviceSelect::Flagged(nullptr, b4, counting, s.dSortedCore, s.dCoreList, s.dNumSel, n, s.st));
    const size_t tempBytes = std::max(std::max(b1, b2), std::max(b3, b4));
    if (tempBytes > s.tempCap) {
        cudaFree(s.dTemp); s.dTemp = nullptr;
        CK(cudaMalloc(&s.dTemp, tempBytes + (tempBytes >> 3)));
        s.tempCap = tempBytes + (tempBytes >> 3);
    }

    const bool prof = P.profile;
    if (prof) cudaEventRecord(s.ev[0], s.st);
    CKQ(cudaMemcpyAsync(s.dXyz, s.hXyz, (size_t)n * 12, cudaMemcpyHostToDevice, s.st));
    CKQ(cudaMemcpyAsync(s.dCore, s.hCore, (size_t)n, cudaMemcpyHostToDevice, s.st));
    if (prof) cudaEventRecord(s.ev[1], s.st);

    const int block = 256, grid = (n + block - 1) / block;
    CKQ(cudaMemsetAsync(s.dHist, 0, (cells + 1) * 4, s.st));
    cellKeyKernel<<<grid, block, 0, s.st>>>(s.dXyz, n, g, s.dKeys, s.dIdx, s.dHist);
    CKQ(cudaGetLastError());
    CKQ(cub::DeviceScan::ExclusiveSum(s.dTemp, s.tempCap, s.dHist, s.dHist, (int)(cells + 1), s.st));   // dHist = cell offsets
    CKQ(cub::DeviceScan::ExclusiveSum(s.dTemp, s.tempCap, coreIt, s.dRank, n, s.st));                  // row of each core point
    if (prof) cudaEventRecord(s.ev[2], s.st);

    CKQ(cub::DeviceRadixSort::SortPairs(s.dTemp, s.tempCap, s.dKeys, s.dKeysS, s.dIdx, s.dIdxS, n, 0, bits, s.st));
    reorderKernel<<<grid, block, 0, s.st>>>(s.dXyz, s.dCore, s.dIdxS, n, s.dPts, s.dSortedCore);
    CKQ(cudaGetLastError());
    CKQ(cub::DeviceSelect::Flagged(s.dTemp, s.tempCap, counting, s.dSortedCore, s.dCoreList, s.dNumSel, n, s.st));
    if (prof) cudaEventRecord(s.ev[3], s.st);

    LaunchArgs la = {s.st, s.dPts, s.dIdxS, s.dCoreList, nCore, s.dRank, s.dHist, g, ks, s.dRows, P.stride};
    launchFeature<fm::kMaxScales>(P.nsc, la);
    CKQ(cudaGetLastError());
    if (prof) cudaEventRecord(s.ev[4], s.st);
    CKQ(cudaMemcpyAsync(s.hRows, s.dRows, (size_t)nCore * P.stride * 4, cudaMemcpyDeviceToHost, s.st));
    CKQ(cudaEventRecord(s.ev[5], s.st));
    s.pending = true;
    return 0;
}

int GpuEngine::wait(int slot) {
    Impl& P = *p_;
    Slot& s = P.slots[(size_t)slot];
    if (!s.pending) return 0;
    s.pending = false;
    CK(cudaEventSynchronize(s.ev[5]));
    CK(cudaGetLastError());
    if (P.profile) {
        float t[5];
        for (int k = 0; k < 5; k++) cudaEventElapsedTime(&t[k], s.ev[k], s.ev[k + 1]);
        P.times.h2d += t[0] * 1e-3; P.times.grid += t[1] * 1e-3; P.times.sort += t[2] * 1e-3;
        P.times.kernel += t[3] * 1e-3; P.times.d2h += t[4] * 1e-3;
        P.times.tiles++;
    }
    return 0;
}

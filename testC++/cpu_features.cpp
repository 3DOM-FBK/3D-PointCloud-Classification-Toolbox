// CPU feature engine, see cpu_features.h.
//
// Per tile: integer ticks -> grid of cells (edge = largest radius / 2) -> counting sort by cell (x fastest, so a row of
// cells is one contiguous range of points) -> for every core point ONE pass over the neighbours of the largest radius,
// each neighbour added to the shell of the smallest scale that contains it, shells summed cumulatively, closed formulas.
// The distance computation of a chunk of candidates is a separate branch-free loop compiled for several instruction sets
// (target_clones: SSE2 baseline, AVX2, AVX-512; the best one is chosen when the program starts, so ONE binary is both
// portable and fast).
#include "cpu_features.h"
#include "ooc/cpu_info.h"
#include "ooc/feature_math.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#define CLONES __attribute__((target_clones("avx512f", "avx2", "default")))
#else
#define CLONES
#endif

namespace {

// squared distances (int32) from (px,py,pz) to n points of the SoA arrays; the caller guarantees no overflow
CLONES void dist2_i32(const int32_t* __restrict__ x, const int32_t* __restrict__ y, const int32_t* __restrict__ z, int n,
                      int32_t px, int32_t py, int32_t pz, int32_t* __restrict__ out) {
    for (int k = 0; k < n; k++) {
        const int32_t dx = x[k] - px, dy = y[k] - py, dz = z[k] - pz;
        out[k] = dx * dx + dy * dy + dz * dz;
    }
}
// same in int64 (large radii in ticks)
CLONES void dist2_i64(const int32_t* __restrict__ x, const int32_t* __restrict__ y, const int32_t* __restrict__ z, int n,
                      int32_t px, int32_t py, int32_t pz, int64_t* __restrict__ out) {
    for (int k = 0; k < n; k++) {
        const int64_t dx = (int64_t)x[k] - px, dy = (int64_t)y[k] - py, dz = (int64_t)z[k] - pz;
        out[k] = dx * dx + dy * dy + dz * dz;
    }
}

struct Slot {
    size_t cap = 0;
    std::vector<int32_t> xyz;
    std::vector<uint8_t> core;
    std::vector<float> rows;
    // per-tile workspace, kept between tiles (no allocation and page faults for every tile)
    std::vector<uint32_t> key, off, pos, perm, rank, coreList;
    std::vector<int32_t> sx, sy, sz;
};

class CpuEngine : public FeatureEngine {
public:
    CpuEngine(const FeatureConfig& cfg, int numSlots, int numThreads) : cfg_(cfg), slots_((size_t)std::max(1, numSlots)) {
        nsc_ = cfg.numScales;
        stride_ = (cfg.numSlots + 3) / 4 * 4;
        for (int i = 0; i < nsc_; i++) order_[i] = i;
        std::sort(order_, order_ + nsc_, [&](int a, int b) { return cfg.scales[a] < cfg.scales[b]; });
        threads_ = numThreads > 0 ? numThreads : ooc::available_cpus();
    }
    const char* name() const override { return "CPU"; }
    int rowStride() const override { return stride_; }
    bool reserve(int slot, size_t capacity) override {
        Slot& s = slots_[(size_t)slot];
        if (capacity <= s.cap) return true;
        try {
            s.xyz.assign(capacity * 3, 0);
            s.core.assign(capacity, 0);
            s.rows.assign(capacity * (size_t)stride_, 0.0f);
        } catch (const std::exception& e) {
            error_ = std::string("cannot allocate the tile buffers: ") + e.what();
            return false;
        }
        s.cap = capacity;
        return true;
    }
    int32_t* hostXyz(int slot) override { return slots_[(size_t)slot].xyz.data(); }
    uint8_t* hostCore(int slot) override { return slots_[(size_t)slot].core.data(); }
    const float* hostRows(int slot) override { return slots_[(size_t)slot].rows.data(); }
    int wait(int) override { return 0; }
    const char* lastError() const override { return error_.c_str(); }
    StageTimes times() const override { return times_; }

    int submit(int slot, int n, int nCore, double unit, const int64_t extent[3]) override {
        Slot& s = slots_[(size_t)slot];
        if ((size_t)n > s.cap) { error_ = "tile larger than the reserved capacity"; return -1; }
        if (n == 0 || nCore == 0) return 0;
        try {
            compute(s, n, nCore, unit, extent);
        } catch (const std::exception& e) {
            error_ = std::string("CPU engine: ") + e.what();
            return -1;
        }
        times_.tiles++;
        return 0;
    }

private:
    void compute(Slot& s, int n, int nCore, double unit, const int64_t extent[3]) {
        // squared radii in ticks^2 (ascending, inclusive): d2 * unit^2 <= r^2 + 1e-7, as in the previous versions
        int64_t r2t[fm::kMaxScales];
        for (int i = 0; i < nsc_; i++) {
            const double r = cfg_.scales[order_[i]];
            r2t[i] = (int64_t)std::floor((r * r + 1e-7) / (unit * unit));
        }
        const int64_t rmax2 = r2t[nsc_ - 1];
        const int rt = (int)std::ceil(std::sqrt((double)rmax2)) + 1;

        // grid: cell edge = radius / 2 (ticks), enlarged when the grid would have far more cells than points
        double cell = std::max(std::ceil(cfg_.scales[order_[nsc_ - 1]] / unit * 0.5), 1.0);
        const size_t maxCells = (size_t)std::min<double>(100e6, std::max<double>(8.0 * n, 4e6));
        int dimX, dimY, dimZ;
        size_t cells;
        for (;;) {
            const double dx = std::floor((double)extent[0] / cell) + 4, dy = std::floor((double)extent[1] / cell) + 4, dz = std::floor((double)extent[2] / cell) + 4;
            if (dx * dy * dz <= (double)maxCells) { dimX = (int)dx; dimY = (int)dy; dimZ = (int)dz; cells = (size_t)(dx * dy * dz); break; }
            cell = std::ceil(cell * 1.25);
        }
        const int cellT = (int)cell;
        const int origin = -cellT;                       // one empty cell of margin
        const double invCell = 1.0 / cellT;
        auto coord = [&](int v, int dim) {               // exact floor((v - origin) / cellT), clamped
            const int64_t d = (int64_t)v - origin;
            if (d < 0) return 0;
            int c = (int)((double)d * invCell);
            if ((int64_t)(c + 1) * cellT <= d) c++;
            else if ((int64_t)c * cellT > d) c--;
            return c >= dim ? dim - 1 : c;
        };

        const int32_t* xyz = s.xyz.data();
        auto& key = s.key; key.resize((size_t)n);
        auto& off = s.off; off.assign(cells + 1, 0);
#pragma omp parallel for schedule(static) num_threads(threads_)
        for (int i = 0; i < n; i++) {
            const int cx = coord(xyz[3 * (size_t)i], dimX), cy = coord(xyz[3 * (size_t)i + 1], dimY), cz = coord(xyz[3 * (size_t)i + 2], dimZ);
            key[(size_t)i] = (uint32_t)(cx + (size_t)dimX * ((size_t)cy + (size_t)dimY * cz));
        }
        for (int i = 0; i < n; i++) off[key[(size_t)i] + 1]++;
        for (size_t c = 0; c < cells; c++) off[c + 1] += off[c];
        auto &sx = s.sx, &sy = s.sy, &sz = s.sz;
        sx.resize((size_t)n); sy.resize((size_t)n); sz.resize((size_t)n);
        auto& perm = s.perm; perm.resize((size_t)n);            // sorted position -> original index
        {
            auto& pos = s.pos; pos.assign(off.begin(), off.end() - 1);
            for (int i = 0; i < n; i++) {
                const uint32_t p = pos[key[(size_t)i]]++;
                sx[p] = xyz[3 * (size_t)i]; sy[p] = xyz[3 * (size_t)i + 1]; sz[p] = xyz[3 * (size_t)i + 2];
                perm[p] = (uint32_t)i;
            }
        }
        auto& rank = s.rank; rank.resize((size_t)n);            // original index -> row among the core points
        {
            uint32_t r = 0;
            for (int i = 0; i < n; i++) { rank[(size_t)i] = r; r += s.core[(size_t)i] ? 1u : 0u; }
        }
        auto& coreList = s.coreList; coreList.clear();                 // sorted positions of the core points
        coreList.reserve((size_t)nCore);
        for (int p = 0; p < n; p++) if (s.core[perm[(size_t)p]]) coreList.push_back((uint32_t)p);

        const bool useI32 = 3.0 * (3.0 * rt + 2.0 * cellT) * (3.0 * rt + 2.0 * cellT) < 2.0e9;
        float* rows = s.rows.data();
        const int nsc = nsc_, stride = stride_;
        const size_t nList = coreList.size();
        const float fu = (float)unit;

#pragma omp parallel num_threads(threads_)
        {
            std::vector<int32_t> d32(256);
            std::vector<int64_t> d64(useI32 ? 0 : 256);
#pragma omp for schedule(dynamic, 64)
            for (int64_t t = 0; t < (int64_t)nList; t++) {
                const uint32_t pos = coreList[(size_t)t];
                const int32_t px = sx[pos], py = sy[pos], pz = sz[pos];
                fm::Acc acc[fm::kMaxScales];
                for (int k = 0; k < nsc; k++) fm::acc_init(acc[k]);
                // adds the candidates of a chunk that are inside the largest radius to the shell of their smallest radius
                auto consume = [&](const auto* d2s, uint32_t j0, int m) {
                    for (int k = 0; k < m; k++) {
                        const int64_t d2 = d2s[k];
                        if (d2 > rmax2) continue;
                        for (int sc = 0; sc < nsc; sc++)
                            if (d2 <= r2t[sc]) { fm::acc_add(acc[sc], (float)(sx[j0 + k] - px) * fu, (float)(sy[j0 + k] - py) * fu, (float)(sz[j0 + k] - pz) * fu); break; }
                    }
                };
                const int ix0 = coord(px - rt, dimX), ix1 = coord(px + rt, dimX);
                const int iy0 = coord(py - rt, dimY), iy1 = coord(py + rt, dimY);
                const int iz0 = coord(pz - rt, dimZ), iz1 = coord(pz + rt, dimZ);
                for (int iz = iz0; iz <= iz1; iz++) {
                    const int64_t z0 = (int64_t)origin + (int64_t)iz * cellT;
                    const int64_t gz = pz < z0 ? z0 - pz : (pz > z0 + cellT - 1 ? pz - (z0 + cellT - 1) : 0);
                    if (gz * gz > rmax2) continue;
                    for (int iy = iy0; iy <= iy1; iy++) {
                        const int64_t y0 = (int64_t)origin + (int64_t)iy * cellT;
                        const int64_t gy = py < y0 ? y0 - py : (py > y0 + cellT - 1 ? py - (y0 + cellT - 1) : 0);
                        if (gz * gz + gy * gy > rmax2) continue;
                        const size_t row = (size_t)dimX * ((size_t)iy + (size_t)dimY * iz);
                        const uint32_t b = off[row + (size_t)ix0], e = off[row + (size_t)ix1 + 1];
                        for (uint32_t j0 = b; j0 < e; j0 += 256) {
                            const int m = (int)std::min<uint32_t>(256, e - j0);
                            if (useI32) {
                                dist2_i32(&sx[j0], &sy[j0], &sz[j0], m, px, py, pz, d32.data());
                                consume(d32.data(), j0, m);
                            } else {
                                dist2_i64(&sx[j0], &sy[j0], &sz[j0], m, px, py, pz, d64.data());
                                consume(d64.data(), j0, m);
                            }
                        }
                    }
                }
                for (int sc = 1; sc < nsc; sc++) fm::acc_merge(acc[sc], acc[sc - 1]);
                float* out = rows + (size_t)rank[perm[pos]] * stride;
                for (int sc = 0; sc < nsc; sc++) {
                    float f[fm::kFeatureCount];
                    fm::finalize(acc[sc], f);
                    const int o = order_[sc];
                    for (int k = 0; k < fm::kFeatureCount; k++) {
                        const int slot = cfg_.slotOf[k][o];
                        if (slot >= 0) out[slot] = f[k];
                    }
                }
            }
        }
    }

    FeatureConfig cfg_;
    std::vector<Slot> slots_;
    int nsc_ = 0, stride_ = 0, threads_ = 1;
    int order_[fm::kMaxScales] = {};
    std::string error_;
    StageTimes times_;
};

}  // namespace

std::unique_ptr<FeatureEngine> makeCpuEngine(const FeatureConfig& cfg, int numSlots, int numThreads) {
    return std::unique_ptr<FeatureEngine>(new CpuEngine(cfg, numSlots, numThreads));
}

double cpuBytesPerPoint(const FeatureConfig& cfg) {
    const int stride = (cfg.numSlots + 3) / 4 * 4;
    // staging xyz 12 + core 1 + rows 4*stride; workspace: key 4, perm 4, sorted xyz 12, rank 4, core list 4, grid ~4
    // + the cell offsets (up to 8 cells per point, 4 B)
    return 77.0 + 4.0 * stride;
}

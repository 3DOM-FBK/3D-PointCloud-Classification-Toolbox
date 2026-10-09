// las2pc — converts a canonical features.las into the chunked point cloud format ("pck")
// used by the viewer. See docs/POINTCLOUD_FORMAT.md.
//
// Output folder layout:
//   geom.bin         20 B records: int32 x,y,z | uint8 r,g,b,pad | uint32 POINT_ID
//   point_order.bin  uint32 POINT_ID in geom.bin order
//   meta.json        chunks, levels, bbox (no columns: they are written by pc_columns.py)
// and, with --ordered-las, the input LAS rewritten in the order of geom.bin (row r of the LAS is point r of
// the geometry): this is the "canonical" features.las every later tool preserves.
//
// Algorithm:
//   1. Counting grid (128^3, 256^3, 512^3 by size) over the cubic bbox (the bbox of the LAS header, checked
//      while counting; measured again only if a point falls outside), merged bottom-up into an implicit octree
//      whose leaves hold at most --max-chunk points (chunks). A grid cell that is still denser is split into
//      octants (down to --max-chunk), so every chunk has bounded size.
//   2. Distribution of the points by chunk.
//   3. Inside every chunk: the points are sorted by POINT_ID (the result must not depend on the order of the
//      input records), then a deterministic random permutation, then stratified levels: level l keeps the first
//      point (in permutation order) that finds its cell free in a 2^(base+l) grid. Points that never find a free
//      cell go to the last level.
//   4. Points are laid out as [head block: levels < H of every chunk][per chunk: levels >= H].
//
// Two execution modes with the SAME output, byte for byte:
//   memory  the LAS is mapped and the points are reordered through an index array (about 28 B/point of RAM and the
//           whole LAS in the page cache): fastest when everything fits in the memory budget;
//   ooc     out-of-core: every pass reads the input sequentially, the points are distributed in one temporary
//           file per chunk (bucket files), each chunk is processed in memory (threads x chunk), and the outputs
//           are written with pwrite at their final position. Memory is bounded by --memory-budget.
// --mode auto picks "memory" when LAS size + 28 B/point fits in the budget.
//
// Only LAS 1.x uncompressed files are supported.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <malloc.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ooc/bucket_writer.h"
#include "ooc/las_stream.h"
#include "ooc/memory_budget.h"
#include "ooc/positional_writer.h"
#include "ooc/temp_dir.h"

namespace {

using ooc::rd;

// ------------------------------------------------------------------ helpers

struct MappedFile {
    const uint8_t* data = nullptr;
    size_t size = 0;
    int fd = -1;
    explicit MappedFile(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open: " + path);
        struct stat st;
        if (fstat(fd, &st) != 0) throw std::runtime_error("Cannot stat: " + path);
        size = (size_t)st.st_size;
        void* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) throw std::runtime_error("Cannot mmap: " + path);
        data = (const uint8_t*)p;
        madvise((void*)data, size, MADV_WILLNEED);
    }
    ~MappedFile() {
        if (data) munmap((void*)data, size);
        if (fd >= 0) ::close(fd);
    }
};

// Output file written through a writable mapping (memory mode only: the pages are dirty until written back).
struct OutFile {
    uint8_t* data = nullptr;
    size_t size = 0;
    int fd = -1;
    OutFile(const std::string& path, size_t bytes) : size(bytes) {
        fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) throw std::runtime_error("Cannot create: " + path);
        if (bytes == 0) return;
        if (ftruncate(fd, (off_t)bytes) != 0) throw std::runtime_error("Cannot resize: " + path);
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) throw std::runtime_error("Cannot mmap for writing: " + path);
        data = (uint8_t*)p;
    }
    ~OutFile() {
        if (data) { msync(data, size, MS_ASYNC); munmap(data, size); }
        if (fd >= 0) ::close(fd);
    }
};

// Output file written with pwrite (ooc mode).
struct PosFile {
    int fd = -1;
    std::string path;
    PosFile(const std::string& p, uint64_t bytes) : path(p) {
        fd = ::open(p.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) throw std::runtime_error("Cannot create: " + p);
        if (ftruncate(fd, (off_t)bytes) != 0) throw std::runtime_error("Cannot resize " + p + " (disk full?)");
    }
    ~PosFile() { if (fd >= 0) ::close(fd); }
};

struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    // Unbiased enough for shuffling: multiply-shift on the 32 high bits.
    uint32_t below(uint32_t n) { return (uint32_t)(((next() >> 32) * (uint64_t)n) >> 32); }
};

std::string num(double v) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

using clk = std::chrono::steady_clock;
clk::time_point g_t0;

// Peak resident set of the process so far (VmHWM), in MB; 0 when /proc is not available.
long peak_rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string key;
    while (f >> key) {
        if (key == "VmHWM:") { long kb = 0; f >> kb; return kb / 1024; }
        f.ignore(1 << 20, '\n');
    }
    return 0;
}

void lap(const char* what) {
    double s = std::chrono::duration<double>(clk::now() - g_t0).count();
    std::cout << "  [" << what << "] " << s << " s  (peak RSS " << peak_rss_mb() << " MB)" << std::endl;
}

// Progress for the backend ("[progress] phase 40%"), printed when the integer percentage changes.
struct Progress {
    std::string phase;
    uint64_t total;
    std::atomic<uint64_t> done{0};
    std::atomic<int> last{-1};
    Progress(std::string p, uint64_t t) : phase(std::move(p)), total(t ? t : 1) {}
    void add(uint64_t n) {
        uint64_t d = done.fetch_add(n) + n;
        int pct = (int)(100 * d / total);
        int prev = last.load();
        while (pct / 10 > prev / 10 && !last.compare_exchange_weak(prev, pct)) {}
        if (pct / 10 > prev / 10) { std::cout << "[progress] " << phase << " " << (pct / 10) * 10 << "%" << std::endl; }
    }
};

// First exception raised inside an OpenMP region (exceptions must not leave it).
struct ErrorBox {
    std::exception_ptr e;
    std::atomic<bool> set{false};
    void capture() {
#pragma omp critical(errbox)
        { if (!e) e = std::current_exception(); }
        set = true;
    }
    void rethrow() { if (e) std::rethrow_exception(e); }
};

// ------------------------------------------------------------------ chunking

struct Chunk {
    double cubeMin[3];    // octree node cube (relative to bbox cube origin)
    double size;
    uint32_t points = 0;
    uint64_t begin = 0;   // memory mode: offset into the chunk-grouped index array
    std::string file;     // ooc mode: temporary file with the full records of the chunk
    std::vector<uint32_t> levelPoints;
    uint64_t headOffset = 0, bodyOffset = 0;
    double tmin[3], tmax[3];  // tight AABB (relative to the bbox min)
};

struct Params {
    std::string input, output, orderedLas, tempDir;
    uint32_t maxChunk = 250000;
    int base = 3;
    int levels = 6;                // number of regular levels (the last, remainder level is extra)
    uint64_t headBudget = 1000000;
    uint64_t seed = 12345;
    int mode = 0;                  // 0 auto, 1 memory, 2 ooc
    std::string stopAfter;         // benchmarking: count | distribute | levels
    double memoryBudgetMb = 0;
};

void usage() {
    std::cerr << "Usage: las2pc --input features.las --output out_dir [--ordered-las out.las] [--max-chunk 250000]\n"
                 "              [--base 3] [--levels 6] [--head-budget 1000000] [--seed 12345]\n"
                 "              [--mode auto|memory|ooc] [--memory-budget MB] [--temp-dir DIR]\n";
}

// Geometry shared by both modes
struct Space {
    int32_t q0[3];        // integer origin of the cube (= qMin of the output)
    double bmin[3];       // real coordinate of q0
    double cubeSize;
    int G, depth;
    double invCube;
};

const int MAX_REFINE = 14;   // octant subdivisions below the counting grid for chunks that are still too dense

// Cell of the counting grid (clamped: only meaningful when the point is inside the bbox)
inline uint32_t cell_of(const uint8_t* r, const Space& S, const ooc::LasInfo& li) {
    int c[3];
    for (int k = 0; k < 3; k++) {
        double rel = (double)((int64_t)rd<int32_t>(r + 4 * k) - S.q0[k]) * li.scale[k];
        int v = (int)(rel * S.invCube);
        c[k] = v < 0 ? 0 : (v >= S.G ? S.G - 1 : v);
    }
    return (uint32_t)(((uint64_t)c[2] * S.G + c[1]) * S.G + c[0]);
}

inline int octant_of(const uint8_t* r, const Space& S, const ooc::LasInfo& li, const Chunk& c) {
    int o = 0;
    for (int k = 0; k < 3; k++) {
        double rel = (double)((int64_t)rd<int32_t>(r + 4 * k) - S.q0[k]) * li.scale[k];
        if (rel >= c.cubeMin[k] + c.size * 0.5) o |= 1 << k;
    }
    return o;
}

Chunk child_chunk(const Chunk& p, int o) {
    Chunk c;
    c.size = p.size * 0.5;
    for (int k = 0; k < 3; k++) c.cubeMin[k] = p.cubeMin[k] + ((o >> k) & 1 ? c.size : 0.0);
    return c;
}

// Sorts the entries by POINT_ID; points with the same id (never in a valid file) by their record bytes, so the
// result never depends on the order in which the records were read.
struct PI { uint32_t pid, idx; };

template <class RecOf>
void sort_by_pid(std::vector<PI>& v, size_t recLen, RecOf recOf) {
    std::sort(v.begin(), v.end(), [](const PI& a, const PI& b) { return a.pid < b.pid; });
    for (size_t i = 0; i < v.size();) {
        size_t j = i + 1;
        while (j < v.size() && v[j].pid == v[i].pid) j++;
        if (j - i > 1)
            std::sort(v.begin() + i, v.begin() + j, [&](const PI& a, const PI& b) {
                int c = std::memcmp(recOf(a.idx), recOf(b.idx), recLen);
                return c != 0 ? c < 0 : a.idx < b.idx;
            });
        i = j;
    }
}

// Per-thread scratch of the level computation
struct LevelScratch {
    std::vector<uint8_t> occupied;
    std::vector<uint32_t> perm, remaining, nextRemaining, levelOrder;
};

// q: (point - origin) integer coordinates of the sorted points, n*3. Fills chunk levelPoints / AABB and returns the
// points in final order as indices into the sorted list.
void levelize(Chunk& c, size_t ci, const std::vector<int32_t>& q, const double* scale, const Params& P,
              LevelScratch& s, std::vector<uint32_t>& order) {
    const uint32_t n = c.points;
    s.perm.resize(n);
    std::iota(s.perm.begin(), s.perm.end(), 0u);
    SplitMix64 rng(P.seed * 0x9E3779B97F4A7C15ULL + (uint64_t)ci * 0xD1B54A32D192ED03ULL + 1);
    for (uint32_t i = n; i > 1; i--) std::swap(s.perm[i - 1], s.perm[rng.below(i)]);

    double tmin[3] = {1e300, 1e300, 1e300}, tmax[3] = {-1e300, -1e300, -1e300};
    for (uint32_t i = 0; i < n; i++)
        for (int k = 0; k < 3; k++) {
            double rel = (double)q[(size_t)i * 3 + k] * scale[k];
            tmin[k] = std::min(tmin[k], rel);
            tmax[k] = std::max(tmax[k], rel);
        }
    for (int k = 0; k < 3; k++) { c.tmin[k] = tmin[k]; c.tmax[k] = tmax[k]; }

    const int L = P.levels;
    const int nLevels = L + 1;
    s.remaining = s.perm;
    s.levelOrder.clear();
    s.levelOrder.reserve(n);
    c.levelPoints.assign(nLevels, 0);
    const double inv = 1.0 / c.size;
    for (int l = 0; l < L && !s.remaining.empty(); l++) {
        const int R = 1 << (P.base + l);
        s.occupied.assign((size_t)R * R * R, 0);
        s.nextRemaining.clear();
        uint32_t taken = 0;
        for (size_t j = 0; j < s.remaining.size(); j++) {
            const int32_t* qq = &q[(size_t)s.remaining[j] * 3];
            int cc[3];
            for (int k = 0; k < 3; k++) {
                double rel = ((double)qq[k] * scale[k] - c.cubeMin[k]) * inv * R;
                int v = (int)rel;
                cc[k] = v < 0 ? 0 : (v >= R ? R - 1 : v);
            }
            size_t cell = ((size_t)cc[2] * R + cc[1]) * R + cc[0];
            if (!s.occupied[cell]) {
                s.occupied[cell] = 1;
                s.levelOrder.push_back(s.remaining[j]);
                taken++;
            } else {
                s.nextRemaining.push_back(s.remaining[j]);
            }
        }
        c.levelPoints[l] = taken;
        s.remaining.swap(s.nextRemaining);
    }
    c.levelPoints[L] = (uint32_t)s.remaining.size();
    s.levelOrder.insert(s.levelOrder.end(), s.remaining.begin(), s.remaining.end());
    order = s.levelOrder;
}

// ------------------------------------------------------------------ geometry record

inline void geom_record(const uint8_t* r, const ooc::LasInfo& li, const int32_t* q0, bool color16, uint32_t pid,
                        uint8_t* o) {
    for (int k = 0; k < 3; k++) {
        int32_t q = (int32_t)((int64_t)rd<int32_t>(r + 4 * k) - q0[k]);
        std::memcpy(o + 4 * k, &q, 4);
    }
    if (li.rgbOff >= 0) {
        for (int k = 0; k < 3; k++) {
            uint16_t v = rd<uint16_t>(r + li.rgbOff + 2 * k);
            o[12 + k] = color16 ? (uint8_t)(v >> 8) : (uint8_t)v;
        }
    } else {
        o[12] = o[13] = o[14] = 255;
    }
    o[15] = 0;
    std::memcpy(o + 16, &pid, 4);
}

}  // namespace

int main(int argc, char** argv) {
    g_t0 = clk::now();

    Params P;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--input" || a == "-i") P.input = next();
        else if (a == "--output" || a == "-o") P.output = next();
        else if (a == "--ordered-las") P.orderedLas = next();
        else if (a == "--max-chunk") P.maxChunk = (uint32_t)std::stoul(next());
        else if (a == "--base") P.base = std::stoi(next());
        else if (a == "--levels") P.levels = std::stoi(next());
        else if (a == "--head-budget") P.headBudget = std::stoull(next());
        else if (a == "--seed") P.seed = std::stoull(next());
        else if (a == "--memory-budget") P.memoryBudgetMb = std::stod(next());
        else if (a == "--temp-dir") P.tempDir = next();
        else if (a == "--stop-after") P.stopAfter = next();
        else if (a == "--mode") {
            std::string m = next();
            if (m == "auto") P.mode = 0; else if (m == "memory") P.mode = 1; else if (m == "ooc") P.mode = 2;
            else { usage(); return 2; }
        }
        else { usage(); return 2; }
    }
    if (P.input.empty() || P.output.empty() || P.levels < 1 || P.base < 1 || P.base + P.levels > 9 ||
        P.maxChunk < 1000) { usage(); return 2; }

    // On SIGTERM (/stop_process/) the temporary folder and the partial outputs are removed
    ooc::TempWorkDir::install_handlers();
    for (const char* f : {"/geom.bin", "/point_order.bin", "/meta.json"}) ooc::TempWorkDir::extra_paths().push_back(P.output + f);
    if (!P.orderedLas.empty()) ooc::TempWorkDir::extra_paths().push_back(P.orderedLas);

    try {
        std::cout << "[las2pc] Input: " << P.input << std::endl;
        ooc::LasInfo li = ooc::read_las_info(P.input);
        const uint64_t N = li.numPoints;
        if (N == 0) throw std::runtime_error("The LAS file has no points");
        if (N >= 0xFFFFFFFFULL) throw std::runtime_error("Too many points (> 4G)");
        const size_t RL = (size_t)li.recordLength;
        int numThreads = 1;
#ifdef _OPENMP
        numThreads = omp_get_max_threads();
#endif
        std::cout << "  points=" << N << " format=" << li.format << " record=" << RL
                  << " rgb=" << (li.rgbOff >= 0) << " point_id=" << (li.pidOff >= 0)
                  << " threads=" << numThreads << std::endl;
        if (li.pidOff < 0)
            std::cout << "  [Warning] POINT_ID extra byte not found: the point index is used instead" << std::endl;

        // ---- mode and budget
        const uint64_t budget = ooc::resolve_budget_bytes(P.memoryBudgetMb);
        const double memNeed = (double)li.fileSize + (double)N * 28.0;
        bool memoryMode;
        if (P.mode == 1) memoryMode = true;
        else if (P.mode == 2) memoryMode = false;
        else memoryMode = memNeed <= (double)budget;
        if (P.mode == 1) {
            double avail = ooc::available_memory_bytes();
            if (avail > 0 && memNeed > avail * 0.9)
                throw std::runtime_error("Not enough memory for in-memory conversion: needs ~" +
                                         std::to_string((uint64_t)(memNeed / 1e6)) + " MB, available ~" +
                                         std::to_string((uint64_t)(avail / 1e6)) + " MB (use --mode ooc)");
        }
        if (!memoryMode) {
            // Big buffers straight from mmap and returned to the OS when freed: the per-thread arenas of glibc would
            // otherwise keep them and the peak RSS would exceed the budget (measured: 626 MB -> 347 MB for 500 MB).
            mallopt(M_MMAP_THRESHOLD, 1 << 20);
            mallopt(M_TRIM_THRESHOLD, 1 << 20);
        }
        std::cout << "  mode=" << (memoryMode ? "memory" : "ooc") << " budget=" << budget / 1048576 << " MB (needs ~"
                  << (uint64_t)(memNeed / 1e6) << " MB in memory)" << std::endl;

        const bool pidInRecord = li.pidOff >= 0;
        const size_t TRL = RL + (pidInRecord ? 0 : 4);   // record length in the temporary chunk files
        auto pidOfTmp = [&](const uint8_t* r) -> uint32_t {
            return pidInRecord ? rd<uint32_t>(r + li.pidOff) : rd<uint32_t>(r + RL);
        };

        std::unique_ptr<MappedFile> mapped;
        std::unique_ptr<ooc::LasStreamReader> reader;
        const uint8_t* rec0 = nullptr;
        // Block size of the sequential passes: bounded by the budget (one read buffer + one sorted buffer per thread)
        uint64_t blockBytes = std::min<uint64_t>(32ULL << 20, std::max<uint64_t>(1ULL << 20, budget / 8 / (uint64_t)(2 * numThreads)));
        if (memoryMode) {
            mapped.reset(new MappedFile(P.input));
            rec0 = mapped->data + li.offsetToData;
        } else {
            reader.reset(new ooc::LasStreamReader(P.input, li, blockBytes));
        }
        auto rec = [&](uint64_t i) { return rec0 + i * RL; };

        // Runs fn(recs, firstRow, rows) over all the records, in parallel over blocks. In memory mode the blocks point into the
        // mapping; in ooc mode each thread preads its own block.
        auto forEachBlock = [&](const char* phase, auto&& fn) {
            Progress prog(phase, N);
            ErrorBox err;
            if (memoryMode) {
                const uint64_t rows = 1 << 20;
                const int64_t nb = (int64_t)((N + rows - 1) / rows);
#pragma omp parallel for schedule(dynamic, 1)
                for (int64_t b = 0; b < nb; b++) {
                    if (err.set) continue;
                    try {
                        uint64_t first = (uint64_t)b * rows;
                        size_t m = (size_t)std::min<uint64_t>(rows, N - first);
                        fn(rec(first), first, m);
                        prog.add(m);
                    } catch (...) { err.capture(); }
                }
            } else {
                const int64_t nb = (int64_t)reader->numBlocks();
#pragma omp parallel
                {
                    std::vector<uint8_t> buf;
                    try { buf.resize((size_t)reader->blockRows() * RL); } catch (...) { err.capture(); }
#pragma omp for schedule(dynamic, 1)
                    for (int64_t b = 0; b < nb; b++) {
                        if (err.set) continue;
                        try {
                            size_t m = reader->read((uint64_t)b, buf.data());
                            fn(buf.data(), reader->firstRow((uint64_t)b), m);
                            prog.add(m);
                        } catch (...) { err.capture(); }
                    }
                }
            }
            err.rethrow();
        };

        // ---- Pass 1: counting grid. The bbox of the header (widened by one cell) is used as an estimate and
        //      checked against every point while counting; if a point falls outside, the bbox is measured and
        //      the counting restarts. The exact integer bbox and the colour range are measured in the same pass.
        int32_t mn[3] = {INT32_MAX, INT32_MAX, INT32_MAX}, mx[3] = {INT32_MIN, INT32_MIN, INT32_MIN};
        uint32_t maxColor = 0;
        const int G = N >= 500000000ULL ? 512 : (N > 100000000ULL ? 256 : 128);
        const int depth = G == 512 ? 9 : (G == 256 ? 8 : 7);
        const uint64_t G3 = (uint64_t)G * G * G;
        Space S;
        S.G = G; S.depth = depth;
        std::vector<uint32_t> grid;
        std::vector<uint32_t> pointCell;
        if (memoryMode) pointCell.resize(N);

        auto setSpace = [&](const int64_t lo[3], const int64_t hi[3]) {
            double ext[3];
            for (int k = 0; k < 3; k++) ext[k] = (double)(hi[k] - lo[k]) * li.scale[k];
            S.cubeSize = std::max({ext[0], ext[1], ext[2], 1e-9});
            S.invCube = (double)G / S.cubeSize;
            for (int k = 0; k < 3; k++) {
                S.q0[k] = (int32_t)lo[k];
                S.bmin[k] = (double)S.q0[k] * li.scale[k] + li.offset[k];
            }
        };
        auto countPass = [&](bool& outside) {
            grid.assign(G3, 0);
            int32_t gmn[3] = {INT32_MAX, INT32_MAX, INT32_MAX}, gmx[3] = {INT32_MIN, INT32_MIN, INT32_MIN};
            uint32_t gmc = 0;
            std::atomic<bool> out{false};
            // per-thread min/max merged under a lock at the end of each block
            forEachBlock("las2pc count", [&](const uint8_t* recs, uint64_t first, size_t m) {
                if (out.load(std::memory_order_relaxed)) return;
                int32_t lmin[3] = {INT32_MAX, INT32_MAX, INT32_MAX}, lmax[3] = {INT32_MIN, INT32_MIN, INT32_MIN};
                uint32_t lmc = 0;
                bool bad = false;
                for (size_t i = 0; i < m; i++) {
                    const uint8_t* r = recs + i * RL;
                    int32_t q[3];
                    for (int k = 0; k < 3; k++) {
                        q[k] = rd<int32_t>(r + 4 * k);
                        lmin[k] = std::min(lmin[k], q[k]);
                        lmax[k] = std::max(lmax[k], q[k]);
                    }
                    if (li.rgbOff >= 0) {
                        lmc = std::max<uint32_t>(lmc, rd<uint16_t>(r + li.rgbOff));
                        lmc = std::max<uint32_t>(lmc, rd<uint16_t>(r + li.rgbOff + 2));
                        lmc = std::max<uint32_t>(lmc, rd<uint16_t>(r + li.rgbOff + 4));
                    }
                    uint32_t c = cell_of(r, S, li);
                    for (int k = 0; k < 3; k++)
                        if ((int64_t)q[k] < S.q0[k] || (double)((int64_t)q[k] - S.q0[k]) * li.scale[k] > S.cubeSize * (1 + 1e-12)) bad = true;
                    if (memoryMode) pointCell[first + i] = c;
#pragma omp atomic
                    grid[c]++;
                }
#pragma omp critical(minmax)
                {
                    for (int k = 0; k < 3; k++) { gmn[k] = std::min(gmn[k], lmin[k]); gmx[k] = std::max(gmx[k], lmax[k]); }
                    gmc = std::max(gmc, lmc);
                }
                if (bad) out = true;
            });
            outside = out;
            for (int k = 0; k < 3; k++) { mn[k] = gmn[k]; mx[k] = gmx[k]; }
            maxColor = gmc;
        };
        auto measurePass = [&]() {
            int32_t gmn[3] = {INT32_MAX, INT32_MAX, INT32_MAX}, gmx[3] = {INT32_MIN, INT32_MIN, INT32_MIN};
            uint32_t gmc = 0;
            forEachBlock("las2pc bbox", [&](const uint8_t* recs, uint64_t, size_t m) {
                int32_t lmin[3] = {INT32_MAX, INT32_MAX, INT32_MAX}, lmax[3] = {INT32_MIN, INT32_MIN, INT32_MIN};
                uint32_t lmc = 0;
                for (size_t i = 0; i < m; i++) {
                    const uint8_t* r = recs + i * RL;
                    for (int k = 0; k < 3; k++) {
                        int32_t q = rd<int32_t>(r + 4 * k);
                        lmin[k] = std::min(lmin[k], q);
                        lmax[k] = std::max(lmax[k], q);
                    }
                    if (li.rgbOff >= 0)
                        for (int k = 0; k < 3; k++) lmc = std::max<uint32_t>(lmc, rd<uint16_t>(r + li.rgbOff + 2 * k));
                }
#pragma omp critical(minmax)
                {
                    for (int k = 0; k < 3; k++) { gmn[k] = std::min(gmn[k], lmin[k]); gmx[k] = std::max(gmx[k], lmax[k]); }
                    gmc = std::max(gmc, lmc);
                }
            });
            for (int k = 0; k < 3; k++) { mn[k] = gmn[k]; mx[k] = gmx[k]; }
            maxColor = gmc;
        };

        {
            // estimate from the header, widened by one grid cell
            int64_t lo[3], hi[3];
            bool headerOk = true;
            double ext[3];
            for (int k = 0; k < 3; k++) {
                ext[k] = li.hdrMax[k] - li.hdrMin[k];
                if (!(ext[k] >= 0) || !std::isfinite(ext[k])) headerOk = false;
            }
            if (headerOk) {
                double cell = std::max({ext[0], ext[1], ext[2], 1e-9}) / G;
                for (int k = 0; k < 3; k++) {
                    double l = std::floor((li.hdrMin[k] - cell - li.offset[k]) / li.scale[k]);
                    double h = std::ceil((li.hdrMax[k] + cell - li.offset[k]) / li.scale[k]);
                    if (l < (double)INT32_MIN || h > (double)INT32_MAX) headerOk = false;
                    lo[k] = (int64_t)l; hi[k] = (int64_t)h;
                }
            }
            bool outside = true;
            if (headerOk) {
                setSpace(lo, hi);
                countPass(outside);
                if (outside) std::cout << "  [Info] the bbox of the LAS header does not contain every point: measuring the real one" << std::endl;
            } else {
                std::cout << "  [Info] the LAS header has no usable bbox: measuring it" << std::endl;
            }
            if (outside) {
                measurePass();
                int64_t l2[3], h2[3];
                for (int k = 0; k < 3; k++) { l2[k] = mn[k]; h2[k] = mx[k]; }
                setSpace(l2, h2);
                bool again = false;
                countPass(again);
                if (again) throw std::runtime_error("Internal error: points outside their own bbox");
            }
        }
        const bool color16 = maxColor > 255;
        lap("count grid");
        if (P.stopAfter == "count") { std::cout << "stopped after the count pass" << std::endl; return 0; }

        // ---- Bottom-up merge into an implicit octree; leaves with <= maxChunk points become chunks
        std::vector<std::vector<uint32_t>> pyramid(depth + 1);  // pyramid[d] has (2^d)^3 cells, d = depth is the grid
        pyramid[depth] = std::move(grid);
        for (int d = depth - 1; d >= 0; d--) {
            int n = 1 << d;
            pyramid[d].assign((size_t)n * n * n, 0);
            int nc = n * 2;
            for (int z = 0; z < nc; z++)
                for (int y = 0; y < nc; y++)
                    for (int x = 0; x < nc; x++)
                        pyramid[d][((size_t)(z >> 1) * n + (y >> 1)) * n + (x >> 1)] +=
                            pyramid[d + 1][((size_t)z * nc + y) * nc + x];
        }
        std::vector<uint32_t> chunkOfCell(G3, 0xFFFFFFFFu);
        std::vector<Chunk> chunks;
        {
            struct Node { int d, x, y, z; };
            std::vector<Node> stack{{0, 0, 0, 0}};
            // Iterative DFS, children visited in Z-order so chunks are spatially coherent in the file
            while (!stack.empty()) {
                Node nd = stack.back(); stack.pop_back();
                int n = 1 << nd.d;
                uint32_t cnt = pyramid[nd.d][((size_t)nd.z * n + nd.y) * n + nd.x];
                if (cnt == 0) continue;
                if (cnt <= P.maxChunk || nd.d == depth) {
                    Chunk c;
                    double sz = S.cubeSize / n;
                    c.cubeMin[0] = nd.x * sz; c.cubeMin[1] = nd.y * sz; c.cubeMin[2] = nd.z * sz;
                    c.size = sz;
                    c.points = cnt;
                    uint32_t id = (uint32_t)chunks.size();
                    chunks.push_back(c);
                    int span = 1 << (depth - nd.d);
                    for (int z = nd.z * span; z < (nd.z + 1) * span; z++)
                        for (int y = nd.y * span; y < (nd.y + 1) * span; y++)
                            for (int x = nd.x * span; x < (nd.x + 1) * span; x++)
                                chunkOfCell[((size_t)z * G + y) * G + x] = id;
                } else {
                    for (int k = 7; k >= 0; k--)
                        stack.push_back({nd.d + 1, nd.x * 2 + (k & 1), nd.y * 2 + ((k >> 1) & 1), nd.z * 2 + ((k >> 2) & 1)});
                }
            }
        }
        pyramid.clear();
        pyramid.shrink_to_fit();
        lap("octree merge");

        // ---- Pass 2: distribute the points by chunk (memory: counting sort of indices; ooc: bucket files)
        std::vector<uint32_t> grouped;
        std::unique_ptr<ooc::TempWorkDir> tmp;
        std::unique_ptr<ooc::BucketWriter> buckets;
        const size_t C0 = chunks.size();
        uint64_t tmpBytes = 0;
        if (memoryMode) {
            uint64_t run = 0;
            for (auto& c : chunks) { c.begin = run; run += c.points; }
            grouped.resize(N);
            std::vector<uint64_t> cursor(C0);
            for (size_t c = 0; c < C0; c++) cursor[c] = chunks[c].begin;
            for (uint64_t i = 0; i < N; i++) grouped[cursor[chunkOfCell[pointCell[i]]]++] = (uint32_t)i;
            std::vector<uint32_t>().swap(pointCell);
            std::vector<uint32_t>().swap(chunkOfCell);
        } else {
            uint64_t need = N * (uint64_t)(TRL + 4) + (64ULL << 20);
            tmp.reset(new ooc::TempWorkDir(P.tempDir, "las2pc_", need));
            std::cout << "  temp folder: " << tmp->path() << " (~" << need / 1000000 << " MB needed)" << std::endl;
            buckets.reset(new ooc::BucketWriter(tmp->path(), "c", C0, std::max<uint64_t>(16ULL << 20, budget / 8)));   // staged bytes; the buffers of the buckets can hold up to twice as much
            const uint32_t* cellToChunk = chunkOfCell.data();
            forEachBlock("las2pc distribute", [&](const uint8_t* recs, uint64_t first, size_t m) {
                std::vector<uint32_t> cid(m);
                std::vector<uint32_t> count(C0 + 1, 0);
                for (size_t i = 0; i < m; i++) {
                    uint32_t id = cellToChunk[cell_of(recs + i * RL, S, li)];
                    cid[i] = id;
                    count[id + 1]++;
                }
                for (size_t c = 0; c < C0; c++) count[c + 1] += count[c];
                std::vector<uint8_t> sorted(m * TRL);
                std::vector<uint32_t> pos(count.begin(), count.end() - 1);
                for (size_t i = 0; i < m; i++) {
                    uint8_t* d = sorted.data() + (size_t)pos[cid[i]]++ * TRL;
                    std::memcpy(d, recs + i * RL, RL);
                    if (!pidInRecord) { uint32_t row = (uint32_t)(first + i); std::memcpy(d + RL, &row, 4); }
                }
                for (size_t c = 0; c < C0; c++)
                    if (count[c + 1] > count[c])
                        buckets->append(c, sorted.data() + (size_t)count[c] * TRL, (size_t)(count[c + 1] - count[c]) * TRL);
            });
            buckets->flushAll();
            std::vector<uint32_t>().swap(chunkOfCell);
            for (size_t c = 0; c < C0; c++) {
                chunks[c].file = buckets->path(c);
                if (buckets->bytes(c) != (uint64_t)chunks[c].points * TRL)
                    throw std::runtime_error("The input changed while it was being read (chunk size mismatch)");
                tmpBytes += buckets->bytes(c);
            }
        }
        lap("distribute");
        if (P.stopAfter == "distribute") { std::cout << "stopped after the distribution" << std::endl; return 0; }

        // ---- Refinement: a grid cell that is still denser than --max-chunk is split into octants (recursively),
        //      in place of its chunk (the order of the chunks stays the depth-first one).
        {
            std::vector<Chunk> out;
            out.reserve(chunks.size());
            size_t refined = 0;
            uint32_t unsplittable = 0;
            std::vector<uint32_t> scratchIdx;
            std::function<void(const Chunk&, int)> refine = [&](const Chunk& c, int dpt) {
                if (c.points <= P.maxChunk) { out.push_back(c); return; }
                const double minScale = std::min({li.scale[0], li.scale[1], li.scale[2]});
                if (dpt >= MAX_REFINE || c.size <= 2 * minScale) { out.push_back(c); unsplittable++; return; }
                std::vector<Chunk> kids(8);
                for (int o = 0; o < 8; o++) kids[o] = child_chunk(c, o);
                uint32_t cnt[8] = {0};
                if (memoryMode) {
                    uint32_t* idx = grouped.data() + c.begin;
                    std::vector<uint8_t> oc(c.points);
                    for (uint32_t i = 0; i < c.points; i++) { oc[i] = (uint8_t)octant_of(rec(idx[i]), S, li, c); cnt[oc[i]]++; }
                    uint64_t start[8], run = c.begin;
                    for (int o = 0; o < 8; o++) { start[o] = run; run += cnt[o]; }
                    scratchIdx.assign(idx, idx + c.points);
                    uint64_t cur[8];
                    for (int o = 0; o < 8; o++) cur[o] = start[o];
                    for (uint32_t i = 0; i < c.points; i++) grouped[cur[oc[i]]++] = scratchIdx[i];
                    for (int o = 0; o < 8; o++) { kids[o].points = cnt[o]; kids[o].begin = start[o]; }
                } else {
                    ooc::BucketWriter w(tmp->path(), "r" + std::to_string(refined) + "_", 8, 64ULL << 20);
                    int fd = ::open(c.file.c_str(), O_RDONLY);
                    if (fd < 0) throw std::runtime_error("Cannot open " + c.file);
                    std::vector<uint8_t> buf((size_t)std::min<uint64_t>(c.points, 1 << 18) * TRL), sorted(buf.size());
                    uint64_t doneRows = 0;
                    while (doneRows < c.points) {
                        size_t m = (size_t)std::min<uint64_t>(c.points - doneRows, buf.size() / TRL);
                        ooc::read_all_at(fd, buf.data(), m * TRL, doneRows * TRL);
                        uint32_t bc[9] = {0};
                        std::vector<uint8_t> oc(m);
                        for (size_t i = 0; i < m; i++) { oc[i] = (uint8_t)octant_of(buf.data() + i * TRL, S, li, c); bc[oc[i] + 1]++; }
                        for (int o = 0; o < 8; o++) bc[o + 1] += bc[o];
                        uint32_t pos[8];
                        for (int o = 0; o < 8; o++) pos[o] = bc[o];
                        for (size_t i = 0; i < m; i++) std::memcpy(sorted.data() + (size_t)pos[oc[i]]++ * TRL, buf.data() + i * TRL, TRL);
                        for (int o = 0; o < 8; o++)
                            if (bc[o + 1] > bc[o]) w.append(o, sorted.data() + (size_t)bc[o] * TRL, (size_t)(bc[o + 1] - bc[o]) * TRL);
                        doneRows += m;
                    }
                    ::close(fd);
                    w.flushAll();
                    ::unlink(c.file.c_str());
                    for (int o = 0; o < 8; o++) {
                        kids[o].points = (uint32_t)(w.bytes(o) / TRL);
                        kids[o].file = w.path(o);
                        cnt[o] = kids[o].points;
                    }
                    refined++;
                }
                for (int o = 0; o < 8; o++)
                    if (kids[o].points) refine(kids[o], dpt + 1);
            };
            size_t before = chunks.size();
            for (const Chunk& c : chunks) {
                if (c.points > P.maxChunk) {
                    std::cout << "  [Info] a grid cell holds " << c.points << " points (> max-chunk): splitting it" << std::endl;
                    if (memoryMode) refined++;
                }
                refine(c, 0);
            }
            chunks.swap(out);
            if (chunks.size() != before) std::cout << "  chunks: " << before << " -> " << chunks.size() << " after splitting dense cells" << std::endl;
            if (unsplittable) std::cout << "  [Warning] " << unsplittable << " chunk(s) could not be split below max-chunk (coincident points)" << std::endl;
        }
        const size_t C = chunks.size();
        uint32_t maxChunkPoints = 0;
        for (auto& c : chunks) maxChunkPoints = std::max(maxChunkPoints, c.points);
        lap("chunks");

        // ---- Stratified levels inside each chunk (points sorted by POINT_ID first)
        const int L = P.levels;           // regular levels 0..L-1, remainder is level L
        const int nLevels = L + 1;
        // Threads used by the per-chunk passes of the ooc mode: each holds ~130 B per point of its chunk
        int workThreads = numThreads;
        if (!memoryMode) {
            uint64_t perThread = (uint64_t)maxChunkPoints * (2 * TRL + 40) + (4ULL << 20);
            uint64_t avail = budget > (budget / 4) ? budget - budget / 4 : budget;   // keep a quarter of the budget for the rest
            workThreads = (int)std::max<uint64_t>(1, std::min<uint64_t>((uint64_t)numThreads, avail / perThread));
            if (workThreads < numThreads)
                std::cout << "  [Info] chunk passes limited to " << workThreads << " thread(s) by the memory budget" << std::endl;
        }
        {
            Progress prog("las2pc levels", N);
            ErrorBox err;
#pragma omp parallel num_threads(workThreads)
            {
                LevelScratch ls;
                std::vector<PI> ent;
                std::vector<int32_t> q;
                std::vector<uint32_t> order;
                std::vector<uint8_t> file;
#pragma omp for schedule(dynamic, 1)
                for (int64_t ci = 0; ci < (int64_t)C; ci++) {
                    if (err.set) continue;
                    try {
                        Chunk& c = chunks[ci];
                        const uint32_t n = c.points;
                        ent.resize(n);
                        if (memoryMode) {
                            uint32_t* idx = grouped.data() + c.begin;
                            for (uint32_t i = 0; i < n; i++) ent[i] = {pidInRecord ? rd<uint32_t>(rec(idx[i]) + li.pidOff) : idx[i], idx[i]};
                            sort_by_pid(ent, RL, [&](uint32_t g) { return rec(g); });
                            q.resize((size_t)n * 3);
                            for (uint32_t i = 0; i < n; i++) {
                                const uint8_t* r = rec(ent[i].idx);
                                for (int k = 0; k < 3; k++) q[(size_t)i * 3 + k] = (int32_t)((int64_t)rd<int32_t>(r + 4 * k) - S.q0[k]);
                            }
                            levelize(c, (size_t)ci, q, li.scale, P, ls, order);
                            for (uint32_t i = 0; i < n; i++) idx[i] = ent[order[i]].idx;
                        } else {
                            file.resize((size_t)n * TRL);
                            int fd = ::open(c.file.c_str(), O_RDONLY);
                            if (fd < 0) throw std::runtime_error("Cannot open " + c.file);
                            ooc::read_all_at(fd, file.data(), file.size(), 0);
                            ::close(fd);
                            for (uint32_t i = 0; i < n; i++) ent[i] = {pidOfTmp(file.data() + (size_t)i * TRL), i};
                            sort_by_pid(ent, TRL, [&](uint32_t p) { return file.data() + (size_t)p * TRL; });
                            q.resize((size_t)n * 3);
                            for (uint32_t i = 0; i < n; i++) {
                                const uint8_t* r = file.data() + (size_t)ent[i].idx * TRL;
                                for (int k = 0; k < 3; k++) q[(size_t)i * 3 + k] = (int32_t)((int64_t)rd<int32_t>(r + 4 * k) - S.q0[k]);
                            }
                            levelize(c, (size_t)ci, q, li.scale, P, ls, order);
                            // order file: position in the chunk file of the i-th output point
                            for (uint32_t i = 0; i < n; i++) order[i] = ent[order[i]].idx;
                            int ofd = ::open((tmp->path() + "/o" + std::to_string(ci) + ".bin").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                            if (ofd < 0) throw std::runtime_error("Cannot create an order file");
                            ooc::write_all_at(ofd, order.data(), (size_t)n * 4, 0);
                            ::close(ofd);
                        }
                        prog.add(n);
                    } catch (...) { err.capture(); }
                }
            }
            err.rethrow();
        }
        lap("levels");
        if (P.stopAfter == "levels") { std::cout << "stopped after the levels" << std::endl; return 0; }

        // ---- Head block size: largest H whose levels < H (all chunks) fit in the budget
        std::vector<uint64_t> levelTotals(nLevels, 0);
        for (auto& c : chunks) for (int l = 0; l < nLevels; l++) levelTotals[l] += c.levelPoints[l];
        int H = 1;
        {
            uint64_t acc = levelTotals[0];
            while (H < nLevels && acc + levelTotals[H] <= P.headBudget) acc += levelTotals[H++];
        }
        uint64_t headPoints = 0;
        for (int l = 0; l < H; l++) headPoints += levelTotals[l];
        uint64_t headRun = 0, bodyRun = headPoints;
        for (auto& c : chunks) {
            uint64_t hp = 0;
            for (int l = 0; l < H; l++) hp += c.levelPoints[l];
            c.headOffset = headRun;  headRun += hp;
            c.bodyOffset = bodyRun;  bodyRun += c.points - hp;
        }

        // ---- Write geom.bin / point_order.bin (/ the ordered LAS)
        const std::string outDir = P.output;
        ooc::mkdir_p(outDir);
        const bool wantLas = !P.orderedLas.empty();
        std::vector<uint8_t> lasHeader;
        if (wantLas) {
            lasHeader = li.headerBytes;
            // the points are the same: only the bbox / counters of the header may need a fix
            double vmin[3], vmax[3];
            for (int k = 0; k < 3; k++) { vmin[k] = (double)mn[k] * li.scale[k] + li.offset[k]; vmax[k] = (double)mx[k] * li.scale[k] + li.offset[k]; }
            for (int k = 0; k < 3; k++) {
                std::memcpy(lasHeader.data() + 179 + 16 * k, &vmax[k], 8);
                std::memcpy(lasHeader.data() + 187 + 16 * k, &vmin[k], 8);
            }
            uint32_t n32 = (uint32_t)N;
            std::memcpy(lasHeader.data() + 107, &n32, 4);
            if (li.versionMinor >= 4 && li.headerSize >= 375) {
                uint64_t n64 = N, zero64 = 0; uint32_t zero32 = 0;
                std::memcpy(lasHeader.data() + 247, &n64, 8);
                std::memcpy(lasHeader.data() + 235, &zero64, 8);   // the EVLRs are not copied
                std::memcpy(lasHeader.data() + 243, &zero32, 4);
            }
        }
        const uint64_t lasBytes = (uint64_t)li.offsetToData + N * RL;
        {
            Progress prog("las2pc write", N);
            ErrorBox err;
            auto headPointsOf = [&](const Chunk& c) { uint64_t hp = 0; for (int l = 0; l < H; l++) hp += c.levelPoints[l]; return hp; };
            if (memoryMode) {
                OutFile geom(outDir + "/geom.bin", N * 20);
                OutFile order(outDir + "/point_order.bin", N * 4);
                std::unique_ptr<OutFile> las;
                if (wantLas) {
                    las.reset(new OutFile(P.orderedLas, lasBytes));
                    std::memcpy(las->data, lasHeader.data(), lasHeader.size());
                }
                uint8_t* g = geom.data;
                uint32_t* po = (uint32_t*)order.data;
#pragma omp parallel for schedule(dynamic, 1)
                for (int64_t ci = 0; ci < (int64_t)C; ci++) {
                    if (err.set) continue;
                    try {
                        const Chunk& c = chunks[ci];
                        const uint32_t* idx = grouped.data() + c.begin;
                        uint64_t hp = headPointsOf(c);
                        for (uint32_t i = 0; i < c.points; i++) {
                            uint64_t outPos = i < hp ? c.headOffset + i : c.bodyOffset + (i - hp);
                            const uint8_t* r = rec(idx[i]);
                            uint32_t pid = pidInRecord ? rd<uint32_t>(r + li.pidOff) : idx[i];
                            geom_record(r, li, S.q0, color16, pid, g + outPos * 20);
                            po[outPos] = pid;
                            if (las) std::memcpy(las->data + li.offsetToData + outPos * RL, r, RL);
                        }
                        prog.add(c.points);
                    } catch (...) { err.capture(); }
                }
            } else {
                PosFile geom(outDir + "/geom.bin", N * 20);
                PosFile order(outDir + "/point_order.bin", N * 4);
                std::unique_ptr<PosFile> las;
                if (wantLas) {
                    las.reset(new PosFile(P.orderedLas, lasBytes));
                    ooc::write_all_at(las->fd, lasHeader.data(), lasHeader.size(), 0);
                }
#pragma omp parallel num_threads(workThreads)
                {
                    ooc::PositionalWriter pw;
                    std::vector<uint8_t> file, gbuf, lbuf;
                    std::vector<uint32_t> ord, pbuf;
#pragma omp for schedule(dynamic, 1)
                    for (int64_t ci = 0; ci < (int64_t)C; ci++) {
                        if (err.set) continue;
                        try {
                            const Chunk& c = chunks[ci];
                            const uint32_t n = c.points;
                            uint64_t hp = headPointsOf(c);
                            file.resize((size_t)n * TRL);
                            ord.resize(n);
                            int fd = ::open(c.file.c_str(), O_RDONLY);
                            if (fd < 0) throw std::runtime_error("Cannot open " + c.file);
                            ooc::read_all_at(fd, file.data(), file.size(), 0);
                            ::close(fd);
                            std::string ofile = tmp->path() + "/o" + std::to_string(ci) + ".bin";
                            int ofd = ::open(ofile.c_str(), O_RDONLY);
                            if (ofd < 0) throw std::runtime_error("Cannot open " + ofile);
                            ooc::read_all_at(ofd, ord.data(), (size_t)n * 4, 0);
                            ::close(ofd);
                            gbuf.resize((size_t)n * 20);
                            pbuf.resize(n);
                            if (wantLas) lbuf.resize((size_t)n * RL);
                            for (uint32_t i = 0; i < n; i++) {
                                const uint8_t* r = file.data() + (size_t)ord[i] * TRL;
                                uint32_t pid = pidOfTmp(r);
                                geom_record(r, li, S.q0, color16, pid, gbuf.data() + (size_t)i * 20);
                                pbuf[i] = pid;
                                if (wantLas) std::memcpy(lbuf.data() + (size_t)i * RL, r, RL);
                            }
                            // head part [0,hp) and body part [hp,n) are contiguous in every output file
                            auto put = [&](PosFile& f, const uint8_t* data, size_t elem, uint64_t base) {
                                if (hp) pw.write(f.fd, base + c.headOffset * elem, data, (size_t)hp * elem);
                                if (n > hp) pw.write(f.fd, base + c.bodyOffset * elem, data + (size_t)hp * elem, (size_t)(n - hp) * elem);
                            };
                            put(geom, gbuf.data(), 20, 0);
                            put(order, (const uint8_t*)pbuf.data(), 4, 0);
                            if (wantLas) put(*las, lbuf.data(), RL, li.offsetToData);
                            ::unlink(c.file.c_str());
                            ::unlink(ofile.c_str());
                            prog.add(n);
                        } catch (...) { err.capture(); }
                    }
                    pw.finish();
                }
            }
            err.rethrow();
        }
        lap("write");

        // ---- meta.json
        std::ostringstream js;
        auto vec3 = [&](const double* v) { return "[" + num(v[0]) + ", " + num(v[1]) + ", " + num(v[2]) + "]"; };
        std::string version = std::to_string(
            (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        double qminD[3] = {(double)S.q0[0], (double)S.q0[1], (double)S.q0[2]};
        double bboxMin[3], bboxMax[3];
        for (int k = 0; k < 3; k++) { bboxMin[k] = S.bmin[k]; bboxMax[k] = S.bmin[k] + S.cubeSize; }
        // The boundingBox is the box of the points that really exist
        for (int k = 0; k < 3; k++) {
            bboxMin[k] = (double)mn[k] * li.scale[k] + li.offset[k];
            bboxMax[k] = (double)mx[k] * li.scale[k] + li.offset[k];
        }
        js << "{\n  \"format\": \"pck\",\n  \"formatVersion\": 1,\n  \"version\": \"" << version << "\",\n"
           << "  \"points\": " << N << ",\n"
           << "  \"scale\": " << vec3(li.scale) << ",\n  \"offset\": " << vec3(li.offset) << ",\n"
           << "  \"qMin\": " << vec3(qminD) << ",\n"
           << "  \"boundingBox\": {\"min\": " << vec3(bboxMin) << ", \"max\": " << vec3(bboxMax) << "},\n"
           << "  \"hasColor\": " << (li.rgbOff >= 0 ? "true" : "false") << ",\n"
           << "  \"canonicalLas\": " << (wantLas ? "true" : "false") << ",\n"
           << "  \"geom\": {\"file\": \"geom.bin\", \"recordSize\": 20},\n"
           << "  \"order\": {\"file\": \"point_order.bin\", \"type\": \"uint32\"},\n"
           << "  \"levels\": {\"count\": " << nLevels << ", \"base\": " << P.base << ", \"head\": " << H << "},\n"
           << "  \"head\": {\"points\": " << headPoints << "},\n  \"chunks\": [\n";
        for (size_t ci = 0; ci < C; ci++) {
            const Chunk& c = chunks[ci];
            js << "    {\"id\": " << ci << ", \"min\": " << vec3(c.tmin) << ", \"max\": " << vec3(c.tmax)
               << ", \"cubeMin\": " << vec3(c.cubeMin) << ", \"size\": " << num(c.size)
               << ", \"levelPoints\": [";
            for (int l = 0; l < nLevels; l++) js << (l ? ", " : "") << c.levelPoints[l];
            js << "], \"headOffset\": " << c.headOffset << ", \"bodyOffset\": " << c.bodyOffset
               << ", \"points\": " << c.points << "}" << (ci + 1 < C ? "," : "") << "\n";
        }
        js << "  ],\n  \"columns\": {}\n}\n";
        {
            std::ofstream mf(outDir + "/meta.json", std::ios::binary);
            mf << js.str();
            if (!mf) throw std::runtime_error("Cannot write meta.json");
        }
        lap("done");
        std::cout << "  mode=" << (memoryMode ? "memory" : "ooc") << " chunks=" << C << " levels=" << nLevels << " head_levels=" << H
                  << " head_points=" << headPoints << " temp_chunk_bytes=" << tmpBytes << " version=" << version << std::endl;
        ooc::TempWorkDir::extra_paths().clear();
        std::cout << "las2pc completed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

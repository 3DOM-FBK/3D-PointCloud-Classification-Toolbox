// las2pc — converts a canonical features.las into the chunked point cloud format ("pck")
// used by the viewer. See docs/POINTCLOUD_FORMAT.md.
//
// Output folder layout:
//   geom.bin         20 B records: int32 x,y,z | uint8 r,g,b,pad | uint32 POINT_ID
//   point_order.bin  uint32 POINT_ID in geom.bin order
//   meta.json        chunks, levels, bbox (no columns: they are written by pc_columns.py)
//
// Algorithm:
//   1. Counting grid (128^3, 256^3 for huge clouds) over the cubic bbox, merged bottom-up into an
//      implicit octree whose leaves hold at most --max-chunk points (chunks).
//   2. Counting sort of the points by chunk.
//   3. Inside every chunk: deterministic random permutation, then stratified levels: level l keeps
//      the first point (in permutation order) that finds its cell free in a 2^(base+l) grid.
//      Points that never find a free cell go to the last level.
//   4. Points are laid out as [head block: levels < H of every chunk][per chunk: levels >= H].
//
// Only LAS 1.x uncompressed files are supported. The whole conversion runs in memory.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

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

template <typename T> T rd(const uint8_t* p) { T v; std::memcpy(&v, p, sizeof(T)); return v; }

int base_record_size(int fmt) {
    static const int sizes[] = {20, 28, 26, 34, 57, 63, 30, 36, 38, 59, 67};
    return (fmt >= 0 && fmt <= 10) ? sizes[fmt] : -1;
}

// Offset of the RGB triplet inside the point record, -1 if the format has no colour.
int rgb_offset(int fmt) {
    switch (fmt) {
        case 2: return 20;
        case 3: case 5: return 28;
        case 7: case 8: case 10: return 30;
        default: return -1;
    }
}

int extra_type_size(int t) {
    switch (t) {
        case 1: case 2: return 1;
        case 3: case 4: return 2;
        case 5: case 6: return 4;
        case 7: case 8: return 8;
        case 9: return 4;
        case 10: return 8;
        case 11: case 12: return 2;
        case 13: case 14: return 4;
        case 15: case 16: return 8;
        case 17: case 18: return 16;
        case 19: return 8;
        case 20: return 16;
        case 21: case 22: return 3;
        case 23: case 24: return 6;
        case 25: case 26: return 12;
        case 27: case 28: return 24;
        case 29: return 12;
        case 30: return 24;
        default: return 0;
    }
}

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

double available_memory_bytes() {
    double avail = 0;
    std::ifstream mi("/proc/meminfo");
    std::string key, unit;
    double val;
    while (mi >> key >> val >> unit) {
        if (key == "MemAvailable:") { avail = val * 1024.0; break; }
    }
    for (const char* cg : {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"}) {
        std::ifstream f(cg);
        std::string s;
        if (f && (f >> s) && s != "max") {
            double lim = std::atof(s.c_str());
            if (lim > 0 && lim < 1e15 && (avail == 0 || lim < avail)) avail = lim;
        }
    }
    return avail;
}

std::string num(double v) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

// ------------------------------------------------------------------ LAS input

struct LasInfo {
    uint64_t numPoints = 0;
    uint32_t offsetToData = 0;
    int recordLength = 0;
    int format = 0;
    double scale[3] = {1, 1, 1};
    double offset[3] = {0, 0, 0};
    int rgbOff = -1;
    int pidOff = -1;
};

LasInfo parse_las(const MappedFile& f) {
    if (f.size < 227 || std::memcmp(f.data, "LASF", 4) != 0)
        throw std::runtime_error("Not a valid LAS file");
    LasInfo li;
    const uint8_t* h = f.data;
    int versionMinor = h[25];
    uint16_t headerSize = rd<uint16_t>(h + 94);
    li.offsetToData = rd<uint32_t>(h + 96);
    uint32_t numVlrs = rd<uint32_t>(h + 100);
    uint8_t fmtRaw = h[104];
    if (fmtRaw & 0xC0) throw std::runtime_error("Compressed LAS (LAZ) is not supported: decompress it first");
    li.format = fmtRaw & 0x3F;
    li.recordLength = rd<uint16_t>(h + 105);
    li.numPoints = rd<uint32_t>(h + 107);
    for (int i = 0; i < 3; i++) li.scale[i] = rd<double>(h + 131 + 8 * i);
    for (int i = 0; i < 3; i++) li.offset[i] = rd<double>(h + 155 + 8 * i);
    if (li.numPoints == 0 && versionMinor >= 4 && headerSize >= 375) li.numPoints = rd<uint64_t>(h + 247);

    int baseSize = base_record_size(li.format);
    if (baseSize < 0) throw std::runtime_error("Unsupported LAS point format " + std::to_string(li.format));
    if (li.recordLength < baseSize) throw std::runtime_error("Invalid record length for point format");
    li.rgbOff = rgb_offset(li.format);

    // Walk the VLRs looking for the Extra Bytes description (LASF_Spec, id 4)
    size_t pos = headerSize;
    for (uint32_t v = 0; v < numVlrs && pos + 54 <= f.size; v++) {
        std::string userId((const char*)f.data + pos + 2, 16);
        uint16_t recordId = rd<uint16_t>(f.data + pos + 18);
        uint16_t len = rd<uint16_t>(f.data + pos + 20);
        size_t body = pos + 54;
        if (recordId == 4 && userId.find("LASF_Spec") != std::string::npos) {
            int cur = 0;
            for (int r = 0; r < len / 192; r++) {
                const uint8_t* rec = f.data + body + 192 * r;
                int type = rec[2];
                int options = rec[3];
                std::string name((const char*)rec + 4, strnlen((const char*)rec + 4, 32));
                int size = type == 0 ? options : extra_type_size(type);
                if (name == "POINT_ID" && (type == 5 || type == 6)) li.pidOff = baseSize + cur;
                cur += size;
            }
        }
        pos = body + len;
    }
    uint64_t avail = f.size > li.offsetToData ? (f.size - li.offsetToData) / li.recordLength : 0;
    if (li.numPoints > avail) throw std::runtime_error("LAS file is truncated");
    return li;
}

// ------------------------------------------------------------------ chunking

struct Chunk {
    double cubeMin[3];   // octree node cube (relative to bbox cube origin)
    double size;
    uint32_t points = 0;
    uint64_t begin = 0;  // offset into the chunk-grouped index array
    std::vector<uint32_t> levelPoints;
    uint64_t headOffset = 0, bodyOffset = 0;
    double tmin[3], tmax[3];  // tight AABB (relative to the bbox min)
};

struct Params {
    std::string input, output;
    uint32_t maxChunk = 250000;
    int base = 3;
    int levels = 6;                // number of regular levels (the last, remainder level is extra)
    uint64_t headBudget = 1000000;
    uint64_t seed = 12345;
};

void usage() {
    std::cerr << "Usage: las2pc --input features.las --output out_dir [--max-chunk 250000] [--base 3]\n"
                 "              [--levels 6] [--head-budget 1000000] [--seed 12345]\n";
}

}  // namespace

int main(int argc, char** argv) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    auto lap = [&](const char* what) {
        double s = std::chrono::duration<double>(clk::now() - t0).count();
        std::cout << "  [" << what << "] " << s << " s" << std::endl;
    };

    Params P;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--input" || a == "-i") P.input = next();
        else if (a == "--output" || a == "-o") P.output = next();
        else if (a == "--max-chunk") P.maxChunk = (uint32_t)std::stoul(next());
        else if (a == "--base") P.base = std::stoi(next());
        else if (a == "--levels") P.levels = std::stoi(next());
        else if (a == "--head-budget") P.headBudget = std::stoull(next());
        else if (a == "--seed") P.seed = std::stoull(next());
        else { usage(); return 2; }
    }
    if (P.input.empty() || P.output.empty() || P.levels < 1 || P.base < 1 || P.base + P.levels > 9 ||
        P.maxChunk < 1000) { usage(); return 2; }

    try {
        std::cout << "[las2pc] Input: " << P.input << std::endl;
        MappedFile las(P.input);
        LasInfo li = parse_las(las);
        const uint64_t N = li.numPoints;
        if (N == 0) throw std::runtime_error("The LAS file has no points");
        if (N >= 0xFFFFFFFFULL) throw std::runtime_error("Too many points (> 4G)");
        const int numThreads =
#ifdef _OPENMP
            omp_get_max_threads();
#else
            1;
#endif
        std::cout << "  points=" << N << " format=" << li.format << " record=" << li.recordLength
                  << " rgb=" << (li.rgbOff >= 0) << " point_id=" << (li.pidOff >= 0)
                  << " threads=" << numThreads << std::endl;
        if (li.pidOff < 0)
            std::cout << "  [Warning] POINT_ID extra byte not found: the point index is used instead" << std::endl;

        double need = (double)N * 32.0;
        double avail = available_memory_bytes();
        if (avail > 0 && need > avail * 0.9)
            throw std::runtime_error("Not enough memory for in-memory conversion: needs ~" +
                                     std::to_string((uint64_t)(need / 1e6)) + " MB, available ~" +
                                     std::to_string((uint64_t)(avail / 1e6)) + " MB (out-of-core mode not implemented)");

        const uint8_t* rec0 = las.data + li.offsetToData;
        const size_t RL = (size_t)li.recordLength;
        auto rec = [&](uint64_t i) { return rec0 + i * RL; };

        // ---- Pass 0: integer bbox + colour range
        int32_t qmin[3] = {INT32_MAX, INT32_MAX, INT32_MAX}, qmax[3] = {INT32_MIN, INT32_MIN, INT32_MIN};
        uint32_t maxColor = 0;
#pragma omp parallel
        {
            int32_t lmin[3] = {INT32_MAX, INT32_MAX, INT32_MAX}, lmax[3] = {INT32_MIN, INT32_MIN, INT32_MIN};
            uint32_t lmc = 0;
#pragma omp for schedule(static) nowait
            for (int64_t i = 0; i < (int64_t)N; i++) {
                const uint8_t* r = rec((uint64_t)i);
                for (int k = 0; k < 3; k++) {
                    int32_t v = rd<int32_t>(r + 4 * k);
                    lmin[k] = std::min(lmin[k], v);
                    lmax[k] = std::max(lmax[k], v);
                }
                if (li.rgbOff >= 0) {
                    lmc = std::max<uint32_t>(lmc, rd<uint16_t>(r + li.rgbOff));
                    lmc = std::max<uint32_t>(lmc, rd<uint16_t>(r + li.rgbOff + 2));
                    lmc = std::max<uint32_t>(lmc, rd<uint16_t>(r + li.rgbOff + 4));
                }
            }
#pragma omp critical
            {
                for (int k = 0; k < 3; k++) { qmin[k] = std::min(qmin[k], lmin[k]); qmax[k] = std::max(qmax[k], lmax[k]); }
                maxColor = std::max(maxColor, lmc);
            }
        }
        const bool color16 = maxColor > 255;
        double ext[3];
        double bmin[3], bmax[3];
        for (int k = 0; k < 3; k++) {
            bmin[k] = (double)qmin[k] * li.scale[k] + li.offset[k];
            bmax[k] = (double)qmax[k] * li.scale[k] + li.offset[k];
            ext[k] = bmax[k] - bmin[k];
        }
        double cubeSize = std::max({ext[0], ext[1], ext[2], 1e-9});
        lap("bbox");

        // ---- Pass 1: counting grid on the bbox cube (coordinates relative to bmin)
        int G = N > 100000000ULL ? 256 : 128;
        int depth = (G == 256) ? 8 : 7;
        const uint64_t G3 = (uint64_t)G * G * G;
        std::vector<uint32_t> grid(G3, 0);
        const double invCube = (double)G / cubeSize;
        auto cellOf = [&](const uint8_t* r) -> uint32_t {
            int c[3];
            for (int k = 0; k < 3; k++) {
                double rel = (double)(rd<int32_t>(r + 4 * k) - qmin[k]) * li.scale[k];
                int v = (int)(rel * invCube);
                c[k] = v < 0 ? 0 : (v >= G ? G - 1 : v);
            }
            return (uint32_t)(((uint64_t)c[2] * G + c[1]) * G + c[0]);
        };
        std::vector<uint32_t> pointCell(N);
#pragma omp parallel
        {
            // Per-thread private counting would need G^3 * threads; atomic increments are fast enough
#pragma omp for schedule(static)
            for (int64_t i = 0; i < (int64_t)N; i++) {
                uint32_t c = cellOf(rec((uint64_t)i));
                pointCell[i] = c;
#pragma omp atomic
                grid[c]++;
            }
        }
        lap("count grid");

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
                    if (cnt > P.maxChunk)
                        std::cout << "  [Warning] a single grid cell holds " << cnt << " points (> max-chunk)" << std::endl;
                    Chunk c;
                    double sz = cubeSize / n;
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
        const size_t C = chunks.size();
        lap("octree merge");

        // ---- Pass 2: counting sort of the points by chunk
        uint64_t run = 0;
        for (auto& c : chunks) { c.begin = run; run += c.points; }
        std::vector<uint32_t> grouped(N);
        {
            std::vector<uint64_t> cursor(C);
            for (size_t c = 0; c < C; c++) cursor[c] = chunks[c].begin;
            for (uint64_t i = 0; i < N; i++) grouped[cursor[chunkOfCell[pointCell[i]]]++] = (uint32_t)i;
        }
        std::vector<uint32_t>().swap(pointCell);
        std::vector<uint32_t>().swap(chunkOfCell);
        lap("chunk sort");

        // ---- Stratified levels inside each chunk
        const int L = P.levels;           // regular levels 0..L-1, remainder is level L
        const int nLevels = L + 1;
#pragma omp parallel
        {
            std::vector<uint8_t> occupied;
            std::vector<uint32_t> remaining, nextRemaining, levelOrder;
            std::vector<int32_t> xyz;
#pragma omp for schedule(dynamic, 1)
            for (int64_t ci = 0; ci < (int64_t)C; ci++) {
                Chunk& c = chunks[ci];
                uint32_t* idx = grouped.data() + c.begin;
                const uint32_t n = c.points;
                SplitMix64 rng(P.seed * 0x9E3779B97F4A7C15ULL + (uint64_t)ci * 0xD1B54A32D192ED03ULL + 1);
                for (uint32_t i = n; i > 1; i--) std::swap(idx[i - 1], idx[rng.below(i)]);

                // Local gather: relative coordinates inside the chunk cube + tight AABB
                xyz.resize((size_t)n * 3);
                double tmin[3] = {1e300, 1e300, 1e300}, tmax[3] = {-1e300, -1e300, -1e300};
                for (uint32_t i = 0; i < n; i++) {
                    const uint8_t* r = rec(idx[i]);
                    for (int k = 0; k < 3; k++) {
                        int32_t q = rd<int32_t>(r + 4 * k) - qmin[k];
                        xyz[(size_t)i * 3 + k] = q;
                        double rel = (double)q * li.scale[k];
                        tmin[k] = std::min(tmin[k], rel);
                        tmax[k] = std::max(tmax[k], rel);
                    }
                }
                for (int k = 0; k < 3; k++) { c.tmin[k] = tmin[k]; c.tmax[k] = tmax[k]; }

                remaining.assign(idx, idx + n);
                std::vector<uint32_t> remPos(n);  // position in xyz for each remaining entry
                for (uint32_t i = 0; i < n; i++) remPos[i] = i;
                std::vector<uint32_t> nextPos;
                levelOrder.clear();
                levelOrder.reserve(n);
                c.levelPoints.assign(nLevels, 0);
                const double inv = 1.0 / c.size;
                for (int l = 0; l < L && !remaining.empty(); l++) {
                    const int R = 1 << (P.base + l);
                    occupied.assign((size_t)R * R * R, 0);
                    nextRemaining.clear();
                    nextPos.clear();
                    uint32_t taken = 0;
                    for (size_t j = 0; j < remaining.size(); j++) {
                        const int32_t* q = &xyz[(size_t)remPos[j] * 3];
                        int cc[3];
                        for (int k = 0; k < 3; k++) {
                            double rel = ((double)q[k] * li.scale[k] - c.cubeMin[k]) * inv * R;
                            int v = (int)rel;
                            cc[k] = v < 0 ? 0 : (v >= R ? R - 1 : v);
                        }
                        size_t cell = ((size_t)cc[2] * R + cc[1]) * R + cc[0];
                        if (!occupied[cell]) {
                            occupied[cell] = 1;
                            levelOrder.push_back(remaining[j]);
                            taken++;
                        } else {
                            nextRemaining.push_back(remaining[j]);
                            nextPos.push_back(remPos[j]);
                        }
                    }
                    c.levelPoints[l] = taken;
                    remaining.swap(nextRemaining);
                    remPos.swap(nextPos);
                }
                c.levelPoints[L] = (uint32_t)remaining.size();
                levelOrder.insert(levelOrder.end(), remaining.begin(), remaining.end());
                std::memcpy(idx, levelOrder.data(), (size_t)n * sizeof(uint32_t));
            }
        }
        lap("levels");

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

        // ---- Write geom.bin / point_order.bin
        const std::string outDir = P.output;
        std::string mk = "mkdir -p '" + outDir + "'";
        if (std::system(mk.c_str()) != 0) throw std::runtime_error("Cannot create output directory");
        {
            OutFile geom(outDir + "/geom.bin", N * 20);
            OutFile order(outDir + "/point_order.bin", N * 4);
            uint8_t* g = geom.data;
            uint32_t* po = (uint32_t*)order.data;
#pragma omp parallel for schedule(dynamic, 1)
            for (int64_t ci = 0; ci < (int64_t)C; ci++) {
                const Chunk& c = chunks[ci];
                const uint32_t* idx = grouped.data() + c.begin;
                uint64_t hp = 0;
                for (int l = 0; l < H; l++) hp += c.levelPoints[l];
                for (uint32_t i = 0; i < c.points; i++) {
                    uint64_t outPos = i < hp ? c.headOffset + i : c.bodyOffset + (i - hp);
                    const uint8_t* r = rec(idx[i]);
                    uint8_t* o = g + outPos * 20;
                    for (int k = 0; k < 3; k++) {
                        int32_t q = rd<int32_t>(r + 4 * k) - qmin[k];
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
                    uint32_t pid = li.pidOff >= 0 ? rd<uint32_t>(r + li.pidOff) : idx[i];
                    std::memcpy(o + 16, &pid, 4);
                    po[outPos] = pid;
                }
            }
        }
        lap("write geometry");

        // ---- meta.json
        std::ostringstream js;
        auto vec3 = [&](const double* v) { return "[" + num(v[0]) + ", " + num(v[1]) + ", " + num(v[2]) + "]"; };
        std::string version = std::to_string(
            (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        double qminD[3] = {(double)qmin[0], (double)qmin[1], (double)qmin[2]};
        js << "{\n  \"format\": \"pck\",\n  \"formatVersion\": 1,\n  \"version\": \"" << version << "\",\n"
           << "  \"points\": " << N << ",\n"
           << "  \"scale\": " << vec3(li.scale) << ",\n  \"offset\": " << vec3(li.offset) << ",\n"
           << "  \"qMin\": " << vec3(qminD) << ",\n"
           << "  \"boundingBox\": {\"min\": " << vec3(bmin) << ", \"max\": " << vec3(bmax) << "},\n"
           << "  \"hasColor\": " << (li.rgbOff >= 0 ? "true" : "false") << ",\n"
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
        std::cout << "  chunks=" << C << " levels=" << nLevels << " head_levels=" << H
                  << " head_points=" << headPoints << " version=" << version << std::endl;
        std::cout << "las2pc completed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

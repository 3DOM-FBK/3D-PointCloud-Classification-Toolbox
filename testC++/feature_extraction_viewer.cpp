// feature_extraction_viewer.cpp (one program: GPU or CPU backend)
// Tiled feature extraction for massive point clouds
// No PCL dependency: the features are computed by ooc/feature_math.h on the GPU (gpu_features.cu) or the CPU (cpu_features.cpp)
//
// Out-of-core: the cloud is never loaded as a whole. The tile size comes from a point cap derived from the memory
// budget (and the free VRAM on the GPU), the points are distributed into one temporary file per tile (core + buffer), and
// the tiles are computed by a pipeline (reader thread, up to --slots tiles in flight on the GPU or one on the CPU, writer thread). The record of every core point is written
// at the SAME ROW it has in the input, so the output keeps the order of the input. A tile that fails is split and
// retried; if a point cannot be written the run fails (it never leaves holes). See ooc/tile_pipeline.h.

#include <iostream>
#include <memory>
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <vector>
#include <map>
#include <set>
#include <iomanip>
#include <algorithm>
#include <atomic>
#include <array>
#include <cstring>
#include <sstream>
#include <thread>
#include <exception>


#include "cpu_features.h"
#include "ooc/cpu_info.h"
#ifdef FEATURES_WITH_CUDA
#include "gpu_features.cuh"
#endif
#include "ooc/async_queue.h"
#include "ooc/tile_pipeline.h"

static const int kExitNoGpu = 3;      // no usable CUDA device: the caller falls back to the CPU extractor

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
TimePoint now_t() { return std::chrono::high_resolution_clock::now(); }
double elapsed(TimePoint start) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
}

// Global settings
const int MAX_SCALES = 10;
int scalesCount = 4;
float scales[MAX_SCALES] = { 0.8f, 1.2f, 2.0f, 3.0f };

// Tiling settings
double TILE_SIZE   = 50.0;
double BUFFER_SIZE = 4.0;
int TARGET_TILES   = 16;

struct LasHeaderInfo;
static double computeDynamicTileSizeFromBbox(const LasHeaderInfo& hdr, double bufferSize, int targetTiles);

// ============================================================
// Feature definitions
// ============================================================

const std::set<std::string> AVAILABLE_SCALE_FEATURES = {
    "anisotropy", "omnivariance", "sphericity",
    "planarity", "linearity", "verticality", "surface_variation",
    "neighbours", "vertical_range", "height_above", "height_below"
};

const std::set<std::string> AVAILABLE_SINGLE_FEATURES = {
    "height"
};

enum FeatureId {
    F_ANISOTROPY, F_HEIGHT_ABOVE, F_HEIGHT_BELOW, F_LINEARITY,
    F_NEIGHBOURS, F_OMNIVARIANCE, F_PLANARITY, F_SPHERICITY,
    F_SURFACE_VARIATION, F_VERTICALITY, F_VERTICAL_RANGE,
    F_COUNT
};

static const char* FEATURE_NAMES[F_COUNT] = {
    "anisotropy", "height_above", "height_below", "linearity",
    "neighbours", "omnivariance", "planarity", "sphericity",
    "surface_variation", "verticality", "vertical_range"
};

// ============================================================
// LAS header reader
// ============================================================

struct LasHeaderInfo {
    uint8_t  ver_major = 1, ver_minor = 4;
    uint8_t  point_fmt = 0;
    uint16_t header_size = 0;
    uint32_t offset_to_data = 0;
    uint16_t point_record_length = 0;
    uint64_t point_count = 0;
    double   scaleX = 0.001, scaleY = 0.001, scaleZ = 0.001;
    double   offX = 0, offY = 0, offZ = 0;
    double   minx = 0, maxx = 0, miny = 0, maxy = 0, minz = 0, maxz = 0;
    int      base_size = 0;
};
static double computeDynamicTileSizeFromBbox(const LasHeaderInfo& hdr, double bufferSize, int targetTiles)
{
    double spanX = std::max(0.0, hdr.maxx - hdr.minx);
    double spanY = std::max(0.0, hdr.maxy - hdr.miny);

    if (spanX == 0.0 && spanY == 0.0) {
        return std::max(1.0, 2.0 * bufferSize);
    }

    if (targetTiles < 1) targetTiles = 1;

    double safeSpanX = std::max(spanX, 1e-9);
    double safeSpanY = std::max(spanY, 1e-9);
    double aspect = safeSpanX / safeSpanY;

    int targetNx = std::max(1, (int)std::round(std::sqrt((double)targetTiles * aspect)));
    int targetNy = std::max(1, (int)std::ceil((double)targetTiles / targetNx));

    double tileSizeX = safeSpanX / targetNx;
    double tileSizeY = safeSpanY / targetNy;
    double tileSize = std::max(tileSizeX, tileSizeY);

    // Keep a soft lower bound only to avoid pathological tiny tiles.
    // Using a strict bound tied to buffer can collapse small models to 1x1.
    double minTileSize = std::max(0.25, 0.5 * bufferSize);
    return std::max(tileSize, minTileSize);
}

static const int kLasBaseSizes[] = { 20, 28, 26, 34, 57, 63, 30, 36, 38, 59, 67 };

LasHeaderInfo readLasHeader(const std::string& fileName)
{
    LasHeaderInfo h;
    std::ifstream f(fileName, std::ios::binary);
    if (!f) return h;

    f.seekg(24); f.read((char*)&h.ver_major, 1); f.read((char*)&h.ver_minor, 1);
    f.seekg(94);  f.read((char*)&h.header_size, 2);
    f.seekg(96);  f.read((char*)&h.offset_to_data, 4);
    f.seekg(104); f.read((char*)&h.point_fmt, 1); h.point_fmt &= 0x0F;
    f.seekg(105); f.read((char*)&h.point_record_length, 2);

    if (h.ver_major == 1 && h.ver_minor >= 4) {
        f.seekg(247); f.read((char*)&h.point_count, 8);
    } else {
        uint32_t c = 0;
        f.seekg(107); f.read((char*)&c, 4);
        h.point_count = c;
    }

    f.seekg(131);
    f.read((char*)&h.scaleX, 8); f.read((char*)&h.scaleY, 8); f.read((char*)&h.scaleZ, 8);
    f.read((char*)&h.offX, 8);   f.read((char*)&h.offY, 8);   f.read((char*)&h.offZ, 8);

    f.seekg(179);
    f.read((char*)&h.maxx, 8); f.read((char*)&h.minx, 8);
    f.seekg(195);
    f.read((char*)&h.maxy, 8); f.read((char*)&h.miny, 8);
    f.seekg(211);
    f.read((char*)&h.maxz, 8); f.read((char*)&h.minz, 8);

    h.base_size = (h.point_fmt <= 10) ? kLasBaseSizes[h.point_fmt] : 20;

    // If bounds are all zero, scan a sample of points
    if (h.minx == 0 && h.maxx == 0 && h.miny == 0 && h.maxy == 0) {
        std::cout << "Header bounds are zero — scanning points..." << std::endl;
        const uint64_t STEP = h.point_count > 2000000UL ? h.point_count / 2000000UL : 1UL;
        // Sampled scan over large sequential blocks. Seeking before every sampled point costs one
        // syscall per point, which takes minutes on network or bind-mounted file systems.
        const uint64_t rec = h.point_record_length;
        const uint64_t blockRecs = std::max<uint64_t>(1, (64ULL << 20) / rec);
        std::vector<char> blk(blockRecs * rec);
        bool first = true;
        for (uint64_t b0 = 0; b0 < h.point_count; b0 += blockRecs) {
            const uint64_t cnt = std::min<uint64_t>(blockRecs, h.point_count - b0);
            f.seekg(h.offset_to_data + b0 * rec);
            f.read(blk.data(), cnt * rec);
            if (!f) break;
            // First sampled index (multiple of STEP) that falls inside this block
            for (uint64_t i = ((b0 + STEP - 1) / STEP) * STEP; i < b0 + cnt; i += STEP) {
                int32_t xyz[3];
                std::memcpy(xyz, &blk[(i - b0) * rec], 12);
                double x = xyz[0] * h.scaleX + h.offX;
                double y = xyz[1] * h.scaleY + h.offY;
                double z = xyz[2] * h.scaleZ + h.offZ;
                if (first) { h.minx=h.maxx=x; h.miny=h.maxy=y; h.minz=h.maxz=z; first=false; }
                else {
                    h.minx=std::min(h.minx,x); h.maxx=std::max(h.maxx,x);
                    h.miny=std::min(h.miny,y); h.maxy=std::max(h.maxy,y);
                    h.minz=std::min(h.minz,z); h.maxz=std::max(h.maxz,z);
                }
            }
        }
        double mx=(h.maxx-h.minx)*0.001, my=(h.maxy-h.miny)*0.001;
        h.minx-=mx; h.maxx+=mx; h.miny-=my; h.maxy+=my;
    }

    return h;
}

// ============================================================
// LAS binary write helpers
// ============================================================

template<typename T>
static void wLE(std::vector<uint8_t>& buf, size_t off, T val) {
    std::memcpy(buf.data() + off, &val, sizeof(T));
}

static std::array<uint8_t, 192> makeVlrDimRecord(const std::string& name, uint8_t dtype) {
    std::array<uint8_t, 192> rec{};
    rec[2] = dtype;
    std::memcpy(rec.data() + 4, name.c_str(), std::min(name.size(), size_t(31)));
    return rec;
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv)
{
    if (argc >= 2 && std::string(argv[1]) == "--probe") {      // GPU available? (exit code 0) -- used by the web app to choose the backend
        std::cout << "cpus " << ooc::available_cpus() << std::endl;
#ifdef FEATURES_WITH_CUDA
        GpuDeviceInfo d = gpuQueryDevice();
        if (!d.ok) { std::cout << "gpu none: " << d.error << std::endl; return kExitNoGpu; }
        std::cout << "gpu " << d.name << " cc " << d.ccMajor << "." << d.ccMinor << " free_mb " << d.freeBytes / 1048576 << " total_mb " << d.totalBytes / 1048576 << std::endl;
        return 0;
#else
        std::cout << "gpu none: built without CUDA" << std::endl;
        return kExitNoGpu;
#endif
    }
    if (argc < 3) {
        std::cout << "\nUsage: " << argv[0] << " <input_file> <output_file> [options]\n"
                  << "Options:\n"
                  << "  --features f1,f2,...    Comma-separated list of features\n"
                  << "  --tile_size S           Tile size in meters (manual override)\n"
                  << "  --target_tiles N        Auto-tiling target tile count (default 16)\n"
                  << "  --buffer B              Buffer in meters (default 4)\n"
                  << "  --radius r1,r2,...      Radius scales (default 0.8,1.2,2.0,3.0)\n"
                  << "  --backend auto|gpu|cpu  Compute on the GPU when there is one (auto, default), always, or never\n"
                  << "  --threads N             Host threads (default: the CPUs this process may use)\n"
                  << "  --slots N               Tiles in flight on the GPU (default 2)\n"
                  << "  --memory-budget MB      Peak host memory (default: 50% of the available memory)\n"
                  << "  --temp-dir DIR          Scratch folder for the tile files\n";
        return 0;
    }

    const std::string inputFile  = argv[1];
    const std::string outputFile = argv[2];
    std::set<std::string> requestedFeatures;
    bool useAllFeatures = true;
    bool manualTileSize = false;
    double memoryBudgetMb = 0;
    std::string tempDir;
    int spillMode = -1;                          // --spill auto|always|never (always/never: tests)
    int slotsOpt = 0;                            // --slots N: tiles in flight on the GPU
    size_t failAbove = 0;                        // --test-fail-above N: tiles of more than N points "fail" (tests of the retry/split path)
    std::string backendOpt = "auto";             // --backend auto|gpu|cpu
    int threadsOpt = 0;                          // --threads N (default: the CPUs the process may use)

    for (int a = 3; a < argc; a++) {
        std::string arg(argv[a]);
        if (arg == "--features" && a + 1 < argc) {
            std::stringstream ss(argv[++a]); std::string t;
            while (std::getline(ss, t, ',')) requestedFeatures.insert(t);
            useAllFeatures = false;
        } else if (arg == "--tile_size" && a + 1 < argc) {
            TILE_SIZE = std::stod(argv[++a]);
            manualTileSize = true;
        } else if (arg == "--target_tiles" && a + 1 < argc) {
            TARGET_TILES = std::max(1, std::stoi(argv[++a]));
        } else if (arg == "--buffer" && a + 1 < argc) {
            BUFFER_SIZE = std::stod(argv[++a]);
        } else if (arg == "--radius" && a + 1 < argc) {
            std::stringstream ss(argv[++a]); std::string t; int c = 0;
            while (std::getline(ss, t, ',') && c < MAX_SCALES) scales[c++] = std::stof(t);
            if (c > 0) scalesCount = c;
        } else if (arg == "--memory-budget" && a + 1 < argc) {
            memoryBudgetMb = std::stod(argv[++a]);
        } else if (arg == "--spill" && a + 1 < argc) {
            const std::string v = argv[++a];
            spillMode = v == "always" ? 1 : (v == "never" ? 0 : -1);
        } else if (arg == "--backend" && a + 1 < argc) {
            backendOpt = argv[++a];
        } else if (arg == "--threads" && a + 1 < argc) {
            threadsOpt = std::stoi(argv[++a]);
        } else if (arg == "--test-fail-above" && a + 1 < argc) {
            failAbove = (size_t)std::stoull(argv[++a]);
        } else if (arg == "--slots" && a + 1 < argc) {
            slotsOpt = std::stoi(argv[++a]);
        } else if (arg == "--temp-dir" && a + 1 < argc) {
            tempDir = argv[++a];
        }
    }
    if (useAllFeatures) {
        for (auto& f : AVAILABLE_SCALE_FEATURES)  requestedFeatures.insert(f);
        for (auto& f : AVAILABLE_SINGLE_FEATURES) requestedFeatures.insert(f);
    }

    // Ensure buffer >= max scale
    float maxScale = *std::max_element(scales, scales + scalesCount);
    if (BUFFER_SIZE < maxScale) {
        std::cout << "WARNING: buffer (" << BUFFER_SIZE << ") < max radius ("
                  << maxScale << "), adjusting to " << (maxScale + 0.5) << std::endl;
        BUFFER_SIZE = maxScale + 0.5;
    }

    auto global_start = now_t();
    ooc::TempWorkDir::install_handlers();       // SIGTERM (/stop_process/): remove the scratch folder and the partial output
    ooc::TempWorkDir::extra_paths().push_back(outputFile);

    try {
    // ------------------------------------------------------------------
    // 1. Read LAS header
    // ------------------------------------------------------------------
    LasHeaderInfo hdr = readLasHeader(inputFile);
    if (hdr.point_count == 0) {
        std::cerr << "ERROR: No points in file." << std::endl;
        return 1;
    }
    ooc::LasInfo li = ooc::read_las_info(inputFile);
    std::cout << "Input: " << hdr.point_count << " points" << std::endl;
    std::cout << "Bounds: X[" << hdr.minx << ", " << hdr.maxx << "] Y["
              << hdr.miny << ", " << hdr.maxy << "] Z["
              << hdr.minz << ", " << hdr.maxz << "]" << std::endl;

    // ------------------------------------------------------------------
    // 2. Build output feature layout (pre-computed offsets)
    // ------------------------------------------------------------------
    auto dimName = [](const std::string& feat, int s) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1) << scales[s];
        std::string r = ss.str();
        std::replace(r.begin(), r.end(), '.', '_');
        return feat + "_" + r;
    };

    bool featureActive[F_COUNT] = {};
    for (int f = 0; f < F_COUNT; f++)
        featureActive[f] = requestedFeatures.count(FEATURE_NAMES[f]) > 0;

    std::vector<std::pair<std::string, int>> outExtra;
    int extraOffset = 0;
    int off_pid = extraOffset;
    outExtra.push_back({"POINT_ID", extraOffset}); extraOffset += 4;

    int featureByteOffset[F_COUNT][MAX_SCALES] = {};
    for (int f = 0; f < F_COUNT; f++) {
        if (featureActive[f]) {
            for (int s = 0; s < scalesCount; s++) {
                std::string dn = dimName(FEATURE_NAMES[f], s);
                featureByteOffset[f][s] = extraOffset;
                outExtra.push_back({dn, extraOffset});
                extraOffset += 4;
            }
        }
    }

    int off_nx = extraOffset; extraOffset += 4;
    int off_ny = extraOffset; extraOffset += 4;
    int off_nz = extraOffset; extraOffset += 4;
    outExtra.push_back({"normal_x", off_nx});
    outExtra.push_back({"normal_y", off_ny});
    outExtra.push_back({"normal_z", off_nz});

    const int BASE_SIZE = 34;
    const int REC_LEN   = BASE_SIZE + extraOffset;

    // ------------------------------------------------------------------
    // 3. Output LAS header + VLR (the point count is known up front: every point is written at its own row)
    // ------------------------------------------------------------------
    const uint64_t N = hdr.point_count;
    uint32_t numVlrs     = 1;
    uint32_t vlrBodySize = (uint32_t)(outExtra.size() * 192);
    uint32_t headerSize  = 227;
    uint32_t offsetToData = headerSize + 54 + vlrBodySize;

    std::vector<uint8_t> headerBuf(headerSize, 0);
    std::memcpy(headerBuf.data(), "LASF", 4);
    wLE<uint16_t>(headerBuf, 6, 0x0011u);
    headerBuf[24] = 1; headerBuf[25] = 2;
    wLE<uint16_t>(headerBuf, 94,  (uint16_t)headerSize);
    wLE<uint32_t>(headerBuf, 96,  offsetToData);
    wLE<uint32_t>(headerBuf, 100, numVlrs);
    headerBuf[104] = 3;
    wLE<uint16_t>(headerBuf, 105, (uint16_t)REC_LEN);
    wLE<uint32_t>(headerBuf, 107, (uint32_t)std::min<uint64_t>(N, 0xFFFFFFFFull));
    wLE<uint32_t>(headerBuf, 111, (uint32_t)std::min<uint64_t>(N, 0xFFFFFFFFull));
    wLE<double>(headerBuf, 131, hdr.scaleX); wLE<double>(headerBuf, 139, hdr.scaleY); wLE<double>(headerBuf, 147, hdr.scaleZ);
    wLE<double>(headerBuf, 155, hdr.offX);   wLE<double>(headerBuf, 163, hdr.offY);   wLE<double>(headerBuf, 171, hdr.offZ);
    wLE<double>(headerBuf, 179, hdr.maxx);   wLE<double>(headerBuf, 187, hdr.minx);
    wLE<double>(headerBuf, 195, hdr.maxy);   wLE<double>(headerBuf, 203, hdr.miny);
    wLE<double>(headerBuf, 211, hdr.maxz);   wLE<double>(headerBuf, 219, hdr.minz);

    std::vector<uint8_t> vlrBuf(54 + vlrBodySize, 0);
    std::memcpy(vlrBuf.data() + 2, "LASF_Spec", 9);
    wLE<uint16_t>(vlrBuf, 18, 4);
    wLE<uint16_t>(vlrBuf, 20, (uint16_t)vlrBodySize);
    for (size_t k = 0; k < outExtra.size(); k++) {
        uint8_t dtype = (outExtra[k].first == "POINT_ID") ? 5 : 9;
        auto rec = makeVlrDimRecord(outExtra[k].first, dtype);
        std::memcpy(vlrBuf.data() + 54 + k * 192, rec.data(), 192);
    }

    // ------------------------------------------------------------------
    // 4. Tile size from the memory caps (host budget and free VRAM)
    // ------------------------------------------------------------------
    const uint64_t budget = ooc::resolve_budget_bytes(memoryBudgetMb);

    // Threads: the CPUs this process may really use (affinity and container quota), also for the host stages of the GPU path.
    const int numThreads = threadsOpt > 0 ? threadsOpt : ooc::available_cpus();
#ifdef _OPENMP
    omp_set_num_threads(numThreads);
#endif

    // Backend: the GPU when a usable CUDA device exists (and --backend allows it), the CPU otherwise.
    bool useGpu = false;
#ifdef FEATURES_WITH_CUDA
    GpuDeviceInfo dev;
    if (backendOpt != "cpu") {
        dev = gpuQueryDevice();
        useGpu = dev.ok;
        if (!dev.ok) {
            if (backendOpt == "gpu") {
                std::cerr << "ERROR: no usable CUDA device: " << dev.error << std::endl;
                return kExitNoGpu;
            }
            std::cout << "No usable CUDA device (" << dev.error << "): using the CPU" << std::endl;
        }
    }
#else
    if (backendOpt == "gpu") {
        std::cerr << "ERROR: this build has no CUDA support" << std::endl;
        return kExitNoGpu;
    }
#endif
    if (useGpu) {
#ifdef FEATURES_WITH_CUDA
        std::cout << "Backend: GPU " << dev.name << " (cc " << dev.ccMajor << "." << dev.ccMinor << ", " << dev.multiprocessors
                  << " SMs, " << dev.freeBytes / 1048576 << " MB free), " << numThreads << " host threads" << std::endl;
#endif
    } else {
        std::cout << "Backend: CPU, " << numThreads << " threads" << std::endl;
    }

    // output columns of the (feature, scale) pairs, in the order of the extra bytes of the record
    FeatureConfig gcfg;
    gcfg.numScales = scalesCount;
    for (int s = 0; s < scalesCount; s++) gcfg.scales[s] = scales[s];
    for (int f = 0; f < FE_F_COUNT; f++) for (int s = 0; s < 10; s++) gcfg.slotOf[f][s] = -1;
    for (int f = 0; f < F_COUNT; f++)
        if (featureActive[f])
            for (int s = 0; s < scalesCount; s++) gcfg.slotOf[f][s] = gcfg.numSlots++;

    // Tiles in flight: on the GPU (--slots, default 2) the host prepares tile k+1 and writes tile k-1 while tile k
    // computes; the CPU computes with all its threads, only the reading and the writing overlap with it.
    const int numSlotsInFlight = useGpu ? std::max(1, std::min(4, slotsOpt > 0 ? slotsOpt : 2)) : 1;
    // bytes per tile point: host (tile records of the tiles in flight, output record + row index of 2 tiles, engine staging)
    // and device (workspace of every slot)
    double hostPerPoint = 36.0 * (numSlotsInFlight + 1) + 2.0 * (REC_LEN + 12.0);
    double vramPerPoint = 0;
    if (useGpu) {
#ifdef FEATURES_WITH_CUDA
        hostPerPoint += (13.0 + 4.0 * ((gcfg.numSlots + 3) / 4 * 4)) * numSlotsInFlight;      // pinned staging
        vramPerPoint = GpuEngine::deviceBytesPerPoint(gcfg) * numSlotsInFlight;
#endif
    } else {
        hostPerPoint += cpuBytesPerPoint(gcfg) * numSlotsInFlight;
    }
    const uint64_t capHost = (uint64_t)(0.6 * (double)budget / hostPerPoint);
    uint64_t capVram = ~0ULL;
#ifdef FEATURES_WITH_CUDA
    if (useGpu) capVram = (uint64_t)(0.7 * (double)dev.freeBytes / vramPerPoint);
#endif
    const uint64_t capPoints = std::max<uint64_t>(100000, std::min(capHost, capVram));
    std::cout << "Tile cap: " << capPoints << " points (host budget " << budget / 1048576 << " MB -> " << capHost;
#ifdef FEATURES_WITH_CUDA
    if (useGpu) std::cout << ", free VRAM " << dev.freeBytes / 1048576 << " MB -> " << capVram;
#endif
    std::cout << ")" << std::endl;

    if (!manualTileSize) {
        TILE_SIZE = computeDynamicTileSizeFromBbox(hdr, BUFFER_SIZE, TARGET_TILES);
        std::cout << "Auto tile size from bbox: " << TILE_SIZE
                  << " (target tiles: " << TARGET_TILES << ")" << std::endl;
    }

    double spanX = std::max(0.0, hdr.maxx - hdr.minx);
    double spanY = std::max(0.0, hdr.maxy - hdr.miny);
    ooc::TileGrid grid;
    grid.minx = hdr.minx; grid.miny = hdr.miny; grid.buffer = BUFFER_SIZE;
    grid.make(spanX, spanY, TILE_SIZE);

    // Same tiling as before when the largest tile fits the cap; otherwise smaller tiles, from an XY histogram of the cloud.
    if ((double)N * 1.3 > (double)capPoints) {
        std::cout << "Counting points per area to size the tiles ..." << std::endl;
        ooc::XYHistogram hist = ooc::build_histogram(inputFile, li, hdr.minx, hdr.maxx, hdr.miny, hdr.maxy, BUFFER_SIZE);
        const double minTile = std::max(2.0 * BUFFER_SIZE, 1.0);
        double t = grid.tile;
        uint64_t worst = ooc::max_tile_points(hist, hdr.minx, hdr.miny, spanX, spanY, t, BUFFER_SIZE);
        const double t0 = t;
        while (worst > capPoints && t > minTile) {
            t = std::max(minTile, t / 1.5);
            worst = ooc::max_tile_points(hist, hdr.minx, hdr.miny, spanX, spanY, t, BUFFER_SIZE);
        }
        if (worst > capPoints)
            std::cout << "  [Warning] the densest tile (" << worst << " points) is above the cap even at the minimum tile size "
                      << minTile << " m: continuing, the tile will be split if its computation fails" << std::endl;
        if (t < t0) {
            grid.make(spanX, spanY, t);
            std::cout << "Tile size reduced from " << t0 << " m to " << t << " m (densest tile ~" << worst << " points)" << std::endl;
        }
    }
    const size_t numTiles = grid.numTiles();
    std::cout << "Grid: " << grid.gnx << " x " << grid.gny << " = " << numTiles << " tiles (tile "
              << grid.tile << " m, buffer " << BUFFER_SIZE << " m)" << std::endl;

    // ------------------------------------------------------------------
    // 5. Distribute the points into tile files
    // ------------------------------------------------------------------
    auto t_part = now_t();
    const double dup = std::min(4.0, std::pow((grid.tile + 2 * BUFFER_SIZE) / std::max(grid.tile, 1e-6), 2.0));
    uint64_t needDisk = (uint64_t)((double)N * dup * sizeof(ooc::TilePt)) + (uint64_t)N * ((uint64_t)REC_LEN + 4) + (64ULL << 20);   // tile files + worst-case row spill
    ooc::TempWorkDir tmp(tempDir, "feat_", needDisk);
    std::cout << "Temporary folder: " << tmp.path() << " (~" << needDisk / 1000000 << " MB)" << std::endl;
    ooc::BucketWriter writer(tmp.path(), "t", numTiles, std::max<uint64_t>(16ULL << 20, budget / 8));
    const uint64_t blockBytes = std::min<uint64_t>(16ULL << 20, std::max<uint64_t>(1ULL << 20, budget / 8 / (uint64_t)(4 * numThreads)));
    std::vector<uint64_t> tileSize = ooc::partition_input(inputFile, li, grid, writer, blockBytes);
    std::cout << "Distribution: " << elapsed(t_part) << "s" << std::endl;

    // ------------------------------------------------------------------
    // 6. Compute the tiles on the GPU, one at a time
    // ------------------------------------------------------------------
    ooc::RowOutput out(outputFile, offsetToData, (size_t)REC_LEN, N);
    out.set_spill_mode(spillMode);
    if (spillMode < 0) out.set_memory_mode(budget / 5);   // --spill always/never are test hooks: they keep the old paths
    out.set_spill(tmp.path(), std::min<uint64_t>(128ULL << 20, std::max<uint64_t>(8ULL << 20, budget / 16)), std::max<uint64_t>(16ULL << 20, budget / 8));
    out.write_header(headerBuf.data(), headerBuf.size());
    out.patch(headerBuf.size(), vlrBuf.data(), vlrBuf.size());

    uint64_t maxTilePts = 0;
    size_t numNonEmpty = 0;
    for (uint64_t v : tileSize) { maxTilePts = std::max(maxTilePts, v); if (v) numNonEmpty++; }
    // slots 0..S-1 run the pipeline; the last one is only used to retry (and split) a tile that failed
    std::unique_ptr<FeatureEngine> enginePtr;
#ifdef FEATURES_WITH_CUDA
    if (useGpu) enginePtr.reset(new GpuEngine(gcfg, numSlotsInFlight + 1));
#endif
    if (!enginePtr) enginePtr = makeCpuEngine(gcfg, numSlotsInFlight + 1, numThreads);
    FeatureEngine& engine = *enginePtr;
    // A tile above the point cap (the densest tile can be, see the tile size warning) does not get a pipeline slot: it goes
    // straight to the synchronous path, which allocates for it and splits it when that fails.
    const size_t slotCap = (size_t)std::max<uint64_t>(1, std::min<uint64_t>(maxTilePts, capPoints));
    for (int k = 0; k < numSlotsInFlight; k++)
        if (!engine.reserve(k, slotCap))
            throw std::runtime_error(std::string("cannot reserve memory for the tiles: ") + engine.lastError());
    const int fbSlot = numSlotsInFlight;
    const int rowStride = engine.rowStride();
    const size_t featBytes = (size_t)gcfg.numSlots * 4;

    // ---- host stages (all OpenMP-parallel inside the tile)
    struct Prepared { int nCore = 0; double unit = 0; int64_t extent[3] = {0, 0, 0}; };
    // bounding box, integer coordinates (ticks) and core flags into the staging buffers of the slot
    auto prepare = [&](int slot, const std::vector<ooc::TilePt>& pts) -> Prepared {
        const int tileN = (int)pts.size();
        double mn[3] = {1e300, 1e300, 1e300}, mx[3] = {-1e300, -1e300, -1e300};
        int coreCount = 0;
#pragma omp parallel
        {
            double lmn[3] = {1e300, 1e300, 1e300}, lmx[3] = {-1e300, -1e300, -1e300};
            int lc = 0;
#pragma omp for nowait
            for (int j = 0; j < tileN; j++) {
                const double c[3] = {ooc::tp_x(pts[j], li), ooc::tp_y(pts[j], li), ooc::tp_z(pts[j], li)};
                for (int k = 0; k < 3; k++) { lmn[k] = std::min(lmn[k], c[k]); lmx[k] = std::max(lmx[k], c[k]); }
                lc += pts[j].core;
            }
#pragma omp critical(feat_bbox)
            {
                for (int k = 0; k < 3; k++) { mn[k] = std::min(mn[k], lmn[k]); mx[k] = std::max(mx[k], lmx[k]); }
                coreCount += lc;
            }
        }
        // ticks from the minimum of the tile: the unit is the smallest LAS scale (exact for data quantised on the LAS
        // grid), enlarged when the tile is so large that the ticks would overflow int32
        const double lasUnit = std::min(li.scale[0], std::min(li.scale[1], li.scale[2]));
        const double ext = std::max(mx[0] - mn[0], std::max(mx[1] - mn[1], mx[2] - mn[2]));
        Prepared pr;
        pr.nCore = coreCount;
        pr.unit = std::max(lasUnit, ext / 1.0e9);
        int32_t* hx = engine.hostXyz(slot);
        uint8_t* hc = engine.hostCore(slot);
#pragma omp parallel for schedule(static)
        for (int j = 0; j < tileN; j++) {
            hx[3 * (size_t)j + 0] = (int32_t)std::llround((ooc::tp_x(pts[j], li) - mn[0]) / pr.unit);
            hx[3 * (size_t)j + 1] = (int32_t)std::llround((ooc::tp_y(pts[j], li) - mn[1]) / pr.unit);
            hx[3 * (size_t)j + 2] = (int32_t)std::llround((ooc::tp_z(pts[j], li) - mn[2]) / pr.unit);
            hc[j] = pts[j].core ? 1 : 0;
        }
        for (int k = 0; k < 3; k++) pr.extent[k] = (int64_t)std::llround((mx[k] - mn[k]) / pr.unit) + 1;
        return pr;
    };
    // LAS records of the core points (zero-filled, then the base fields and the columns computed by the GPU)
    auto pack = [&](int slot, const std::vector<ooc::TilePt>& pts, int coreCount, std::vector<uint8_t>& outRecs,
                    std::vector<std::pair<uint32_t, uint32_t>>* rowsOut) {
        const int tileN = (int)pts.size();
        outRecs.assign((size_t)coreCount * REC_LEN, 0);
        std::vector<uint32_t> corePos((size_t)coreCount);
        {
            size_t w = 0;
            for (int j = 0; j < tileN; j++) if (pts[j].core) corePos[w++] = (uint32_t)j;
        }
        if (rowsOut) {
            rowsOut->resize((size_t)coreCount);
            for (int c = 0; c < coreCount; c++) (*rowsOut)[(size_t)c] = {pts[corePos[(size_t)c]].row, (uint32_t)c};
        }
        const float* rows = engine.hostRows(slot);
#pragma omp parallel for schedule(static)
        for (int64_t c = 0; c < (int64_t)coreCount; c++) {
            const ooc::TilePt& rp = pts[corePos[(size_t)c]];
            uint8_t* rec = outRecs.data() + (size_t)c * REC_LEN;
            auto put = [&](size_t off, const auto& v) { std::memcpy(rec + off, &v, sizeof v); };

            const double px = ooc::tp_x(rp, li), py = ooc::tp_y(rp, li), pz = ooc::tp_z(rp, li);
            put(0, (int32_t)std::round((px - hdr.offX) / hdr.scaleX));
            put(4, (int32_t)std::round((py - hdr.offY) / hdr.scaleY));
            put(8, (int32_t)std::round((pz - hdr.offZ) / hdr.scaleZ));
            put(12, rp.intensity);
            rec[14] = (rp.return_num & 0x07) | ((rp.num_returns & 0x07) << 3);
            rec[15] = rp.class_id;
            int scan_angle_rank = std::max(-128, std::min(127, (int)rp.scan_angle));
            rec[16] = (uint8_t)((int8_t)scan_angle_rank);
            put(28, rp.r);
            put(30, rp.g);
            put(32, rp.b);
            put((size_t)BASE_SIZE + off_pid, rp.pid);
            // the requested columns are consecutive in the record (right after POINT_ID), in the order of the rows
            std::memcpy(rec + BASE_SIZE + 4, rows + (size_t)c * rowStride, featBytes);
            // normals stay zero
        }
    };

    // synchronous computation of one tile (the runner calls it for a tile that failed in the pipeline and for its halves)
    ooc::ComputeTile compute = [&](const std::vector<ooc::TilePt>& pts, std::vector<uint8_t>& outRecs, std::string& error) -> bool {
        if (failAbove && pts.size() > failAbove) { error = "simulated failure (--test-fail-above)"; return false; }
        if (!engine.reserve(fbSlot, pts.size())) { error = engine.lastError(); return false; }
        Prepared pr = prepare(fbSlot, pts);
        if (engine.submit(fbSlot, (int)pts.size(), pr.nCore, pr.unit, pr.extent) != 0 || engine.wait(fbSlot) != 0) {
            error = engine.lastError();
            return false;
        }
        pack(fbSlot, pts, pr.nCore, outRecs, nullptr);
        return true;
    };
    auto t_pipeline = now_t();
    ooc::TileRunner runner(out, (size_t)REC_LEN, compute, li);
    runner.set_buffer(BUFFER_SIZE);

    // ---- pipeline: reader thread -> (prepare + compute, `numSlotsInFlight` tiles in flight) -> pack -> writer thread
    struct TileJob { size_t t = 0; std::vector<ooc::TilePt> pts; };
    struct WriteJob { std::vector<std::pair<uint32_t, uint32_t>> rows; std::vector<uint8_t> recs; uint32_t nc = 0; };
    struct Flight { bool active = false; TileJob job; int nCore = 0; };
    ooc::BoundedQueue<TileJob> readQ(1);
    ooc::BoundedQueue<WriteJob> writeQ(1);
    std::exception_ptr readErr, writeErr;
    double loadTotal = 0, prepTotal = 0, submitTotal = 0, waitTotal = 0, packTotal = 0;

    std::thread reader([&] {
        try {
            for (size_t t = 0; t < numTiles; t++) {
                if (tileSize[t] == 0) continue;
                auto t_load = now_t();
                TileJob job;
                job.t = t;
                job.pts = ooc::load_tile(writer.path(t), tileSize[t]);
                ::unlink(writer.path(t).c_str());
                loadTotal += elapsed(t_load);
                if (!readQ.push(std::move(job))) break;
            }
        } catch (...) { readErr = std::current_exception(); }
        readQ.close();
    });
    std::thread writerThread([&] {
        WriteJob w;
        while (writeQ.pop(w)) {
            if (!writeErr) {
                try { runner.commit(w.rows, w.recs.data(), w.nc); } catch (...) { writeErr = std::current_exception(); }
            }
            std::vector<uint8_t>().swap(w.recs);
        }
    });

    auto retrySync = [&](TileJob& job, const std::string& why) {
        std::cout << "  [Warning] tile " << job.t << " failed in the pipeline (" << why << "): retrying it synchronously" << std::endl;
        runner.process(job.pts, 0, std::to_string(job.t));
    };
    std::vector<Flight> fl((size_t)numSlotsInFlight);
    int nextSlot = 0, tileNo = 0;
    auto progress = [&] { std::cout << "[progress] features " << (100 * (uint64_t)tileNo / std::max<size_t>(1, numNonEmpty)) << "%" << std::endl; };
    auto finishFlight = [&](int slot) {
        Flight& f = fl[(size_t)slot];
        if (!f.active) return;
        f.active = false;
        auto t_w = now_t();
        const int rc = engine.wait(slot);
        waitTotal += elapsed(t_w);
        if (rc != 0) { retrySync(f.job, engine.lastError()); return; }
        auto t_p = now_t();
        WriteJob w;
        w.nc = (uint32_t)f.nCore;
        pack(slot, f.job.pts, f.nCore, w.recs, &w.rows);
        std::vector<ooc::TilePt>().swap(f.job.pts);
        packTotal += elapsed(t_p);
        writeQ.push(std::move(w));
    };

    try {
        TileJob job;
        while (readQ.pop(job)) {
            tileNo++;
            if (job.pts.size() > slotCap) {
                retrySync(job, "larger than the tile cap");
                progress();
                continue;
            }
            const int slot = nextSlot;
            nextSlot = (nextSlot + 1) % numSlotsInFlight;
            finishFlight(slot);                       // the slot is reused: collect the tile it was computing
            auto t_p = now_t();
            Prepared pr = prepare(slot, job.pts);
            prepTotal += elapsed(t_p);
            if (pr.nCore > 0) {
                auto t_s = now_t();
                const int rc = (failAbove && job.pts.size() > failAbove) ? -1 : engine.submit(slot, (int)job.pts.size(), pr.nCore, pr.unit, pr.extent);
                submitTotal += elapsed(t_s);
                if (rc != 0) retrySync(job, failAbove ? "simulated failure" : engine.lastError());
                else { Flight& f = fl[(size_t)slot]; f.active = true; f.nCore = pr.nCore; f.job = std::move(job); }
            }
            progress();
            if (writeErr) break;
        }
        for (int k = 0; k < numSlotsInFlight; k++) finishFlight((nextSlot + k) % numSlotsInFlight);   // in submission order
    } catch (...) {
        readQ.close(); writeQ.close();
        reader.join(); writerThread.join();
        throw;
    }
    writeQ.close();
    writerThread.join();
    readQ.close();
    reader.join();
    if (readErr) std::rethrow_exception(readErr);
    if (writeErr) std::rethrow_exception(writeErr);
    const double pipelineSec = elapsed(t_pipeline);
    auto t_finish = now_t();
    runner.finish();
    std::cout << "Pipeline: " << pipelineSec << " s, final row merge (spill): " << elapsed(t_finish) << " s" << std::endl;

    if (runner.stats.written != N) {
        throw std::runtime_error("Only " + std::to_string(runner.stats.written) + " of " + std::to_string(N) +
                                 " points were written: some points belong to no tile");
    }
    std::cout << "Output rows written: " << runner.stats.written << " (tiles split after errors: " << runner.stats.splits << ")" << std::endl;
    std::cout << "Host time: compute " << runner.stats.computeSec << " s, row writes " << runner.stats.writeSec << " s in " << runner.stats.writeRuns << " writes" << std::endl;
    std::cout << "Profile (s): read tiles " << loadTotal << " (reader thread), host prep " << prepTotal << ", gpu submit " << submitTotal << ", waiting for the GPU " << waitTotal << ", record pack " << packTotal << ", row writes " << runner.stats.writeSec << " (writer thread)" << std::endl;
    if (useGpu && gpuProfileEnabled()) { const StageTimes g = engine.times(); std::cout << "GPU stages (s): h2d " << g.h2d << ", grid " << g.grid << ", sort " << g.sort << ", kernel " << g.kernel << ", d2h " << g.d2h << " (" << g.tiles << " tiles)" << std::endl; }

    double total = elapsed(global_start);
    int mn = (int)(total / 60), sc = (int)total % 60;
    std::cout << "\nDone. " << runner.stats.written << " points, "
              << runner.stats.tilesDone << " tiles in ";
    if (mn > 0) std::cout << mn << "m " << sc << "s" << std::endl;
    else std::cout << total << "s" << std::endl;

    ooc::TempWorkDir::extra_paths().clear();
    return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        std::remove(outputFile.c_str());
        return 1;
    }
}

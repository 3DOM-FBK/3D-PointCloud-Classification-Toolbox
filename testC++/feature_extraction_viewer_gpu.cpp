// feature_extraction_viewer_gpu.cpp
// GPU-accelerated tiled feature extraction for massive point clouds
// No PCL dependency — all computation is done via CUDA kernels
//
// Out-of-core: the cloud is never loaded as a whole. The tile size comes from a point cap derived from the memory
// budget and from the free VRAM, the points are distributed into one temporary file per tile (core + buffer), and the
// tiles are computed one at a time (only that tile is uploaded to the GPU). The record of every core point is written
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

#include <cuda_runtime.h>

#include "gpu_features.cuh"
#include "ooc/tile_pipeline.h"

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
    if (argc < 3) {
        std::cout << "\nUsage: " << argv[0] << " <input_file> <output_file> [options]\n"
                  << "Options:\n"
                  << "  --features f1,f2,...    Comma-separated list of features\n"
                  << "  --tile_size S           Tile size in meters (manual override)\n"
                  << "  --target_tiles N        Auto-tiling target tile count (default 16)\n"
                  << "  --buffer B              Buffer in meters (default 4)\n"
                  << "  --radius r1,r2,...      Radius scales (default 0.8,1.2,2.0,3.0)\n"
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
    size_t vramFree = 0, vramTotal = 0;
    if (cudaMemGetInfo(&vramFree, &vramTotal) != cudaSuccess) {
        std::cerr << "ERROR: no CUDA device available" << std::endl;
        return 1;
    }
    // bytes per tile point: host (tile record, xyz, isCore, features, output record, row index, bbox copy) /
    // device (xyz, isCore, cell ids, sorted idx, sorted xyz, features)
    const double hostPerPoint = 36.0 + 24.0 + 4.0 + 4.0 * GPU_F_COUNT * scalesCount + REC_LEN + 8.0 + 24.0;
    const double vramPerPoint = 24.0 + 4.0 + 8.0 + 24.0 + 4.0 * GPU_F_COUNT * scalesCount;
    const uint64_t capHost = (uint64_t)(0.6 * (double)budget / hostPerPoint);
    const uint64_t capVram = (uint64_t)(0.7 * (double)vramFree / vramPerPoint);
    const uint64_t capPoints = std::max<uint64_t>(100000, std::min(capHost, capVram));
    std::cout << "Tile cap: " << capPoints << " points (host budget " << budget / 1048576 << " MB -> " << capHost
              << ", free VRAM " << vramFree / 1048576 << " MB -> " << capVram << ")" << std::endl;

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
    int numThreads = 1;
#ifdef _OPENMP
    numThreads = omp_get_max_threads();
#endif
    ooc::BucketWriter writer(tmp.path(), "t", numTiles, std::max<uint64_t>(16ULL << 20, budget / 8));
    const uint64_t blockBytes = std::min<uint64_t>(16ULL << 20, std::max<uint64_t>(1ULL << 20, budget / 8 / (uint64_t)(4 * numThreads)));
    std::vector<uint64_t> tileSize = ooc::partition_input(inputFile, li, grid, writer, blockBytes);
    std::cout << "Distribution: " << elapsed(t_part) << "s" << std::endl;

    // ------------------------------------------------------------------
    // 6. Compute the tiles on the GPU, one at a time
    // ------------------------------------------------------------------
    ooc::RowOutput out(outputFile, offsetToData, (size_t)REC_LEN, N);
    out.set_spill_mode(spillMode);
    out.set_spill(tmp.path(), std::min<uint64_t>(128ULL << 20, std::max<uint64_t>(8ULL << 20, budget / 16)), std::max<uint64_t>(16ULL << 20, budget / 8));
    out.write_header(headerBuf.data(), headerBuf.size());
    out.patch(headerBuf.size(), vlrBuf.data(), vlrBuf.size());

    double gpuTimeTotal = 0;
    ooc::ComputeTile compute = [&](const std::vector<ooc::TilePt>& pts, std::vector<uint8_t>& outRecs, std::string& error) -> bool {
        const int tileN = (int)pts.size();
        std::vector<double> xyz((size_t)tileN * 3);
        std::vector<int> h_isCore(tileN, 0);
        int coreCount = 0;
        for (int j = 0; j < tileN; j++) {
            xyz[(size_t)j * 3 + 0] = ooc::tp_x(pts[j], li);
            xyz[(size_t)j * 3 + 1] = ooc::tp_y(pts[j], li);
            xyz[(size_t)j * 3 + 2] = ooc::tp_z(pts[j], li);
            h_isCore[j] = pts[j].core;
            coreCount += pts[j].core;
        }
        std::vector<float> h_features((size_t)GPU_F_COUNT * scalesCount * tileN, 0.0f);

        GpuFeatureParams params;
        params.numPoints = tileN;
        params.numScales = scalesCount;
        std::memcpy(params.scales, scales, sizeof(float) * scalesCount);
        params.gridCellSize = maxScale;

        auto t_gpu = now_t();
        int gpuErr = computeFeaturesGPU(xyz.data(), h_isCore.data(), params, h_features.data());
        double gpuTime = elapsed(t_gpu);
        if (gpuErr != 0) {
            cudaGetLastError();
            error = "GPU error";
            return false;
        }
        gpuTimeTotal += gpuTime;
        std::cout << "- " << coreCount << " core / " << tileN << " total, GPU time: " << gpuTime << "s" << std::endl;

        outRecs.assign((size_t)coreCount * REC_LEN, 0);
        std::vector<uint8_t> rec_buf(REC_LEN, 0);
        size_t w = 0;
        for (int j = 0; j < tileN; j++) {
            if (!h_isCore[j]) continue;
            const ooc::TilePt& rp = pts[j];
            std::fill(rec_buf.begin(), rec_buf.end(), 0);

            double px = ooc::tp_x(rp, li), py = ooc::tp_y(rp, li), pz = ooc::tp_z(rp, li);
            int32_t ixr = (int32_t)std::round((px - hdr.offX) / hdr.scaleX);
            int32_t iyr = (int32_t)std::round((py - hdr.offY) / hdr.scaleY);
            int32_t izr = (int32_t)std::round((pz - hdr.offZ) / hdr.scaleZ);

            wLE<int32_t>(rec_buf, 0, ixr);
            wLE<int32_t>(rec_buf, 4, iyr);
            wLE<int32_t>(rec_buf, 8, izr);
            wLE<uint16_t>(rec_buf, 12, rp.intensity);
            rec_buf[14] = (rp.return_num & 0x07) | ((rp.num_returns & 0x07) << 3);
            rec_buf[15] = rp.class_id;
            int scan_angle_rank = (int)rp.scan_angle;
            scan_angle_rank = std::max(-128, std::min(127, scan_angle_rank));
            rec_buf[16] = (uint8_t)((int8_t)scan_angle_rank);
            wLE<uint16_t>(rec_buf, 28, rp.r);
            wLE<uint16_t>(rec_buf, 30, rp.g);
            wLE<uint16_t>(rec_buf, 32, rp.b);

            wLE<uint32_t>(rec_buf, BASE_SIZE + off_pid, rp.pid);

            // Layout of h_features: [featureId * numScales * N + scaleIdx * N + pointIdx]
            for (int f = 0; f < F_COUNT; f++) {
                if (featureActive[f]) {
                    for (int s = 0; s < scalesCount; s++) {
                        float val = h_features[(size_t)f * scalesCount * tileN + (size_t)s * tileN + j];
                        if (!std::isfinite(val)) val = 0.0f;
                        wLE<float>(rec_buf, BASE_SIZE + featureByteOffset[f][s], val);
                    }
                }
            }

            // Normals (zeros)
            wLE<float>(rec_buf, BASE_SIZE + off_nx, 0.0f);
            wLE<float>(rec_buf, BASE_SIZE + off_ny, 0.0f);
            wLE<float>(rec_buf, BASE_SIZE + off_nz, 0.0f);

            std::memcpy(outRecs.data() + w * REC_LEN, rec_buf.data(), REC_LEN);
            w++;
        }
        return true;
    };

    ooc::TileRunner runner(out, (size_t)REC_LEN, compute, li);
    runner.set_buffer(BUFFER_SIZE);
    int tileNo = 0;
    for (size_t t = 0; t < numTiles; t++) {
        if (tileSize[t] == 0) continue;
        tileNo++;
        std::cout << "\nTile [ " << tileNo << " / " << numTiles << " ]:" << std::endl;
        std::vector<ooc::TilePt> pts = ooc::load_tile(writer.path(t), tileSize[t]);
        ::unlink(writer.path(t).c_str());
        runner.process(pts, 0, std::to_string(t));
        std::cout << "[progress] features " << (100 * (t + 1) / numTiles) << "%" << std::endl;
    }
    runner.finish();

    if (runner.stats.written != N) {
        throw std::runtime_error("Only " + std::to_string(runner.stats.written) + " of " + std::to_string(N) +
                                 " points were written: some points belong to no tile");
    }
    std::cout << "Output rows written: " << runner.stats.written << " (tiles split after errors: " << runner.stats.splits << ")" << std::endl;
    std::cout << "Host time: compute " << runner.stats.computeSec << " s, row writes " << runner.stats.writeSec << " s in " << runner.stats.writeRuns << " writes" << std::endl;

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

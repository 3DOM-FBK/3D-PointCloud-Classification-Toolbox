// feature_extraction_viewer_v1.cpp
// GPU-accelerated tiled feature extraction for massive point clouds
// No PCL dependency — all computation is done via CUDA kernels

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
#include <cstring>
#include <sstream>

#include "gpu_features.cuh"

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
TimePoint now_t() { return std::chrono::high_resolution_clock::now(); }
double elapsed(TimePoint start) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
}

// Global settings
const int MAX_SCALES = 10;
int scalesCount = 4;
float scales[MAX_SCALES] = { 0.8f, 1.2f, 2.0f, 3.0f };

float color_bitter = 256.0f / 65536.0f;

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
// Lightweight point for raw-binary I/O (~48 bytes)
// ============================================================

struct RawPoint {
    double x, y, z;
    float  intensity;
    float  scan_angle;
    float  class_id;
    float  return_num;
    float  number_of_returns;
    uint16_t r, g, b;
    uint32_t raw_point_id;
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
        bool first = true;
        for (uint64_t i = 0; i < h.point_count; i += STEP) {
            f.seekg(h.offset_to_data + i * h.point_record_length);
            int32_t ix, iy, iz;
            f.read((char*)&ix, 4); f.read((char*)&iy, 4); f.read((char*)&iz, 4);
            if (!f) break;
            double x = ix * h.scaleX + h.offX;
            double y = iy * h.scaleY + h.offY;
            double z = iz * h.scaleZ + h.offZ;
            if (first) { h.minx=h.maxx=x; h.miny=h.maxy=y; h.minz=h.maxz=z; first=false; }
            else {
                h.minx=std::min(h.minx,x); h.maxx=std::max(h.maxx,x);
                h.miny=std::min(h.miny,y); h.maxy=std::max(h.maxy,y);
                h.minz=std::min(h.minz,z); h.maxz=std::max(h.maxz,z);
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
                  << "  --radius r1,r2,...      Radius scales (default 0.8,1.2,2.0,3.0)\n";
        return 0;
    }

    const std::string inputFile  = argv[1];
    const std::string outputFile = argv[2];
    std::set<std::string> requestedFeatures;
    bool useAllFeatures = true;
    bool manualTileSize = false;

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

    // ------------------------------------------------------------------
    // 1. Read LAS header
    // ------------------------------------------------------------------
    LasHeaderInfo hdr = readLasHeader(inputFile);
    if (hdr.point_count == 0) {
        std::cerr << "ERROR: No points in file." << std::endl;
        return 1;
    }
    std::cout << "Input: " << hdr.point_count << " points" << std::endl;
    std::cout << "Bounds: X[" << hdr.minx << ", " << hdr.maxx << "] Y["
              << hdr.miny << ", " << hdr.maxy << "] Z["
              << hdr.minz << ", " << hdr.maxz << "]" << std::endl;

    if (!manualTileSize) {
        TILE_SIZE = computeDynamicTileSizeFromBbox(hdr, BUFFER_SIZE, TARGET_TILES);
        std::cout << "Auto tile size from bbox: " << TILE_SIZE
                  << " (target tiles: " << TARGET_TILES << ")" << std::endl;
    }

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
    // 3. Write output LAS header + VLR
    // ------------------------------------------------------------------
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

    std::ofstream out(outputFile, std::ios::binary);
    out.write((char*)headerBuf.data(), headerBuf.size());
    out.write((char*)vlrBuf.data(),    vlrBuf.size());

    // ------------------------------------------------------------------
    // 4. Bulk read point data
    // ------------------------------------------------------------------
    auto t_read = now_t();
    std::cout << "Reading point data..." << std::flush;

    const uint64_t N = hdr.point_count;
    const int recLen = hdr.point_record_length;
    std::vector<uint8_t> rawData((size_t)N * recLen);
    {
        std::ifstream fin(inputFile, std::ios::binary);
        fin.seekg(hdr.offset_to_data);
        fin.read((char*)rawData.data(), (std::streamsize)rawData.size());
    }
    std::cout << " done (" << elapsed(t_read) << "s)" << std::endl;

    // ------------------------------------------------------------------
    // 5. Decode points and distribute to tiles
    // ------------------------------------------------------------------
    auto t_decode = now_t();

    double spanX = std::max(0.0, hdr.maxx - hdr.minx);
    double spanY = std::max(0.0, hdr.maxy - hdr.miny);
    int gnx = std::max(1, (int)std::ceil(spanX / TILE_SIZE));
    int gny = std::max(1, (int)std::ceil(spanY / TILE_SIZE));
    int numTiles = gnx * gny;

    // In auto mode, ensure we don't stay at 1 tile when a multi-tile target is requested.
    if (!manualTileSize && numTiles == 1 && TARGET_TILES > 1 && (spanX > 0.0 || spanY > 0.0)) {
        double safeSpanX = std::max(spanX, 1e-9);
        double safeSpanY = std::max(spanY, 1e-9);
        double aspect = safeSpanX / safeSpanY;

        int targetNx = std::max(1, (int)std::round(std::sqrt((double)TARGET_TILES * aspect)));
        int targetNy = std::max(1, (int)std::ceil((double)TARGET_TILES / targetNx));

        gnx = std::max(1, targetNx);
        gny = std::max(1, targetNy);
        numTiles = gnx * gny;

        std::cout << "Auto fallback grid override: " << gnx << " x " << gny
                  << " (target tiles: " << TARGET_TILES << ")" << std::endl;
    }
    std::cout << "Grid: " << gnx << " x " << gny << " = " << numTiles << " tiles" << std::endl;

    std::vector<std::vector<uint64_t>> tileIndices(numTiles);

    // Decode all raw points
    std::vector<RawPoint> allPoints(N);
    std::vector<double> allXYZ((size_t)N * 3);

    bool isNewFormat = (hdr.point_fmt >= 6);
    bool hasRGB = (hdr.point_fmt == 2 || hdr.point_fmt == 3 || hdr.point_fmt == 5 ||
                   hdr.point_fmt == 7 || hdr.point_fmt == 8 || hdr.point_fmt == 10);
    int rgbOffset = 0;
    if (hasRGB) {
        switch (hdr.point_fmt) {
            case 2: rgbOffset = 20; break;
            case 3: rgbOffset = 28; break;
            case 5: rgbOffset = 28; break;
            case 7: case 8: case 10: rgbOffset = 30; break;
            default: hasRGB = false; break;
        }
    }

    // Find POINT_ID in VLR extra bytes
    int extraBytesStart = hdr.base_size;
    int pidExtraOffset = -1;
    {
        std::ifstream vf(inputFile, std::ios::binary);
        uint32_t num_vlrs = 0;
        vf.seekg(100); vf.read((char*)&num_vlrs, 4);
        vf.seekg(hdr.header_size);
        for (uint32_t v = 0; v < num_vlrs; ++v) {
            uint8_t vlrhdr[54] = {};
            vf.read((char*)vlrhdr, 54);
            if (!vf) break;
            char uid[17] = {};
            uint16_t rid = 0, rlen = 0;
            std::memcpy(uid, vlrhdr + 2, 16);
            std::memcpy(&rid, vlrhdr + 18, 2);
            std::memcpy(&rlen, vlrhdr + 20, 2);
            if (std::string(uid) == "LASF_Spec" && rid == 4) {
                int ndims = rlen / 192;
                std::vector<uint8_t> vdata(rlen);
                vf.read((char*)vdata.data(), rlen);
                int runoff = 0;
                for (int di = 0; di < ndims; ++di) {
                    uint8_t dtype = vdata[di * 192 + 2];
                    char nbuf[33] = {};
                    std::memcpy(nbuf, vdata.data() + di * 192 + 4, 32);
                    int sz = 0;
                    switch (dtype) {
                        case 1: case 2: sz=1; break;
                        case 3: case 4: sz=2; break;
                        case 5: case 6: sz=4; break;
                        case 7: case 8: sz=8; break;
                        case 9: sz=4; break;
                        case 10: sz=8; break;
                    }
                    std::string dname(nbuf, strnlen(nbuf, 32));
                    std::string lower = dname;
                    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                    if (lower == "point_id" || lower == "pointid")
                        pidExtraOffset = runoff;
                    runoff += sz;
                }
                break;
            } else {
                vf.seekg(rlen, std::ios::cur);
            }
        }
    }

    // Parallel decode
    #pragma omp parallel for schedule(static)
    for (int64_t pi = 0; pi < (int64_t)N; pi++) {
        const uint8_t* rec = rawData.data() + (size_t)pi * recLen;
        int32_t ix, iy, iz;
        std::memcpy(&ix, rec + 0, 4);
        std::memcpy(&iy, rec + 4, 4);
        std::memcpy(&iz, rec + 8, 4);

        RawPoint& rp = allPoints[pi];
        rp.x = ix * hdr.scaleX + hdr.offX;
        rp.y = iy * hdr.scaleY + hdr.offY;
        rp.z = iz * hdr.scaleZ + hdr.offZ;

        allXYZ[(size_t)pi * 3 + 0] = rp.x;
        allXYZ[(size_t)pi * 3 + 1] = rp.y;
        allXYZ[(size_t)pi * 3 + 2] = rp.z;

        uint16_t rawIntensity;
        std::memcpy(&rawIntensity, rec + 12, 2);
        rp.intensity = (float)rawIntensity;

        if (isNewFormat) {
            rp.return_num = (float)(rec[14] & 0x0F);
            rp.number_of_returns = (float)((rec[14] >> 4) & 0x0F);
            rp.class_id = (float)rec[16];
            int16_t sa; std::memcpy(&sa, rec + 18, 2);
            rp.scan_angle = (float)sa;
        } else {
            rp.return_num = (float)(rec[14] & 0x07);
            rp.number_of_returns = (float)((rec[14] >> 3) & 0x07);
            rp.class_id = (float)rec[15];
            rp.scan_angle = (float)((int8_t)rec[16]);
        }

        if (hasRGB) {
            std::memcpy(&rp.r, rec + rgbOffset, 2);
            std::memcpy(&rp.g, rec + rgbOffset + 2, 2);
            std::memcpy(&rp.b, rec + rgbOffset + 4, 2);
        } else { rp.r = rp.g = rp.b = 0; }

        if (pidExtraOffset >= 0 && extraBytesStart + pidExtraOffset + 4 <= recLen)
            std::memcpy(&rp.raw_point_id, rec + extraBytesStart + pidExtraOffset, 4);
        else
            rp.raw_point_id = (uint32_t)pi;
    }

    // Distribute to tiles
    for (uint64_t pi = 0; pi < N; pi++) {
        const RawPoint& rp = allPoints[pi];
        int ix_min = std::max(0, (int)std::floor((rp.x - BUFFER_SIZE - hdr.minx) / TILE_SIZE));
        int ix_max = std::min(gnx - 1, (int)std::floor((rp.x + BUFFER_SIZE - hdr.minx) / TILE_SIZE));
        int iy_min = std::max(0, (int)std::floor((rp.y - BUFFER_SIZE - hdr.miny) / TILE_SIZE));
        int iy_max = std::min(gny - 1, (int)std::floor((rp.y + BUFFER_SIZE - hdr.miny) / TILE_SIZE));

        for (int tix = ix_min; tix <= ix_max; tix++) {
            for (int tiy = iy_min; tiy <= iy_max; tiy++) {
                double tx0 = hdr.minx + tix * TILE_SIZE;
                double tx1 = tx0 + TILE_SIZE;
                double ty0 = hdr.miny + tiy * TILE_SIZE;
                double ty1 = ty0 + TILE_SIZE;
                if (rp.x >= tx0 - BUFFER_SIZE && rp.x < tx1 + BUFFER_SIZE &&
                    rp.y >= ty0 - BUFFER_SIZE && rp.y < ty1 + BUFFER_SIZE) {
                    tileIndices[tix * gny + tiy].push_back(pi);
                }
            }
        }
    }

    std::cout << "Decode + distribute: " << elapsed(t_decode) << "s" << std::endl;

    std::cout << "Uploading XYZ cloud to GPU..." << std::flush;
    auto t_upload_gpu = now_t();
    if (initGpuPointCloud(allXYZ.data(), N) != 0) {
        std::cerr << " failed." << std::endl;
        return 1;
    }
    std::cout << " done (" << elapsed(t_upload_gpu) << "s)" << std::endl;
    allXYZ.clear();
    allXYZ.shrink_to_fit();

    // Free raw binary data
    rawData.clear();
    rawData.shrink_to_fit();

    // ------------------------------------------------------------------
    // 6. Process tiles on GPU
    // ------------------------------------------------------------------
    uint64_t totalWritten = 0;
    int tilesProcessed = 0;
    auto t_compute = now_t();

    for (int tix = 0; tix < gnx; tix++) {
        for (int tiy = 0; tiy < gny; tiy++) {
            int tileIdx = tix * gny + tiy;
            auto& indices = tileIndices[tileIdx];
            if (indices.empty()) continue;

            double tx0 = hdr.minx + tix * TILE_SIZE;
            double tx1 = tx0 + TILE_SIZE;
            double ty0 = hdr.miny + tiy * TILE_SIZE;
            double ty1 = ty0 + TILE_SIZE;

            int tileN = (int)indices.size();

            std::vector<int>   h_isCore(tileN, 0);
            int coreCount = 0;

            int tileProgress = tix * gny + tiy + 1;  // 1..numTiles
            std::cout << "\nTile [ " << tileProgress << " / " << numTiles << " ]:" << std::endl;

            // Allocate feature output
            size_t featureSize = (size_t)GPU_F_COUNT * scalesCount * tileN;
            std::vector<float> h_features(featureSize, 0.0f);

            // Call GPU
            GpuFeatureParams params;
            params.numPoints = tileN;
            params.numScales = scalesCount;
            std::memcpy(params.scales, scales, sizeof(float) * scalesCount);
            params.gridCellSize = maxScale;

            auto t_gpu = now_t();
            int gpuErr = computeFeaturesGPUFromTileIndices(
                indices.data(),
                tileN,
                tix,
                tiy,
                gnx,
                gny,
                tx0,
                tx1,
                ty0,
                ty1,
                params,
                h_isCore.data(),
                &coreCount,
                h_features.data()
            );
            double gpuTime = elapsed(t_gpu);

            if (coreCount == 0) {
                std::cout << "- 0 core / " << tileN << " total (buffer-only, skipped)" << std::endl;
                continue;
            }


            if (gpuErr != 0) {
                std::cerr << "- " << coreCount << " core / " << tileN << " total GPU ERROR! Skipping tile." << std::endl;
                continue;
            }

            tilesProcessed++;

            std::cout << "- " << coreCount << " core / " << tileN << " total, GPU time: " << gpuTime << "s" << std::endl;

            // Write core points to output
            std::vector<uint8_t> rec_buf(REC_LEN, 0);
            for (int j = 0; j < tileN; j++) {
                if (!h_isCore[j]) continue;

                const RawPoint& rp = allPoints[indices[j]];

                std::fill(rec_buf.begin(), rec_buf.end(), 0);

                int32_t ixr = (int32_t)std::round((rp.x - hdr.offX) / hdr.scaleX);
                int32_t iyr = (int32_t)std::round((rp.y - hdr.offY) / hdr.scaleY);
                int32_t izr = (int32_t)std::round((rp.z - hdr.offZ) / hdr.scaleZ);

                wLE<int32_t>(rec_buf, 0, ixr);
                wLE<int32_t>(rec_buf, 4, iyr);
                wLE<int32_t>(rec_buf, 8, izr);
                wLE<uint16_t>(rec_buf, 12, (uint16_t)rp.intensity);
                rec_buf[14] = ((uint8_t)rp.return_num & 0x07) |
                              (((uint8_t)rp.number_of_returns & 0x07) << 3);
                rec_buf[15] = (uint8_t)rp.class_id;
                int scan_angle_rank = (int)std::lround(rp.scan_angle);
                scan_angle_rank = std::max(-128, std::min(127, scan_angle_rank));
                rec_buf[16] = (uint8_t)((int8_t)scan_angle_rank);
                wLE<uint16_t>(rec_buf, 28, (uint16_t)(rp.r));
                wLE<uint16_t>(rec_buf, 30, (uint16_t)(rp.g));
                wLE<uint16_t>(rec_buf, 32, (uint16_t)(rp.b));

                wLE<uint32_t>(rec_buf, BASE_SIZE + off_pid, rp.raw_point_id);

                // Read features from GPU output array
                // Layout: h_features[featureId * numScales * N + scaleIdx * N + pointIdx]
                for (int f = 0; f < F_COUNT; f++) {
                    if (featureActive[f]) {
                        for (int s = 0; s < scalesCount; s++) {
                            float val = h_features[(size_t)f * scalesCount * tileN + s * tileN + j];
                            if (!std::isfinite(val)) val = 0.0f;
                            wLE<float>(rec_buf, BASE_SIZE + featureByteOffset[f][s], val);
                        }
                    }
                }

                // Normals (zeros)
                wLE<float>(rec_buf, BASE_SIZE + off_nx, 0.0f);
                wLE<float>(rec_buf, BASE_SIZE + off_ny, 0.0f);
                wLE<float>(rec_buf, BASE_SIZE + off_nz, 0.0f);

                out.write((char*)rec_buf.data(), REC_LEN);
                totalWritten++;
            }

            // Free tile data
            indices.clear();
            indices.shrink_to_fit();
        }
    }

    releaseGpuPointCloud();

    // std::cout << "Compute + write: " << elapsed(t_compute) << "s" << std::endl;

    // ------------------------------------------------------------------
    // 7. Patch point count
    // ------------------------------------------------------------------
    uint32_t totalWritten32 = static_cast<uint32_t>(std::min<uint64_t>(totalWritten, 0xFFFFFFFFull));
    out.seekp(107);
    out.write((char*)&totalWritten32, 4);
    out.seekp(111);
    out.write((char*)&totalWritten32, 4);
    out.close();

    double total = elapsed(global_start);
    int mn = (int)(total / 60), sc = (int)total % 60;
    std::cout << "\nDone. " << totalWritten << " points, "
              << tilesProcessed << " tiles in ";
    if (mn > 0) std::cout << mn << "m " << sc << "s" << std::endl;
    else std::cout << total << "s" << std::endl;

    return 0;
}
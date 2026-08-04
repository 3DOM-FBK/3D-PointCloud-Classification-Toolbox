#define PCL_NO_PRECOMPILE

#include <memory>
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <Eigen/Core>
#include <pcl/common/pca.h>
#include <pcl/point_types.h>
#include <pcl/search/octree.h>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <iomanip>
#include <algorithm>
#include <atomic>
#include <cstring>

// No PDAL includes needed — all I/O is raw binary

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
double BUFFER_SIZE = 4.0; // Must be > max(scales)

// ============================================================
// CustomPoint — PCL point with feature arrays
// ============================================================

struct CustomPoint
{
    PCL_ADD_POINT4D;
    float anisotropy[MAX_SCALES];
    float heightAbove[MAX_SCALES];
    float heightBelow[MAX_SCALES];
    float linearity[MAX_SCALES];
    float neighbours[MAX_SCALES];
    float omnivariance[MAX_SCALES];
    float planarity[MAX_SCALES];
    float sphericity[MAX_SCALES];
    float surface_variation[MAX_SCALES];
    float verticality[MAX_SCALES];
    float verticalRange[MAX_SCALES];
    float height;
    float class_id;
    float scan_angle;
    float intensity;
    float number_of_returns;
    float return_num;
    float normal_x;
    float normal_y;
    float normal_z;
    float point_source_id;
    float raw_normal_x;
    float raw_normal_y;
    float raw_normal_z;
    uint32_t raw_point_id;
    PCL_ADD_UNION_RGB;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(
    CustomPoint,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, height, height)
    (float, class_id, class_id)
    (float, scan_angle, scan_angle)
    (float, intensity, intensity)
    (float, number_of_returns, number_of_returns)
    (float, return_num, return_num)
    (float, normal_x, normal_x)
    (float, normal_y, normal_y)
    (float, normal_z, normal_z)
    (float, point_source_id, point_source_id)
    (float, rgb, rgb)
)

const std::set<std::string> AVAILABLE_SCALE_FEATURES = {
    "anisotropy", "omnivariance", "sphericity",
    "planarity", "linearity", "verticality", "surface_variation",
    "neighbours", "vertical_range", "height_above", "height_below"
};

const std::set<std::string> AVAILABLE_SINGLE_FEATURES = {
    "height"
};

// ============================================================
// Lightweight point for raw-binary distribution (~48 bytes)
// ============================================================

struct RawPoint {
    double x, y, z;         // world coordinates
    float  intensity;
    float  scan_angle;
    float  class_id;
    float  return_num;
    float  number_of_returns;
    uint16_t r, g, b;
    uint32_t raw_point_id;
};

// ============================================================
// LAS header info
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
// Shape feature computation (unchanged logic, thread-safe signature)
// ============================================================

bool GetEigenVector(const Eigen::Matrix3f& eigenVectors, unsigned index, double eigenVector[]) {
    if (eigenVector && index < (unsigned)eigenVectors.size()) {
        for (unsigned i = 0; i < (unsigned)eigenVectors.size(); ++i)
            eigenVector[i] = eigenVectors(i, index);
        return true;
    }
    return false;
}

void computeShapeFeatures(int pt_idx,
                          pcl::PointCloud<CustomPoint>::Ptr pcl_in,
                          pcl::search::Search<CustomPoint>::Ptr kdTree)
{
    // Each thread gets its own PCA instance (thread-safe)
    pcl::PCA<CustomPoint> PCA;
    PCA.setInputCloud(pcl_in);

    std::vector<int>   neighborsIndices;
    std::vector<float> neighborsDistances;
    Eigen::Matrix3f eigenvectors;
    Eigen::Vector3f eigenvalues;

    for (int i = 0; i < scalesCount; i++) {
        kdTree->radiusSearch(pcl_in->points[pt_idx], scales[i], neighborsIndices, neighborsDistances);
        pcl_in->points[pt_idx].neighbours[i] = (float)neighborsIndices.size();

        if (neighborsIndices.size() < 3)
            kdTree->nearestKSearch(pcl_in->points[pt_idx], (int)(scales[i] * 8), neighborsIndices, neighborsDistances);

        if (neighborsIndices.size() >= 3) {
            float yMin = pcl_in->points[neighborsIndices[0]].y;
            float yMax = yMin;
            for (size_t n = 1; n < neighborsIndices.size(); ++n) {
                float yy = pcl_in->points[neighborsIndices[n]].y;
                if (yy < yMin) yMin = yy;
                if (yy > yMax) yMax = yy;
            }
            pcl_in->points[pt_idx].verticalRange[i] = yMax - yMin;
            pcl_in->points[pt_idx].heightAbove[i]   = yMax - pcl_in->points[pt_idx].y;
            pcl_in->points[pt_idx].heightBelow[i]   = pcl_in->points[pt_idx].y - yMin;

            PCA.setIndices(std::make_shared<std::vector<int>>(neighborsIndices));
            eigenvalues  = PCA.getEigenValues();
            eigenvectors = PCA.getEigenVectors();

            float e0 = eigenvalues(0), e1 = eigenvalues(1), e2 = eigenvalues(2);
            float sum = e0 + e1 + e2;

            pcl_in->points[pt_idx].linearity[i]         = (e0 - e1) / e0;
            pcl_in->points[pt_idx].planarity[i]         = (e1 - e2) / e0;
            pcl_in->points[pt_idx].surface_variation[i] = e2 / sum;
            pcl_in->points[pt_idx].omnivariance[i]      = (e0 * e1 * e2) / (e0 / e2);
            pcl_in->points[pt_idx].anisotropy[i]        = (e0 - e2) / e0;
            pcl_in->points[pt_idx].sphericity[i]        = e2 / e0;

            Eigen::Vector3d Y(0.0, 1.0, 0.0), e3;
            GetEigenVector(eigenvectors, 2, e3.data());
            pcl_in->points[pt_idx].verticality[i] = 1.0 - std::abs(Y.dot(e3));
        }
    }
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
// Feature getter by index (avoids std::function overhead)
// ============================================================

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

inline float getFeature(const CustomPoint& p, FeatureId fid, int s) {
    switch (fid) {
        case F_ANISOTROPY:        return p.anisotropy[s];
        case F_HEIGHT_ABOVE:      return p.heightAbove[s];
        case F_HEIGHT_BELOW:      return p.heightBelow[s];
        case F_LINEARITY:         return p.linearity[s];
        case F_NEIGHBOURS:        return p.neighbours[s];
        case F_OMNIVARIANCE:      return p.omnivariance[s];
        case F_PLANARITY:         return p.planarity[s];
        case F_SPHERICITY:        return p.sphericity[s];
        case F_SURFACE_VARIATION: return p.surface_variation[s];
        case F_VERTICALITY:       return p.verticality[s];
        case F_VERTICAL_RANGE:    return p.verticalRange[s];
        default: return 0.0f;
    }
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
                  << "  --tile_size S           Tile size in meters (default 50)\n"
                  << "  --buffer B              Buffer in meters (default 4)\n"
                  << "  --radius r1,r2,...      Radius scales (default 0.8,1.2,2.0,3.0)\n";
        return 0;
    }

    const std::string inputFile  = argv[1];
    const std::string outputFile = argv[2];
    std::set<std::string> requestedFeatures;
    bool useAllFeatures = true;

    for (int a = 3; a < argc; a++) {
        std::string arg(argv[a]);
        if (arg == "--features" && a + 1 < argc) {
            std::stringstream ss(argv[++a]); std::string t;
            while (std::getline(ss, t, ',')) requestedFeatures.insert(t);
            useAllFeatures = false;
        } else if (arg == "--tile_size" && a + 1 < argc) {
            TILE_SIZE = std::stod(argv[++a]);
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
        std::cout << "WARNING: buffer (" << BUFFER_SIZE << ") < max radius (" << maxScale
                  << "), adjusting buffer to " << (maxScale + 0.5) << std::endl;
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

    // Which features are active?
    bool featureActive[F_COUNT] = {};
    for (int f = 0; f < F_COUNT; f++)
        featureActive[f] = requestedFeatures.count(FEATURE_NAMES[f]) > 0;

    std::vector<std::pair<std::string, int>> outExtra;
    int extraOffset = 0;
    int off_pid = extraOffset;
    outExtra.push_back({"POINT_ID", extraOffset}); extraOffset += 4;

    // Pre-compute: for each active feature f, for each scale s, store the byte offset
    // featureByteOffset[f][s] = offset within extra-bytes block
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
    // 4. SINGLE bulk read of entire point data block
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
    // 5. Decode raw points into lightweight RawPoint array + assign tiles
    // ------------------------------------------------------------------
    auto t_decode = now_t();

    int gnx = (int)std::ceil((hdr.maxx - hdr.minx) / TILE_SIZE);
    int gny = (int)std::ceil((hdr.maxy - hdr.miny) / TILE_SIZE);
    int numTiles = gnx * gny;
    std::cout << "Grid: " << gnx << " x " << gny << " = " << numTiles << " tiles" << std::endl;

    // For each tile, store indices of points that fall within tile+buffer
    // We use a flat vector of vectors indexed by tileIdx = ix * gny + iy
    std::vector<std::vector<uint64_t>> tileIndices(numTiles);
    // Also store which points are "core" (inside tile without buffer)
    // We'll mark this during processing, not during distribution

    // Decode all points and distribute
    std::vector<RawPoint> allPoints(N);

    // Detect LAS format for field offsets
    // Format 6-10: bytes 14-15 are different from format 0-5
    bool isNewFormat = (hdr.point_fmt >= 6);

    // Determine if file has RGB (format 2,3,5,7,8,10)
    bool hasRGB = (hdr.point_fmt == 2 || hdr.point_fmt == 3 || hdr.point_fmt == 5 ||
                   hdr.point_fmt == 7 || hdr.point_fmt == 8 || hdr.point_fmt == 10);
    int rgbOffset = 0;
    if (hasRGB) {
        // RGB offset depends on format
        switch (hdr.point_fmt) {
            case 2: rgbOffset = 20; break;
            case 3: rgbOffset = 28; break;
            case 5: rgbOffset = 28; break;
            case 7: rgbOffset = 30; break;
            case 8: rgbOffset = 30; break;
            case 10: rgbOffset = 30; break;
            default: hasRGB = false; break;
        }
    }

    // Decode VLR to find POINT_ID extra dim offset
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

        uint16_t rawIntensity;
        std::memcpy(&rawIntensity, rec + 12, 2);
        rp.intensity = (float)rawIntensity;

        if (isNewFormat) {
            // Format 6+: byte 14 = return/numreturns packed, byte 16 = classification
            rp.return_num = (float)(rec[14] & 0x0F);
            rp.number_of_returns = (float)((rec[14] >> 4) & 0x0F);
            rp.class_id = (float)rec[16];
            int16_t sa; std::memcpy(&sa, rec + 18, 2);
            rp.scan_angle = (float)sa;
        } else {
            // Format 0-5: byte 14 = return/numreturns, byte 15 = classification
            rp.return_num = (float)(rec[14] & 0x07);
            rp.number_of_returns = (float)((rec[14] >> 3) & 0x07);
            rp.class_id = (float)rec[15];
            rp.scan_angle = (float)((int8_t)rec[16]);
        }

        if (hasRGB) {
            std::memcpy(&rp.r, rec + rgbOffset, 2);
            std::memcpy(&rp.g, rec + rgbOffset + 2, 2);
            std::memcpy(&rp.b, rec + rgbOffset + 4, 2);
        } else {
            rp.r = rp.g = rp.b = 0;
        }

        // POINT_ID from extra bytes
        if (pidExtraOffset >= 0 && extraBytesStart + pidExtraOffset + 4 <= recLen) {
            std::memcpy(&rp.raw_point_id, rec + extraBytesStart + pidExtraOffset, 4);
        } else {
            rp.raw_point_id = (uint32_t)pi;
        }
    }

    // Distribute points to tiles (sequential — writing to shared vectors)
    for (uint64_t pi = 0; pi < N; pi++) {
        const RawPoint& rp = allPoints[pi];

        // Find all tiles whose (tile+buffer) region contains this point
        int ix_min = (int)std::floor((rp.x - BUFFER_SIZE - hdr.minx) / TILE_SIZE);
        int ix_max = (int)std::floor((rp.x + BUFFER_SIZE - hdr.minx) / TILE_SIZE);
        int iy_min = (int)std::floor((rp.y - BUFFER_SIZE - hdr.miny) / TILE_SIZE);
        int iy_max = (int)std::floor((rp.y + BUFFER_SIZE - hdr.miny) / TILE_SIZE);

        ix_min = std::max(0, ix_min); ix_max = std::min(gnx - 1, ix_max);
        iy_min = std::max(0, iy_min); iy_max = std::min(gny - 1, iy_max);

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

    // Free raw binary data — no longer needed
    rawData.clear();
    rawData.shrink_to_fit();

    // ------------------------------------------------------------------
    // 6. Process tiles with OpenMP on feature computation
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

            // Build PCL cloud for this tile (core + buffer points)
            int tileN = (int)indices.size();
            pcl::PointCloud<CustomPoint>::Ptr pclCloud(new pcl::PointCloud<CustomPoint>);
            pclCloud->width  = tileN;
            pclCloud->height = 1;
            pclCloud->points.resize(tileN);

            // Mark which points are core (inside tile without buffer)
            std::vector<bool> isCore(tileN, false);

            for (int j = 0; j < tileN; j++) {
                const RawPoint& rp = allPoints[indices[j]];
                auto& pt = pclCloud->points[j];
                std::memset(&pt, 0, sizeof(CustomPoint));

                pt.x = (float)(rp.x - hdr.offX);
                pt.y = (float)(rp.y - hdr.offY);
                pt.z = (float)rp.z;
                pt.intensity         = rp.intensity;
                pt.scan_angle        = rp.scan_angle;
                pt.class_id          = rp.class_id;
                pt.return_num        = rp.return_num;
                pt.number_of_returns = rp.number_of_returns;
                pt.r = (float)rp.r * color_bitter;
                pt.g = (float)rp.g * color_bitter;
                pt.b = (float)rp.b * color_bitter;
                pt.raw_point_id = rp.raw_point_id;

                if (rp.x >= tx0 && rp.x < tx1 && rp.y >= ty0 && rp.y < ty1)
                    isCore[j] = true;
            }

            // Count core points
            int coreCount = 0;
            for (int j = 0; j < tileN; j++) if (isCore[j]) coreCount++;
            if (coreCount == 0) continue;

            tilesProcessed++;
            std::cout << "Tile [" << tix << "," << tiy << "] "
                      << coreCount << " core / " << tileN << " total" << std::endl;

            // Build search tree (single-threaded, fast)
            pcl::search::Search<CustomPoint>::Ptr tree =
                std::make_shared<pcl::search::Octree<CustomPoint>>(0.2);
            tree->setInputCloud(pclCloud);

            // Compute features with OpenMP
            #pragma omp parallel for schedule(dynamic, 256)
            for (int j = 0; j < tileN; j++) {
                if (isCore[j]) {
                    computeShapeFeatures(j, pclCloud, tree);
                }
            }

            // Write core points to output (sequential)
            std::vector<uint8_t> rec_buf(REC_LEN, 0);
            for (int j = 0; j < tileN; j++) {
                if (!isCore[j]) continue;

                const CustomPoint& pt = pclCloud->points[j];
                double wx = pt.x + hdr.offX;
                double wy = pt.y + hdr.offY;

                std::fill(rec_buf.begin(), rec_buf.end(), 0);

                int32_t ixr = (int32_t)std::round((wx - hdr.offX) / hdr.scaleX);
                int32_t iyr = (int32_t)std::round((wy - hdr.offY) / hdr.scaleY);
                int32_t izr = (int32_t)std::round((pt.z - hdr.offZ) / hdr.scaleZ);

                wLE<int32_t>(rec_buf, 0, ixr);
                wLE<int32_t>(rec_buf, 4, iyr);
                wLE<int32_t>(rec_buf, 8, izr);
                wLE<uint16_t>(rec_buf, 12, (uint16_t)pt.intensity);
                rec_buf[14] = ((uint8_t)pt.return_num & 0x07) |
                              (((uint8_t)pt.number_of_returns & 0x07) << 3);
                rec_buf[15] = (uint8_t)pt.class_id;
                int scan_angle_rank = (int)std::lround(pt.scan_angle);
                scan_angle_rank = std::max(-128, std::min(127, scan_angle_rank));
                rec_buf[16] = (uint8_t)((int8_t)scan_angle_rank);
                wLE<uint16_t>(rec_buf, 28, (uint16_t)(pt.r / color_bitter));
                wLE<uint16_t>(rec_buf, 30, (uint16_t)(pt.g / color_bitter));
                wLE<uint16_t>(rec_buf, 32, (uint16_t)(pt.b / color_bitter));

                wLE<uint32_t>(rec_buf, BASE_SIZE + off_pid, pt.raw_point_id);

                for (int f = 0; f < F_COUNT; f++) {
                    if (featureActive[f]) {
                        for (int s = 0; s < scalesCount; s++) {
                            float val = getFeature(pt, (FeatureId)f, s);
                            if (!std::isfinite(val)) val = 0.0f;
                            wLE<float>(rec_buf, BASE_SIZE + featureByteOffset[f][s], val);
                        }
                    }
                }

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

    std::cout << "Compute + write: " << elapsed(t_compute) << "s" << std::endl;

    // ------------------------------------------------------------------
    // 7. Patch point count in LAS header
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
    else std::cout << sc << "s" << std::endl;

    return 0;
}
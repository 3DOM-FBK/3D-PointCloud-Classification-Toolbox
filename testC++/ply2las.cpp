// ply2las.cpp
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstring>


// Open3D (only for the PLY variants the streaming reader does not support)
#include <open3d/Open3D.h>

// streaming PLY -> LAS (no Open3D, out-of-core)
#include "ooc/ply2las_stream.h"

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
TimePoint now_t() { return std::chrono::high_resolution_clock::now(); }
double elapsed(TimePoint start) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
}

template<typename T>
void write_val(std::ofstream& f, T val) {
    f.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

void write_str(std::ofstream& f, const char* s, int len) {
    std::vector<char> buf(len, 0);
    int slen = (int)strlen(s);
    memcpy(buf.data(), s, std::min(slen, len));
    f.write(buf.data(), len);
}

// Per le normali: tipo 9 = float
void write_extra_bytes_record(std::ofstream& f, const char* name, const char* description) {
    write_val<uint8_t>(f, 0);
    write_val<uint8_t>(f, 0);
    write_val<uint8_t>(f, 9);   // float
    write_val<uint8_t>(f, 0);
    write_str(f, name, 32);
    for (int i = 0; i < 4;  i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 72; i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 48; i++) write_val<uint8_t>(f, 0);
    write_str(f, description, 32);
}

// Per POINT_ID: tipo 6 = uint32
void write_extra_bytes_record_uint32(std::ofstream& f, const char* name, const char* description) {
    write_val<uint8_t>(f, 0);
    write_val<uint8_t>(f, 0);
    write_val<uint8_t>(f, 6);   // uint32
    write_val<uint8_t>(f, 0);
    write_str(f, name, 32);
    for (int i = 0; i < 4;  i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 72; i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 48; i++) write_val<uint8_t>(f, 0);
    write_str(f, description, 32);
}

void write_las(const std::string& out_file,
               const std::vector<Eigen::Vector3d>& points,
               const std::vector<Eigen::Vector3d>& colors,
               const std::vector<Eigen::Vector3d>& normals,
               bool has_colors,
               bool has_normals)
{
    std::ofstream f(out_file, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << out_file << std::endl; return; }

    uint32_t n = (uint32_t)points.size();

    double min_x = 1e18, min_y = 1e18, min_z = 1e18;
    double max_x = -1e18, max_y = -1e18, max_z = -1e18;
    for (auto& p : points) {
        min_x = std::min(min_x, p[0]); max_x = std::max(max_x, p[0]);
        min_y = std::min(min_y, p[1]); max_y = std::max(max_y, p[1]);
        min_z = std::min(min_z, p[2]); max_z = std::max(max_z, p[2]);
    }

    double scale_xyz = 0.0001;
    // Use data-driven offsets to keep encoded int32 coordinates in range.
    double offset_x = min_x, offset_y = min_y, offset_z = min_z;
    double qmin_x = 1e18, qmin_y = 1e18, qmin_z = 1e18;
    double qmax_x = -1e18, qmax_y = -1e18, qmax_z = -1e18;
    for (const auto& point : points) {
        int32_t ix = (int32_t)std::round((point[0] - offset_x) / scale_xyz);
        int32_t iy = (int32_t)std::round((point[1] - offset_y) / scale_xyz);
        int32_t iz = (int32_t)std::round((point[2] - offset_z) / scale_xyz);
        double qx = ix * scale_xyz + offset_x;
        double qy = iy * scale_xyz + offset_y;
        double qz = iz * scale_xyz + offset_z;
        qmin_x = std::min(qmin_x, qx); qmax_x = std::max(qmax_x, qx);
        qmin_y = std::min(qmin_y, qy); qmax_y = std::max(qmax_y, qy);
        qmin_z = std::min(qmin_z, qz); qmax_z = std::max(qmax_z, qz);
    }

    // Extra bytes: normals (se presenti) + POINT_ID (sempre)
    uint32_t extra_bytes_payload = 0;
    if (has_normals) extra_bytes_payload += 3 * 192; // normal_x, normal_y, normal_z
    extra_bytes_payload += 192;                      // POINT_ID

    uint16_t header_size = 227;
    uint32_t vlr_count   = 1;                        // sempre 1 VLR
    uint32_t vlr_total   = 54 + extra_bytes_payload; // 54 = header VLR fisso
    uint32_t offset_to_data = header_size + vlr_total;

    uint16_t point_data_length = 34;                 // base format 3
    if (has_normals) point_data_length += 12;        // 3x float normals
    point_data_length += 4;                          // uint32 POINT_ID

    // --- LAS Header ---
    write_str(f, "LASF", 4);
    write_val<uint16_t>(f, 0);
    write_val<uint16_t>(f, 0);
    write_val<uint32_t>(f, 0);
    write_val<uint16_t>(f, 0);
    write_val<uint16_t>(f, 0);
    for (int i = 0; i < 8; i++) write_val<uint8_t>(f, 0);
    write_val<uint8_t>(f, 1);
    write_val<uint8_t>(f, 2);
    write_str(f, "OTHER", 32);
    write_str(f, "ply2las_cpp", 32);
    write_val<uint16_t>(f, 0);
    write_val<uint16_t>(f, 0);
    write_val<uint16_t>(f, header_size);
    write_val<uint32_t>(f, offset_to_data);
    write_val<uint32_t>(f, vlr_count);
    write_val<uint8_t>(f, 3);
    write_val<uint16_t>(f, point_data_length);
    write_val<uint32_t>(f, n);
    write_val<uint32_t>(f, n);
    for (int i = 0; i < 4; i++) write_val<uint32_t>(f, 0);
    write_val<double>(f, scale_xyz);
    write_val<double>(f, scale_xyz);
    write_val<double>(f, scale_xyz);
    write_val<double>(f, offset_x);
    write_val<double>(f, offset_y);
    write_val<double>(f, offset_z);
    write_val<double>(f, qmax_x); write_val<double>(f, qmin_x);
    write_val<double>(f, qmax_y); write_val<double>(f, qmin_y);
    write_val<double>(f, qmax_z); write_val<double>(f, qmin_z);

    // --- VLR unico: normals (se presenti) + POINT_ID ---
    write_val<uint16_t>(f, 0);
    write_str(f, "LASF_Spec", 16);
    write_val<uint16_t>(f, 4);
    write_val<uint16_t>(f, (uint16_t)extra_bytes_payload);
    write_str(f, "Extra Bytes Record", 32);

    if (has_normals) {
        write_extra_bytes_record(f, "NormalX", "Normal X");
        write_extra_bytes_record(f, "NormalY", "Normal Y");
        write_extra_bytes_record(f, "NormalZ", "Normal Z");
    }
    write_extra_bytes_record_uint32(f, "POINT_ID", "Point ID");

    // --- Point records ---
    for (uint32_t i = 0; i < n; ++i) {
        int32_t ix = (int32_t)std::round((points[i][0] - offset_x) / scale_xyz);
        int32_t iy = (int32_t)std::round((points[i][1] - offset_y) / scale_xyz);
        int32_t iz = (int32_t)std::round((points[i][2] - offset_z) / scale_xyz);
        write_val<int32_t>(f, ix);
        write_val<int32_t>(f, iy);
        write_val<int32_t>(f, iz);
        write_val<uint16_t>(f, 0);  // intensity
        write_val<uint8_t>(f, 0);   // return bits
        write_val<uint8_t>(f, 0);   // classification
        write_val<uint8_t>(f, 0);   // scan angle
        write_val<uint8_t>(f, 0);   // user data
        write_val<uint16_t>(f, 0);  // point source ID
        write_val<double>(f, 0.0);  // GPS time
        if (has_colors) {
            write_val<uint16_t>(f, (uint16_t)(std::clamp(colors[i][0], 0.0, 1.0) * 65535));
            write_val<uint16_t>(f, (uint16_t)(std::clamp(colors[i][1], 0.0, 1.0) * 65535));
            write_val<uint16_t>(f, (uint16_t)(std::clamp(colors[i][2], 0.0, 1.0) * 65535));
        } else {
            write_val<uint16_t>(f, 0);
            write_val<uint16_t>(f, 0);
            write_val<uint16_t>(f, 0);
        }
        if (has_normals) {
            write_val<float>(f, (float)normals[i][0]);
            write_val<float>(f, (float)normals[i][1]);
            write_val<float>(f, (float)normals[i][2]);
        }
        write_val<uint32_t>(f, i);  // POINT_ID
    }

    f.close();
}

int main(int argc, char* argv[]) {
    // optional "--memory-budget MB" and "--temp-dir DIR" anywhere; the input and output paths are positional
    double memory_budget_mb = 0;
    std::string temp_dir;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--memory-budget" && i + 1 < argc) memory_budget_mb = std::stod(argv[++i]);
        else if (a == "--temp-dir" && i + 1 < argc) temp_dir = argv[++i];
        else pos.push_back(a);
    }
    if (pos.size() < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.ply> <output.las> [--memory-budget MB] [--temp-dir DIR]" << std::endl;
        return 1;
    }

    std::string ply_path = pos[0];
    std::string out_path = pos[1];

    auto global_start = now_t();

    // Streaming path: the PLY is read in blocks and the normals are computed tile by tile (memory bounded by the budget).
    try {
        std::cout << "Loading PLY from " << ply_path << std::endl;
        std::string why;
        if (ooc::ply_to_las_streaming(ply_path, out_path, memory_budget_mb, temp_dir, why)) {
            double total = elapsed(global_start);
            std::cout << "Point cloud saved as: " << out_path << std::endl;
            std::cout << "Processing time: " << (int)total / 60 << " min " << (int)total % 60 << " sec" << std::endl;
            std::cout << out_path << std::endl;
            return 0;
        }
        std::cout << "[Info] streaming reader not used (" << why << "): falling back to Open3D" << std::endl;
        // The Open3D path holds the whole cloud, its normals, colours and the KD-tree: about 330 bytes per point.
        ooc::PlyReader probe(ply_path);
        const uint64_t budget = ooc::resolve_budget_bytes(memory_budget_mb);
        if (probe.numVertices() > 0 && 330.0 * (double)probe.numVertices() > (double)budget) {
            std::cerr << "ERROR: this PLY variant needs the whole cloud in memory (~" << (uint64_t)(330.0 * (double)probe.numVertices() / 1e6)
                      << " MB for " << probe.numVertices() << " points) but the memory budget is " << budget / 1048576
                      << " MB. Convert it to a binary PLY with x y z (and red green blue) properties only." << std::endl;
            return 3;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 2;
    }

    std::cout << "Loading PLY from " << ply_path << std::endl;
    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(ply_path, *pcd)) {
        std::cerr << "Failed to read: " << ply_path << std::endl;
        return 1;
    }
    if (pcd->points_.empty()) {
        std::cerr << "Empty point cloud" << std::endl;
        return 1;
    }

    bool has_colors  = pcd->HasColors();
    bool has_normals = false;

    std::cout << "Computing normals..." << std::endl;
    pcd->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(0.02, 30));
    pcd->OrientNormalsTowardsCameraLocation();
    has_normals = pcd->HasNormals();

    // Convert NaN normals to a (0,0,0)
    for (auto& n : pcd->normals_) {
        if (std::isnan(n[0]) || std::isnan(n[1]) || std::isnan(n[2]))
            n = {0.0, 0.0, 0.0};
    }

    write_las(out_path, pcd->points_, pcd->colors_, pcd->normals_,
              has_colors, has_normals);

    double total = elapsed(global_start);
    int mn = (int)(total / 60);
    int sc = (int)total % 60;
    std::cout << "Point cloud saved as: " << out_path
              << ", with " << pcd->points_.size() << " points" << std::endl;
    if (mn > 0)
        std::cout << "Processing time: " << mn << " min " << sc << " sec" << std::endl;
    else
        std::cout << "Processing time: " << sc << " sec" << std::endl;

    std::cout << out_path << std::endl;
    return 0;
}
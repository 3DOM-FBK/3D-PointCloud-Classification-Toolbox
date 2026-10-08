#include <open3d/Open3D.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstring>

// ============================================================
// Read/Write helpers
// ============================================================
template<typename T>
void write_val(std::ofstream& f, T val) {
    f.write(reinterpret_cast<const char*>(&val), sizeof(T));
}
template<typename T>
T read_val(std::ifstream& f) {
    T val;
    f.read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
}
void skip(std::ifstream& f, int n) {
    f.seekg(n, std::ios::cur);
}
void write_str(std::ofstream& f, const char* s, int len) {
    std::vector<char> buf(len, 0);
    int slen = (int)strlen(s);
    memcpy(buf.data(), s, std::min(slen, len));
    f.write(buf.data(), len);
}
void write_extra_bytes_record(std::ofstream& f, const char* name, const char* description) {
    write_val<uint8_t>(f, 0); write_val<uint8_t>(f, 0);
    write_val<uint8_t>(f, 9); write_val<uint8_t>(f, 0);
    write_str(f, name, 32);
    for (int i = 0; i < 4;  i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 72; i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 48; i++) write_val<uint8_t>(f, 0);
    write_str(f, description, 32);
}
void write_extra_bytes_record_uint32(std::ofstream& f, const char* name, const char* description) {
    write_val<uint8_t>(f, 0); write_val<uint8_t>(f, 0);
    write_val<uint8_t>(f, 5); write_val<uint8_t>(f, 0);
    write_str(f, name, 32);
    for (int i = 0; i < 4;  i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 72; i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 48; i++) write_val<uint8_t>(f, 0);
    write_str(f, description, 32);
}

uint8_t base_point_format(uint8_t point_format) {
    return point_format & 0x3F;
}

int las_base_point_size(uint8_t point_format) {
    switch (point_format) {
        case 0: return 20;
        case 1: return 28;
        case 2: return 26;
        case 3: return 34;
        case 4: return 57;
        case 5: return 63;
        case 6: return 30;
        case 7: return 36;
        case 8: return 38;
        case 9: return 59;
        case 10: return 67;
        default: return -1;
    }
}

bool format_has_rgb(uint8_t point_format) {
    return point_format == 2 || point_format == 3 || point_format == 5 ||
           point_format == 7 || point_format == 8 || point_format == 10;
}

// ============================================================
// read_las — legge header sequenzialmente (niente seekg intermedi)
// LAS 1.2, Point Format 3 + extra bytes NormalX/Y/Z + POINT_ID
// ============================================================
struct LasData {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> colors;
    std::vector<Eigen::Vector3d> normals;
    bool has_colors  = false;
    bool has_normals = false;
};

LasData read_las(const std::string& path) {
    LasData data;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    // Leggi header sequenzialmente seguendo la spec LAS 1.2
    // offset 0
    char sig[4]; f.read(sig, 4);
    if (std::string(sig, 4) != "LASF")
        throw std::runtime_error("Not a valid LAS file: " + path);

    // offset 4
    skip(f, 2);  // file_source_id
    skip(f, 2);  // global_encoding
    skip(f, 4);  // project_id_1
    skip(f, 2);  // project_id_2
    skip(f, 2);  // project_id_3
    skip(f, 8);  // project_id_4
    skip(f, 1);  // version_major
    skip(f, 1);  // version_minor
    skip(f, 32); // system_identifier
    skip(f, 32); // generating_software
    skip(f, 2);  // file_creation_day
    skip(f, 2);  // file_creation_year
    // offset 94
    uint16_t header_size    = read_val<uint16_t>(f); // 94
    uint32_t offset_to_data = read_val<uint32_t>(f); // 96
    skip(f, 4);  // num_vlrs                          // 100
    uint8_t  point_format_raw = read_val<uint8_t>(f); // 104
    uint16_t point_length   = read_val<uint16_t>(f); // 105
    uint32_t num_points     = read_val<uint32_t>(f); // 107
    skip(f, 20); // num_points_by_return (5*4)        // 111
    // offset 131
    double scale_x = read_val<double>(f); // 131
    double scale_y = read_val<double>(f); // 139
    double scale_z = read_val<double>(f); // 147
    double off_x   = read_val<double>(f); // 155
    double off_y   = read_val<double>(f); // 163
    double off_z   = read_val<double>(f); // 171
    // skip max/min (6*8=48 bytes)        // 179
    skip(f, 48);
    // offset 227 = fine header

    const uint8_t fmt_base = base_point_format(point_format_raw);
    const int base_point_size = las_base_point_size(fmt_base);
    if (base_point_size < 0) {
        throw std::runtime_error("Unsupported LAS point format: " + std::to_string((int)fmt_base));
    }

    // extra bytes = point_length - base_size
    int extra_size = (int)point_length - base_point_size;
    if (extra_size < 0) {
        throw std::runtime_error("Invalid LAS point length for format " + std::to_string((int)fmt_base));
    }

    // 12 (normali) + 4 (POINT_ID) = 16 -> has normals
    // 4 (solo POINT_ID) -> no normals
    bool has_normals_in_file = (extra_size >= 16);
    bool has_point_id        = (extra_size >= 4);

    data.has_colors  = format_has_rgb(fmt_base);
    data.has_normals = has_normals_in_file;

    std::cout << "  LAS header: " << num_points << " points"
              << "  format=" << (int)fmt_base
              << "  point_length=" << point_length
              << "  extra=" << extra_size
              << "  offset_to_data=" << offset_to_data
              << "  scale=(" << scale_x << "," << scale_y << "," << scale_z << ")"
              << "  offset=(" << off_x << "," << off_y << "," << off_z << ")"
              << std::endl;

    // Read the point records in large blocks and decode them from memory.
    // Walking the file field by field with ifstream (every skip() is a seekg that drops the
    // stream buffer) costs several syscalls per point: minutes for millions of points on network
    // or bind-mounted file systems.
    data.points.resize(num_points);
    data.colors.resize(num_points);
    if (has_normals_in_file) data.normals.resize(num_points, {0,0,0});

    // Offset of the RGB triplet inside the record (-1: the format has no colour)
    int rgb_off = -1;
    if (format_has_rgb(fmt_base)) {
        if (fmt_base <= 5) rgb_off = (fmt_base == 3 || fmt_base == 5) ? 28 : 20;
        else               rgb_off = 30;
    }

    f.seekg(offset_to_data);
    const size_t rec = point_length;
    const size_t block_points = std::max<size_t>(1, (64u << 20) / rec);
    std::vector<char> block(block_points * rec);
    for (uint32_t start = 0; start < num_points; start += (uint32_t)block_points) {
        const size_t cnt = std::min<size_t>(block_points, num_points - start);
        f.read(block.data(), cnt * rec);
        if ((size_t)f.gcount() != cnt * rec) throw std::runtime_error("Truncated LAS file: " + path);
        for (size_t k = 0; k < cnt; ++k) {
            const char* p = block.data() + k * rec;
            const size_t i = start + k;
            // XYZ int32 -> double
            int32_t xyz[3];
            std::memcpy(xyz, p, 12);
            data.points[i] = { xyz[0] * scale_x + off_x, xyz[1] * scale_y + off_y, xyz[2] * scale_z + off_z };
            if (rgb_off >= 0) {
                uint16_t rgb[3];
                std::memcpy(rgb, p + rgb_off, 6);
                data.colors[i] = { rgb[0] / 65535.0, rgb[1] / 65535.0, rgb[2] / 65535.0 };
            } else {
                data.colors[i] = {0.0, 0.0, 0.0};
            }
            // Extra bytes: normals (3 x float) come first, POINT_ID (ignored here) last
            if (has_normals_in_file) {
                float n[3];
                std::memcpy(n, p + base_point_size, 12);
                data.normals[i] = { (double)n[0], (double)n[1], (double)n[2] };
            }
        }
    }

    f.close();
    return data;
}

// ============================================================
// write_las
// ============================================================
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
    uint32_t extra_bytes_payload = 0;
    if (has_normals) extra_bytes_payload += 3 * 192;
    extra_bytes_payload += 192; // POINT_ID
    uint16_t header_size    = 227;
    uint32_t vlr_total      = 54 + extra_bytes_payload;
    uint32_t offset_to_data = header_size + vlr_total;
    uint16_t point_data_length = 34;
    if (has_normals) point_data_length += 12;
    point_data_length += 4; // POINT_ID

    write_str(f, "LASF", 4);
    write_val<uint16_t>(f, 0); write_val<uint16_t>(f, 0);
    write_val<uint32_t>(f, 0); write_val<uint16_t>(f, 0); write_val<uint16_t>(f, 0);
    for (int i = 0; i < 8; i++) write_val<uint8_t>(f, 0);
    write_val<uint8_t>(f, 1); write_val<uint8_t>(f, 2);
    write_str(f, "OTHER", 32);
    write_str(f, "subsample_cpp", 32);
    write_val<uint16_t>(f, 0); write_val<uint16_t>(f, 0);
    write_val<uint16_t>(f, header_size);
    write_val<uint32_t>(f, offset_to_data);
    write_val<uint32_t>(f, 1);
    write_val<uint8_t>(f, 3);
    write_val<uint16_t>(f, point_data_length);
    write_val<uint32_t>(f, n); write_val<uint32_t>(f, n);
    for (int i = 0; i < 4; i++) write_val<uint32_t>(f, 0);
    write_val<double>(f, scale_xyz); write_val<double>(f, scale_xyz); write_val<double>(f, scale_xyz);
    write_val<double>(f, 0.0); write_val<double>(f, 0.0); write_val<double>(f, 0.0);
    write_val<double>(f, max_x); write_val<double>(f, min_x);
    write_val<double>(f, max_y); write_val<double>(f, min_y);
    write_val<double>(f, max_z); write_val<double>(f, min_z);

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

    for (uint32_t i = 0; i < n; ++i) {
        write_val<int32_t>(f, (int32_t)std::round(points[i][0] / scale_xyz));
        write_val<int32_t>(f, (int32_t)std::round(points[i][1] / scale_xyz));
        write_val<int32_t>(f, (int32_t)std::round(points[i][2] / scale_xyz));
        write_val<uint16_t>(f, 0); write_val<uint8_t>(f, 0); write_val<uint8_t>(f, 0);
        write_val<uint8_t>(f, 0);  write_val<uint8_t>(f, 0); write_val<uint16_t>(f, 0);
        write_val<double>(f, 0.0);
        if (has_colors) {
            write_val<uint16_t>(f, (uint16_t)(std::clamp(colors[i][0], 0.0, 1.0) * 65535));
            write_val<uint16_t>(f, (uint16_t)(std::clamp(colors[i][1], 0.0, 1.0) * 65535));
            write_val<uint16_t>(f, (uint16_t)(std::clamp(colors[i][2], 0.0, 1.0) * 65535));
        } else {
            write_val<uint16_t>(f, 0); write_val<uint16_t>(f, 0); write_val<uint16_t>(f, 0);
        }
        if (has_normals) {
            write_val<float>(f, (float)normals[i][0]);
            write_val<float>(f, (float)normals[i][1]);
            write_val<float>(f, (float)normals[i][2]);
        }
        write_val<uint32_t>(f, i);
    }
    f.close();
}

// ============================================================
// Estensione
// ============================================================
std::string get_extension(const std::string& path) {
    size_t pos = path.rfind('.');
    if (pos == std::string::npos) return "";
    std::string ext = path.substr(pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

// ============================================================
// Subsample PLY → PLY (invariato)
// ============================================================
std::string subsample_ply(const std::string& file_path, const std::string& output_path, double voxel_size) {
    std::cout << "Loading PLY: " << file_path << std::endl;
    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(file_path, *pcd))
        throw std::runtime_error("Failed to read: " + file_path);
    std::cout << "Original Points N: " << pcd->points_.size() << std::endl;

    auto pcd_down = pcd->VoxelDownSample(voxel_size);

    int voxel_size_cm = (int)(voxel_size * 100);
    int voxel_size_mm = (int)(voxel_size * 1000);
    if (voxel_size_cm >= 1) {
        std::cout << "Points N after voxel_down_sample (" << voxel_size_cm << " cm): "
                  << pcd_down->points_.size() << std::endl;
    } else {
        std::cout << "Points N after voxel_down_sample (" << voxel_size_mm << " mm): "
                  << pcd_down->points_.size() << std::endl;
    }

    if (!open3d::io::WritePointCloud(output_path, *pcd_down))
        throw std::runtime_error("Failed to write: " + output_path);

    std::cout << "Subsampled point cloud saved to: " << std::endl;
    return output_path;
}

// ============================================================
// Subsample LAS → LAS
// ============================================================
std::string subsample_las(const std::string& file_path, const std::string& output_path, double voxel_size) {
    std::cout << "Loading LAS: " << file_path << std::endl;

    LasData las_data = read_las(file_path);
    std::cout << "Original Points N: " << las_data.points.size() << std::endl;

    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    pcd->points_ = las_data.points;
    if (las_data.has_colors) pcd->colors_ = las_data.colors;

    auto pcd_down = pcd->VoxelDownSample(voxel_size);

    int voxel_size_cm = (int)(voxel_size * 100);
    int voxel_size_mm = (int)(voxel_size * 1000);
    if (voxel_size_cm >= 1) {
        std::cout << "Points N after voxel_down_sample (" << voxel_size_cm << " cm): "
                  << pcd_down->points_.size() << std::endl;
    } else {
        std::cout << "Points N after voxel_down_sample (" << voxel_size_mm << " mm): "
                  << pcd_down->points_.size() << std::endl;
    }

    bool has_colors = pcd_down->HasColors();

    std::cout << "Computing normals..." << std::endl;
    pcd_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(0.02, 30));
    pcd_down->OrientNormalsTowardsCameraLocation();
    bool has_normals = pcd_down->HasNormals();

    for (auto& n : pcd_down->normals_) {
        if (std::isnan(n[0]) || std::isnan(n[1]) || std::isnan(n[2]))
            n = {0.0, 0.0, 0.0};
    }

    write_las(output_path, pcd_down->points_, pcd_down->colors_, pcd_down->normals_,
              has_colors, has_normals);

    std::cout << "Subsampled point cloud saved to: " << std::endl;
    return output_path;
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <file.ply|file.las> <output.ply|output.las> [voxel_size]" << std::endl;
            return 1;
        }
        std::string file_path = argv[1];
        std::string output_path = argv[2];
        double voxel_size = (argc >= 4) ? std::stod(argv[3]) : 0.002;

        std::string ext = get_extension(file_path);
        std::string output;

        if (ext == "ply") {
            output = subsample_ply(file_path, output_path, voxel_size);
        } else if (ext == "las") {
            output = subsample_las(file_path, output_path, voxel_size);
        } else {
            std::cerr << "ERROR: unsupported format '" << ext << "' (use .ply or .las)" << std::endl;
            return 1;
        }

        std::cout << output << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 2;
    }
}
#include <open3d/Open3D.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <filesystem>

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

// ============================================================
// Extra Bytes Record: float (type=9) per normali
// ============================================================
void write_extra_bytes_record(std::ofstream& f, const char* name, const char* description) {
    write_val<uint8_t>(f, 0); write_val<uint8_t>(f, 0);
    write_val<uint8_t>(f, 9); write_val<uint8_t>(f, 0);
    write_str(f, name, 32);
    for (int i = 0; i < 4;  i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 72; i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 48; i++) write_val<uint8_t>(f, 0);
    write_str(f, description, 32);
}

// ============================================================
// Extra Bytes Record: uint32 (type=5) per POINT_ID
// ============================================================
void write_extra_bytes_record_uint32(std::ofstream& f, const char* name, const char* description) {
    write_val<uint8_t>(f, 0); write_val<uint8_t>(f, 0);
    write_val<uint8_t>(f, 5); write_val<uint8_t>(f, 0);
    write_str(f, name, 32);
    for (int i = 0; i < 4;  i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 72; i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 48; i++) write_val<uint8_t>(f, 0);
    write_str(f, description, 32);
}

// ============================================================
// LAS structure with flags indicating which extra fields are present.
// ============================================================
struct LasData {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> colors;
    std::vector<Eigen::Vector3d> normals;
    bool has_colors   = false;
    bool has_normals  = false;
    bool has_point_id = false;
    // Header information preserved for writing back the file.
    double scale_x = 0.0001, scale_y = 0.0001, scale_z = 0.0001;
    double off_x = 0.0, off_y = 0.0, off_z = 0.0;
    uint8_t  point_format = 3;
    uint16_t point_length = 0;
};

// ============================================================
// Parse the VLR to discover which extra bytes are present.
// ============================================================
struct ExtraByteInfo {
    bool has_normal_x  = false;
    bool has_normal_y  = false;
    bool has_normal_z  = false;
    bool has_point_id  = false;
    int  normal_x_off  = -1;
    int  normal_y_off  = -1;
    int  normal_z_off  = -1;
    int  point_id_off  = -1;
};

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

int extra_data_type_size(uint8_t data_type) {
    switch (data_type) {
        case 1:  case 2:  return 1;
        case 3:  case 4:  return 2;
        case 5:  case 6:  return 4;
        case 7:  case 8:  return 8;
        case 9:           return 4;
        case 10:          return 8;
        case 11: case 12: return 2;
        case 13: case 14: return 4;
        case 15: case 16: return 8;
        case 17: case 18: return 16;
        case 19:          return 8;
        case 20:          return 16;
        case 21: case 22: return 3;
        case 23: case 24: return 6;
        case 25: case 26: return 12;
        case 27: case 28: return 24;
        case 29:          return 12;
        case 30:          return 24;
        default:          return 0;
    }
}

// Read the Extra Bytes VLR and locate the offsets for normals and POINT_ID.
ExtraByteInfo parse_extra_bytes_vlr(std::ifstream& f,
                                    uint32_t offset_to_data,
                                    uint16_t header_size,
                                    uint32_t num_vlrs,
                                    uint16_t point_length)
{
    ExtraByteInfo info;
    (void)offset_to_data;
    (void)point_length;

    f.seekg(header_size);
    for (uint32_t v = 0; v < num_vlrs; ++v) {
        uint16_t reserved   = read_val<uint16_t>(f);
        (void)reserved;
        char user_id[16];   f.read(user_id, 16);
        uint16_t record_id  = read_val<uint16_t>(f);
        uint16_t vlr_length = read_val<uint16_t>(f);
        char description[32]; f.read(description, 32);

        std::string uid(user_id, 16);
        if (record_id == 4 && uid.find("LASF_Spec") != std::string::npos) {
            int num_records = vlr_length / 192;
            int current_offset = 0;
            for (int r = 0; r < num_records; ++r) {
                uint8_t res0 = read_val<uint8_t>(f);
                uint8_t res1 = read_val<uint8_t>(f);
                (void)res0; (void)res1;
                uint8_t data_type = read_val<uint8_t>(f);
                uint8_t options   = read_val<uint8_t>(f);
                (void)options;
                char name[33] = {};
                f.read(name, 32);
                char rest[156]; f.read(rest, 156);

                std::string field_name(name);
                int field_size = extra_data_type_size(data_type);

                if (field_name == "NormalX") {
                    info.has_normal_x = true;
                    info.normal_x_off = current_offset;
                } else if (field_name == "NormalY") {
                    info.has_normal_y = true;
                    info.normal_y_off = current_offset;
                } else if (field_name == "NormalZ") {
                    info.has_normal_z = true;
                    info.normal_z_off = current_offset;
                } else if (field_name == "POINT_ID") {
                    if (data_type == 5 || data_type == 6) {
                        info.has_point_id = true;
                        info.point_id_off = current_offset;
                    }
                }
                current_offset += field_size;
            }
        } else {
            f.seekg(vlr_length, std::ios::cur);
        }
    }
    return info;
}

// ============================================================
// read_las — rileva POINT_ID e normali dai VLR
//            con rilevamento automatico scala colori (8 vs 16 bit)
// ============================================================
LasData read_las(const std::string& path, bool header_only = false) {
    LasData data;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    // --- Header ---
    char sig[4]; f.read(sig, 4);
    if (std::string(sig, 4) != "LASF")
        throw std::runtime_error("Not a valid LAS file: " + path);

    skip(f, 2); skip(f, 2);
    skip(f, 4); skip(f, 2); skip(f, 2); skip(f, 8);
    skip(f, 1); skip(f, 1);
    skip(f, 32); skip(f, 32);
    skip(f, 2); skip(f, 2);

    uint16_t header_size    = read_val<uint16_t>(f);
    uint32_t offset_to_data = read_val<uint32_t>(f);
    uint32_t num_vlrs       = read_val<uint32_t>(f);
    uint8_t  point_format   = read_val<uint8_t>(f);
    uint16_t point_length   = read_val<uint16_t>(f);
    uint32_t num_points     = read_val<uint32_t>(f);
    skip(f, 20);

    double scale_x = read_val<double>(f);
    double scale_y = read_val<double>(f);
    double scale_z = read_val<double>(f);
    double off_x   = read_val<double>(f);
    double off_y   = read_val<double>(f);
    double off_z   = read_val<double>(f);
    skip(f, 48);

    data.scale_x = scale_x; data.scale_y = scale_y; data.scale_z = scale_z;
    data.off_x   = off_x;   data.off_y   = off_y;   data.off_z   = off_z;
    const uint8_t fmt_base = base_point_format(point_format);
    const int base_point_size = las_base_point_size(fmt_base);
    if (base_point_size < 0) {
        throw std::runtime_error("Unsupported LAS point format: " + std::to_string((int)fmt_base));
    }

    data.point_format = fmt_base;
    data.point_length = point_length;

    // --- Analisi VLR per extra bytes ---
    ExtraByteInfo eb = parse_extra_bytes_vlr(f, offset_to_data, header_size, num_vlrs, point_length);
    data.has_normals  = eb.has_normal_x && eb.has_normal_y && eb.has_normal_z;
    data.has_point_id = eb.has_point_id;
    data.has_colors   = format_has_rgb(fmt_base);

    int extra_size = (int)point_length - base_point_size;
    if (extra_size < 0) {
        throw std::runtime_error("Invalid LAS point length for format " + std::to_string((int)fmt_base));
    }

    std::cout << "  LAS: " << num_points << " points"
              << "  format=" << (int)fmt_base
              << "  point_length=" << point_length
              << "  extra=" << extra_size
              << "  has_normals=" << data.has_normals
              << "  has_point_id=" << data.has_point_id
              << std::endl;

    if (header_only) { f.close(); return data; }

    // --- Lettura point records ---
    // The whole record block is read in big chunks and parsed from memory: reading field by field
    // with ifstream (seekg flushes its buffer) costs several syscalls per point, which takes
    // minutes on 5 M points when the file lives on a network/bind-mounted file system.
    data.points.resize(num_points);
    data.colors.resize(num_points);
    if (data.has_normals) data.normals.resize(num_points, {0, 0, 0});

    // Offsets inside the record (same layout the old field-by-field parser walked)
    int rgb_off = -1;
    if (format_has_rgb(fmt_base)) rgb_off = (fmt_base <= 5) ? ((fmt_base == 1 || fmt_base == 3 || fmt_base == 4 || fmt_base == 5) ? 28 : 20)
                                                            : 30;
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
            int32_t xyz[3];
            memcpy(xyz, p, 12);
            data.points[i] = { xyz[0] * scale_x + off_x, xyz[1] * scale_y + off_y, xyz[2] * scale_z + off_z };
            if (rgb_off >= 0) {
                uint16_t rgb[3];
                memcpy(rgb, p + rgb_off, 6);
                data.colors[i] = { rgb[0] / 65535.0, rgb[1] / 65535.0, rgb[2] / 65535.0 };
            } else {
                data.colors[i] = {0.0, 0.0, 0.0};
            }
            if (extra_size > 0 && data.has_normals) {
                const char* ex = p + base_point_size;
                float nx, ny, nz;
                memcpy(&nx, ex + eb.normal_x_off, 4);
                memcpy(&ny, ex + eb.normal_y_off, 4);
                memcpy(&nz, ex + eb.normal_z_off, 4);
                data.normals[i] = { (double)nx, (double)ny, (double)nz };
            }
            // POINT_ID is ignored on read and recomputed on write.
        }
    }

    f.close();
    return data;
}

// ============================================================
// Write a LAS file, always including normals and POINT_ID when available.
// Colors are always saved as 16-bit values in the 0-65535 range.
// ============================================================
void write_las(const std::string& out_file,
               const LasData& data)
{
    std::ofstream f(out_file, std::ios::binary);
    if (!f) { std::cerr << "Cannot open output: " << out_file << std::endl; return; }

    const auto& points  = data.points;
    const auto& colors  = data.colors;
    const auto& normals = data.normals;
    bool has_normals    = data.has_normals;
    uint32_t n = (uint32_t)points.size();

    double min_x = 1e18, min_y = 1e18, min_z = 1e18;
    double max_x = -1e18, max_y = -1e18, max_z = -1e18;
    for (auto& p : points) {
        min_x = std::min(min_x, p[0]); max_x = std::max(max_x, p[0]);
        min_y = std::min(min_y, p[1]); max_y = std::max(max_y, p[1]);
        min_z = std::min(min_z, p[2]); max_z = std::max(max_z, p[2]);
    }

    const double scale_x = (data.scale_x != 0.0) ? data.scale_x : 0.0001;
    const double scale_y = (data.scale_y != 0.0) ? data.scale_y : 0.0001;
    const double scale_z = (data.scale_z != 0.0) ? data.scale_z : 0.0001;
    const double off_x   = data.off_x;
    const double off_y   = data.off_y;
    const double off_z   = data.off_z;
    double qmin_x = 1e18, qmin_y = 1e18, qmin_z = 1e18;
    double qmax_x = -1e18, qmax_y = -1e18, qmax_z = -1e18;
    for (const auto& point : points) {
        int32_t ix = (int32_t)std::round((point[0] - off_x) / scale_x);
        int32_t iy = (int32_t)std::round((point[1] - off_y) / scale_y);
        int32_t iz = (int32_t)std::round((point[2] - off_z) / scale_z);
        double qx = ix * scale_x + off_x;
        double qy = iy * scale_y + off_y;
        double qz = iz * scale_z + off_z;
        qmin_x = std::min(qmin_x, qx); qmax_x = std::max(qmax_x, qx);
        qmin_y = std::min(qmin_y, qy); qmax_y = std::max(qmax_y, qy);
        qmin_z = std::min(qmin_z, qz); qmax_z = std::max(qmax_z, qz);
    }

    // Extra Bytes VLR: normals (if present) + POINT_ID.
    uint32_t extra_bytes_payload = 0;
    if (has_normals) extra_bytes_payload += 3 * 192; // NormalX/Y/Z
    extra_bytes_payload += 192;                       // POINT_ID

    uint16_t header_size    = 227;
    uint32_t vlr_total      = 54 + extra_bytes_payload;
    uint32_t offset_to_data = header_size + vlr_total;

    uint16_t point_data_length = 34;
    if (has_normals) point_data_length += 12; // 3 x float
    point_data_length += 4;                   // uint32 POINT_ID

    // ---- Header LAS ----
    write_str(f, "LASF", 4);
    write_val<uint16_t>(f, 0); write_val<uint16_t>(f, 0);
    write_val<uint32_t>(f, 0); write_val<uint16_t>(f, 0); write_val<uint16_t>(f, 0);
    for (int i = 0; i < 8; i++) write_val<uint8_t>(f, 0);
    write_val<uint8_t>(f, 1); write_val<uint8_t>(f, 2);
    write_str(f, "OTHER", 32);
    write_str(f, "check_point_id", 32);
    write_val<uint16_t>(f, 0); write_val<uint16_t>(f, 0);
    write_val<uint16_t>(f, header_size);
    write_val<uint32_t>(f, offset_to_data);
    write_val<uint32_t>(f, 1);     // num_vlrs
    write_val<uint8_t>(f, 3);      // point format
    write_val<uint16_t>(f, point_data_length);
    write_val<uint32_t>(f, n);     // Number of point records
    write_val<uint32_t>(f, n);     // Points by return[0]
    for (int i = 0; i < 4; i++) write_val<uint32_t>(f, 0); // Points by return[1-4]
    write_val<double>(f, scale_x); write_val<double>(f, scale_y); write_val<double>(f, scale_z);
    write_val<double>(f, off_x); write_val<double>(f, off_y); write_val<double>(f, off_z);
    write_val<double>(f, qmax_x); write_val<double>(f, qmin_x);
    write_val<double>(f, qmax_y); write_val<double>(f, qmin_y);
    write_val<double>(f, qmax_z); write_val<double>(f, qmin_z);

    // ---- VLR: normali (se presenti) + POINT_ID ----
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

    // ---- Point records ----
    for (uint32_t i = 0; i < n; ++i) {
        // 1. COORDINATE (12 byte totali)
        write_val<int32_t>(f, (int32_t)std::round((points[i][0] - off_x) / scale_x)); // X
        write_val<int32_t>(f, (int32_t)std::round((points[i][1] - off_y) / scale_y)); // Y
        write_val<int32_t>(f, (int32_t)std::round((points[i][2] - off_z) / scale_z)); // Z

        // 2. ATTRIBUTI STANDARD (8 byte totali)
        write_val<uint16_t>(f, 0); // Intensity (2 byte)
        write_val<uint8_t>(f, 0);  // Return Number & Number of Returns (1 byte)
        write_val<uint8_t>(f, 0);  // Classification (1 byte)
        write_val<uint8_t>(f, 0);  // Scan Angle Rank (1 byte)
        write_val<uint8_t>(f, 0);  // User Data (1 byte)
        write_val<uint16_t>(f, 0); // Point Source ID (2 byte)

        // 3. GPS TIME (8 bytes total) - required in format 3.
        // Skipping this would shift the RGB fields by 8 bytes.
        write_val<double>(f, 0.0); 

        // 4. RGB COLORS (6 bytes total).
        // They must be stored as uint16 values in the 0-65535 range.
        uint16_t r = (uint16_t)(std::clamp(colors[i][0], 0.0, 1.0) * 65535);
        uint16_t g = (uint16_t)(std::clamp(colors[i][1], 0.0, 1.0) * 65535);
        uint16_t b = (uint16_t)(std::clamp(colors[i][2], 0.0, 1.0) * 65535);
        write_val<uint16_t>(f, r);
        write_val<uint16_t>(f, g);
        write_val<uint16_t>(f, b);

        // Total so far: 34 bytes (standard Record Format 3).

        // 5. EXTRA BYTES (normals + ID).
        if (has_normals) {
            write_val<float>(f, (float)normals[i][0]); // 4 byte
            write_val<float>(f, (float)normals[i][1]); // 4 byte
            write_val<float>(f, (float)normals[i][2]); // 4 byte
        }
        write_val<uint32_t>(f, i); // POINT_ID (4 byte)
    }
    f.close();
}

// ============================================================
// check_and_fix: return a valid LAS path.
//   - if everything is already present, return input_path (no I/O)
//   - otherwise write output_path and return output_path
// ============================================================
std::string check_and_fix(const std::string& input_path,
                           const std::string& output_path)
{
    std::cout << "Reading: " << input_path << std::endl;
    // Header only first: when nothing has to change the points are never loaded
    LasData data = read_las(input_path, true);

    bool needs_normals   = !data.has_normals;
    bool needs_point_id  = !data.has_point_id;
    bool needs_canonical = !(data.point_format == 3 && data.point_length == 50);

    if (!needs_normals && !needs_point_id && !needs_canonical) {
        std::cout << "Normals and POINT_ID already present with canonical format. No modification needed." << std::endl;

        if (input_path != output_path) {
            std::filesystem::path in_path(input_path);
            std::filesystem::path out_path(output_path);

            if (std::filesystem::exists(out_path)) {
                std::filesystem::remove(out_path);
            }
            std::filesystem::rename(in_path, out_path);
            std::cout << "Save into output path: " << output_path << std::endl;
            return output_path;
        }

        return input_path;
    }

    data = read_las(input_path);

    // Add normals if they are missing.
    if (needs_normals) {
        std::cout << "Computing Normals..." << std::endl;
        auto pcd = std::make_shared<open3d::geometry::PointCloud>();
        pcd->points_ = data.points;
        pcd->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(0.02, 30));
        pcd->OrientNormalsTowardsCameraLocation();
        data.normals.resize(data.points.size(), {0, 0, 0});
        for (size_t i = 0; i < pcd->normals_.size(); ++i) {
            auto& n = pcd->normals_[i];
            data.normals[i] = (std::isnan(n[0]) || std::isnan(n[1]) || std::isnan(n[2]))
                              ? Eigen::Vector3d(0, 0, 0) : n;
        }
        data.has_normals = true;
    }

    if (needs_point_id) {
        std::cout << "Adding POINT_ID..." << std::endl;
        data.has_point_id = true;
    }

    if (needs_canonical) {
        std::cout << "Rewriting LAS to canonical format (PointFormat=3 + NormalX/Y/Z + POINT_ID)..." << std::endl;
    }

    std::cout << "Saving..." << std::endl;
    write_las(output_path, data);
    return output_path;
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <input.las> <output.las>" << std::endl;
            return 1;
        }
        std::string input_path  = argv[1];
        std::string output_path = argv[2];
        std::string result = check_and_fix(input_path, output_path);
        std::cout << "Output: " << result << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 2;
    }
}
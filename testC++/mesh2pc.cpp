// mesh2pc.cpp
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <random>
#include <unordered_map>
#include <numeric>
// CGAL
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
// Open3D
#include <open3d/Open3D.h>
// tinygltf (header-only)
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"
// OpenMP
#include <omp.h>

typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Point_3;
typedef K::Triangle_3 Triangle_3;
typedef std::vector<Triangle_3>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

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
    write_val<uint8_t>(f, 9);   // float32
    write_val<uint8_t>(f, 0);
    write_str(f, name, 32);
    for (int i = 0; i < 4;  i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 72; i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 48; i++) write_val<uint8_t>(f, 0);
    write_str(f, description, 32);
}

// Per POINT_ID: tipo 5 = uint32 (LAS Extra Bytes standard)
void write_extra_bytes_record_uint32(std::ofstream& f, const char* name, const char* description) {
    write_val<uint8_t>(f, 0);
    write_val<uint8_t>(f, 0);
    write_val<uint8_t>(f, 5);   // uint32
    write_val<uint8_t>(f, 0);
    write_str(f, name, 32);
    for (int i = 0; i < 4;  i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 72; i++) write_val<uint8_t>(f, 0);
    for (int i = 0; i < 48; i++) write_val<uint8_t>(f, 0);
    write_str(f, description, 32);
}

void write_las(const std::string& out_file,
               const std::vector<std::array<double,3>>& points,
               const std::vector<std::array<double,3>>& colors,
               const std::vector<std::array<double,3>>& normals)
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

    // Margine di sicurezza: assorbe il rumore in virgola mobile che PotreeConverter
    // introduce ricalcolando le coordinate dagli int32, evitando "point outside bounding box".
    qmin_x -= scale_xyz; qmax_x += scale_xyz;
    qmin_y -= scale_xyz; qmax_y += scale_xyz;
    qmin_z -= scale_xyz; qmax_z += scale_xyz;

    // Extra bytes: 3 normali + POINT_ID
    uint32_t extra_bytes_payload = 3 * 192 + 192;
    uint16_t header_size    = 227;
    uint32_t vlr_total      = 54 + extra_bytes_payload;
    uint32_t offset_to_data = header_size + vlr_total;
    uint16_t point_data_length = 46 + 4; // 46 base con normals + 4 POINT_ID

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
    write_str(f, "mesh2pc_cpp", 32);
    write_val<uint16_t>(f, 0);
    write_val<uint16_t>(f, 0);
    write_val<uint16_t>(f, header_size);
    write_val<uint32_t>(f, offset_to_data);
    write_val<uint32_t>(f, 1);
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

    // --- VLR unico: normals + POINT_ID ---
    write_val<uint16_t>(f, 0);
    write_str(f, "LASF_Spec", 16);
    write_val<uint16_t>(f, 4);
    write_val<uint16_t>(f, (uint16_t)extra_bytes_payload);
    write_str(f, "Extra Bytes Record", 32);
    write_extra_bytes_record(f, "NormalX", "Normal X");
    write_extra_bytes_record(f, "NormalY", "Normal Y");
    write_extra_bytes_record(f, "NormalZ", "Normal Z");
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
        write_val<uint8_t>(f, 0);   // user data (mandatory LAS field)
        write_val<uint16_t>(f, 0);  // point source ID (mandatory LAS field)
        write_val<double>(f, 0.0);  // GPS time
        write_val<uint16_t>(f, (uint16_t)(std::clamp(colors[i][0], 0.0, 1.0) * 65535));
        write_val<uint16_t>(f, (uint16_t)(std::clamp(colors[i][1], 0.0, 1.0) * 65535));
        write_val<uint16_t>(f, (uint16_t)(std::clamp(colors[i][2], 0.0, 1.0) * 65535));
        write_val<float>(f, (float)normals[i][0]);
        write_val<float>(f, (float)normals[i][1]);
        write_val<float>(f, (float)normals[i][2]);
        write_val<uint32_t>(f, i);  // POINT_ID
    }

    f.close();
}

struct SubMeshResult {
    std::vector<std::array<double,3>> points;
    std::vector<std::array<double,3>> colors;
};

struct GLBMesh {
    std::vector<std::array<float,3>> vertices;
    std::vector<std::array<int,3>>   faces;
    std::vector<std::array<float,2>> uvs;
    bool has_uvs = false;
    bool is_point_cloud = false;  // primitive senza triangoli/edge: nuvola di punti gia' pronta
    std::vector<std::array<float,4>> vertex_colors;
    int texture_index = -1;
    std::array<float,3> base_color = {1.0f, 1.0f, 1.0f};
    int wrap_s = TINYGLTF_TEXTURE_WRAP_REPEAT;
    int wrap_t = TINYGLTF_TEXTURE_WRAP_REPEAT;
};

Eigen::Matrix4d get_node_local_transform(const tinygltf::Node& node) {
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    if (node.matrix.size() == 16) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                transform(row, col) = node.matrix[col * 4 + row];
            }
        }
        return transform;
    }

    Eigen::Vector3d translation(0.0, 0.0, 0.0);
    Eigen::Vector3d scale(1.0, 1.0, 1.0);
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();

    if (node.translation.size() == 3) {
        translation = Eigen::Vector3d(node.translation[0], node.translation[1], node.translation[2]);
    }
    if (node.scale.size() == 3) {
        scale = Eigen::Vector3d(node.scale[0], node.scale[1], node.scale[2]);
    }
    if (node.rotation.size() == 4) {
        rotation = Eigen::Quaterniond(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
    }

    Eigen::Matrix4d translation_matrix = Eigen::Matrix4d::Identity();
    translation_matrix.block<3,1>(0, 3) = translation;

    Eigen::Matrix4d rotation_matrix = Eigen::Matrix4d::Identity();
    rotation_matrix.block<3,3>(0, 0) = rotation.normalized().toRotationMatrix();

    Eigen::Matrix4d scale_matrix = Eigen::Matrix4d::Identity();
    scale_matrix(0, 0) = scale.x();
    scale_matrix(1, 1) = scale.y();
    scale_matrix(2, 2) = scale.z();

    return translation_matrix * rotation_matrix * scale_matrix;
}

std::array<float,3> transform_position(const Eigen::Matrix4d& transform, const std::array<float,3>& position) {
    Eigen::Vector4d p(position[0], position[1], position[2], 1.0);
    Eigen::Vector4d out = transform * p;
    return {
        static_cast<float>(out.x()),
        static_cast<float>(out.y()),
        static_cast<float>(out.z())
    };
}

double apply_wrap_mode(double uv, int wrap_mode) {
    switch (wrap_mode) {
        case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
            return std::clamp(uv, 0.0, 1.0);
        case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT: {
            double t = std::fmod(uv, 2.0);
            if (t < 0.0) t += 2.0;
            return (t <= 1.0) ? t : (2.0 - t);
        }
        case TINYGLTF_TEXTURE_WRAP_REPEAT:
        default: {
            double t = std::fmod(uv, 1.0);
            if (t < 0.0) t += 1.0;
            return t;
        }
    }
}

std::array<double,3> sample_texture_color(const std::vector<uint8_t>& tex,
                                         int width,
                                         int height,
                                         int channels,
                                         double u,
                                         double v)
{
    if (width <= 0 || height <= 0 || channels < 3) {
        return {1.0, 1.0, 1.0};
    }

    double x = u * (width - 1);
    double y = v * (height - 1);
    int x0 = std::clamp((int)std::floor(x), 0, width - 1);
    int y0 = std::clamp((int)std::floor(y), 0, height - 1);
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);
    double tx = x - x0;
    double ty = y - y0;

    auto read_texel = [&](int px, int py) {
        int offset = (py * width + px) * channels;
        double alpha = (channels >= 4) ? tex[offset + 3] / 255.0 : 1.0;
        return std::array<double,4>{
            tex[offset] / 255.0,
            tex[offset + 1] / 255.0,
            tex[offset + 2] / 255.0,
            alpha
        };
    };

    auto c00 = read_texel(x0, y0);
    auto c10 = read_texel(x1, y0);
    auto c01 = read_texel(x0, y1);
    auto c11 = read_texel(x1, y1);

    double w00 = (1.0 - tx) * (1.0 - ty);
    double w10 = tx * (1.0 - ty);
    double w01 = (1.0 - tx) * ty;
    double w11 = tx * ty;

    double aw00 = w00 * c00[3];
    double aw10 = w10 * c10[3];
    double aw01 = w01 * c01[3];
    double aw11 = w11 * c11[3];
    double alpha_weight_sum = aw00 + aw10 + aw01 + aw11;

    if (alpha_weight_sum > 1e-8) {
        return {
            (c00[0] * aw00 + c10[0] * aw10 + c01[0] * aw01 + c11[0] * aw11) / alpha_weight_sum,
            (c00[1] * aw00 + c10[1] * aw10 + c01[1] * aw01 + c11[1] * aw11) / alpha_weight_sum,
            (c00[2] * aw00 + c10[2] * aw10 + c01[2] * aw01 + c11[2] * aw11) / alpha_weight_sum
        };
    }

    double weight_sum = w00 + w10 + w01 + w11;
    if (weight_sum <= 1e-8) {
        return {c00[0], c00[1], c00[2]};
    }

    return {
        (c00[0] * w00 + c10[0] * w10 + c01[0] * w01 + c11[0] * w11) / weight_sum,
        (c00[1] * w00 + c10[1] * w10 + c01[1] * w01 + c11[1] * w11) / weight_sum,
        (c00[2] * w00 + c10[2] * w10 + c01[2] * w01 + c11[2] * w11) / weight_sum
    };
}

int component_size_bytes(int component_type) {
    switch (component_type) {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  return 1;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: return 2;
        case TINYGLTF_COMPONENT_TYPE_INT:
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        case TINYGLTF_COMPONENT_TYPE_FLOAT:          return 4;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:         return 8;
        default:                                     return 0;
    }
}

double read_component_as_double(const uint8_t* ptr, int component_type, bool normalized) {
    switch (component_type) {
        case TINYGLTF_COMPONENT_TYPE_FLOAT: {
            float v;
            std::memcpy(&v, ptr, sizeof(float));
            return static_cast<double>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
            uint8_t v;
            std::memcpy(&v, ptr, sizeof(uint8_t));
            return normalized ? static_cast<double>(v) / 255.0 : static_cast<double>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
            uint16_t v;
            std::memcpy(&v, ptr, sizeof(uint16_t));
            return normalized ? static_cast<double>(v) / 65535.0 : static_cast<double>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_BYTE: {
            int8_t v;
            std::memcpy(&v, ptr, sizeof(int8_t));
            if (!normalized) return static_cast<double>(v);
            return std::max(-1.0, static_cast<double>(v) / 127.0);
        }
        case TINYGLTF_COMPONENT_TYPE_SHORT: {
            int16_t v;
            std::memcpy(&v, ptr, sizeof(int16_t));
            if (!normalized) return static_cast<double>(v);
            return std::max(-1.0, static_cast<double>(v) / 32767.0);
        }
        default:
            return 0.0;
    }
}

int read_index_as_int(const uint8_t* ptr, int component_type) {
    switch (component_type) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
            uint32_t v;
            std::memcpy(&v, ptr, sizeof(uint32_t));
            return static_cast<int>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
            uint16_t v;
            std::memcpy(&v, ptr, sizeof(uint16_t));
            return static_cast<int>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
            uint8_t v;
            std::memcpy(&v, ptr, sizeof(uint8_t));
            return static_cast<int>(v);
        }
        default:
            return 0;
    }
}

bool load_glb(const std::string& path,
              std::vector<GLBMesh>& meshes,
              std::vector<std::vector<uint8_t>>& textures,
              std::vector<std::array<int,2>>& tex_sizes)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    if (!loader.LoadBinaryFromFile(&model, &err, &warn, path)) {
        std::cerr << "Failed to load GLB: " << err << std::endl;
        return false;
    }
    for (auto& img : model.images) {
        textures.push_back(img.image);
        tex_sizes.push_back({img.width, img.height});
    }

    std::function<void(int, const Eigen::Matrix4d&)> visit_node;
    visit_node = [&](int node_index, const Eigen::Matrix4d& parent_transform) {
        const auto& node = model.nodes[node_index];
        Eigen::Matrix4d world_transform = parent_transform * get_node_local_transform(node);

        if (node.mesh >= 0 && node.mesh < (int)model.meshes.size()) {
            auto& mesh = model.meshes[node.mesh];
            for (auto& prim : mesh.primitives) {
            GLBMesh m;
            auto& pa = model.accessors[prim.attributes.at("POSITION")];
            auto& pbv = model.bufferViews[pa.bufferView];
            int pos_stride = (pbv.byteStride > 0) ? pbv.byteStride : 3 * sizeof(float);
            const uint8_t* pbase = model.buffers[pbv.buffer].data.data() + pbv.byteOffset + pa.byteOffset;
            m.vertices.resize(pa.count);
            for (size_t i = 0; i < pa.count; ++i) {
                const float* pd = reinterpret_cast<const float*>(pbase + i * pos_stride);
                m.vertices[i] = transform_position(world_transform, {pd[0], pd[1], pd[2]});
            }

            int prim_mode = (prim.mode < 0) ? TINYGLTF_MODE_TRIANGLES : prim.mode;
            m.is_point_cloud = (prim_mode == TINYGLTF_MODE_POINTS);

            if (m.is_point_cloud) {
                // Nessun triangolo/edge: i vertici sono gia' una nuvola di punti, niente sampling.
            } else if (prim.indices >= 0) {
                auto& ia = model.accessors[prim.indices];
                auto& ibv = model.bufferViews[ia.bufferView];
                const uint8_t* ibase = model.buffers[ibv.buffer].data.data() + ibv.byteOffset + ia.byteOffset;
                int idx_comp_size = component_size_bytes(ia.componentType);
                int idx_stride = (ibv.byteStride > 0) ? ibv.byteStride : idx_comp_size;
                m.faces.resize(ia.count / 3);
                for (size_t i = 0; i < ia.count / 3; ++i) {
                    const uint8_t* i0 = ibase + (i * 3 + 0) * idx_stride;
                    const uint8_t* i1 = ibase + (i * 3 + 1) * idx_stride;
                    const uint8_t* i2 = ibase + (i * 3 + 2) * idx_stride;
                    m.faces[i] = {
                        read_index_as_int(i0, ia.componentType),
                        read_index_as_int(i1, ia.componentType),
                        read_index_as_int(i2, ia.componentType)
                    };
                }
            } else {
                // Non-indexed primitive: assume triangles in-order (mode TRIANGLES).
                size_t tri_count = m.vertices.size() / 3;
                m.faces.resize(tri_count);
                for (size_t i = 0; i < tri_count; ++i) {
                    m.faces[i] = {static_cast<int>(i * 3), static_cast<int>(i * 3 + 1), static_cast<int>(i * 3 + 2)};
                }
            }
            m.texture_index = -1;  // nessuna texture di default
            if (prim.material >= 0) {
                auto& mat = model.materials[prim.material];
                if (mat.pbrMetallicRoughness.baseColorFactor.size() >= 3) {
                    m.base_color = {
                        static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[0]),
                        static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[1]),
                        static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[2])
                    };
                }
                int tex_idx = -1;
                int uv_set = 0;

                // Preferred source for color in PBR workflow.
                if (mat.pbrMetallicRoughness.baseColorTexture.index >= 0) {
                    tex_idx = mat.pbrMetallicRoughness.baseColorTexture.index;
                    uv_set = mat.pbrMetallicRoughness.baseColorTexture.texCoord;
                } else if (mat.emissiveTexture.index >= 0) {
                    // Fallback often used by some DCC exports.
                    tex_idx = mat.emissiveTexture.index;
                    uv_set = mat.emissiveTexture.texCoord;
                } else if (model.textures.size() == 1) {
                    // Last-resort fallback for files with a single texture but no explicit binding.
                    tex_idx = 0;
                    uv_set = 0;
                }

                if (tex_idx >= 0 && tex_idx < (int)model.textures.size()) {
                    const auto& tex = model.textures[tex_idx];
                    int img_idx = tex.source;
                    if (img_idx >= 0 && img_idx < (int)textures.size()) {
                        m.texture_index = img_idx;
                    }

                    if (tex.sampler >= 0 && tex.sampler < (int)model.samplers.size()) {
                        const auto& s = model.samplers[tex.sampler];
                        if (s.wrapS != 0) m.wrap_s = s.wrapS;
                        if (s.wrapT != 0) m.wrap_t = s.wrapT;
                    }

                    std::string uv_attr = "TEXCOORD_" + std::to_string(uv_set);
                    int uv_accessor_index = -1;
                    if (prim.attributes.count(uv_attr)) {
                        uv_accessor_index = prim.attributes.at(uv_attr);
                    } else if (prim.attributes.count("TEXCOORD_0")) {
                        uv_accessor_index = prim.attributes.at("TEXCOORD_0");
                    } else if (prim.attributes.count("TEXCOORD_1")) {
                        uv_accessor_index = prim.attributes.at("TEXCOORD_1");
                    }

                    m.uvs.resize(pa.count, {0.0f, 0.0f});
                    m.has_uvs = false;
                    if (uv_accessor_index >= 0) {
                        auto& ua = model.accessors[uv_accessor_index];
                        auto& ubv = model.bufferViews[ua.bufferView];
                        int uv_comp_size = component_size_bytes(ua.componentType);
                        int uv_stride = (ubv.byteStride > 0) ? ubv.byteStride : 2 * uv_comp_size;
                        const uint8_t* ubase = model.buffers[ubv.buffer].data.data() + ubv.byteOffset + ua.byteOffset;
                        size_t uv_count = std::min((size_t)ua.count, m.uvs.size());
                        for (size_t i = 0; i < uv_count; ++i) {
                            const uint8_t* uv_ptr = ubase + i * uv_stride;
                            double u = read_component_as_double(uv_ptr, ua.componentType, ua.normalized);
                            double v = read_component_as_double(uv_ptr + uv_comp_size, ua.componentType, ua.normalized);
                            m.uvs[i] = {
                                static_cast<float>(u),
                                static_cast<float>(v)
                            };
                        }
                        m.has_uvs = (uv_count > 0);
                    }
                }
            }

            m.vertex_colors.resize(pa.count, {1.0f, 1.0f, 1.0f, 1.0f});
            if (prim.attributes.count("COLOR_0")) {
                auto& ca = model.accessors[prim.attributes.at("COLOR_0")];
                auto& cbv = model.bufferViews[ca.bufferView];
                int color_comp_size = component_size_bytes(ca.componentType);
                int color_components = (ca.type == TINYGLTF_TYPE_VEC4) ? 4 : 3;
                int color_stride = (cbv.byteStride > 0) ? cbv.byteStride : color_components * color_comp_size;
                const uint8_t* cbase = model.buffers[cbv.buffer].data.data() + cbv.byteOffset + ca.byteOffset;
                size_t color_count = std::min((size_t)ca.count, m.vertex_colors.size());
                for (size_t i = 0; i < color_count; ++i) {
                    const uint8_t* color_ptr = cbase + i * color_stride;
                    double r = read_component_as_double(color_ptr, ca.componentType, ca.normalized);
                    double g = read_component_as_double(color_ptr + color_comp_size, ca.componentType, ca.normalized);
                    double b = read_component_as_double(color_ptr + 2 * color_comp_size, ca.componentType, ca.normalized);
                    double a = 1.0;
                    if (color_components == 4) {
                        a = read_component_as_double(color_ptr + 3 * color_comp_size, ca.componentType, ca.normalized);
                    }
                    m.vertex_colors[i] = {
                        static_cast<float>(std::clamp(r, 0.0, 1.0)),
                        static_cast<float>(std::clamp(g, 0.0, 1.0)),
                        static_cast<float>(std::clamp(b, 0.0, 1.0)),
                        static_cast<float>(std::clamp(a, 0.0, 1.0))
                    };
                }
            }
            if (m.uvs.empty()) {
                m.uvs.resize(pa.count, {0.0f, 0.0f});
            }

            meshes.push_back(std::move(m));
        }
        }

        for (int child_index : node.children) {
            visit_node(child_index, world_transform);
        }
    };

    std::vector<int> root_nodes;
    if (model.defaultScene >= 0 && model.defaultScene < (int)model.scenes.size()) {
        root_nodes = model.scenes[model.defaultScene].nodes;
    } else if (!model.scenes.empty()) {
        root_nodes = model.scenes[0].nodes;
    } else {
        root_nodes.resize(model.nodes.size());
        for (int i = 0; i < (int)model.nodes.size(); ++i) root_nodes[i] = i;
    }

    for (int node_index : root_nodes) {
        visit_node(node_index, Eigen::Matrix4d::Identity());
    }

    return true;
}

// Sceglie k punti senza reinserimento tramite Fisher-Yates parziale; se k >= n restituisce tutto.
void subsample_points(const std::vector<std::array<double,3>>& pts,
                       const std::vector<std::array<double,3>>& cols,
                       size_t k,
                       std::vector<std::array<double,3>>& out_pts,
                       std::vector<std::array<double,3>>& out_cols)
{
    size_t n = pts.size();
    if (k >= n) {
        out_pts = pts;
        out_cols = cols;
        return;
    }
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937_64 rng(1234567ULL);
    for (size_t i = 0; i < k; ++i) {
        std::uniform_int_distribution<size_t> dist(i, n - 1);
        size_t j = dist(rng);
        std::swap(idx[i], idx[j]);
    }
    out_pts.resize(k);
    out_cols.resize(k);
    for (size_t i = 0; i < k; ++i) {
        out_pts[i] = pts[idx[i]];
        out_cols[i] = cols[idx[i]];
    }
}

SubMeshResult process_submesh(
    int idx,
    const GLBMesh& mesh,
    int points_per_mesh,
    const std::vector<std::vector<uint8_t>>& textures,
    const std::vector<std::array<int,2>>& tex_sizes)
{
    SubMeshResult result;
    auto t = now_t();

    std::vector<double> cumulative_areas;
    cumulative_areas.reserve(mesh.faces.size());
    double total_area = 0.0;
    for (const auto& f : mesh.faces) {
        const auto& v0 = mesh.vertices[f[0]];
        const auto& v1 = mesh.vertices[f[1]];
        const auto& v2 = mesh.vertices[f[2]];
        Eigen::Vector3d e1(v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]);
        Eigen::Vector3d e2(v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]);
        double area = 0.5 * e1.cross(e2).norm();
        total_area += std::max(area, 0.0);
        cumulative_areas.push_back(total_area);
    }

    if (mesh.faces.empty() || total_area <= 0.0 || points_per_mesh <= 0) {
        return result;
    }

    int n = points_per_mesh;
    result.points.resize(n);
    result.colors.resize(n);

    bool has_texture = mesh.texture_index >= 0 && mesh.texture_index < (int)textures.size() && mesh.has_uvs;
    bool has_vertex_colors = !mesh.vertex_colors.empty();
    const std::vector<uint8_t>* tex_ptr = nullptr;
    int tw = 0, th = 0, channels = 0;
    if (has_texture) {
        tex_ptr = &textures[mesh.texture_index];
        tw = tex_sizes[mesh.texture_index][0];
        th = tex_sizes[mesh.texture_index][1];
        if (tw > 0 && th > 0) {
            channels = (int)tex_ptr->size() / (tw * th);
        }
        if (channels < 3) has_texture = false;
    }
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int i = 0; i < n; ++i) {
        std::mt19937_64 rng(static_cast<uint64_t>(idx + 1) * 1000003ULL + static_cast<uint64_t>(i));
        std::uniform_real_distribution<double> dist01(0.0, 1.0);
        double area_pick = dist01(rng) * total_area;
        int tri_id = (int)(std::lower_bound(cumulative_areas.begin(), cumulative_areas.end(), area_pick) - cumulative_areas.begin());
        tri_id = std::clamp(tri_id, 0, (int)mesh.faces.size() - 1);

        auto& f = mesh.faces[tri_id];
        auto& v0 = mesh.vertices[f[0]];
        auto& v1 = mesh.vertices[f[1]];
        auto& v2 = mesh.vertices[f[2]];

        double r1 = dist01(rng);
        double r2 = dist01(rng);
        double sqrt_r1 = std::sqrt(r1);
        double bu = 1.0 - sqrt_r1;
        double bv = sqrt_r1 * (1.0 - r2);
        double bw = sqrt_r1 * r2;

        result.points[i] = {
            bu * v0[0] + bv * v1[0] + bw * v2[0],
            bu * v0[1] + bv * v1[1] + bw * v2[1],
            bu * v0[2] + bv * v1[2] + bw * v2[2]
        };

        if (has_texture) {
            auto& uv0 = mesh.uvs[f[0]];
            auto& uv1 = mesh.uvs[f[1]];
            auto& uv2 = mesh.uvs[f[2]];
            double u = bu*uv0[0] + bv*uv1[0] + bw*uv2[0];
            double v = bu*uv0[1] + bv*uv1[1] + bw*uv2[1];
            u = apply_wrap_mode(u, mesh.wrap_s);
            v = apply_wrap_mode(v, mesh.wrap_t);
            const auto& tex = *tex_ptr;
            auto sampled = sample_texture_color(tex, tw, th, channels, u, v);
            result.colors[i] = {
                sampled[0] * std::clamp((double)mesh.base_color[0], 0.0, 1.0),
                sampled[1] * std::clamp((double)mesh.base_color[1], 0.0, 1.0),
                sampled[2] * std::clamp((double)mesh.base_color[2], 0.0, 1.0)
            };
        } else if (has_vertex_colors) {
            auto& c0 = mesh.vertex_colors[f[0]];
            auto& c1 = mesh.vertex_colors[f[1]];
            auto& c2 = mesh.vertex_colors[f[2]];
            result.colors[i] = {
                std::clamp(bu*c0[0] + bv*c1[0] + bw*c2[0], 0.0, 1.0),
                std::clamp(bu*c0[1] + bv*c1[1] + bw*c2[1], 0.0, 1.0),
                std::clamp(bu*c0[2] + bv*c1[2] + bw*c2[2], 0.0, 1.0)
            };
        } else {
            result.colors[i] = {
                (double)std::clamp(mesh.base_color[0], 0.0f, 1.0f),
                (double)std::clamp(mesh.base_color[1], 0.0f, 1.0f),
                (double)std::clamp(mesh.base_color[2], 0.0f, 1.0f)
            };
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.glb> <output.las> [num_points]" << std::endl;
        return 1;
    }

    std::string mesh_path = argv[1];
    std::string out_path = argv[2];
    int num_points = (argc >= 4) ? std::stoi(argv[3]) : 5000000;

    auto global_start = now_t();

    std::vector<GLBMesh> meshes;
    std::vector<std::vector<uint8_t>> textures;
    std::vector<std::array<int,2>> tex_sizes;

    if (!load_glb(mesh_path, meshes, textures, tex_sizes)) return 1;
    std::cout << "Load GLB from " << mesh_path << std::endl;
    

    if (textures.empty()) {
        textures.push_back({128, 128, 128, 255});
        tex_sizes.push_back({1, 1});
    }

    // Submesh gia' senza triangoli/edge: e' una nuvola di punti pronta, va copiata cosi' com'e'.
    std::vector<std::array<double,3>> passthrough_points, passthrough_colors;
    for (const auto& m : meshes) {
        if (!m.is_point_cloud) continue;
        for (size_t i = 0; i < m.vertices.size(); ++i) {
            passthrough_points.push_back({m.vertices[i][0], m.vertices[i][1], m.vertices[i][2]});
            if (!m.vertex_colors.empty()) {
                passthrough_colors.push_back({m.vertex_colors[i][0], m.vertex_colors[i][1], m.vertex_colors[i][2]});
            } else {
                passthrough_colors.push_back({(double)m.base_color[0], (double)m.base_color[1], (double)m.base_color[2]});
            }
        }
    }
    if (!passthrough_points.empty()) {
        std::cout << "Detected " << passthrough_points.size()
                  << " points already in point-cloud form" << std::endl;
    }else {
        std::cout << "Iterate " << meshes.size() 
                << " submeshes, " << textures.size() << " textures" << std::endl;
    }

    // Distribute samples proportionally to submesh area (only non-empty meshes).
    // This avoids severe sampling bias and preserves the exact requested count.
    int n_meshes = (int)meshes.size();
    std::vector<double> mesh_areas(n_meshes, 0.0);
    double total_area = 0.0;
    for (int i = 0; i < n_meshes; ++i) {
        const auto& m = meshes[i];
        if (m.is_point_cloud) continue;
        double a_sum = 0.0;
        for (const auto& f : m.faces) {
            const auto& v0 = m.vertices[f[0]];
            const auto& v1 = m.vertices[f[1]];
            const auto& v2 = m.vertices[f[2]];
            Eigen::Vector3d e1(v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]);
            Eigen::Vector3d e2(v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]);
            a_sum += std::max(0.0, 0.5 * e1.cross(e2).norm());
        }
        mesh_areas[i] = a_sum;
        total_area += a_sum;
    }

    double passthrough_weight = (double)passthrough_points.size();
    double total_weight = total_area + passthrough_weight;

    if (total_weight <= 0.0) {
        std::cerr << "No valid mesh area found for sampling." << std::endl;
        return 1;
    }

    // Il peso della point-cloud passthrough concorre proporzionalmente al target richiesto,
    // cosi' il numero finale di punti rispetta num_points invece di copiare sempre tutto.
    int passthrough_target = (int)std::llround((passthrough_weight / total_weight) * (double)num_points);
    passthrough_target = std::clamp(passthrough_target, 0, (int)passthrough_points.size());

    std::vector<int> points_for_mesh(n_meshes, 0);
    std::vector<std::pair<double, int>> remainders;
    remainders.reserve(n_meshes);
    int assigned = passthrough_target;
    for (int i = 0; i < n_meshes; ++i) {
        if (mesh_areas[i] <= 0.0) continue;
        double exact = (mesh_areas[i] / total_weight) * (double)num_points;
        int base = (int)std::floor(exact);
        points_for_mesh[i] = base;
        assigned += base;
        remainders.push_back({exact - base, i});
    }

    int missing = std::max(0, num_points - assigned);
    std::sort(remainders.begin(), remainders.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    for (int k = 0; k < missing && k < (int)remainders.size(); ++k) {
        points_for_mesh[remainders[k].second]++;
    }

    if (passthrough_target < (int)passthrough_points.size()) {
        std::vector<std::array<double,3>> sampled_points, sampled_colors;
        std::cout << "Sampling the point-cloud to target: " << num_points << " points" << std::endl;
        subsample_points(passthrough_points, passthrough_colors, (size_t)passthrough_target, sampled_points, sampled_colors);
        passthrough_points = std::move(sampled_points);
        passthrough_colors = std::move(sampled_colors);
    }

    int total_assigned = passthrough_target;
    for (int v : points_for_mesh) total_assigned += v;
   
    std::vector<SubMeshResult> results(n_meshes);
    int omp_threads = std::max(1, std::min(omp_get_max_threads(), n_meshes));
    #pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < n_meshes; ++i) {
        if (points_for_mesh[i] > 0) {
            results[i] = process_submesh(i, meshes[i], points_for_mesh[i], textures, tex_sizes);
        }
    }

    if (passthrough_points.empty()) {
        std::cout << "Merging submeshes into final point cloud" << std::endl;
    }
    std::vector<std::array<double,3>> all_points, all_colors;
    all_points.insert(all_points.end(), passthrough_points.begin(), passthrough_points.end());
    all_colors.insert(all_colors.end(), passthrough_colors.begin(), passthrough_colors.end());
    for (auto& r : results) {
        all_points.insert(all_points.end(), r.points.begin(), r.points.end());
        all_colors.insert(all_colors.end(), r.colors.begin(), r.colors.end());
    }

    std::cout << "Computing normals" << std::endl;
    auto pcd_final = std::make_shared<open3d::geometry::PointCloud>();
    pcd_final->points_.resize(all_points.size());
    pcd_final->colors_.resize(all_colors.size());
    for (size_t i = 0; i < all_points.size(); ++i) {
        pcd_final->points_[i] = {all_points[i][0], all_points[i][1], all_points[i][2]};
        pcd_final->colors_[i] = {all_colors[i][0], all_colors[i][1], all_colors[i][2]};
    }
    pcd_final->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(0.02, 30));
    pcd_final->OrientNormalsTowardsCameraLocation();

    // Convert NaN normals to a (0,0,0)
    std::vector<std::array<double,3>> all_normals(all_points.size());
    for (size_t i = 0; i < all_points.size(); ++i) {
        auto& n = pcd_final->normals_[i];
        if (std::isnan(n[0]) || std::isnan(n[1]) || std::isnan(n[2]))
            all_normals[i] = {0.0, 0.0, 0.0};
        else
            all_normals[i] = {n[0], n[1], n[2]};
    }

    write_las(out_path, all_points, all_colors, all_normals);

    double total = elapsed(global_start);
    int mn = (int)(total / 60);
    int sc = (int)total % 60;
    std::cout << "Saved: " << out_path << " with " << all_points.size() << " points" << std::endl;
    if (mn > 0)
        std::cout << "Processing time: " << mn << " min " << sc << " sec" << std::endl;
    else
        std::cout << "Processing time: " << sc << " sec" << std::endl;

    std::cout << out_path << std::endl;
    return 0;
}
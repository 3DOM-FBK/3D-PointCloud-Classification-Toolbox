// subsample_pc — voxel-grid subsampling of a point cloud.
//
//   subsample_pc <file.ply|file.las> <output.ply|output.las> [voxel_size] [--memory-budget MB] [--temp-dir DIR]
//
// LAS -> LAS: out-of-core (ooc/subsample_stream.h, no Open3D). The cloud is read in blocks, distributed in XY tiles aligned to the
//   voxel grid, subsampled tile by tile (average of the points and colours of every voxel, as Open3D's VoxelDownSample), and the
//   normals are estimated on the subsampled cloud (radius 0.02, 30 neighbours, oriented towards the origin), also tile by tile.
//   Memory is bounded by --memory-budget; the output is a LAS with NormalX/Y/Z and POINT_ID.
// PLY -> PLY: Open3D (the whole cloud is loaded: the memory needed is checked before reading it).

#include <open3d/Open3D.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstring>

#include "ooc/memory_budget.h"
#include "ooc/ply_reader.h"
#include "ooc/subsample_stream.h"

static double g_memory_budget_mb = 0;    // --memory-budget (0: 50% of the available memory)
static std::string g_temp_dir;           // --temp-dir

std::string get_extension(const std::string& path) {
    size_t pos = path.rfind('.');
    if (pos == std::string::npos) return "";
    std::string ext = path.substr(pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

static void print_downsampled(double voxel_size, size_t n) {
    int voxel_size_cm = (int)(voxel_size * 100);
    int voxel_size_mm = (int)(voxel_size * 1000);
    if (voxel_size_cm >= 1) {
        std::cout << "Points N after voxel_down_sample (" << voxel_size_cm << " cm): " << n << std::endl;
    } else {
        std::cout << "Points N after voxel_down_sample (" << voxel_size_mm << " mm): " << n << std::endl;
    }
}

// ============================================================
// Subsample PLY → PLY (Open3D, the whole cloud is in memory)
// ============================================================
std::string subsample_ply(const std::string& file_path, const std::string& output_path, double voxel_size) {
    {
        // points + colours + normals of the input, the same of the output, the hash map of the voxels: ~150 bytes per point
        ooc::PlyReader probe(file_path);
        const uint64_t budget = ooc::resolve_budget_bytes(g_memory_budget_mb);
        if (probe.numVertices() > 0 && 150.0 * (double)probe.numVertices() > (double)budget)
            throw std::runtime_error("Subsampling a PLY of " + std::to_string(probe.numVertices()) + " points needs about " +
                                     std::to_string((uint64_t)(150.0 * (double)probe.numVertices() / 1e6)) + " MB (the whole cloud is loaded) but the memory budget is " +
                                     std::to_string(budget / 1048576) + " MB. Convert it to LAS: LAS subsampling is out-of-core.");
    }
    std::cout << "Loading PLY: " << file_path << std::endl;
    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(file_path, *pcd))
        throw std::runtime_error("Failed to read: " + file_path);
    std::cout << "Original Points N: " << pcd->points_.size() << std::endl;

    auto pcd_down = pcd->VoxelDownSample(voxel_size);
    print_downsampled(voxel_size, pcd_down->points_.size());

    if (!open3d::io::WritePointCloud(output_path, *pcd_down))
        throw std::runtime_error("Failed to write: " + output_path);

    std::cout << "Subsampled point cloud saved to: " << std::endl;
    return output_path;
}

// ============================================================
// Subsample LAS → LAS (out-of-core)
// ============================================================
std::string subsample_las(const std::string& file_path, const std::string& output_path, double voxel_size) {
    std::cout << "Loading LAS: " << file_path << std::endl;
    std::string why;
    if (!ooc::subsample_las_streaming(file_path, output_path, voxel_size, g_memory_budget_mb, g_temp_dir, why))
        throw std::runtime_error("Nothing to subsample: " + why);
    std::cout << "Subsampled point cloud saved to: " << std::endl;
    return output_path;
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[]) {
    try {
        std::vector<std::string> pos;
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if (a == "--memory-budget" && i + 1 < argc) g_memory_budget_mb = std::stod(argv[++i]);
            else if (a == "--temp-dir" && i + 1 < argc) g_temp_dir = argv[++i];
            else pos.push_back(a);
        }
        if (pos.size() < 2) {
            std::cerr << "Usage: " << argv[0] << " <file.ply|file.las> <output.ply|output.las> [voxel_size] [--memory-budget MB] [--temp-dir DIR]" << std::endl;
            return 1;
        }
        std::string file_path = pos[0];
        std::string output_path = pos[1];
        double voxel_size = (pos.size() >= 3) ? std::stod(pos[2]) : 0.002;

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

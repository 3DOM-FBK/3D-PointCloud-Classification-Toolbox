// check_point_id — makes sure a LAS has the canonical layout used by the pipeline:
//   LAS 1.2, point format 3 (34 bytes) + NormalX/NormalY/NormalZ (float) + POINT_ID (uint32) = 50 bytes per point.
//
//   check_point_id <input.las> <output.las> [--memory-budget MB] [--temp-dir DIR]
//
//   * already canonical          -> the input is moved to the output path (no I/O on the points);
//   * normals present            -> rewritten in blocks (ooc/canonical_las.h), memory O(block);
//   * normals missing            -> estimated tile by tile (ooc/las_normals_stream.h, radius 0.02, 30 neighbours, oriented
//                                   towards the origin as Open3D did), memory bounded by --memory-budget.
// The tool no longer uses Open3D: nothing is loaded as a whole.
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "ooc/canonical_las.h"
#include "ooc/las_normals_stream.h"
#include "ooc/las_stream.h"
#include "ooc/memory_budget.h"
#include "ooc/temp_dir.h"

namespace fs = std::filesystem;

static double g_memory_budget_mb = 0;    // --memory-budget (0: 50% of the available memory)
static std::string g_temp_dir;           // --temp-dir

static std::string check_and_fix(const std::string& input_path, const std::string& output_path) {
    std::cout << "Reading: " << input_path << std::endl;
    ooc::LasInfo li = ooc::read_las_info(input_path);        // header and Extra Bytes only
    const bool has_normals = li.find_extra("NormalX") && li.find_extra("NormalY") && li.find_extra("NormalZ");
    const bool has_point_id = li.find_extra("POINT_ID") && (li.find_extra("POINT_ID")->type == 5 || li.find_extra("POINT_ID")->type == 6);
    const int extra_size = li.recordLength - ooc::base_record_size(li.format);
    std::cout << "  LAS: " << li.numPoints << " points  format=" << li.format << "  point_length=" << li.recordLength
              << "  extra=" << extra_size << "  has_normals=" << has_normals << "  has_point_id=" << has_point_id << std::endl;

    const bool needs_normals = !has_normals;
    const bool needs_point_id = !has_point_id;
    const bool needs_canonical = !(li.format == 3 && li.recordLength == 50);

    if (!needs_normals && !needs_point_id && !needs_canonical) {
        std::cout << "Normals and POINT_ID already present with canonical format. No modification needed." << std::endl;
        if (input_path != output_path) {
            fs::path in_path(input_path), out_path(output_path);
            if (fs::exists(out_path)) fs::remove(out_path);
            fs::rename(in_path, out_path);
            std::cout << "Save into output path: " << output_path << std::endl;
            return output_path;
        }
        return input_path;
    }

    // never write the output while the input is still being read (they may be the same path)
    const std::string tmp = output_path + ".tmp";
    ooc::TempWorkDir::install_handlers();                       // SIGTERM (/stop_process/): remove the partial output
    ooc::TempWorkDir::extra_paths().push_back(tmp);
    if (!needs_normals) {
        std::cout << "Rewriting LAS to canonical format (PointFormat=3 + NormalX/Y/Z + POINT_ID), in blocks..." << std::endl;
        uint64_t n = ooc::rewrite_canonical_streaming(input_path, tmp, li);
        std::cout << "  Written " << n << " points" << std::endl;
    } else {
        std::cout << "Computing Normals and rewriting LAS to canonical format (PointFormat=3 + NormalX/Y/Z + POINT_ID)..." << std::endl;
        ooc::las_add_normals_streaming(input_path, tmp, g_memory_budget_mb, g_temp_dir);
    }
    fs::rename(tmp, output_path);
    return output_path;
}

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
            std::cerr << "Usage: " << argv[0] << " <input.las> <output.las> [--memory-budget MB] [--temp-dir DIR]" << std::endl;
            return 1;
        }
        std::string result = check_and_fix(pos[0], pos[1]);
        std::cout << "Output: " << result << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 2;
    }
}

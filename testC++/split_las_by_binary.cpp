/**
 * split_las_by_binary.cpp  (v5 — annotations.bin, streaming, no PDAL)
 *
 * Splits a LAS by the annotations of the viewer (annotations.bin: 2 bytes per POINT_ID = segment+1, class).
 *
 *   split_las_by_binary <las> <annotations.bin> <out_dir> [--exclude-unclassified]
 *       one segment_<id>.las per annotated segment, each with a `labels` attribute (uint8: the class, 255 = unclassified)
 *   split_las_by_binary <las> <annotations.bin> --extract-segment <seg_id> <out_path>
 *       the points of one segment, no `labels`
 *   (both accept --memory-budget MB: only used to check that annotations.bin and the output buffers fit)
 *
 * The LAS is read once, sequentially, in blocks. Every output record is the input record byte for byte (coordinates,
 * scale/offset, every attribute, POINT_ID) plus `labels`, and the order of the points is preserved: an extracted segment
 * is an ordered subset of the cloud, which pc_columns.py merges back without a join. Memory: the annotations
 * (2 bytes per POINT_ID) + one block + a small write buffer per segment, whatever the size of the LAS.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "ooc/las_stream.h"
#include "ooc/las_write.h"
#include "ooc/memory_budget.h"
#include "ooc/positional_writer.h"
#include "ooc/temp_dir.h"

namespace fs = std::filesystem;

namespace {

struct AnnotationStore {
    std::vector<uint8_t> raw;
    uint32_t pointCount = 0;   // number of POINT_ID slots (raw.size() / 2)
    size_t annotated = 0;      // points with a segment assigned
};

AnnotationStore read_annotations_bin(const fs::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open annotations.bin: " + path.string());
    const std::streamsize size = f.tellg();
    if (size <= 0 || (size % 2) != 0)
        throw std::runtime_error("Invalid annotations.bin size (" + std::to_string(size) +
                                 " bytes, expected 2 bytes per point) in: " + path.string());
    f.seekg(0, std::ios::beg);
    AnnotationStore store;
    store.raw.resize(static_cast<size_t>(size));
    f.read(reinterpret_cast<char*>(store.raw.data()), size);
    if (!f) throw std::runtime_error("Failed reading annotations.bin: " + path.string());
    store.pointCount = static_cast<uint32_t>(store.raw.size() / 2);
    for (uint32_t pid = 0; pid < store.pointCount; ++pid)
        if (store.raw[2 * static_cast<size_t>(pid)] != 0) ++store.annotated;
    std::cout << "  Loaded " << store.annotated << " annotated points from annotations.bin (N="
              << store.pointCount << ").\n";
    return store;
}

// One output LAS: header written at the end (count and bbox are known only then), records appended through a small buffer.
class SegmentOutput {
public:
    SegmentOutput(const fs::path& path, const ooc::OutHeader& hdr, bool addLabels)
        : path_(path), hdr_(hdr), addLabels_(addLabels), pw_(32ULL << 20) {
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0) throw std::runtime_error("Cannot create: " + path.string());
        pos_ = hdr.offsetToData;      // the header is written by finish()
        buf_.reserve(kBuf);
        for (int k = 0; k < 3; k++) { mn_[k] = INT32_MAX; mx_[k] = INT32_MIN; }
    }
    ~SegmentOutput() { if (fd_ >= 0) ::close(fd_); }
    SegmentOutput(const SegmentOutput&) = delete;
    SegmentOutput& operator=(const SegmentOutput&) = delete;

    uint64_t count() const { return count_; }
    const fs::path& path() const { return path_; }

    void add(const uint8_t* rec, size_t inLen, uint8_t label) {
        int32_t q[3];
        std::memcpy(q, rec, 12);
        for (int k = 0; k < 3; k++) { mn_[k] = std::min(mn_[k], q[k]); mx_[k] = std::max(mx_[k], q[k]); }
        size_t at = buf_.size();
        buf_.resize(at + hdr_.recordLength);
        std::memcpy(buf_.data() + at, rec, inLen);
        if (hdr_.addedOffset >= 0) buf_[at + hdr_.addedOffset] = addLabels_ ? label : buf_[at + hdr_.addedOffset];
        count_++;
        if (buf_.size() >= kBuf) flush();
    }

    void finish(const ooc::LasInfo& li) {
        flush();
        pw_.finish();
        std::vector<uint8_t> h = hdr_.bytes;
        ooc::patch_count_and_bbox(h, li, count_, mn_, mx_);
        ooc::write_all_at(fd_, h.data(), h.size(), 0);
        ::close(fd_);
        fd_ = -1;
    }
    void abort() {
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        std::error_code ec;
        fs::remove(path_, ec);
    }

private:
    static constexpr size_t kBuf = 1 << 20;
    void flush() {
        if (buf_.empty()) return;
        pw_.write(fd_, pos_, buf_.data(), buf_.size());
        pos_ += buf_.size();
        buf_.clear();
    }
    fs::path path_;
    ooc::OutHeader hdr_;
    bool addLabels_;
    int fd_ = -1;
    uint64_t pos_ = 0, count_ = 0;
    std::vector<uint8_t> buf_;
    ooc::PositionalWriter pw_;
    int32_t mn_[3], mx_[3];
};

void run_split(const fs::path& lasPath, const fs::path& annotationsPath, const std::map<int, fs::path>& outputMap,
               bool addLabels, bool excludeUnclassified, double memoryBudgetMb) {
    std::cout << "Loading LAS: " << lasPath << "\n";
    ooc::LasInfo li = ooc::read_las_info(lasPath.string());
    std::cout << "  Total points: " << li.numPoints << "\n";
    if (li.pidOff < 0)
        throw std::runtime_error("No POINT_ID dimension found in source LAS (an Extra Byte named POINT_ID, uint32, is required).");
    const size_t RL = (size_t)li.recordLength;

    std::cout << "Loading annotations.bin: " << annotationsPath << "\n";
    AnnotationStore ann = read_annotations_bin(annotationsPath);
    {
        std::map<int, size_t> segCounts;
        for (size_t p = 0; p < ann.raw.size(); p += 2)
            if (ann.raw[p] != 0) ++segCounts[static_cast<int>(ann.raw[p]) - 1];
        std::cout << "  Annotation breakdown:";
        for (auto& [seg, cnt] : segCounts) std::cout << " seg" << seg << "=" << cnt;
        std::cout << "\n";
    }

    const uint64_t budget = ooc::resolve_budget_bytes(memoryBudgetMb);
    const uint64_t need = ann.raw.size() + outputMap.size() * (1ULL << 20) + (64ULL << 20);
    if (need > budget)
        throw std::runtime_error("Not enough memory budget: annotations.bin and the write buffers need ~" +
                                 std::to_string(need / 1048576) + " MB, the budget is " + std::to_string(budget / 1048576) + " MB");

    ooc::OutHeader outHdr;
    if (addLabels) outHdr = ooc::header_with_u8_attribute(li, "labels");
    else {
        outHdr.bytes = li.headerBytes; outHdr.offsetToData = li.offsetToData; outHdr.recordLength = li.recordLength;
        // no attribute is added: records are copied as they are
    }

    std::map<int, std::unique_ptr<SegmentOutput>> outputs;
    // On SIGTERM (/stop_process/) the partial outputs are removed
    for (auto const& [id, path] : outputMap) ooc::TempWorkDir::extra_paths().push_back(path.string());
    ooc::TempWorkDir::install_handlers();

    uint64_t unmapped = 0, outOfRange = 0, unclassifiedFiltered = 0;
    ooc::LasStreamReader reader(lasPath.string(), li, 32ULL << 20);
    std::vector<uint8_t> buf((size_t)reader.blockRows() * RL);
    std::cout << "Distributing points from annotations.bin...\n";
    uint64_t done = 0;
    int lastPct = -1;
    try {
        for (uint64_t b = 0; b < reader.numBlocks(); b++) {
            const size_t m = reader.read(b, buf.data());
            for (size_t i = 0; i < m; i++) {
                const uint8_t* rec = buf.data() + i * RL;
                const uint32_t pid = ooc::rd<uint32_t>(rec + li.pidOff);
                if (pid >= ann.pointCount) { ++outOfRange; continue; }
                const uint8_t segBuf = ann.raw[2 * static_cast<size_t>(pid)];
                if (segBuf == 0) { ++unmapped; continue; }          // not annotated: skipped
                const uint8_t rawClass = ann.raw[2 * static_cast<size_t>(pid) + 1];
                if (excludeUnclassified && rawClass == 0) { ++unclassifiedFiltered; continue; }
                const int segId = static_cast<int>(segBuf) - 1;
                const uint8_t classId = (rawClass == 0) ? 0xFF : rawClass;
                auto path = outputMap.find(segId);
                if (path == outputMap.end()) continue;               // segment not requested
                auto& out = outputs[segId];
                if (!out) out.reset(new SegmentOutput(path->second, outHdr, addLabels));
                out->add(rec, RL, classId);
            }
            done += m;
            int pct = (int)(100 * done / li.numPoints);
            if (pct / 10 != lastPct / 10) { std::cout << "[progress] split " << (pct / 10) * 10 << "%" << std::endl; lastPct = pct; }
        }
        std::cout << "Writing output files:\n";
        for (auto const& [id, path] : outputMap) {
            auto it = outputs.find(id);
            if (it == outputs.end()) { std::cout << "  No points — skipped: " << path << "\n"; continue; }
            it->second->finish(li);
            std::cout << "  Saved " << it->second->count() << " points → " << path << "\n";
        }
    } catch (...) {
        for (auto& kv : outputs) kv.second->abort();
        throw;
    }
    ooc::TempWorkDir::extra_paths().clear();

    if (unmapped > 0) std::cerr << "INFO: " << unmapped << " points not annotated (skipped).\n";
    if (outOfRange > 0)
        std::cerr << "WARNING: " << outOfRange << " points have POINT_ID >= annotations.bin point count (" << ann.pointCount
                  << "); the annotations file does not match this LAS.\n";
    if (unclassifiedFiltered > 0)
        std::cerr << "INFO: " << unclassifiedFiltered << " points with unclassified label excluded (--exclude-unclassified).\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    // optional "--memory-budget MB" anywhere; the other arguments keep their positions
    double memoryBudgetMb = 0;
    std::vector<std::string> a;
    for (int i = 0; i < argc; i++) {
        std::string s = argv[i];
        if (s == "--memory-budget" && i + 1 < argc) memoryBudgetMb = std::stod(argv[++i]);
        else a.push_back(s);
    }
    try {
        // Mode 2: extract single segment
        if (a.size() >= 6 && a[3] == "--extract-segment") {
            fs::path las = a[1], annotations = a[2], out = a[5];
            int segId = std::stoi(a[4]);
            if (!fs::exists(annotations)) throw std::runtime_error("annotations.bin file not found: " + annotations.string());
            if (out.has_parent_path()) fs::create_directories(out.parent_path());
            std::cout << "Extract-segment mode: annotations.bin (segment " << segId << ")\n";
            std::map<int, fs::path> m = {{segId, out}};
            run_split(las, annotations, m, /*addLabels=*/false, /*excludeUnclassified=*/false, memoryBudgetMb);
            return 0;
        }
        // Mode 1: split all segments
        if (a.size() >= 4) {
            fs::path las = a[1], annotations = a[2], outDir = a[3];
            bool excludeUnclassified = (a.size() > 4 && a[4] == "--exclude-unclassified");
            if (!fs::exists(annotations)) throw std::runtime_error("annotations.bin file not found: " + annotations.string());
            fs::create_directories(outDir);
            std::cout << "Mode: Split all annotated segments";
            if (excludeUnclassified) std::cout << " (excluding unclassified points)";
            std::cout << "\n";
            // All possible segment ids (0–254; 255 = unassigned)
            std::map<int, fs::path> outMap;
            for (int sid = 0; sid < 255; ++sid) outMap[sid] = outDir / ("segment_" + std::to_string(sid) + ".las");
            run_split(las, annotations, outMap, /*addLabels=*/true, excludeUnclassified, memoryBudgetMb);
            std::cout << "\nProcess completed!\n";
            return 0;
        }
        std::cerr << "ERROR: Invalid arguments.\n\n"
                  << "Usage (split all segments):\n"
                  << "  " << a[0] << " <las_path> <annotations.bin> <output_dir> [--exclude-unclassified]\n\n"
                  << "Usage (extract single segment):\n"
                  << "  " << a[0] << " <las_path> <annotations.bin> --extract-segment <seg_id> <out_path>\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}

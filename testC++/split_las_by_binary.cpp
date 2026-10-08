/**
 * split_las_by_binary.cpp  (v4 — annotations.bin, robust single-pass)
 *
 * Key improvements over original:
 * 1. O(N) single-pass: source LAS read once; all segments populated simultaneously.
 * 2. Type-safe copy: native types preserved for Extra Bytes (no intermediate double cast).
 * 3. Strict validation: binary buffer size checked against point count.
 * 4. Safe PID access: out-of-bounds PIDs are counted and reported; never silently remapped.
 * 5. Clean layout reuse: output layout built once per segment from a common helper.
 */

#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/io/LasReader.hpp>
#include <pdal/io/LasWriter.hpp>
#include <pdal/io/BufferReader.hpp>
#include <pdal/Options.hpp>
#include <pdal/Dimension.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
//  LAS Extra Bytes VLR parsing
//  (raw binary header read to recover non-UTF8 names that PDAL would drop)
// ─────────────────────────────────────────────────────────────────────────────

struct ExtraBytesEntry {
    std::string           name;
    pdal::Dimension::Type type;
    int                   size;
};

static const char* pdalTypeToReaderTypeName(pdal::Dimension::Type t)
{
    switch (t) {
        case pdal::Dimension::Type::Unsigned8:  return "uint8";
        case pdal::Dimension::Type::Signed8:    return "int8";
        case pdal::Dimension::Type::Unsigned16: return "uint16";
        case pdal::Dimension::Type::Signed16:   return "int16";
        case pdal::Dimension::Type::Unsigned32: return "uint32";
        case pdal::Dimension::Type::Signed32:   return "int32";
        case pdal::Dimension::Type::Unsigned64: return "uint64";
        case pdal::Dimension::Type::Signed64:   return "int64";
        case pdal::Dimension::Type::Float:      return "float";
        case pdal::Dimension::Type::Double:     return "double";
        default:                                return nullptr;
    }
}

static std::string buildReaderExtraDimsSpec(const std::vector<ExtraBytesEntry>& extras)
{
    std::string spec;
    for (const auto& e : extras) {
        const char* tname = pdalTypeToReaderTypeName(e.type);
        if (!tname || e.name.empty()) {
            continue;
        }
        if (!spec.empty()) {
            spec += ",";
        }
        spec += e.name;
        spec += "=";
        spec += tname;
    }
    return spec;
}

static pdal::Dimension::Type lasTypeToPdal(uint8_t data_type, int& size)
{
    switch (data_type) {
        case  1: size=1;  return pdal::Dimension::Type::Unsigned8;
        case  2: size=1;  return pdal::Dimension::Type::Signed8;
        case  3: size=2;  return pdal::Dimension::Type::Unsigned16;
        case  4: size=2;  return pdal::Dimension::Type::Signed16;
        case  5: size=4;  return pdal::Dimension::Type::Unsigned32;
        case  6: size=4;  return pdal::Dimension::Type::Signed32;
        case  7: size=8;  return pdal::Dimension::Type::Unsigned64;
        case  8: size=8;  return pdal::Dimension::Type::Signed64;
        case  9: size=4;  return pdal::Dimension::Type::Float;
        case 10: size=8;  return pdal::Dimension::Type::Double;
        default: size=0;  return pdal::Dimension::Type::None;
    }
}

static std::vector<ExtraBytesEntry> readExtraBytesVLR(const fs::path& las_path)
{
    std::vector<ExtraBytesEntry> result;
    std::ifstream f(las_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open LAS: " + las_path.string());

    f.seekg(100); uint32_t num_vlrs; f.read(reinterpret_cast<char*>(&num_vlrs), 4);
    f.seekg(94);  uint16_t hdr_size; f.read(reinterpret_cast<char*>(&hdr_size), 2);
    f.seekg(hdr_size);

    for (uint32_t v = 0; v < num_vlrs; ++v) {
        uint8_t vlr_header[54];
        f.read(reinterpret_cast<char*>(vlr_header), 54);
        if (!f) break;

        char     user_id[17] = {}; std::memcpy(user_id,       vlr_header + 2,  16);
        uint16_t record_id;        std::memcpy(&record_id,    vlr_header + 18,  2);
        uint16_t record_length;    std::memcpy(&record_length, vlr_header + 20,  2);

        if (std::string(user_id) == "LASF_Spec" && record_id == 4) {
            int num_extra = record_length / 192;
            std::vector<uint8_t> data(record_length);
            f.read(reinterpret_cast<char*>(data.data()), record_length);
            for (int i = 0; i < num_extra; ++i) {
                const uint8_t* rec = data.data() + i * 192;
                int sz = 0;
                pdal::Dimension::Type pdal_type = lasTypeToPdal(rec[2], sz);
                if (pdal_type != pdal::Dimension::Type::None) {
                    char name_buf[33] = {};
                    std::memcpy(name_buf, rec + 4, 32);
                    result.push_back({std::string(name_buf), pdal_type, sz});
                }
            }
        } else {
            f.seekg(record_length, std::ios::cur);
        }
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers & metadata
// ─────────────────────────────────────────────────────────────────────────────

// Annotation store (annotations.bin): raw buffer, 2 bytes per POINT_ID
//   byte[2*pid]   = segment_id + 1   (0 = point not annotated)
//   byte[2*pid+1] = class_id         (0 = no class assigned)
struct AnnotationStore {
    std::vector<uint8_t> raw;
    uint32_t             pointCount = 0;   // number of POINT_ID slots (raw.size() / 2)
    size_t               annotated  = 0;   // points with segment assigned
};

// Read annotations.bin into a flat vector (O(1) lookup by POINT_ID).
static AnnotationStore read_annotations_bin(const fs::path& path)
{
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


// ─────────────────────────────────────────────────────────────────────────────
//  Type-safe field copy  (native type preserved — no intermediate double cast)
// ─────────────────────────────────────────────────────────────────────────────

static void copyFieldTyped(const pdal::PointViewPtr& src, pdal::PointId si,
                                 pdal::PointViewPtr& dst, pdal::PointId di,
                           pdal::Dimension::Id srcId,
                           pdal::Dimension::Id dstId,
                           pdal::Dimension::Type type)
{
    using T = pdal::Dimension::Type;
    switch (type) {
        case T::Unsigned8:  dst->setField(dstId, di, src->getFieldAs<uint8_t> (srcId, si)); break;
        case T::Signed8:    dst->setField(dstId, di, src->getFieldAs<int8_t>  (srcId, si)); break;
        case T::Unsigned16: dst->setField(dstId, di, src->getFieldAs<uint16_t>(srcId, si)); break;
        case T::Signed16:   dst->setField(dstId, di, src->getFieldAs<int16_t> (srcId, si)); break;
        case T::Unsigned32: dst->setField(dstId, di, src->getFieldAs<uint32_t>(srcId, si)); break;
        case T::Signed32:   dst->setField(dstId, di, src->getFieldAs<int32_t> (srcId, si)); break;
        case T::Unsigned64: dst->setField(dstId, di, src->getFieldAs<uint64_t>(srcId, si)); break;
        case T::Signed64:   dst->setField(dstId, di, src->getFieldAs<int64_t> (srcId, si)); break;
        case T::Float:      dst->setField(dstId, di, src->getFieldAs<float>   (srcId, si)); break;
        case T::Double:     dst->setField(dstId, di, src->getFieldAs<double>  (srcId, si)); break;
        default: break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Dimension info (dim mapping between source and output)
// ─────────────────────────────────────────────────────────────────────────────

struct DimMapping {
    pdal::Dimension::Id   srcId;
    pdal::Dimension::Id   dstId;
    pdal::Dimension::Type type;
};
// Standard PDAL dimension names — used to separate standard from extra dims
static const std::set<std::string> kStandardDimNames = {
    "X","Y","Z","Intensity","ReturnNumber","NumberOfReturns",
    "GpsTime","Red","Green","Blue",
    "ScannerChannel","ScanChannel","ClassificationFlags","ScanAngle",
    "NormalX","NormalY","NormalZ","Synthetic","KeyPoint","Withheld","Overlap"
};

// Normalise a dimension name: lowercase, strip underscores/spaces.
// Lets us compare VLR "normal_x" with PDAL "NormalX" → both become "normalx".
static std::string normaliseDimName(const std::string& s)
{
    std::string r;
    for (char c : s)
        if (c != '_' && c != ' ')
            r += static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    return r;
}

// Decide whether a VLR extra-bytes entry was already absorbed by PDAL as a
// standard/known dimension (e.g. VLR "normal_x" → PDAL standard "NormalX").
static bool vlrMatchesStandardPdalDim(
        const std::string& vlrName,
        const pdal::PointViewPtr& view)
{
    std::string vnorm = normaliseDimName(vlrName);
    for (auto id : view->dims()) {
        std::string dname = pdal::Dimension::name(id);
        if (dname.empty()) continue;                     // unnamed → not standard
        if (!kStandardDimNames.count(dname)) continue;   // not a known standard
        if (normaliseDimName(dname) == vnorm) return true;
    }
    return false;
}

// Find POINT_ID — first try by PDAL dim name, then by VLR + positional matching.
// Returns the PDAL Dimension::Id that holds the file-level POINT_ID.
static pdal::Dimension::Id findPointIdDim(
        const pdal::PointViewPtr& view,
        const std::vector<ExtraBytesEntry>& vlrExtras)
{
    // ── Strategy 1: search for a PDAL dim named "POINT_ID" or "point_id" ──────
    pdal::Dimension::Id candidate = pdal::Dimension::Id::Unknown;
    for (auto id : view->dims()) {
        std::string n = pdal::Dimension::name(id);
        if (n == "POINT_ID") {
            std::cout << "  [Info] Found POINT_ID dimension (exact): '" << n << "'\n";
            return id;
        }
        std::string low = n;
        std::transform(low.begin(), low.end(), low.begin(), ::tolower);
        // Accept "point_id" but NOT PDAL's built-in "PointId" (no underscore)
        if (low == "point_id" && n != "PointId")
            candidate = id;
    }
    if (candidate != pdal::Dimension::Id::Unknown) {
        std::cout << "  [Info] Found POINT_ID dimension (case-insensitive): '"
                  << pdal::Dimension::name(candidate) << "'\n";
        return candidate;
    }

    // ── Strategy 2: positional matching with unnamed PDAL dims ────────────────
    // PDAL may create unnamed (empty-string) dims for extra bytes that it could
    // not map to any standard name. These unnamed dims correspond positionally
    // to the VLR entries that PDAL did NOT absorb as standard dims.
    // We find which VLR position holds POINT_ID, then return the matching
    // unnamed PDAL dim.
    std::vector<pdal::Dimension::Id> unnamedDims;
    for (auto id : view->dims()) {
        std::string dname = pdal::Dimension::name(id);
        if (dname.empty()) unnamedDims.push_back(id);
    }

    // Build list of VLR entries NOT absorbed into standard PDAL dims
    size_t unnamedIdx = 0;
    for (size_t k = 0; k < vlrExtras.size(); ++k) {
        const std::string& vname = vlrExtras[k].name;
        std::string vlow = vname;
        std::transform(vlow.begin(), vlow.end(), vlow.begin(), ::tolower);

        if (vlrMatchesStandardPdalDim(vname, view)) continue; // absorbed

        if ((vlow == "point_id" || vlow == "pointid") && unnamedIdx < unnamedDims.size()) {
            std::cout << "  [Info] Found POINT_ID via VLR positional match → unnamed PDAL dim #"
                      << unnamedIdx << "\n";
            return unnamedDims[unnamedIdx];
        }
        ++unnamedIdx;
    }

    return pdal::Dimension::Id::Unknown;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Per-segment output container
// ─────────────────────────────────────────────────────────────────────────────

struct SegmentOutput {
    std::unique_ptr<pdal::PointTable> table;
    pdal::PointViewPtr                view;
    fs::path                          outPath;
    pdal::PointId                     count    = 0;
    std::vector<DimMapping>           mappings;   // all dims to copy each point
    pdal::Dimension::Id               dimLabels = pdal::Dimension::Id::Unknown;
    pdal::Dimension::Id               srcPointId = pdal::Dimension::Id::Unknown;
    pdal::Dimension::Id               dstPointId = pdal::Dimension::Id::Unknown;
};

// Build the layout and DimMappings for one output segment, mirroring the source.
// Uses a hybrid approach: name-based matching first, then positional matching
// between unnamed PDAL dims and VLR entries that PDAL didn't absorb.
static std::unique_ptr<SegmentOutput> makeOutput(
        const pdal::PointViewPtr&           srcView,
        const std::vector<ExtraBytesEntry>& vlrExtras,
        const fs::path&                     outPath,
        bool                                addLabels)
{
    auto out  = std::make_unique<SegmentOutput>();
    out->outPath  = outPath;
    out->table    = std::make_unique<pdal::PointTable>();
    auto& layout  = *out->table->layout();

    pdal::PointLayoutPtr srcLayout = srcView->table().layout();

    // ── Standard dims ─────────────────────────────────────────────────────────
    for (auto srcId : srcView->dims()) {
        const std::string dname = pdal::Dimension::name(srcId);
        if (!kStandardDimNames.count(dname)) continue;
        pdal::Dimension::Type tp = srcLayout->dimType(srcId);
        pdal::Dimension::Id dstId = layout.registerOrAssignDim(dname, tp);
        out->mappings.push_back({srcId, dstId, tp});
    }

    // ── Extra dims ────────────────────────────────────────────────────────────
    // PDAL may give extra-bytes dims proper names or leave them unnamed (empty).
    // Build a name-based lookup for named PDAL dims...
    std::map<std::string, pdal::Dimension::Id> pdalByNorm;
    for (auto srcId : srcView->dims()) {
        std::string dname = pdal::Dimension::name(srcId);
        if (dname.empty() || kStandardDimNames.count(dname)) continue;
        pdalByNorm[normaliseDimName(dname)] = srcId;
    }

    // ...and collect unnamed dims for positional fallback
    std::vector<pdal::Dimension::Id> unnamedDims;
    for (auto id : srcView->dims()) {
        if (pdal::Dimension::name(id).empty()) unnamedDims.push_back(id);
    }

    // Walk VLR entries. For each non-POINT_ID, non-standard entry:
    // try name-based match first, then fall back to positional against unnamed.
    size_t unnamedIdx = 0;
    for (size_t k = 0; k < vlrExtras.size(); ++k) {
        const std::string vname = vlrExtras[k].name;
        std::string vlow = vname;
        std::transform(vlow.begin(), vlow.end(), vlow.begin(), ::tolower);

        // Skip POINT_ID — handled separately
        bool isPointId = (vlow == "point_id" || vlow == "pointid");

        // Was this VLR entry absorbed into a standard PDAL dim?
        bool absorbedByStandard = vlrMatchesStandardPdalDim(vname, srcView);

        if (absorbedByStandard) {
            // Already copied above as a standard dim; don't advance unnamedIdx
            continue;
        }

        // This VLR entry corresponds to the next unnamed PDAL dim
        pdal::Dimension::Id srcId = pdal::Dimension::Id::Unknown;

        // Try name-based match (for PDAL versions that name extras)
        std::string vnorm = normaliseDimName(vname);
        auto pit = pdalByNorm.find(vnorm);
        if (pit != pdalByNorm.end()) {
            srcId = pit->second;
        } else if (unnamedIdx < unnamedDims.size()) {
            // Positional fallback
            srcId = unnamedDims[unnamedIdx];
        }
        ++unnamedIdx;  // always advance for non-absorbed VLR entries

        if (isPointId) continue;  // don't add mapping for POINT_ID itself

        if (srcId == pdal::Dimension::Id::Unknown) continue;

        pdal::Dimension::Type tp = srcLayout->dimType(srcId);
        pdal::Dimension::Id dstId = layout.registerOrAssignDim(vname, tp);
        out->mappings.push_back({srcId, dstId, tp});
    }

    // ── POINT_ID ─────────────────────────────────────────────────────────────
    out->srcPointId = findPointIdDim(srcView, vlrExtras);
    out->dstPointId = layout.registerOrAssignDim("POINT_ID", pdal::Dimension::Type::Unsigned32);

    // ── labels (training only) ────────────────────────────────────────────────
    if (addLabels)
        out->dimLabels = layout.registerOrAssignDim("labels", pdal::Dimension::Type::Unsigned8);

    out->view = std::make_shared<pdal::PointView>(*out->table);
    return out;
}

// Write a completed output segment to disk.
static void writeOutput(SegmentOutput& out)
{
    if (out.count == 0) {
        std::cout << "  No points — skipped: " << out.outPath << "\n";
        return;
    }

    pdal::BufferReader reader;
    reader.addView(out.view);

    pdal::Options wOpts;
    wOpts.add("filename",      out.outPath.string());
    wOpts.add("extra_dims",    "all");
    wOpts.add("minor_version", 2);
    wOpts.add("dataformat_id", 3);

    pdal::LasWriter writer;
    writer.setOptions(wOpts);
    writer.setInput(reader);
    writer.prepare(*out.table);
    writer.execute(*out.table);

    std::cout << "  Saved " << out.count << " points → " << out.outPath << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Core: single-pass O(N) distribution using annotations.bin
// ─────────────────────────────────────────────────────────────────────────────

static void processPointsFromMapping(
        const pdal::PointViewPtr&                         srcView,
        const AnnotationStore&                            annotations,
        std::map<int, std::unique_ptr<SegmentOutput>>&    outputs,
        bool                                              addLabels,
        bool                                              excludeUnclassified = false)
{
    const size_t nPts = srcView->size();
    pdal::PointId unmapped = 0;
    pdal::PointId outOfRange = 0;
    pdal::PointId unclassified_filtered = 0;

    // annotations.bin is indexed by POINT_ID value — not by PDAL read-order index.
    // We must read the POINT_ID attribute from each point.
    // Using sequential index `i` is wrong: PDAL may reorder points and, even
    // when it doesn't, POINT_ID values may not start at 0 or be contiguous.
    const pdal::Dimension::Id pidSrc = outputs.empty()
        ? pdal::Dimension::Id::Unknown
        : outputs.begin()->second->srcPointId;

    if (pidSrc == pdal::Dimension::Id::Unknown) {
        throw std::runtime_error(
            "No POINT_ID dimension found in source LAS. "
            "Ensure readers.las loads extra dimensions (extra_dims=all)."
        );
    }

    // Diagnostic: print first 10 POINT_ID values
    {
        size_t diag = std::min<size_t>(nPts, 10);
        std::cout << "  First " << diag << " POINT_ID values:";
        for (size_t d = 0; d < diag; ++d) {
            uint32_t dpid = srcView->getFieldAs<uint32_t>(pidSrc, d);
            std::cout << " " << dpid;
        }
        std::cout << "\n";
    }

    for (pdal::PointId i = 0; i < (pdal::PointId)nPts; ++i) {
        // Read POINT_ID attribute → index into annotations.bin
        uint32_t pid = srcView->getFieldAs<uint32_t>(pidSrc, i);

        if (pid >= annotations.pointCount) {
            ++outOfRange;
            continue;
        }

        const uint8_t segBuf = annotations.raw[2 * static_cast<size_t>(pid)];
        if (segBuf == 0) {
            ++unmapped;
            continue;  // Point not annotated → skip
        }

        // 0 = no class assigned → keep the historical 0xFF "no class" label
        const uint8_t rawClass = annotations.raw[2 * static_cast<size_t>(pid) + 1];

        // Skip points without manual class assignment (if requested)
        if (excludeUnclassified && rawClass == 0) {
            ++unclassified_filtered;
            continue;
        }

        int segId = static_cast<int>(segBuf) - 1;
        uint8_t classId = (rawClass == 0) ? 0xFF : rawClass;

        auto out_it = outputs.find(segId);
        if (out_it == outputs.end()) continue;  // segment not in requested output list

        SegmentOutput& out = *out_it->second;
        pdal::PointId  di  = out.count;

        // Copy all standard + extra dims with native types
        for (const auto& m : out.mappings)
            copyFieldTyped(srcView, i, out.view, di, m.srcId, m.dstId, m.type);

        // POINT_ID
        out.view->setField(out.dstPointId, di,
                           srcView->getFieldAs<uint32_t>(out.srcPointId, i));

        // labels (training mode only)
        if (addLabels && out.dimLabels != pdal::Dimension::Id::Unknown)
            out.view->setField(out.dimLabels, di, classId);

        ++out.count;
    }

    if (unmapped > 0)
        std::cerr << "INFO: " << unmapped
                  << " points not annotated (skipped).\n";
    if (outOfRange > 0)
        std::cerr << "WARNING: " << outOfRange
                  << " points have POINT_ID >= annotations.bin point count (" << annotations.pointCount
                  << "); the annotations file does not match this LAS.\n";
    if (unclassified_filtered > 0)
        std::cerr << "INFO: " << unclassified_filtered
                  << " points with unclassified label excluded (--exclude-unclassified).\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public entry point: split all annotated segments from an annotations.bin store
// ─────────────────────────────────────────────────────────────────────────────

static void run_split_from_annotations(const fs::path&                las_path,
                                  const fs::path&                annotations_path,
                                  const std::map<int, fs::path>& output_map,
                                  bool                           addLabels,
                                  bool                           excludeUnclassified = false)
{
    // ── Parse Extra Bytes VLR first (used to configure reader extra_dims) ───
    auto vlrExtras = readExtraBytesVLR(las_path);
    const std::string extraDimsSpec = buildReaderExtraDimsSpec(vlrExtras);

    // ── Read source LAS once ──────────────────────────────────────────────────
    std::cout << "Loading LAS: " << las_path << "\n";
    pdal::PointTable srcTable;
    pdal::LasReader  srcReader;
    {
        pdal::Options opts;
        opts.add("filename", las_path.string());
        // Older PDAL versions don't accept extra_dims=all for readers.las,
        // so pass explicit <name>=<type> list from VLR.
        if (!extraDimsSpec.empty()) {
            opts.add("extra_dims", extraDimsSpec);
        }
        srcReader.setOptions(opts);
        srcReader.prepare(srcTable);
    }
    pdal::PointViewPtr srcView = *srcReader.execute(srcTable).begin();
    std::cout << "  Total points: " << srcView->size() << "\n";

    // ── Read annotations from annotations.bin ──────────────────────────────────────────
    std::cout << "Loading annotations.bin: " << annotations_path << "\n";
    auto mapping = read_annotations_bin(annotations_path);

    // ── Diagnostic: dump all PDAL dimension names ─────────────────────────────
    std::cout << "  PDAL dimensions (" << srcView->dims().size() << "):";
    for (auto id : srcView->dims())
        std::cout << " [" << pdal::Dimension::name(id) << "]";
    std::cout << "\n";

    std::cout << "  VLR extras (" << vlrExtras.size() << "):";
    for (auto& e : vlrExtras)
        std::cout << " [" << e.name << "]";
    std::cout << "\n";

    // ── Diagnostic: dump annotation breakdown per segment ─────────────────────
    {
        std::map<int, int> segCounts;
        for (size_t p = 0; p < mapping.raw.size(); p += 2)
            if (mapping.raw[p] != 0) ++segCounts[static_cast<int>(mapping.raw[p]) - 1];
        std::cout << "  Annotation breakdown:";
        for (auto& [seg, cnt] : segCounts)
            std::cout << " seg" << seg << "=" << cnt;
        std::cout << "\n";
    }

    // ── Build per-segment outputs (name-based VLR↔PDAL matching) ──────────────
    std::map<int, std::unique_ptr<SegmentOutput>> outputs;
    for (auto const& [id, path] : output_map) {
        outputs[id] = makeOutput(srcView, vlrExtras, path, addLabels);
    }

    // ── Single-pass point distribution ───────────────────────────────────────
    std::cout << "Distributing points from annotations.bin...\n";
    processPointsFromMapping(srcView, mapping, outputs, addLabels, excludeUnclassified);

    // ── Write results ─────────────────────────────────────────────────────────
    std::cout << "Writing output files:\n";
    for (auto& [id, out] : outputs)
        writeOutput(*out);
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    // Usage modes (all use annotations.bin for annotations):
    // Mode 1: split_las_by_binary <las> <annotations.bin> <out_dir>
    //         Split all annotated segments into separate LAS files (addLabels=true)
    // Mode 2: split_las_by_binary <las> <annotations.bin> --extract-segment <seg_id> <out_path>
    //         Extract a single segment's points into one LAS file (addLabels=false)

    try {
        // Mode 2: extract single segment
        if (argc >= 6 && std::string(argv[3]) == "--extract-segment") {
            fs::path las    = argv[1];
            fs::path annotations  = argv[2];
            int      segId  = std::stoi(argv[4]);
            fs::path out    = argv[5];

            if (!fs::exists(annotations))
                throw std::runtime_error("annotations.bin file not found: " + annotations.string());

            fs::create_directories(out.parent_path());
            std::cout << "Extract-segment mode: annotations.bin (segment " << segId << ")\n";

            std::map<int, fs::path> m = {{segId, out}};
            run_split_from_annotations(las, annotations, m, /*addLabels=*/false, /*excludeUnclassified=*/false);
            return 0;
        }

        // Mode 1: split all segments
        if (argc >= 4) {
            fs::path las     = argv[1];
            fs::path annotations   = argv[2];
            fs::path out_dir = argv[3];
            bool excludeUnclassified = (argc > 4 && std::string(argv[4]) == "--exclude-unclassified");

            if (!fs::exists(annotations))
                throw std::runtime_error("annotations.bin file not found: " + annotations.string());

            fs::create_directories(out_dir);
            std::cout << "Mode: Split all annotated segments";
            if (excludeUnclassified) std::cout << " (excluding unclassified points)";
            std::cout << "\n";

            // Build output map for all possible segment IDs (0–254; 255 = unassigned)
            std::map<int, fs::path> out_map;
            for (int sid = 0; sid < 255; ++sid)
                out_map[sid] = out_dir / ("segment_" + std::to_string(sid) + ".las");

            run_split_from_annotations(las, annotations, out_map, /*addLabels=*/true, excludeUnclassified);
            std::cout << "\nProcess completed!\n";
            return 0;
        }

        std::cerr << "ERROR: Invalid arguments.\n\n"
                  << "Usage (split all segments):\n"
                  << "  " << argv[0] << " <las_path> <annotations.bin> <output_dir> [--exclude-unclassified]\n\n"
                  << "Usage (extract single segment):\n"
                  << "  " << argv[0] << " <las_path> <annotations.bin> --extract-segment <seg_id> <out_path>\n";
        return 1;

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

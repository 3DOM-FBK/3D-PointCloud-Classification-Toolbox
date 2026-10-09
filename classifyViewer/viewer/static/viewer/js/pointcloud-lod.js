// =====================================================================
// LOD model of a chunk of the `pck` format (docs/POINTCLOUD_FORMAT.md), shared by the loader (main
// thread: how many points a chunk needs, picking) and the worker (per-point `lodSpacing`).
//
// The points of a chunk are stored in "sequence" order: level 0, level 1, ... (the head is the
// first `prefix[head]` points, the body follows). Inside a level the order is random, so ANY prefix
// of the sequence is a uniform subsample of the chunk. The spacing of the sample made by the first
// k points is therefore a continuous, decreasing function of k:
//   * level l covers the sequence positions [prefix[l], prefix[l+1]) with cell size
//     size / 2^(base+l); for the point at fraction u of the level the spacing is
//     size / 2^(base + l + u)   (the spacing halves from the start to the end of a level);
//   * the last level is the "remainder" of the build (no nominal spacing): its spacing is
//     extrapolated with the power law  n(s) ~ s^-D  estimated from the last regular levels
//     (D = 2 for a surface, 1 for a line, 3 for a volume; clamped to 1..3).
// =====================================================================

/** lodSpacing written for the points of level 0: the overview is never dropped by the shader. */
export const LOD_NEVER_DROP = 1e9;

/**
 * Builds the (structured-cloneable) model of a chunk.
 *   levelPoints: points of every level (the last one is the remainder), size: edge of the chunk cube,
 *   base: log2 of the cells per side of level 0.
 */
export function makeLodModel(size, base, levelPoints) {
    const L = levelPoints.length;
    const prefix = new Array(L + 1).fill(0);
    for (let l = 0; l < L; l++) prefix[l + 1] = prefix[l] + levelPoints[l];
    const Lr = L - 1;                                   // index of the remainder level
    // Exponent: each regular level halves the spacing, so P(l) / P(l-1) = 2^D with P(l) = points of levels
    // <= l. The growth is largest while the levels are not saturated (a finer grid than the data's own
    // spacing adds fewer and fewer points), so the exponent is the largest ratio among the regular levels
    // 1 .. Lr-1 that already hold a few points.
    let D = null;
    for (let l = 1; l < Lr; l++) {
        const a = prefix[l], b = prefix[l + 1];
        if (a >= 64 && b > a) { const r = Math.log2(b / a); D = D === null ? r : Math.max(D, r); }
    }
    if (D === null || !Number.isFinite(D)) D = 2;
    D = Math.min(3, Math.max(1, D));
    // The remainder extrapolation starts where the last NON-EMPTY regular level ends (an empty level halves the
    // nominal spacing without adding points, which would make the spacing jump)
    let lastFull = -1;
    for (let l = 0; l < Lr; l++) if (levelPoints[l] > 0) lastFull = l;
    return {
        size, base, levelPoints, prefix, Lr, D,
        xEnd: lastFull + 1,                             // fractional level where the regular levels end
        sRef: size / Math.pow(2, base + lastFull + 1),  // spacing at the end of the last regular level
        points: prefix[L]
    };
}

/** Spacing of the sample made by the first k points (k is a position, possibly fractional). */
export function lodSpacingAt(m, k) {
    const { prefix, Lr } = m;
    if (k >= prefix[Lr]) {
        const k0 = Math.max(1, prefix[Lr]);
        return m.sRef * Math.pow(Math.max(k, 1) / k0, -1 / m.D);
    }
    // level l with prefix[l] <= k < prefix[l+1]
    let l = 0;
    while (l < Lr - 1 && k >= prefix[l + 1]) l++;
    const n = m.levelPoints[l];
    const u = n > 0 ? (k - prefix[l]) / n : 0;
    return m.size / Math.pow(2, m.base + l + u);
}

/** Number of points of the chunk whose spacing is >= sMin (the points a viewer needs to resolve sMin). */
export function lodCountFor(m, sMin) {
    if (!(sMin > 0)) return m.points;
    const x = Math.log2(m.size / sMin) - m.base;        // fractional level at which the spacing is sMin
    if (x <= 0) return 0;
    let count;
    if (x < m.xEnd) {
        const l = Math.floor(x);
        count = m.prefix[l] + (x - l) * m.levelPoints[l];
    } else {
        count = Math.max(1, m.prefix[m.Lr]) * Math.pow(m.sRef / sMin, m.D);
    }
    return Math.min(m.points, Math.ceil(count));
}

/**
 * Fills out[2j] = lodSpacing and out[2j+1] = fullSpacing for the n points at sequence positions
 * seqStart .. seqStart+n-1. Level-0 points get LOD_NEVER_DROP.
 */
export function fillLodInfo(m, seqStart, n, out) {
    const full = lodSpacingAt(m, m.points);
    const n0 = m.prefix[1];
    for (let j = 0; j < n; j++) {
        const k = seqStart + j;
        out[2 * j] = k < n0 ? LOD_NEVER_DROP : lodSpacingAt(m, k);
        out[2 * j + 1] = full;
    }
}

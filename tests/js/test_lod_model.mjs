// Run: node tests/js/test_lod_model.mjs  (node >= 18; the module is imported as an ES module).
// Unit checks of pointcloud-lod.js (node): the spacing is continuous and decreasing, lodCountFor inverts lodSpacingAt,
// fillLodInfo agrees with lodSpacingAt, level 0 is never dropped.
import { makeLodModel, lodSpacingAt, lodCountFor, fillLodInfo, LOD_NEVER_DROP } from '../../classifyViewer/viewer/static/viewer/js/pointcloud-lod.js';
let bad = 0; const ck = (c, m) => { if (!c) { bad++; console.log('FAIL', m); } };
for (const lp of [[80, 313, 1339, 5313, 20363, 68086, 151614], [80, 313, 1339, 5313, 20363, 68086, 110428, 38681, 2505], [500, 1900, 7000, 300], [900], [10, 40, 0, 0]]) {
    const m = makeLodModel(65.0, 3, lp);
    let prev = Infinity, maxJump = 1;
    for (let k = 1; k <= m.points; k = Math.max(k + 1, Math.floor(k * 1.01))) {      // geometric sampling: a 1 % step in k is <= 1 % in spacing
        if (k > 1 && k < 40) { prev = lodSpacingAt(m, k); continue; }
        const s = lodSpacingAt(m, k);
        ck(s <= prev * (1 + 1e-9), `non increasing at k=${k} (${s} > ${prev}) levels=${lp}`);
        if (prev !== Infinity) maxJump = Math.max(maxJump, prev / s);
        prev = s;
    }
    ck(maxJump < 1.1, `jump ${maxJump} levels=${lp}`);       // a 1 % step in k moves the spacing by at most a few % (a small level halves it over few points)
    for (const f of [0.9, 0.5, 0.2, 0.05]) {
        const s = lodSpacingAt(m, Math.max(1, m.points * f));
        const k = lodCountFor(m, s);
        ck(Math.abs(k - m.points * f) <= 2 + 1e-3 * m.points || lp.length === 1, `inverse f=${f} got ${k} want ${m.points * f} levels=${lp}`);
    }
    const info = new Float32Array(2 * m.points); fillLodInfo(m, 0, m.points, info);
    ck(info[0] === LOD_NEVER_DROP || m.prefix[1] === 0, 'level 0 never dropped');
    for (let k = m.prefix[1]; k < m.points; k += 1 + Math.floor(m.points / 997))      // every sampled index
        ck(Math.abs(info[2 * k] / lodSpacingAt(m, k) - 1) < 1e-5, `fillLodInfo != lodSpacingAt at ${k}`);
    // a chunk sliced in two requests gives the same values as one
    const a = new Float32Array(2 * 1000); fillLodInfo(m, Math.floor(m.points / 3), Math.min(1000, m.points - Math.floor(m.points / 3)), a);
    ck(a[0] === info[2 * Math.floor(m.points / 3)], 'slice start');
    console.log(`levels=${lp.length} points=${m.points} D=${m.D.toFixed(2)} sRef=${m.sRef.toFixed(3)} ok`);
}
{   // empty trailing regular levels (formatVersion 1 tiny chunk): count is non-decreasing and has no 2^D jump
    const m = makeLodModel(65.0, 3, [100, 400, 0, 0, 0, 300]);     // 3 empty regular levels, then the remainder
    let prev = -1;      // checked around and after the end of the regular levels
    for (let x = 1.8; x < 6; x += 0.05) { const k = lodCountFor(m, 65 / Math.pow(2, 3 + x)); ck(k >= prev, 'count decreases at x=' + x); ck(prev < 0 || k - prev <= 0.12 * prev + 3, 'count jumps at x=' + x + ' (' + prev + ' -> ' + k + ')'); prev = k; }
}
console.log(bad ? `${bad} FAILURES` : 'ALL OK'); process.exit(bad ? 1 : 0);

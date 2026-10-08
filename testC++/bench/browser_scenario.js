// Repeatable manual scenario for the editing logic of the chunked point cloud loader.
// Paste it in the browser console of the viewer with a cloud loaded (File > Load Data).
//
// What it does (all numbers come out of the loader, nothing is eyeballed):
//   1. rect selection over the middle of the canvas + CUT into a new segment;
//   2. exportAllTrainingData (what exportAnnotations sends to /api/export-mapping/);
//   3. brute force over the WHOLE geom.bin: a point belongs to the segment iff it falls in the AABB
//      of the cut entry AND inside the frozen 2D selection prism (exactly the rule the replay uses);
//   4. compares the two sets point by point -> `mismatches` must be 0.
// Then it repeats the check after evicting every loaded body mesh (data reloaded from the server) to
// show that segment membership survives LOD eviction (hidden points stay hidden).
(async () => {
    const scene = window.__babylonScene;
    const loader = scene.pointCloudLoader;
    const meta = loader.metadata;
    const eng = scene.getEngine();
    const W = eng.getRenderWidth(), H = eng.getRenderHeight();
    const area = { x: W * 0.3, y: H * 0.3, width: W * 0.3, height: H * 0.3 };

    const nSel = loader.applySelection('rect', area);
    const sel = loader.selectionHistory[0];
    const cut = loader.cutSelection();
    const entry = loader.cutHistory.find(e => e.segmentId === cut.segmentId);

    const exp = await loader.exportAllTrainingData({ 0: 'main', [cut.segmentId]: 'seg' });
    const buf = exp.buffer;

    const geom = new Int32Array(await (await fetch(`/pointcloud-data/${loader.rangeBasePath}/geom.bin?v=${meta.version}`)).arrayBuffer());
    const world = loader.rootTransform.getWorldMatrix();
    const v = new BABYLON.Vector3(), p = new BABYLON.Vector3();
    const [sx, sy, sz] = meta.scale;
    const cx = meta.qMin[0] * sx + meta.offset[0] - meta.boundingBox.min[0];
    const cy = meta.qMin[1] * sy + meta.offset[1] - meta.boundingBox.min[1];
    const cz = meta.qMin[2] * sz + meta.offset[2] - meta.boundingBox.min[2];
    let truth = 0, exported = 0, mismatches = 0;
    for (let i = 0; i < meta.points; i++) {
        const w = i * 5;
        const px = Math.fround(geom[w] * sx + cx), py = Math.fround(geom[w + 1] * sy + cy), pz = Math.fround(geom[w + 2] * sz + cz);
        let inside = false;
        if (!(px < entry.minX || px > entry.maxX || py < entry.minY || py > entry.maxY || pz < entry.minZ || pz > entry.maxZ)) {
            v.set(px, py, pz);
            BABYLON.Vector3.ProjectToRef(v, world, sel.transformMatrix, sel.viewport, p);
            inside = p.x >= area.x && p.x <= area.x + area.width && p.y >= area.y && p.y <= area.y + area.height;
        }
        const got = buf[geom[w + 4] * 2] === cut.segmentId + 1;
        if (inside) truth++;
        if (got) exported++;
        if (inside !== got) mismatches++;
    }

    // Eviction round trip
    for (const c of loader.chunks) {
        while (c.stack.length) { const t = c.stack.pop(); c.loadedCount = t.metadata.nodeInfo.level; loader._disposeMesh(t); }
    }
    loader.update(scene.activeCamera);
    await new Promise(r => setTimeout(r, 3000));
    let hiddenWrong = 0, inSegment = 0;
    for (const m of loader.loadedNodes.values()) {
        const pos = m.getVerticesData('position'), ids = m.metadata.segmentIds;
        for (let i = 0; i < ids.length; i++) {
            const isSeg = ids[i] === cut.segmentId;
            if (isSeg) inSegment++;
            if (isSeg !== Number.isNaN(pos[3 * i])) hiddenWrong++;
        }
    }
    console.log({ selected: nSel, cut, truth, exported, mismatches, afterReload: { inSegment, hiddenWrong } });
})();

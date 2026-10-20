// Diagnostics of the splat layer inside the app: loads a file through the UI (same as gs_import_e2e.js), then
// prints camera, mesh and material state. Env: APP_URL, SAMPLE.
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log('console', m.type() + ':', m.text().slice(0, 400)); });
  page.on('pageerror', (e) => console.log('pageerror:', e.message.slice(0, 400)));
  await page.goto(process.env.APP_URL, { waitUntil: 'load', timeout: 120000 });
  await page.waitForTimeout(2000);
  await page.getByText('File', { exact: false }).first().click();
  await page.waitForTimeout(400);
  await page.locator('#menu-load-data').click();
  await page.waitForTimeout(600);
  await page.locator('input[type=file][accept*=".sog"]').setInputFiles(process.env.SAMPLE);
  await page.waitForTimeout(500);
  await page.getByText('Upload & Process', { exact: true }).click();
  await page.waitForFunction(() => window.__babylonScene && window.__babylonScene.gaussianSplats, null, { timeout: 900000, polling: 1000 });
  await page.waitForTimeout(3000);
  const d = await page.evaluate(() => {
    const s = window.__babylonScene;
    const cam = s.activeCamera;
    const mesh = s.meshes.find((m) => m.name === 'gsSplats');
    const mat = mesh && mesh.material;
    return {
      camera: { type: cam.getClassName(), mode: cam.mode, pos: cam.position.asArray(), target: cam.target.asArray(),
        radius: cam.radius, fov: cam.fov, minZ: cam.minZ, maxZ: cam.maxZ },
      mesh: mesh ? { visible: mesh.isVisible, enabled: mesh.isEnabled(), instances: mesh.thinInstanceCount,
        active: s.getActiveMeshes().data.indexOf(mesh) >= 0, pos: mesh.position.asArray(), vertices: mesh.getTotalVertices(),
        bbox: mesh.getBoundingInfo().boundingBox.extendSize.asArray(), alwaysSelect: mesh.alwaysSelectAsActiveMesh } : null,
      material: mat ? { ready: mat.isReady(mesh), alphaMode: mat.alphaMode, needAlpha: mat.needAlphaBlending(), backFace: mat.backFaceCulling } : null,
      meshCount: s.meshes.length,
      activeCount: s.getActiveMeshes().length,
      splatMedian: s.gaussianSplats ? s.gaussianSplats.n : null,
    };
  });
  console.log(JSON.stringify(d, null, 1));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });

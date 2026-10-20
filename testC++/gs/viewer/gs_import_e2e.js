// End to end: opens the app, loads a Gaussian splat file through "Load Data" > "Upload & Process", waits for the
// splats to appear in the main viewer, and writes a screenshot. Logs the relevant console lines.
// Env: APP_URL (e.g. http://localhost:8000/ in the shared network namespace), SAMPLE (path in the container), SHOT
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const log = [];
  page.on('console', (m) => {
    const t = m.text();
    if (/gs|splat|Gaussian|error|Error/i.test(t)) log.push(`[${m.type()}] ${t.slice(0, 240)}`);
  });
  page.on('pageerror', (e) => log.push(`[pageerror] ${e.message.slice(0, 300)}`));
  await page.goto(process.env.APP_URL, { waitUntil: 'load', timeout: 120000 });
  await page.waitForTimeout(2000);

  await page.getByText('File', { exact: false }).first().click();
  await page.waitForTimeout(400);
  await page.locator('#menu-load-data').click();
  await page.waitForTimeout(600);
  await page.locator('input[type=file][accept*=".sog"]').setInputFiles(process.env.SAMPLE);
  await page.waitForTimeout(500);
  await page.getByText('Upload & Process', { exact: true }).click();

  // the splat layer registers itself on the scene (EXPECT=gs, the default), or the point cloud loader (EXPECT=pc)
  const expectPc = process.env.EXPECT === 'pc';
  try {
    await page.waitForFunction((pc) => window.__babylonScene && (pc ? !!window.__babylonScene.pointCloudLoader
      : !!window.__babylonScene.gaussianSplats), expectPc, { timeout: 1500000, polling: 1000 });
    console.log(expectPc ? 'point cloud loader registered' : 'splat layer registered');
  } catch (e) {
    console.log('not registered:', e.message.split('\n')[0]);
  }
  await page.waitForTimeout(4000);
  const state = await page.evaluate(() => {
    const s = window.__babylonScene;
    const g = s && s.gaussianSplats;
    return { splats: g ? { n: g.n, kind: g.kind, sh: g.shDegree, loadMs: Math.round(g.loadMs), stats: g.stats() } : null,
      pointCloud: !!(s && s.pointCloudLoader), meshes: s ? s.meshes.length : 0 };
  });
  console.log('state:', JSON.stringify(state));
  // software rendering of ~1M splats is slow: the screenshot waits for a frame, so allow a long timeout
  await page.screenshot({ path: process.env.SHOT || '/shots/app_gs.png', timeout: 600000 });
  console.log(log.slice(0, 60).join('\n'));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });

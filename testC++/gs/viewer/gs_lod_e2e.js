// LOD check in the app: loads a splat file through the UI, then for several camera distances renders with LOD off and
// on (handle.setLod), and records the drawn splat count and sort time. Writes one screenshot per case.
// Env: APP_URL, SAMPLE, RADII (comma separated camera radii), OUT_PREFIX (screenshot prefix in /shots)
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.on('pageerror', (e) => console.log('pageerror:', e.message.slice(0, 300)));
  page.on('console', (m) => { if (m.type() === 'error') console.log('console error:', m.text().slice(0, 300)); });
  await page.goto(process.env.APP_URL, { waitUntil: 'load', timeout: 120000 });
  await page.waitForTimeout(2000);
  await page.getByText('File', { exact: false }).first().click();
  await page.waitForTimeout(400);
  await page.locator('#menu-load-data').click();
  await page.waitForTimeout(600);
  await page.locator('input[type=file][accept*=".sog"]').setInputFiles(process.env.SAMPLE);
  await page.waitForTimeout(500);
  await page.getByText('Upload & Process', { exact: true }).click();
  await page.waitForFunction(() => window.__babylonScene && window.__babylonScene.gaussianSplats, null, { timeout: 1500000, polling: 1000 });
  console.log('loaded:', await page.evaluate(() => JSON.stringify(window.__babylonScene.gaussianSplats.stats())));

  const radii = (process.env.RADII || '1.2,6,25').split(',').map(Number);
  const prefix = process.env.OUT_PREFIX || 'lod';
  for (const r of radii) {
    for (const lod of [false, true]) {
      const res = await page.evaluate(async ({ r, lod }) => {
        const g = window.__babylonScene.gaussianSplats;
        window.__babylonScene.activeCamera.radius = r;
        const before = g.stats().sortCount;
        g.setLod(lod);
        // wait until a sort started after the change has arrived, and the drawn count has settled
        const t0 = performance.now();
        let last = -1, stable = 0;
        while (performance.now() - t0 < 300000) {
          await new Promise((res) => setTimeout(res, 700));
          const st = g.stats();
          if (st.sortCount <= before) continue;
          stable = st.drawn === last ? stable + 1 : 0;
          last = st.drawn;
          if (stable >= 3) break;
        }
        return g.stats();
      }, { r, lod });
      const name = `${prefix}_r${r}_${lod ? 'lod' : 'full'}.png`;
      await page.screenshot({ path: `/shots/${name}`, timeout: 900000 });
      console.log(`radius ${r} lod ${lod ? 'on ' : 'off'}: drawn ${res.drawn} of ${res.total}, sort ${res.sortMs.toFixed(1)} ms, screenshot ${name}`);
    }
  }
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });

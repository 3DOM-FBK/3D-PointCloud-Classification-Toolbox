// Debug of the LOD selection inside the app: loads the sample through the UI and prints chunk data and the choice.
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.on('pageerror', (e) => console.log('pageerror:', e.message.slice(0, 300)));
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
  await page.waitForTimeout(4000);
  const d = await page.evaluate(() => window.__babylonScene.gaussianSplats.debug());
  console.log(JSON.stringify(d.chunks.map((c) => ({ start: c.start, count: c.count, centre: c.centre.map((v) => +v.toFixed(3)), radius: +c.radius.toFixed(3), size: +c.size.toExponential(2) }))));
  console.log(JSON.stringify({ keep: d.state.keep, frac: d.state.frac.map((f) => +f.toFixed(3)), focal: d.state.focal, fovy: d.state.fovy, aspect: d.state.aspect, view: d.state.view }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });

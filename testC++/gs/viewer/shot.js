// Opens the test renderer in headless Chromium (software WebGL2), waits for the decode and
// writes a screenshot. Prints the viewer statistics and the decode checksum as JSON.
// Env: PAGE_URL, SHOT (output png, optional), W, H, YAW, PITCH, DIST, STRIDE, TIMEOUT_MS
const { chromium } = require('playwright');

(async () => {
  const W = +(process.env.W || 900), H = +(process.env.H || 600);
  const q = new URLSearchParams();
  for (const k of ['YAW', 'PITCH', 'DIST']) if (process.env[k]) q.set(k.toLowerCase(), process.env[k]);
  q.set('w', String(W)); q.set('h', String(H));
  if (process.env.SET) q.set('set', process.env.SET);
  if (process.env.NOSORT) q.set('nosort', '1');
  if (process.env.STRIDE) q.set('stride', process.env.STRIDE);
  if (process.env.NODRAW) q.set('nodraw', '1');
  if (process.env.BENCH) q.set('bench', process.env.BENCH);
  if (process.env.SPIN) q.set('spin', process.env.SPIN);
  const url = `${process.env.PAGE_URL}?${q.toString()}`;

  const browser = await chromium.launch({
    args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
  });
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log('console:', m.text()); });
  page.on('pageerror', (e) => console.log('pageerror:', e.message));
  await page.goto(url);
  await page.waitForFunction(() => window.__viewer && window.__viewer.ready, null, {
    timeout: +(process.env.TIMEOUT_MS || 900000), polling: 500,
  });
  const v = await page.evaluate(() => window.__viewer);
  if (v.error) { console.log('viewer error:', v.error); await browser.close(); process.exit(1); }
  await page.waitForTimeout(+(process.env.SETTLE_MS || 1500));
  await page.evaluate(() => window.__viewer.redraw());
  await page.waitForTimeout(1500);
  if (process.env.SPIN) {
    await page.waitForFunction(() => window.__viewer.spinResult, null, { timeout: 900000, polling: 500 });
    console.log('spin:', JSON.stringify(await page.evaluate(() => window.__viewer.spinResult)));
  }
  if (process.env.BENCH) {
    await page.waitForFunction(() => window.__viewer.benchResult, null, { timeout: 900000, polling: 500 });
    const b = await page.evaluate(() => window.__viewer.benchResult);
    console.log('bench:', JSON.stringify(b));
  }
  const stats = await page.evaluate(() => window.__viewer.stats());
  if (process.env.SHOT) await page.screenshot({ path: process.env.SHOT, timeout: 240000 });
  console.log(JSON.stringify({ n: v.n, kind: v.kind, shDegree: v.shDegree, loadMs: Math.round(v.loadMs),
    radius: v.radius, median: v.median, checksum: v.checksum, stats }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });

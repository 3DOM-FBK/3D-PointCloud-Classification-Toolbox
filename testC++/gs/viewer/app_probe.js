// Probe for the app: opens "Load Data" and lists the file inputs and buttons of the dialog.
// Env: APP_URL (e.g. http://localhost:8000/ inside the shared network namespace)
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.on('pageerror', (e) => console.log('pageerror:', e.message.slice(0, 300)));
  await page.goto(process.env.APP_URL, { waitUntil: 'load', timeout: 120000 });
  await page.waitForTimeout(2000);
  await page.getByText('File', { exact: false }).first().click();
  await page.waitForTimeout(500);
  await page.locator('#menu-load-data').click();
  await page.waitForTimeout(1500);
  const info = await page.evaluate(() => ({
    inputs: [...document.querySelectorAll('input[type=file]')].map((i) => ({ id: i.id, accept: i.accept, cls: i.className })),
    buttons: [...document.querySelectorAll('button, [role=button]')]
      .filter((b) => b.offsetParent !== null)
      .map((b) => (b.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 40)).filter((t) => t).slice(0, 60),
  }));
  console.log(JSON.stringify(info, null, 1));
  await page.screenshot({ path: '/shots/app_load_dialog.png' });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });

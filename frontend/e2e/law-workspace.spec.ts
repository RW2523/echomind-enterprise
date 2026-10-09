import { expect, Page, test } from '@playwright/test';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/** EchoMind Law end-to-end (spec test T01 in a real browser) + design-review screenshots. */
const HERE = path.dirname(fileURLToPath(import.meta.url));   // the frontend is an ES module package
const DATA = path.resolve(HERE, '../../.law-data-e2e');
const SHOTS = path.resolve(HERE, '../../docs/echomind-law/design/screens');
const creds = (): Record<string, { password: string; totp_secret: string }> =>
  JSON.parse(fs.readFileSync(path.join(DATA, 'legal/demo_credentials.json'), 'utf8'));

function totp(secretB32: string): string {
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567';
  let bits = '';
  for (const ch of secretB32.replace(/=+$/, '')) bits += alphabet.indexOf(ch).toString(2).padStart(5, '0');
  const key = Buffer.from(bits.match(/.{8}/g)!.map((b) => parseInt(b, 2)));
  const counter = Buffer.alloc(8);
  counter.writeBigUInt64BE(BigInt(Math.floor(Date.now() / 1000 / 30)));
  const h = crypto.createHmac('sha1', key).update(counter).digest();
  const o = h[h.length - 1] & 0xf;
  return String((h.readUInt32BE(o) & 0x7fffffff) % 1_000_000).padStart(6, '0');
}

async function signIn(page: Page, name: string) {
  const c = creds()[name];
  await page.goto('/signin');
  await page.getByLabel('Username').fill(name);
  await page.getByLabel('Password').fill(c.password);
  await page.getByRole('button', { name: 'Sign in' }).click();
  await page.getByLabel('Authenticator code').fill(totp(c.totp_secret));
  await page.getByRole('button', { name: 'Verify' }).click();
  await page.waitForURL('**/home');
}

async function shot(page: Page, name: string) {
  fs.mkdirSync(SHOTS, { recursive: true });
  await page.mouse.move(1439, 899);             // no hover state in review screenshots
  await page.waitForTimeout(250);
  await page.screenshot({ path: path.join(SHOTS, `${name}.png`) });
}

test('alice sees only her matters; a matter she is not on is simply not available', async ({ page }) => {
  await page.goto('/signin');
  await expect(page.getByText('Private legal workspace')).toBeVisible();
  await shot(page, '01-sign-in-light');
  await signIn(page, 'alice');
  await expect(page.getByRole('heading', { name: 'Your open matters' })).toBeVisible();
  await shot(page, '02-home-light');

  await page.getByRole('link', { name: 'Matters', exact: true }).click();
  const rows = page.locator('table.lw-table tbody tr');
  await expect(rows).toHaveCount(2);
  await expect(page.getByText('Harrowgate MSA renewal')).toBeVisible();
  await expect(page.getByText('Series B due diligence')).toBeVisible();
  for (const hidden of ['M-202', 'Board investigation', 'M-215', 'Beta Shipping']) {
    await expect(page.getByText(hidden)).toHaveCount(0);
  }
  await shot(page, '03-matters-light');

  await page.goto('/m/M-202/overview');       // walled-off / expired: never visible
  await expect(page.getByText("This item isn't available.")).toBeVisible();
  const hiddenText = await page.locator('main').innerText();
  await page.goto('/m/DOES-NOT-EXIST/overview');
  await expect(page.getByText("This item isn't available.")).toBeVisible();
  expect(await page.locator('main').innerText()).toBe(hiddenText);   // indistinguishable
  await shot(page, '04-not-available-light');

  await page.goto('/m/M-101/overview');
  await expect(page.getByRole('heading', { name: 'Harrowgate MSA renewal' })).toBeVisible();
  await shot(page, '05-matter-overview-light');
  await page.getByRole('link', { name: 'Members' }).first().click();
  await expect(page.getByText('carol')).toBeVisible();
  await shot(page, '06-matter-members-light');

  await page.getByRole('button', { name: /Switch matter/ }).click();
  const switcher = page.getByRole('listbox', { name: 'Your matters' });
  await expect(switcher.getByRole('option')).toHaveCount(2);         // only alice's own matters
  await shot(page, '07-matter-switcher-light');
  await switcher.getByRole('option', { name: /M-114/ }).click();
  await page.waitForURL('**/m/M-114/overview');

  await page.emulateMedia({ colorScheme: 'dark' });
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Series B due diligence' })).toBeVisible();
  await shot(page, '08-matter-overview-dark');
  await page.goto('/matters');
  await expect(rows).toHaveCount(2);
  await shot(page, '09-matters-dark');
});

test('bob sees the restricted matter that alice cannot', async ({ page }) => {
  await signIn(page, 'bob');
  await page.goto('/matters');
  await expect(page.getByText('Board investigation')).toBeVisible();
  await expect(page.getByText('Highly restricted')).toBeVisible();
  await expect(page.getByText('Harrowgate MSA renewal')).toHaveCount(0);
});

test('signing out ends the session for every tab', async ({ page, context }) => {
  await signIn(page, 'carol');
  const other = await context.newPage();
  await other.goto('/matters');
  await expect(other.getByText('Harrowgate MSA renewal')).toBeVisible();
  await page.getByRole('button', { name: /Sign out/ }).click();
  await page.waitForURL('**/signin');
  await other.reload();
  await other.waitForURL('**/signin**');
});

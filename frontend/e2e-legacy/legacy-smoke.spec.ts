import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';

/** T14: the standard edition's interface still works — its own app (not the Law workspace), every
 *  view, Knowledge Chat answering from a document with the real model, and each vertical. */
const ROOT = new URL('../../', import.meta.url);
const password = () => readFileSync(new URL('.law-data-legacy/dev_admin_password', ROOT), 'utf8').trim();
const FAILED = 'The rest of the app is still working.';          // the ErrorBoundary fallback
const DOC = 'Northwind Logistics fleet policy (fictional). Section 3. Every delivery van is serviced every 12,000 kilometres or every six months, whichever comes first.';
const SHOTS = new URL('docs/echomind-law/design/screens/', ROOT);

function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  return errors;
}

async function signIn(page: Page) {
  await page.goto('/');
  await page.getByLabel('Username').fill('admin');
  await page.getByLabel('Password').fill(password());
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page.getByRole('button', { name: 'Knowledge Chat' }).first()).toBeVisible({ timeout: 30_000 });
}

test('the standard edition opens its own interface, and every view renders', async ({ page }) => {
  const errors = watchErrors(page);
  await signIn(page);
  await expect(page.getByText('Matters', { exact: true })).toHaveCount(0);           // not the Law workspace
  for (const view of ['Live Transcript', 'Conversation', 'Document Studio', 'Settings', 'Knowledge Chat']) {
    await page.getByRole('button', { name: view }).first().click();
    await page.waitForTimeout(800);
    await expect(page.getByText(FAILED)).toHaveCount(0);
  }
  expect(errors).toEqual([]);
});

test('Knowledge Chat answers from an uploaded document with the real model', async ({ page, request }) => {
  const errors = watchErrors(page);
  const login = await request.post('http://127.0.0.1:8130/api/auth/login', { data: { username: 'admin', password: password() } });
  const token = (await login.json()).token as string;
  const up = await request.post('http://127.0.0.1:8130/api/docs/upload', {
    headers: { Authorization: `Bearer ${token}` },
    multipart: { file: { name: 'northwind-fleet-policy.txt', mimeType: 'text/plain', buffer: Buffer.from(DOC) } },
  });
  expect(up.ok()).toBeTruthy();
  await signIn(page);
  await page.reload();               // pre-existing defect (see the test below): the first chat only loads on a signed-in page load
  const box = page.getByPlaceholder(/Ask /).first();
  const sendButton = page.getByRole('button', { name: 'Send', exact: true });
  await expect(sendButton).toBeEnabled({ timeout: 30_000 });
  await box.fill('How often is a Northwind delivery van serviced?');
  await sendButton.click();
  await expect(box).toHaveValue('');
  await expect(page.getByText(/12,000/).last()).toBeVisible({ timeout: 180_000 });
  await expect(sendButton).toHaveText('Send', { timeout: 180_000 });          // the answer has finished
  await page.mouse.move(0, 0);
  await page.screenshot({ path: new URL('20-legacy-knowledge-chat.png', SHOTS).pathname });
  expect(errors).toEqual([]);
});

test('KNOWN DEFECT, also on main: right after signing in, Knowledge Chat cannot send until a reload', async ({ page }) => {
  // useKnowledgeChat() loads or creates the first chat once, when the app mounts — before the sign-in
  // gate — so with sign-in required that request fails and Send stays disabled until the page is
  // reloaded or "New chat" is pressed. Not introduced by the Law work (same code on main). Marked as
  // an expected failure: Playwright reports it the day it is fixed.
  test.fail();
  await signIn(page);
  await page.getByRole('button', { name: 'Knowledge Chat' }).first().click();
  await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeEnabled({ timeout: 15_000 });
});

test('Document Studio lists its templates and Live Transcript offers Boardroom', async ({ page }) => {
  await signIn(page);
  await page.getByRole('button', { name: 'Document Studio' }).first().click();
  await expect(page.getByText(FAILED)).toHaveCount(0);
  await expect(page.getByText(/template/i).first()).toBeVisible({ timeout: 30_000 });
  await page.getByRole('button', { name: 'Live Transcript' }).first().click();
  await page.getByRole('button', { name: 'Start', exact: true }).first().click();          // opens the start dialog only
  await expect(page.getByText('Boardroom mode')).toBeVisible({ timeout: 30_000 });
  const boardroom = page.getByRole('switch');
  await boardroom.click();
  await expect(boardroom).toHaveAttribute('aria-checked', 'true');
  await boardroom.click();
  await page.getByRole('button', { name: 'Cancel' }).click();                               // nothing is recorded
});

for (const vertical of ['health', 'meetings', 'law', 'retail', 'bank']) {
  test(`the ${vertical} vertical renders`, async ({ page }) => {
    const errors = watchErrors(page);
    await page.goto(`/?vertical=${vertical}`);
    await page.waitForLoadState('networkidle');
    await expect(page.getByText(FAILED)).toHaveCount(0);
    expect((await page.locator('body').innerText()).trim().length).toBeGreaterThan(20);
    expect(errors).toEqual([]);
  });
}

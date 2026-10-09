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
  for (let attempt = 0; attempt < 2; attempt++) {
    await page.getByLabel('Authenticator code').fill(totp(c.totp_secret));
    await page.getByRole('button', { name: 'Verify' }).click();
    const outcome = await Promise.race([
      page.waitForURL('**/home').then(() => 'ok'),
      page.getByRole('alert').waitFor().then(() => 'refused'),
    ]);
    if (outcome === 'ok') return;
    // A code is single-use: if this person already signed in during this 30 s step, wait for the next one.
    await page.waitForTimeout(31_000 - (Date.now() % 30_000));
  }
  throw new Error(`could not sign ${name} in`);
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

test('a wall approved by a second managing partner hides a matter from its own lead', async ({ browser }) => {
  test.setTimeout(120_000);
  const morgan = await (await browser.newContext()).newPage();
  await signIn(morgan, 'morgan');
  await morgan.goto('/admin/walls');
  await morgan.getByRole('button', { name: 'New wall' }).click();
  await morgan.getByLabel('Name').fill('Series B conflict screen');
  await morgan.getByLabel('Rationale').fill('Alice previously advised the investor.');
  await morgan.getByLabel('Screened people').selectOption('alice');
  await morgan.getByLabel('Protect').fill('M-114');
  await morgan.getByRole('button', { name: 'Matter', exact: true }).click();
  await morgan.getByRole('button', { name: 'Request wall' }).click();
  await expect(morgan.getByText('Awaiting approval')).toBeVisible();
  await expect(morgan.getByRole('button', { name: 'Approve' })).toHaveCount(0);       // not by the requester

  const nakamura = await (await browser.newContext()).newPage();
  await signIn(nakamura, 'nakamura');
  await nakamura.goto('/admin/walls');
  await nakamura.getByRole('button', { name: 'Approve' }).click();
  await expect(nakamura.getByText('In force')).toBeVisible();
  await shot(nakamura, '10-ethical-walls-light');

  const alice = await (await browser.newContext()).newPage();
  await signIn(alice, 'alice');
  await alice.goto('/matters');
  await expect(alice.getByText('Harrowgate MSA renewal')).toBeVisible();
  await expect(alice.getByText('Series B due diligence')).toHaveCount(0);            // she leads it, but is walled
  await alice.goto('/m/M-114/overview');
  await expect(alice.getByText("This item isn't available.")).toBeVisible();
});

test('emergency access: requested by a firm admin, approved by a managing partner, read only', async ({ browser }) => {
  const fiona = await (await browser.newContext()).newPage();
  await signIn(fiona, 'fiona');
  await fiona.goto('/admin/emergency-access');
  await fiona.getByLabel('Matter number').fill('M-215');
  await fiona.getByLabel('Reason').fill('Lead unreachable; landlord deadline today');
  await fiona.getByRole('button', { name: /Request emergency access/ }).click();
  await expect(fiona.getByText('Awaiting approval')).toBeVisible();

  const morgan = await (await browser.newContext()).newPage();
  await signIn(morgan, 'morgan');
  await morgan.goto('/admin/emergency-access');
  await expect(morgan.getByText('Warehouse lease dispute · Beta Shipping Corp')).toBeVisible();
  await morgan.getByRole('button', { name: 'Approve' }).click();
  await expect(morgan.getByText('Active')).toBeVisible();

  await fiona.goto('/m/M-215/overview');
  await expect(fiona.getByText('Emergency access.')).toBeVisible();
  await expect(fiona.getByRole('button', { name: 'Edit details' })).toHaveCount(0);
  await shot(fiona, '11-emergency-access-banner-light');
  await fiona.getByRole('button', { name: 'End now' }).click();
  await fiona.reload();
  await expect(fiona.getByText("This item isn't available.")).toBeVisible();
});

test('the audit trail shows the emergency access, intact, and the lead sees it on the matter', async ({ browser }) => {
  test.setTimeout(120_000);
  const morgan = await (await browser.newContext()).newPage();
  await signIn(morgan, 'morgan');
  await morgan.goto('/admin/audit');
  await morgan.getByLabel('Activity').selectOption('breakglass');
  await expect(morgan.getByText(/asked for emergency access to M-215/)).toBeVisible();
  await expect(morgan.getByText(/approved emergency access to M-215/)).toBeVisible();
  await morgan.getByLabel('Activity').selectOption('');
  const integrity = morgan.getByRole('region', { name: 'Integrity' });
  await integrity.getByRole('button', { name: 'Check now' }).click();
  await expect(integrity.getByText('Intact')).toBeVisible();
  await shot(morgan, '12-audit-trail-light');

  const bob = await (await browser.newContext()).newPage();
  await signIn(bob, 'bob');
  await bob.goto('/m/M-215/activity');
  await expect(bob.getByText(/fiona viewed M-215/).first()).toBeVisible();
  await expect(bob.getByText('Emergency access').first()).toBeVisible();
  await shot(bob, '13-matter-activity-light');

  const alice = await (await browser.newContext()).newPage();
  await signIn(alice, 'alice');
  await alice.goto('/admin/audit');
  await expect(alice.getByText("This item isn't available.")).toBeVisible();
});

test('documents: imported into M-101, read in the background, searchable only inside M-101', async ({ browser }) => {
  test.setTimeout(240_000);
  const alice = await (await browser.newContext()).newPage();
  await signIn(alice, 'alice');
  await alice.goto('/m/M-101/documents');
  await alice.getByRole('button', { name: /Import a document/ }).click();
  const dialog = alice.getByRole('dialog', { name: 'Import a document' });
  await dialog.getByLabel('File').setInputFiles(path.resolve(HERE, '../../eval/fixtures/law/Master Services Agreement (DEMO).md'));
  await dialog.getByRole('button', { name: /^Import$/ }).click();
  await expect(dialog).toHaveCount(0);
  await expect(alice.getByText('Searchable')).toBeVisible({ timeout: 180_000 });     // real embeddings, real worker
  await alice.getByLabel('Search this matter’s documents').fill('cap on aggregate liability');
  await alice.getByRole('button', { name: 'Search', exact: true }).click();
  await expect(alice.getByText(/total aggregate liability/).first()).toBeVisible({ timeout: 60_000 });
  await alice.getByRole('button', { name: 'Master Services Agreement (DEMO)', exact: true }).click();
  await expect(alice.getByRole('complementary', { name: 'Document details' }).getByText('Versions')).toBeVisible();
  await shot(alice, '14-documents-light');

  const bob = await (await browser.newContext()).newPage();
  await signIn(bob, 'bob');
  await bob.goto('/m/M-101/documents');
  await expect(bob.getByText("This item isn't available.")).toBeVisible();
});

test('research: an answer from M-101 only, tied to the clause it relies on', async ({ browser }) => {
  test.setTimeout(240_000);
  const alice = await (await browser.newContext()).newPage();
  await signIn(alice, 'alice');
  await alice.goto('/m/M-101/research');
  await alice.getByLabel('Ask about this matter').fill('What is the cap on each party’s aggregate liability?');
  await alice.getByRole('button', { name: 'Ask' }).click();
  const answer = alice.getByRole('article', { name: 'Answer' });
  const sources = answer.getByRole('region', { name: 'Sources' });
  await expect(sources).toBeVisible({ timeout: 180_000 });                  // real model, real retrieval
  await expect(sources.getByText(/aggregate liability/).first()).toBeVisible();
  await expect(answer.getByRole('button', { name: /^Source \d$/ }).first()).toBeVisible();
  await expect(answer.getByText(/250,000/).first()).toBeVisible();
  await shot(alice, '15-research-light');
  await alice.getByLabel('Ask about this matter').fill('Which merger did the board of Beta Shipping approve?');
  await alice.getByRole('button', { name: 'Ask' }).click();
  await expect(alice.getByText('The documents in this matter do not answer that.')).toBeVisible({ timeout: 120_000 });
  await expect(alice.getByRole('complementary', { name: 'Your earlier questions' }).getByText(/aggregate liability/)).toBeVisible();
});

test('meetings: recording waits for everyone’s consent and stops the moment one is withdrawn', async ({ browser }) => {
  test.setTimeout(180_000);
  const alice = await (await browser.newContext()).newPage();
  await signIn(alice, 'alice');
  await alice.goto('/m/M-101/meetings');
  await alice.getByRole('button', { name: /Plan a meeting/ }).click();
  await alice.getByRole('dialog', { name: 'Plan a meeting' }).getByLabel('Title').fill('Call with Voss-Lindqvist');
  await alice.getByRole('button', { name: 'Create' }).click();
  await alice.getByRole('link', { name: 'Call with Voss-Lindqvist' }).click();
  await alice.getByRole('tab', { name: 'Prepare' }).click();
  for (const [name, role] of [['Alice Hart', 'host'], ['Petra Voss', 'client']]) {
    await alice.getByLabel('Name').fill(name);
    await alice.getByLabel('Role').selectOption(role);
    await alice.getByRole('button', { name: 'Add' }).click();
    await expect(alice.getByText(name, { exact: true })).toBeVisible();
  }
  await alice.getByRole('tab', { name: 'Record' }).click();
  await expect(alice.getByText('Still needed from: Alice Hart, Petra Voss.')).toBeVisible();
  await expect(alice.getByRole('button', { name: /Start recording/ })).toBeDisabled();
  await alice.getByRole('tab', { name: /Consent/ }).click();
  for (const name of ['Alice Hart', 'Petra Voss']) {
    await alice.getByRole('row', { name: new RegExp(name) }).getByRole('button', { name: 'Record consent' }).click();
    const dialog = alice.getByRole('dialog', { name: `Consent of ${name}` });
    await dialog.getByRole('checkbox').check();
    await dialog.getByRole('button', { name: 'Record consent' }).click();
    await expect(dialog).toHaveCount(0);
  }
  await alice.getByRole('tab', { name: 'Record' }).click();
  await alice.getByRole('button', { name: /Start recording/ }).click();
  await expect(alice.getByRole('status').getByText(/^Recording \d\d:\d\d/)).toBeVisible({ timeout: 20_000 });
  await alice.waitForTimeout(1500);                                         // a little synthetic audio
  await shot(alice, '16-meeting-recording-light');

  const carol = await (await browser.newContext()).newPage();
  await signIn(carol, 'carol');
  await carol.goto(new URL(alice.url()).pathname);
  await carol.getByRole('row', { name: /Petra Voss/ }).getByRole('button', { name: 'Withdraw' }).click();
  await expect(alice.getByText(/a participant withdrew consent\. Nothing after that moment was captured\./)).toBeVisible({ timeout: 10_000 });
  await expect(alice.getByRole('region', { name: 'Recording history' }).getByText(/audio receipt/)).toBeVisible();
  await expect(alice.getByText('Still needed from: Petra Voss.')).toBeVisible();
});

test('records: a label, a deletion approved by records and verified, and a two-person legal hold', async ({ browser }) => {
  test.setTimeout(180_000);
  const alice = await (await browser.newContext()).newPage();
  await signIn(alice, 'alice');
  await alice.goto('/m/M-101/documents');
  await alice.getByRole('button', { name: 'Master Services Agreement (DEMO)', exact: true }).click();
  const panel = alice.getByRole('complementary', { name: 'Document details' });
  await panel.getByLabel('Set label').selectOption('confidential');
  await expect(panel.getByText(/set by alice/)).toBeVisible();
  await panel.getByRole('button', { name: /Delete…/ }).click();
  const dialog = alice.getByRole('dialog', { name: /Delete “Master Services Agreement \(DEMO\)”/ });
  await expect(dialog.getByText(/searchable passages?/)).toBeVisible();
  await dialog.getByLabel('Reason').fill('Superseded by the signed copy');
  await dialog.getByRole('button', { name: /Request deletion/ }).click();
  await expect(alice.getByRole('region', { name: 'Waiting for deletion approval' })).toBeVisible();

  const rhea = await (await browser.newContext()).newPage();
  await signIn(rhea, 'rhea');
  await rhea.goto('/admin/records');
  const card = rhea.getByRole('region', { name: 'Master Services Agreement (DEMO)' });
  await expect(card.getByText(/research answers? quoting it/)).toBeVisible();
  await shot(rhea, '17-records-deletion-light');
  await card.getByRole('button', { name: 'Approve deletion' }).click();
  await expect(rhea.getByText('Nothing left')).toBeVisible({ timeout: 20_000 });

  await rhea.getByRole('tab', { name: 'Legal holds' }).click();
  await rhea.getByLabel('Name').fill('Beta Shipping litigation hold');
  await rhea.getByLabel('Matters').fill('M-215');
  await rhea.getByRole('button', { name: /\+ M-215/ }).click();
  await rhea.getByLabel('Reason').fill('Claim anticipated from the landlord');
  await rhea.getByRole('button', { name: /Request hold/ }).click();
  await expect(rhea.getByText('Awaiting approval')).toBeVisible();
  await expect(rhea.getByRole('button', { name: 'Approve hold' })).toHaveCount(0);

  const morgan = await (await browser.newContext()).newPage();
  await signIn(morgan, 'morgan');
  await morgan.goto('/admin/records');
  await morgan.getByRole('tab', { name: 'Legal holds' }).click();
  await morgan.getByRole('button', { name: 'Approve hold' }).click();
  await expect(morgan.getByText('In force')).toBeVisible();

  const bob = await (await browser.newContext()).newPage();
  await signIn(bob, 'bob');
  await bob.goto('/m/M-215/overview');
  await expect(bob.getByText(/Nothing in this matter can be deleted while the hold is in force/)).toBeVisible();
});

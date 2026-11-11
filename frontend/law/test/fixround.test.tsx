/** The 2026-11-12 fix round's small rules: each person's home and rail, refusals and limits in words, devices by name. */
import { describe, expect, it } from 'vitest';
import { failureText, NOT_YOUR_ROLE } from '../api/client';
import { firmEntries } from '../app/nav';
import { homeFor, isLawyer, mayDo } from '../app/session';
import type { Me } from '../api/types';
import { device, stamp, where } from '../pages/Security';

const me = (roles: string[], can?: string[]): Me => ({
  user: { id: 'u', username: 'x', role: 'user', roles: roles as Me['user']['roles'], display_name: 'X Person' },
  session: { auth: 'pwd+mfa', mfa_enrolled: true, must_change_password: false, idle_expires_at: 0, abs_expires_at: 0 },
  features: { intake: true, intake_links: false, regulatory: true }, ...(can ? { can } : {}),
});

describe('each person lands on their own work', () => {
  it('lawyers on the chat, the front desk and conflicts on Enquiries, records on Records, the administrator on Users', () => {
    expect(homeFor(me(['attorney']))).toBe('/chat');
    expect(homeFor(me(['intake'], ['intake.manage']))).toBe('/enquiries');
    expect(homeFor(me(['conflicts'], ['conflicts.review']))).toBe('/enquiries');
    expect(homeFor(me(['records_admin'], ['audit.read.org', 'deletion.review', 'holds.read']))).toBe('/admin/records');
    expect(homeFor(me(['firm_admin'], ['users.manage', 'users.read', 'audit.read.org']))).toBe('/admin/users');
    expect(isLawyer(me(['firm_admin']))).toBe(false);
  });

  it('the menus offer only what the server says the person may do', () => {
    const admin = firmEntries(me(['firm_admin'], ['users.manage', 'users.read', 'connectors.manage', 'breakglass.request', 'audit.read.org']));
    const keys = admin.map((e) => e.key);
    expect(keys).toEqual(expect.arrayContaining(['users', 'connectors', 'audit', 'emergency']));
    expect(keys).not.toContain('chat');
    expect(keys).not.toContain('regulatory');                 // the server refuses the administrator regulatory updates
    expect(keys).not.toContain('enquiries');
    expect(mayDo(me(['attorney']), ['regulatory.read'], ['attorney'])).toBe(true);          // an older server: the role rule
  });
});

describe('failures in words', () => {
  it('a general refusal, a list of field problems, and a busy model', () => {
    expect(failureText(403, { detail: "You don't have permission to do that." })).toBe(NOT_YOUR_ROLE);
    expect(failureText(403, { detail: 'This access is read-only.' })).toBe('This access is read-only.');
    expect(failureText(422, { detail: [{ type: 'string_too_long', ctx: { max_length: 2000 } }] })).toMatch(/at most 2000 characters/);
    expect(failureText(429, {}, '12')).toBe('Too many questions in the last minute. Try again in 12 seconds.');
  });
});

describe('devices and times', () => {
  it('a browser string read as a device, a private address as this network, a time with its zone', () => {
    expect(device('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36')).toBe('Chrome on Linux');
    expect(device('Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Version/17.0 Mobile/15E148 Safari/604.1')).toBe('Safari on iOS');
    expect(where('172.28.0.1')).toBe('This network');
    expect(where('203.0.113.9')).toBe('203.0.113.9');
    expect(stamp('2026-11-11T14:05:00Z')).toMatch(/\d{4}.*[A-Z]{2,5}|GMT|UTC/);
  });
});

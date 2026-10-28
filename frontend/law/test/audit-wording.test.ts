import { describe, expect, it } from 'vitest';
import type { AuditEvent } from '../api/types';
import { describe as say } from '../ui/auditText';

/** Activity in plain words (design review 2026-10-28): no event codes such as "counsel.answer" on screen. */
const ev = (action: string, outcome = 'allow', matter = true): AuditEvent =>
  ({ id: 1, at: '2026-10-28T09:00:00Z', action, outcome, actor: { username: 'alice' }, details: {},
     matter: matter ? { id: 'm1', number: 'MAT-1042', title: 'Morgan v. Apex' } : null } as unknown as AuditEvent);

describe('audit wording', () => {
  it('says what happened in the chat in words', () => {
    expect(say(ev('counsel.turn'))).toBe('asked a question in chat about MAT-1042');
    expect(say(ev('counsel.answer'))).toBe('got an answer about MAT-1042');
    expect(say(ev('counsel.answer', 'error'))).toBe('did not get an answer about MAT-1042 (stopped or failed)');
    expect(say(ev('counsel.citation.open'))).toBe('opened a cited source in MAT-1042');
    expect(say(ev('research.answer'))).toBe('got a research answer about MAT-1042');
  });
  it('never shows a raw event code, even for events without their own sentence', () => {
    for (const a of ['meeting.transcript.read', 'obligation.extract', 'compare.run', 'translate.document', 'panel.facts', 'some.new_thing']) {
      const text = say(ev(a));
      expect(text).not.toContain(a);
      expect(text).not.toMatch(/\bperformed\b/);
    }
    expect(say(ev('meeting.transcript.read'))).toBe('opened a meeting transcript in MAT-1042');
    expect(say(ev('document.import', 'deny', false))).toBe('was refused (imported a document)');
  });
});

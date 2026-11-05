import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { AnswerText } from '../counsel/cards';
import type { CardContext } from '../counsel/cards';
import type { CitationRef } from '../counsel/types';

/** 2026-11-05 brief: a source quoted at clauses 4.1, 4.2 and 4.3 has one citation per place, and each sentence's marker opens
 *  the words that support that sentence, not the first quote of the source. */
const cite = (id: string, clause: string, sentence: string): CitationRef => ({
  id: `cit_${id.repeat(32).slice(0, 32)}`, n: 2, document_id: 'd1', document_title: 'Master Services Agreement', version_number: 1,
  clause, verified: true, removed: false, sentences: [sentence] });
const TEXT = 'The fee is USD 18,500 per month [2]. Invoices are payable net 30 days [2]. Overdue amounts accrue 1% interest per month [2].';
const CITES = [cite('a', '4.1', 'The fee is USD 18,500 per month [2].'), cite('b', '4.2', 'Invoices are payable net 30 days [2].'),
  cite('c', '4.3', 'Overdue amounts accrue 1% interest per month [2].')];

describe('Citations in an answer', () => {
  afterEach(() => cleanup());

  it('each sentence\'s marker opens the citation that supports that sentence', () => {
    const openCitation = vi.fn();
    const ctx = { openCitation } as unknown as CardContext;
    render(<div data-testid="a"><AnswerText text={TEXT} citations={CITES} ctx={ctx} /></div>);
    const chips = within(screen.getByTestId('a')).getAllByRole('button');
    expect(chips.map((c) => c.getAttribute('title'))).toEqual([
      'Master Services Agreement, clause 4.1', 'Master Services Agreement, clause 4.2', 'Master Services Agreement, clause 4.3']);
    chips.forEach((c) => fireEvent.click(c));
    expect(openCitation.mock.calls.map(([c]) => c.clause)).toEqual(['4.1', '4.2', '4.3']);
  });

  it('a source cited once keeps one chip per sentence as before', () => {
    const one = [{ ...CITES[0], sentences: undefined }];
    render(<div data-testid="b"><AnswerText text="The fee is USD 18,500 per month [2]." citations={one} ctx={{ openCitation: vi.fn() } as unknown as CardContext} /></div>);
    expect(within(screen.getByTestId('b')).getAllByRole('button')).toHaveLength(1);
  });
});

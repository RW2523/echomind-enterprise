import type { SummaryCite } from '../api/types';

/** Where a summary point's quote is, as a lawyer would cite it (S4.2): a transcript's time, otherwise the clause,
 *  page and line. Computed by the server from the quote's position, never by the model. */
export function citeLabel(c: SummaryCite | null): string {
  if (!c) return 'Not verified';
  if (c.timecode) return c.timecode;
  return [c.clause && `clause ${c.clause}`, c.page && `p. ${c.page}`, c.line && `l. ${c.line}`].filter(Boolean).join(', ');
}

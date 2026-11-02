/**
 * An answer as plain text for the clipboard (owner's request 2026-11-02, "Copy"). What a card shows travels with it:
 * the source numbers as the reader saw them and a list of those sources, whether each quote was verified, whether the
 * answer was only partly supported, and dates that were calculated rather than stated. Nothing is sent anywhere; the
 * text goes to the person's own clipboard.
 */
import { displayNumbers, MARK } from './cards';
import type { Envelope, SafeComponent, SourceAnswerProps } from './types';

function stateLine(p: SourceAnswerProps): string | null {
  const verified = p.citations.filter((c) => c.verified && !c.removed).length;
  if (p.state === 'abstained') return "No source found in this matter's documents.";
  if (p.state === 'unverified' || (p.state === 'answered' && verified === 0)) return 'Not verified: no quote could be matched to the documents.';
  if (p.state === 'partial') return 'Partly supported: some sentences have no verified source.';
  return null;
}

export function plainAnswer(p: SourceAnswerProps): string {
  const labels = displayNumbers(p.answer, p.citations);
  const known = new Set(p.citations.map((c) => c.n));
  const out: string[] = [];
  if (p.state === 'abstained') {
    out.push(p.absence ?? 'The documents in this matter do not answer that.');
  } else {
    const marked = new RegExp(`([ \\t]*)${MARK.source}`, 'g');      // a marker without a source goes with its space
    const text = p.answer.replace(/\*\*(.+?)\*\*/g, '$1').replace(marked, (_, space: string, list: string) => {
      const ns = [...new Set(list.split(',').map((x) => Number(x.trim())).filter((n) => known.has(n)).map((n) => labels.get(n) ?? n))];
      return ns.length ? `${space}[${ns.join(', ')}]` : '';
    }).replace(/(\[[\d, ]+\])(?:\1)+/g, '$1');
    out.push(text.trim());
  }
  const state = stateLine(p);
  if (state) out.push(state);
  if (p.calculation) {
    const c = p.calculation;
    out.push([`Calculated, not stated in ${c.document_title}:`, ...c.derived.map((d) => `- ${d.label}: ${d.display}`),
              ...c.assumptions.map((a) => `- Assumption: ${a}`)].join('\n'));
  }
  p.notes.forEach((n) => out.push(`Note: ${n}`));
  const sources = Array.from(new Map(p.citations.map((c) => [c.n, c])).values())
    .sort((a, b) => (labels.get(a.n) ?? a.n) - (labels.get(b.n) ?? b.n));
  if (sources.length) {
    out.push(['Sources:', ...sources.map((c) => {
      const where = [c.matter, c.removed ? 'Deleted document' : c.document_title, c.page ? `p. ${c.page}` : '', c.clause ? `cl. ${c.clause}` : '']
        .filter(Boolean).join(' · ');
      return `[${labels.get(c.n) ?? c.n}] ${where} (${c.removed ? 'no longer available' : c.verified ? 'verified quote' : 'quote not verified'})`;
    })].join('\n'));
  }
  return out.join('\n\n');
}

function plainComponent(c: SafeComponent): string {
  switch (c.type) {
    case 'source_answer': return plainAnswer(c.props);
    case 'priority_list':
      return [c.props.summary, ...c.props.items.map((it, i) => `${i + 1}. ${it.matter ? `${it.matter}: ` : ''}${it.title}${it.detail ? ` (${it.detail})` : ''}`),
              ...(c.props.more ? [`and ${c.props.more} more`] : []), ...(c.props.items.length ? [] : [c.props.empty ?? 'Nothing needs your attention.'])].join('\n');
    case 'evidence_comparison': {
      const p = c.props;
      const side = (label: string, s: typeof p.a) => `${label}: ${s.document_title}${s.clause ? `, cl. ${s.clause}` : ''}${s.page ? `, p. ${s.page}` : ''}: “…${s.context}…”`;
      return [p.title, side('A', p.a), side('B', p.b), ...(p.explanation ? [`Why it was raised: ${p.explanation}`] : []), p.caution,
              `Status: ${p.status.replace('_', ' ')}. Nothing is decided until a person reviews it.`].join('\n');
    }
    case 'timeline':
      return [c.props.summary, ...c.props.events.map((e) => {
        const tags = [e.reported ? 'recollection' : '', e.certainty === 'uncertain' ? 'uncertain' : '', e.superseded ? 'replaced' : '', e.status ?? '']
          .filter(Boolean).join(', ');
        const where = [e.source.document_title, e.source.clause ? `cl. ${e.source.clause}` : '', e.source.page ? `p. ${e.source.page}` : ''].filter(Boolean).join(', ');
        return `- ${e.matter ? `${e.matter}: ` : ''}${e.date ? e.date_text : `“${e.date_text}”`}: ${e.text} (${where}${tags ? `; ${tags}` : ''})`;
      }), ...(c.props.more ? [`${c.props.more} more not shown.`] : []), c.props.caution].join('\n');
    case 'case_table':
      return [c.props.summary, ['Matter', 'Title', 'Client', 'Documents', 'Open inconsistencies', 'Next deadline'].join('\t'),
              ...c.props.rows.map((r) => [r.number, r.title, r.client, String(r.documents), r.open_inconsistencies === undefined ? '' : String(r.open_inconsistencies),
                                          r.next_deadline ? `${r.next_deadline}${r.deadline_state ? ` (${r.deadline_state})` : ''}` : ''].join('\t')),
              c.props.caution].join('\n');
    case 'notice': return c.props.text;
    case 'unavailable': return c.props.text;
    default: return plainExtra(c);
  }
}

/** Cards added after the first registry write their own plain text here (kept apart so the switch above stays exhaustive). */
function plainExtra(c: SafeComponent): string {
  const p = (c as { props?: { summary?: string; text?: string; title?: string } }).props;
  return p?.summary ?? p?.title ?? p?.text ?? '';
}

export function plainTurn(question: string, envelope: Envelope | undefined, streamed: string): string {
  const body = envelope ? envelope.components.map(plainComponent).filter((t) => t.trim()).join('\n\n') : streamed;
  return body.trim() ? body.trim() : question;
}

/** Into the clipboard; true when it worked. The fallback serves pages opened without a secure context. */
export async function toClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) { await navigator.clipboard.writeText(text); return true; }
  } catch { /* fall back below */ }
  try {
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand('copy');
    area.remove();
    return ok;
  } catch {
    return false;
  }
}

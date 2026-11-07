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
  if (p.working) {
    const w = p.working;
    out.push(["Worked out from the documents' figures, not stated in them:",
              ...w.steps.map((s) => (s.check ? `- ${s.label}: ${s.working} (${s.result})` : `- ${s.label}: ${s.working} = ${s.result}`)),
              ...w.figures.map((f) => `- Figure: ${f.label}, ${f.value}: “${f.quote}” (${f.document_title}${f.clause ? `, cl. ${f.clause}` : ''})`),
              ...w.assumptions.map((a) => `- Assumption: ${a}`)].join('\n'));
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

/** The cards that work on records (workcards.tsx). */
function plainExtra(c: SafeComponent): string {
  switch (c.type) {
    case 'checklist':
      return [c.props.summary, ...c.props.items.map((i) => `[${i.status === 'done' ? 'x' : ' '}] ${i.title}${i.due_date ? ` (due ${i.due_date}${i.status === 'confirmed' ? ', confirmed' : i.status === 'proposed' ? ', not confirmed' : ''})` : ''}`
        + ` · ${i.document_title}${i.clause ? `, cl. ${i.clause}` : ''}`), c.props.caution].join('\n');
    case 'approval_request':
      return [c.props.summary, ...c.props.items.map((i) => `- ${i.kind}: ${i.title}, version ${i.version} by ${i.author}`)].join('\n');
    case 'guided_questions': return '';
    case 'document_preview':
      return [`${c.props.title} (version ${c.props.version}, ${c.props.status})`, ...c.props.blocks.map((b) => b.text), c.props.review].join('\n\n');
    case 'council_summary':
      return [`Review panel (${c.props.area}): ${c.props.question}`, ...c.props.issues.map((i) => `- ${i.issue}${i.disagreement ? ' (they differ)' : ''}\n`
        + i.positions.map((x) => `  ${x.role}: ${x.text}`).join('\n')), c.props.caution].join('\n');
    case 'numeric_chart':
      return [c.props.summary, ...c.props.bars.map((b) => `- ${b.shown} · ${b.document_title}${b.clause ? `, cl. ${b.clause}` : ''}`), c.props.caution].join('\n');
    case 'graph_excerpt': {
      const label = new Map(c.props.nodes.map((n) => [n.id, n.label]));
      return [c.props.summary, ...c.props.edges.map((e) => `- ${label.get(e.source)} ${e.label} ${label.get(e.target)}`), c.props.caution].join('\n');
    }
    case 'live_transcript':
      return [`${c.props.title} (${c.props.state.replace('_', ' ')})`, ...c.props.segments.map((g) => `${g.speaker ? `${g.speaker}: ` : ''}${g.text}`)].join('\n');
    default: return '';
  }
}

export function plainTurn(question: string, envelope: Envelope | undefined, streamed: string): string {
  const body = envelope ? envelope.components.map(plainComponent).filter((t) => t.trim()).join('\n\n') : streamed;
  return body.trim() ? body.trim() : question;
}

/** What "Read aloud" says for a turn, as voice mode says it: the answer (the server removes its source marks), how well it is
 *  supported when not fully, dates calculated rather than stated, and for a card the server's own summary of it. */
export function spokenTurn(envelope: Envelope | undefined, streamed: string): string {
  if (!envelope) return streamed;
  const answer = envelope.components.find((c) => c.type === 'source_answer');
  if (answer && answer.type === 'source_answer') {
    const p = answer.props;
    if (p.state === 'abstained') return p.absence ?? 'The documents in this matter do not answer that.';
    const out = [p.answer];
    const state = stateLine(p);
    if (state) out.push(state);
    if (p.calculation?.derived.length) {
      out.push(`Calculated, not stated in ${p.calculation.document_title}: ${p.calculation.derived.map((d) => `${d.label}, ${d.display}`).join('; ')}.`);
    }
    if (p.working) {
      out.push(`Worked out from the documents' figures, not stated in them: ${p.working.steps.filter((s) => !s.check).map((s) => `${s.label}, ${s.result}`).join('; ')}.`);
    }
    return out.join('\n\n');
  }
  const notices = envelope.components.filter((c) => c.type === 'notice' || c.type === 'unavailable');
  if (notices.length && notices.length === envelope.components.length) return notices.map(plainComponent).join(' ');
  return envelope.summary ? `${envelope.summary} The details are on your screen.` : '';
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

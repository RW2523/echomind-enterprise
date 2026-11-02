/**
 * The few shapes an answer's text may take besides paragraphs (owner's request 2026-11-02): headings, bulleted and
 * numbered lists, and simple tables, written the way the model writes them (Markdown-like). Parsed here into plain data;
 * cards.tsx renders each piece as React text, so nothing is ever HTML, a link, an image or a style. Anything that is not
 * one of these shapes stays a paragraph, exactly as before.
 */
export type Block =
  | { kind: 'p'; text: string }
  | { kind: 'h'; text: string }
  | { kind: 'ul'; items: string[] }
  | { kind: 'ol'; start: number; items: string[] }
  | { kind: 'table'; head: string[]; rows: string[][] };

const HEADING = /^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/;
const BOLD_LINE = /^\s{0,3}\*\*([^*\n]{1,120})\*\*:?\s*$/;            // "**Payment**" alone on its line
const BULLET = /^\s{0,3}[-*•]\s+(.*)$/;
const NUMBERED = /^\s{0,3}(\d{1,3})[.)]\s+(.*)$/;
const ROW = /^\s*\|(.*)\|\s*$/;
const RULE = /^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$/;
export const MAX_COLUMNS = 8;
export const MAX_ROWS = 60;

const cells = (line: string) => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim()).slice(0, MAX_COLUMNS);

export function blocks(text: string): Block[] {
  const out: Block[] = [];
  const lines = text.replace(/\r\n?/g, '\n').split('\n');
  let para: string[] = [];
  let inItem = false;                                   // the previous line was a list item (a plain line continues it)
  const flush = () => {
    const t = para.join(' ').trim();
    if (t) out.push({ kind: 'p', text: t });
    para = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) { flush(); inItem = false; continue; }
    const last = out[out.length - 1];
    let m: RegExpExecArray | null;
    if ((m = HEADING.exec(line)) || (m = BOLD_LINE.exec(line))) {
      flush(); out.push({ kind: 'h', text: m[1].trim() }); inItem = false; continue;
    }
    if (ROW.test(line) && i + 1 < lines.length && RULE.test(lines[i + 1]) && lines[i + 1].includes('|')) {
      flush();
      const head = cells(line);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && ROW.test(lines[i])) {
        if (rows.length < MAX_ROWS) rows.push(cells(lines[i]));
        i++;
      }
      i--;
      out.push({ kind: 'table', head, rows });
      inItem = false;
      continue;
    }
    if ((m = BULLET.exec(line))) {
      flush();
      const prev = out[out.length - 1];                 // a list goes on across blank lines, not across a paragraph
      if (prev?.kind === 'ul') prev.items.push(m[1]); else out.push({ kind: 'ul', items: [m[1]] });
      inItem = true;
      continue;
    }
    if ((m = NUMBERED.exec(line))) {
      flush();
      const prev = out[out.length - 1];
      if (prev?.kind === 'ol') prev.items.push(m[2]); else out.push({ kind: 'ol', start: Number(m[1]), items: [m[2]] });
      inItem = true;
      continue;
    }
    if (inItem && last && (last.kind === 'ul' || last.kind === 'ol')) {
      last.items[last.items.length - 1] += ` ${line.trim()}`;
      continue;
    }
    para.push(line.trim());
  }
  flush();
  return out;
}

/** Bold spans of one line of text: [text, bold] pieces. An unpaired "**" stays as written. */
export function boldPieces(text: string): [string, boolean][] {
  const out: [string, boolean][] = [];
  let last = 0;
  for (const m of text.matchAll(/\*\*(.+?)\*\*/g)) {
    if ((m.index ?? 0) > last) out.push([text.slice(last, m.index), false]);
    out.push([m[1], true]);
    last = (m.index ?? 0) + m[0].length;
  }
  if (last < text.length) out.push([text.slice(last), false]);
  return out;
}

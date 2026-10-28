/**
 * Chat, the home of EchoMind Legal (docs/echomind-legal/02, 04 Journeys B–D). One question at a time about one matter.
 * Text streams as research releases each checked paragraph; citation marks stay unclickable until the final, validated
 * components arrive. Cards act in place; sources open in the right-hand panel without leaving the conversation.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import type { LucideIcon } from 'lucide-react';
import {
  ArrowUp, BookOpenText, CalendarClock, CalendarRange, ClipboardList, Columns3, FilePen, ListChecks, Lock, Mic, Paperclip, Scale, ScrollText,
  Square, Users2, X,
} from 'lucide-react';
import { ApiError } from '../api/client';
import { useDocuments, useMatters } from '../api/queries';
import type { Matter } from '../api/types';
import { matterEntries } from '../app/nav';
import { useSession } from '../app/session';
import { counselApi, requestId, type StoredTurn } from './api';
import { ActionButtons, AnswerText, ComponentView, type CardContext } from './cards';
import { useChatMatter, usePanel } from './context';
import MatterPicker, { Layer, usePopover } from './MatterPicker';
import { parseComponent, parseEnvelope, parseTaskProgress } from './validate';
import type { ChatTurn, SafeComponent } from './types';

/** Questions any matter's documents can answer (or say they cannot). */
const EXAMPLES = [
  'What are the key dates and deadlines?',
  'Who are the parties, and what are their obligations?',
  'Summarize the latest correspondence',
  'What did the most recent amendment change?',
];

/** What the home offers. Each tile either asks the chat (one of its four kinds of answer) or opens a real page of the matter. */
interface Tile { key: string; title: string; about: string; icon: LucideIcon; hue: string; ask?: string; page?: string }
const ASK_TILES: Tile[] = [
  { key: 'today', title: 'What matters today?', about: 'Deadlines, reviews and new documents, from records', icon: ListChecks, hue: 'gold', ask: 'What matters today?' },
  { key: 'conflicts', title: 'Find conflicting dates or amounts', about: 'Potential inconsistencies, side by side', icon: Scale, hue: 'rose', ask: 'Find conflicting dates or amounts' },
  { key: 'chronology', title: 'Build a chronology of events', about: 'Every stated date in order, with its source', icon: CalendarRange, hue: 'teal', ask: 'Build a chronology of events' },
  { key: 'summary', title: 'Summarize this matter', about: 'A cited overview from its documents', icon: ScrollText, hue: 'indigo', ask: 'Summarize this matter from its documents' },
];
/** With several matters: the same answers across all of them, and the matters side by side instead of one summary. */
const SEVERAL_TILES: Tile[] = [
  { ...ASK_TILES[0], about: 'One briefing across the chosen matters' },
  { ...ASK_TILES[1], about: 'Checked inside each matter, never across' },
  { ...ASK_TILES[2], about: 'Every matter\'s dates in one timeline, labelled' },
  { key: 'cases', title: 'Compare these cases', about: 'Side by side: deadlines, documents, open items', icon: Columns3, hue: 'indigo', ask: 'Compare these cases side by side' },
];
const PAGE_TILES: Tile[] = [
  { key: 'meetings', title: 'Record a meeting', about: 'Live transcript and Silent Assist, with everyone\'s consent', icon: CalendarClock, hue: 'violet', page: 'meetings' },
  { key: 'conversation', title: 'Ask by voice', about: 'Speak a question; hear the cited answer', icon: Mic, hue: 'blue', page: 'conversation' },
  { key: 'drafting', title: 'Draft a document', about: 'Firm templates; a second lawyer approves', icon: FilePen, hue: 'amber', page: 'drafting' },
  { key: 'panel', title: 'Get four perspectives', about: 'Deal, risk, litigation and specialist views', icon: Users2, hue: 'slate', page: 'panel' },
  { key: 'research', title: 'Research in depth', about: 'Longer answers, saved with their sources', icon: BookOpenText, hue: 'slate', page: 'research' },
  { key: 'diligence', title: 'Run due diligence', about: 'A checklist across the data room, with evidence', icon: ClipboardList, hue: 'slate', page: 'diligence' },
];

/** Eight tiles: the chat's four kinds of answer, then the first matter's own pages this person can open (or the matter list). */
function useTiles(matter: Matter | null, several: boolean): (Tile & { to?: string })[] {
  const me = useSession();
  const asks = several ? SEVERAL_TILES : ASK_TILES;
  if (!matter) return [...asks, ...PAGE_TILES.slice(0, 4).map((t) => ({ ...t, to: '/matters' }))];
  const pages = new Map(matterEntries(me, matter).map((e) => [e.key, e.to]));
  const open = PAGE_TILES.filter((t) => pages.has(t.page!)).slice(0, 4);
  return [...asks, ...open.map((t) => ({ ...t, to: pages.get(t.page!) }))];
}
const FINISHED = new Set(['completed', 'abstained']);

function fromStored(t: StoredTurn): ChatTurn {
  const envelope = FINISHED.has(t.status) ? parseEnvelope(t.envelope) ?? undefined : undefined;
  const status = (['completed', 'abstained', 'failed', 'scope_revoked', 'canceled'].includes(t.status) ? t.status : 'failed') as ChatTurn['status'];
  return { key: t.id, id: t.id, question: t.question, understood: t.understood ?? undefined, status, text: '', envelope,
           error: status === 'failed' ? 'This answer could not be completed.' : undefined };
}

interface Attached { id: string; title: string }

export default function ChatPage() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const matters = useMatters();
  const { matter, matterId, matters: chosen, matterIds, setMatterId, setMatterIds } = useChatMatter();
  const several = chosen.length > 1;
  const panel = usePanel();
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [convId, setConvId] = useState<string | undefined>(conversationId);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [attached, setAttached] = useState<Attached[]>([]);
  const [pendingAsk, setPendingAsk] = useState<string | null>(null);
  const [announce, setAnnounce] = useState('');
  const created = useRef<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const tiles = useTiles(matter, several);
  const busy = turns.some((t) => t.status === 'streaming');

  useEffect(() => {
    setLoadError(null);
    if (!conversationId) { abort.current?.abort(); setTurns([]); setConvId(undefined); created.current = null; return; }
    if (created.current === conversationId) return;          // our own new conversation: its first turn is already here
    let alive = true;
    counselApi.read(conversationId).then((r) => {
      if (!alive) return;
      setConvId(r.conversation.id);
      setMatterIds(r.conversation.matter_ids?.length ? r.conversation.matter_ids : [r.conversation.matter_id]);
      setTurns(r.turns.map(fromStored));
    }).catch((e) => {
      if (!alive) return;
      setTurns([]);
      setLoadError(e instanceof ApiError && e.status === 404 ? 'This conversation is not available.' : 'This conversation could not be opened.');
    });
    return () => { alive = false; };
  }, [conversationId, setMatterIds]);

  useEffect(() => () => abort.current?.abort(), []);

  const lastText = turns[turns.length - 1]?.text;
  useEffect(() => {
    const el = end.current;
    const scroller = el?.closest('.qc-main') as HTMLElement | null;
    if (!el || !scroller || typeof el.scrollIntoView !== 'function') return;
    const near = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 160;
    if (near) el.scrollIntoView({ block: 'end' });                // never pull a reader away from earlier context
  }, [turns.length, lastText]);

  const patch = useCallback((key: string, change: Partial<ChatTurn> | ((t: ChatTurn) => Partial<ChatTurn>)) => {
    setTurns((ts) => ts.map((t) => (t.key === key ? { ...t, ...(typeof change === 'function' ? change(t) : change) } : t)));
  }, []);

  const ask = useCallback(async (text: string, opts: { document_ids?: string[]; focus?: string } = {}) => {
    const question = text.trim();
    if (!question || busy) return;
    if (!matterId) { setPendingAsk(question); return; }
    let id = convId;
    if (!id) {
      try {
        id = (await counselApi.create(matterIds)).conversation.id;
      } catch (e) {
        setLoadError(e instanceof ApiError ? e.message : 'The conversation could not be started.');
        return;
      }
      created.current = id;
      setConvId(id);
      navigate(`/chat/${id}`);
    }
    const key = requestId();
    const docs = opts.document_ids ?? attached.map((a) => a.id);
    setTurns((ts) => [...ts, { key, question, status: 'streaming', text: '' }]);
    setDraft('');
    setAttached([]);
    const ctrl = new AbortController();
    abort.current = ctrl;
    let terminal = false;
    try {
      await counselApi.ask(id, { text: question, client_request_id: key, ...(docs.length ? { document_ids: docs } : {}),
                                 ...(opts.focus ? { focus_discrepancy_id: opts.focus } : {}) }, (e) => {
        switch (e.event) {
          case 'turn.started': patch(key, { id: e.data?.turn_id }); break;
          case 'turn.understood': patch(key, { understood: String(e.data?.question ?? '') }); break;
          case 'task.progress': patch(key, { progress: parseTaskProgress(e.data) ?? undefined }); break;
          case 'answer.delta': patch(key, (t) => ({ text: t.text + String(e.data?.text ?? '') })); break;
          case 'ui.partial': {                                    // a card ready before the whole answer
            const comp = parseComponent(e.data);
            if (comp.type !== 'inert') patch(key, (t) => ({ partial: [...(t.partial ?? []).filter((c) => c.id !== comp.id), comp] }));
            break;
          }
          case 'ui.final': {
            const env = parseEnvelope(e.data);
            patch(key, env ? { envelope: env } : { status: 'failed', error: 'The answer could not be shown.' });
            if (env?.summary) setAnnounce(env.summary);
            break;
          }
          case 'turn.completed': terminal = true; patch(key, { status: 'completed', progress: undefined }); break;
          case 'turn.abstained': terminal = true; patch(key, { status: 'abstained', progress: undefined }); break;
          case 'turn.failed': terminal = true; patch(key, { status: 'failed', progress: undefined, error: String(e.data?.message ?? 'The answer could not be completed.') }); break;
          case 'scope_revoked': terminal = true; patch(key, { status: 'scope_revoked', progress: undefined }); break;
        }
      }, ctrl.signal);
      if (!terminal) patch(key, { status: 'failed', progress: undefined, error: 'The answer was cut off. Try again.' });
    } catch (err) {
      if (ctrl.signal.aborted) patch(key, { status: 'canceled', progress: undefined });
      else patch(key, { status: 'failed', progress: undefined, error: err instanceof ApiError ? err.message : 'The answer could not be completed.' });
    } finally {
      abort.current = null;
      void qc.invalidateQueries({ queryKey: ['counsel', 'conversations'] });
    }
  }, [busy, matterId, matterIds, convId, attached, navigate, patch, qc]);

  const location = useLocation();
  const handed = useRef(false);
  useEffect(() => {                                             // a question handed over by another page, asked once
    const st = location.state as { ask?: unknown; focus?: unknown } | null;
    if (handed.current || conversationId || !matterId || typeof st?.ask !== 'string' || !st.ask.trim()) return;
    handed.current = true;
    navigate('/chat', { replace: true, state: null });
    void ask(st.ask.slice(0, 2000), typeof st.focus === 'string' && /^[A-Za-z0-9_-]{1,80}$/.test(st.focus) ? { focus: st.focus } : {});
  }, [location.state, conversationId, matterId, navigate, ask]);

  useEffect(() => {                                             // a question asked before a matter was chosen
    if (pendingAsk && matterId) { const q = pendingAsk; setPendingAsk(null); void ask(q); }
  }, [pendingAsk, matterId, ask]);

  const replace = useCallback((turnKey: string) => (comp: SafeComponent) => {
    patch(turnKey, (t) => (t.envelope ? { envelope: { ...t.envelope, components: t.envelope.components.map((c) => (c.id === comp.id ? comp : c)) } } : {}));
  }, [patch]);

  const ctxFor = (t: ChatTurn): CardContext => ({
    matterId: t.envelope?.matter_id ?? matterId ?? '',
    matterNumber: (id) => (t.envelope?.matters ?? []).find((m) => m.id === id)?.number,
    busy,
    ask: (text, opts) => { void ask(text, opts); },
    openCitation: (c, opener) => panel.open({ kind: 'citation', id: c.id, onUse: c.removed ? undefined : () => {
      setAttached((a) => (a.some((x) => x.id === c.document_id) ? a : [...a, { id: c.document_id, title: c.document_title }].slice(0, 10)));
      panel.close();
    } }, opener),
    openSide: (side, label, opener) => panel.open({ kind: 'side', side, label }, opener),
    go: (path) => navigate(path),
    replace: replace(t.key),
  });

  const start = !conversationId && turns.length === 0;
  const composer = (
    <Composer value={draft} onChange={setDraft} busy={busy} onSend={() => void ask(draft)} onStop={() => abort.current?.abort()} hero={start}
      placeholder={several ? `Ask across ${chosen.map((m) => m.number).join(', ')}: every source says which matter it is from…`
        : matter ? `Ask anything about ${matter.number}: its documents, dates, parties or obligations…` : 'Ask about a matter…'}
      attached={attached} onAttach={setAttached} matterId={several ? null : matterId} inputRef={input}
      onDetach={(id) => setAttached((a) => a.filter((x) => x.id !== id))} />
  );
  return (
    <div className={`qc-chat${start ? ' qc-home' : ''}`}>
      <div className="qc-sr" aria-live="polite">{announce}</div>
      {loadError && <p className="qc-error" role="alert">{loadError}</p>}
      {start ? (
        <div className="qc-start qc-home-panel">
          <div className="qc-hero">
            <span className="qc-eyebrow">Private legal workspace</span>
            <h1>How can I help you today?</h1>
            <p className="qc-context">{several
              ? <>Working across <b>{chosen.map((m) => m.number).join(', ')}</b>. Each answer says which matter every source comes from{
                new Set(chosen.map((m) => m.client.id)).size > 1 ? <>; these matters belong to <b>different clients</b>, so keep each client's information to its own work</> : ''}.</>
              : matter
                ? <>Working on <b>{matter.number} · {matter.title}</b>. Answers come only from its documents, and every quote is checked.</>
                : 'Choose one matter or several, then ask. Answers come only from those matters\' documents, and every quote is checked.'}</p>
          </div>
          <div className="qc-tiles" role="group" aria-label="Start with">
            {tiles.map((t) => (
              <button key={t.key} type="button" className="qc-tile" aria-label={t.title} aria-describedby={`tile-${t.key}`}
                onClick={() => { if (t.ask) void ask(t.ask); else if (t.to) navigate(t.to); }}>
                <span className="qc-tile-icon" data-hue={t.hue} aria-hidden><t.icon /></span>
                <span className="qc-tile-text"><b>{t.title}</b><small id={`tile-${t.key}`}>{t.to === '/matters' ? 'Choose a matter first' : t.about}</small></span>
              </button>
            ))}
          </div>
          {pendingAsk && !matterId && (
            <MatterQuestion matters={(matters.data ?? []).slice(0, 5)} onPick={setMatterId} onCancel={() => setPendingAsk(null)} />
          )}
          {composer}
          <div className="qc-suggestions" role="group" aria-label="Example questions">
            {EXAMPLES.map((q) => <button key={q} type="button" className="qc-suggestion" onClick={() => void ask(q)}>{q}</button>)}
          </div>
        </div>
      ) : (
        <>
          <div className="qc-thread" aria-label="Conversation">
            {turns.map((t) => <TurnView key={t.key} t={t} ctx={ctxFor(t)} onRetry={() => void ask(t.question)} onStop={() => abort.current?.abort()} />)}
          </div>
          {pendingAsk && !matterId && (
            <MatterQuestion matters={(matters.data ?? []).slice(0, 5)} onPick={setMatterId} onCancel={() => setPendingAsk(null)} />
          )}
          <div ref={end} />
          {composer}
        </>
      )}
    </div>
  );
}

function MatterQuestion({ matters, onPick, onCancel }: { matters: { id: string; number: string; title: string }[];
                                                        onPick: (id: string) => void; onCancel: () => void }) {
  return (
    <section className="qc-card qc-pick" aria-label="Which matter is this about?">
      <h3 className="qc-card-title">Which matter is this about?</h3>
      {matters.length ? (
        <div className="qc-actions">
          {matters.map((m) => <button key={m.id} type="button" className="qc-btn" onClick={() => onPick(m.id)}>{m.number} · {m.title}</button>)}
        </div>
      ) : <p className="qc-meta" style={{ margin: 0 }}>You are not a member of any matter yet.</p>}
      <button type="button" className="qc-btn" onClick={onCancel}>Cancel</button>
    </section>
  );
}

function TurnView({ t, ctx, onRetry, onStop }: { t: ChatTurn; ctx: CardContext; onRetry: () => void; onStop: () => void }) {
  const comps = t.envelope?.components ?? (t.status === 'streaming' ? t.partial ?? [] : []);
  return (
    <article className="qc-turn" style={{ display: 'grid', gap: 12 }} aria-busy={t.status === 'streaming'}>
      <div className="qc-q">{t.question}</div>
      <div className="qc-a">
        {t.understood && <p className="qc-understood">Answering: {t.understood}</p>}
        {t.status === 'streaming' && (
          <div className="qc-progress" role="status">
            {t.progress ? (
              <>
                <span>{t.progress.label}{t.progress.total ? `: ${t.progress.done} of ${t.progress.total}` : ''}</span>
                {t.progress.total > 0 && <span className="qc-progress-bar" aria-hidden><span style={{ width: `${Math.round((100 * t.progress.done) / t.progress.total)}%` }} /></span>}
              </>
            ) : <span>{t.text ? 'Checking each quote against the documents…' : 'Searching this matter\'s documents…'}</span>}
            <button type="button" className="qc-btn" onClick={onStop}><Square aria-hidden />Stop</button>
          </div>
        )}
        {t.status === 'streaming' && t.text && !t.envelope && <AnswerText text={t.text} citations={[]} pending />}
        {comps.map((c) => <ComponentView key={c.id} comp={c} ctx={ctx} />)}
        {t.envelope && <ActionButtons actions={t.envelope.suggested_actions} ctx={ctx} />}
        {t.status === 'failed' && (
          <div className="qc-actions"><p className="qc-error" role="alert" style={{ margin: 0 }}>{t.error}</p>
            <button type="button" className="qc-btn" onClick={onRetry}>Try again</button></div>
        )}
        {t.status === 'scope_revoked' && <p className="qc-error" role="alert">You no longer have access to this matter, so the answer was stopped.</p>}
        {t.status === 'canceled' && <p className="qc-meta">Stopped.</p>}
      </div>
    </article>
  );
}

/** "Answer only from these documents": up to ten of the matter's documents (the server checks each one again). */
function DocumentPicker({ matterId, attached, onAttach }: { matterId: string; attached: Attached[]; onAttach: (a: Attached[]) => void }) {
  const { open, setOpen, button, pop, style } = usePopover();
  const docs = useDocuments(matterId);
  const ready = (docs.data ?? []).filter((d) => d.current_version?.index_status === 'indexed');
  const toggle = (id: string, title: string) => onAttach(attached.some((a) => a.id === id)
    ? attached.filter((a) => a.id !== id) : [...attached, { id, title }].slice(0, 10));
  return (
    <>
      <button ref={button} type="button" className="qc-tool" aria-haspopup="dialog" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <Paperclip aria-hidden /><span>{attached.length ? `Only from ${attached.length} document${attached.length === 1 ? '' : 's'}` : 'Choose documents'}</span>
      </button>
      {open && (<Layer>
        <div ref={pop} className="qc-pop" role="dialog" aria-label="Answer only from these documents" style={{ ...style, padding: 10, width: 380 }}>
          <div className="qc-pop-label" style={{ paddingTop: 2 }}>Answer only from…</div>
          {docs.isPending ? <p className="qc-meta" style={{ padding: '4px 8px' }}>Loading the documents…</p> : ready.length ? (
            <div className="qc-docpick">
              {ready.map((d) => (
                <label key={d.id}>
                  <input type="checkbox" checked={attached.some((a) => a.id === d.id)} onChange={() => toggle(d.id, d.title)}
                    disabled={!attached.some((a) => a.id === d.id) && attached.length >= 10} />
                  <span style={{ minWidth: 0 }}>{d.title}<span className="qc-meta" style={{ display: 'block' }}>{d.kind} · version {d.version}</span></span>
                </label>
              ))}
            </div>
          ) : <p className="qc-meta" style={{ padding: '4px 8px' }}>No document of this matter has been read yet.</p>}
          <p className="qc-meta" style={{ borderTop: '1px solid var(--rule)', margin: '8px 0 0', padding: '8px 8px 2px' }}>
            With none ticked, answers use every document of the matter you may read.
          </p>
        </div></Layer>)}
    </>
  );
}

function Composer({ value, onChange, busy, onSend, onStop, placeholder, attached, onAttach, onDetach, matterId, hero, inputRef }: {
  value: string; onChange: (v: string) => void; busy: boolean; onSend: () => void; onStop: () => void; placeholder: string;
  attached: Attached[]; onAttach: (a: Attached[]) => void; onDetach: (id: string) => void; matterId: string | null; hero: boolean;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
}) {
  const ref = inputRef;
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(200, el.scrollHeight)}px`;
  }, [value, ref]);
  return (
    <div className="qc-composer-wrap" data-hero={hero ? 'true' : undefined}>
      <form className="qc-composer" onSubmit={(e) => { e.preventDefault(); if (!busy) onSend(); }}>
        <textarea ref={ref} rows={hero ? 2 : 1} value={value} maxLength={2000} placeholder={placeholder} aria-label="Ask EchoMind"
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); if (!busy) onSend(); } }} />
        <div className="qc-composer-row" style={{ alignItems: 'center' }}>
          <div className="qc-composer-tools">
            {matterId && <DocumentPicker matterId={matterId} attached={attached} onAttach={onAttach} />}
            <MatterPicker variant="composer" />
          </div>
          <span className="qc-private qc-hide-small" title="Answered by this appliance's own model; nothing leaves it."><Lock aria-hidden />On this appliance</span>
          {busy
            ? <button type="button" className="qc-send" onClick={onStop} aria-label="Stop the answer"><Square aria-hidden /></button>
            : <button type="submit" className="qc-send" disabled={!value.trim()} aria-label="Send"><ArrowUp aria-hidden /></button>}
        </div>
      </form>
      <div className="qc-composer-meta">
        {attached.map((a) => (
          <span key={a.id} className="qc-attached">Only from: {a.title}
            <button type="button" className="qc-icon-btn" style={{ width: 22, height: 22 }} onClick={() => onDetach(a.id)} aria-label={`Stop using ${a.title}`}><X aria-hidden /></button>
          </span>
        ))}
        <span>Answers come only from the chosen matter's documents and are drafts for a lawyer's review.</span>
      </div>
    </div>
  );
}

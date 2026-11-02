/**
 * Chat, the home of EchoMind Legal (docs/echomind-legal/02, 04 Journeys B–D). One question at a time about one matter.
 * Text streams as research releases each checked paragraph; citation marks stay unclickable until the final, validated
 * components arrive. Cards act in place; sources open in the right-hand panel without leaving the conversation.
 *
 * Like other chat assistants (owner's request 2026-11-02): each answer can be copied, asked again or rated, and each
 * question edited. Asking again or editing adds another version of that turn (the server keeps every answer that was
 * shown); "‹ 2 / 3 ›" moves between versions, and what follows a version stays with it. Documents can be added from the
 * question box, and questions dictated where voice is turned on.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useLocation, useNavigate, useParams } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { keys } from '../api/queries';
import type { LucideIcon } from 'lucide-react';
import {
  ArrowUp, CalendarRange, Check, ChevronLeft, ChevronRight, Columns3, Copy, ListChecks, Loader2, Lock, Paperclip, Pencil, RefreshCw, Scale,
  ScrollText, Square, ThumbsDown, ThumbsUp, X,
} from 'lucide-react';
import { ApiError } from '../api/client';
import { useDocuments, useMatters, useObligations, useReviews } from '../api/queries';
import type { Matter } from '../api/types';
import { matterEntries } from '../app/nav';
import { useSession } from '../app/session';
import { whereAnswersComeFrom } from './where';
import { counselApi, requestId, type StoredTurn } from './api';
import { ActionButtons, AnswerText, ComponentView, type CardContext } from './cards';
import { AddDocument, addDocument, Dictate, droppedFile } from './ComposerTools';
import { plainTurn, toClipboard } from './copy';
import { useChatMatter, usePanel } from './context';
import MatterPicker, { Layer, usePopover } from './MatterPicker';
import { parseComponent, parseEnvelope, parseTaskProgress } from './validate';
import type { ChatTurn, Feedback, FeedbackReason, SafeComponent } from './types';

/** A few plain starters: each one is one of the chat's kinds of answer, about the chosen matter or matters. */
interface Starter { label: string; ask: string; icon: LucideIcon }
const ONE_MATTER: Starter[] = [
  { label: 'What needs my attention today?', ask: 'What needs my attention today?', icon: ListChecks },
  { label: 'Find conflicting dates or amounts', ask: 'Find conflicting dates or amounts', icon: Scale },
  { label: 'Build a chronology of events', ask: 'Build a chronology of events', icon: CalendarRange },
  { label: 'Summarize this matter', ask: 'Summarize this matter from its documents', icon: ScrollText },
];
const SEVERAL_MATTERS: Starter[] = [
  { label: 'What needs my attention across these matters?', ask: 'What needs my attention across these matters?', icon: ListChecks },
  { label: 'Compare these cases side by side', ask: 'Compare these cases side by side', icon: Columns3 },
  { label: 'Build one chronology for these matters', ask: 'Build a chronology of events', icon: CalendarRange },
  { label: 'Find conflicting dates inside each matter', ask: 'Find conflicting dates or amounts', icon: Scale },
];
/** One quiet line under the matter (review H1): what waits for the person's decision, and the next dated deadline, each a
 *  link. Nothing when there is neither. A deadline read from a document but not confirmed by a lawyer says so. */
function NextUp({ matter }: { matter: Matter }) {
  const reviews = useReviews();
  const obligations = useObligations(matter.id);
  const waiting = (reviews.data?.waiting ?? []).filter((i) => i.matter?.number === matter.number).length;
  const today = new Date().toISOString().slice(0, 10);
  const next = (obligations.data?.obligations ?? [])
    .filter((o) => o.due_date && o.due_date >= today && o.status !== 'dismissed' && o.status !== 'done')
    .sort((a, b) => (a.due_date ?? '').localeCompare(b.due_date ?? ''))[0];
  if (!waiting && !next) return null;
  const base = `/m/${encodeURIComponent(matter.number)}`;
  return (
    <p className="qc-next">
      {waiting > 0 && <Link to="/drafts">{waiting} item{waiting === 1 ? '' : 's'} wait{waiting === 1 ? 's' : ''} for your decision</Link>}
      {waiting > 0 && next && <span aria-hidden> · </span>}
      {next && <Link to={`${base}/obligations`}>Next deadline {new Date(`${next.due_date}T12:00:00`).toLocaleDateString([], { day: 'numeric', month: 'short' })}
        {next.status === 'confirmed' ? '' : ' (not confirmed)'}</Link>}
    </p>
  );
}

/** The matter's own pages that the home links to quietly (the rest are on the matter's tabs). */
const MATTER_LINKS: { key: string; label: string }[] = [
  { key: 'documents', label: 'Documents' }, { key: 'meetings', label: 'Meetings' }, { key: 'conversation', label: 'Ask by voice' },
  { key: 'drafting', label: 'Drafting' },
];

function useMatterLinks(matter: Matter | null): { to: string; label: string }[] {
  const me = useSession();
  if (!matter) return [];
  const pages = new Map(matterEntries(me, matter).map((e) => [e.key, e.to]));
  return [{ to: pages.get('overview') ?? `/m/${encodeURIComponent(matter.number)}/overview`, label: `Open ${matter.number}` },
          ...MATTER_LINKS.filter((l) => pages.has(l.key)).map((l) => ({ to: pages.get(l.key)!, label: l.label }))];
}
const FINISHED = new Set(['completed', 'abstained']);

function fromStored(t: StoredTurn): ChatTurn {
  const envelope = FINISHED.has(t.status) ? parseEnvelope(t.envelope) ?? undefined : undefined;
  const status = (['completed', 'abstained', 'failed', 'scope_revoked', 'canceled'].includes(t.status) ? t.status : 'failed') as ChatTurn['status'];
  return { key: t.id, id: t.id, parent: t.parent_turn_id ?? null, feedback: t.feedback, question: t.question,
           understood: t.understood ?? undefined, status, text: '', envelope,
           error: status === 'failed' ? 'This answer could not be completed.' : undefined };
}

interface Attached { id: string; title: string }

/** One step of the conversation as shown: the chosen version, and all versions of that step (oldest first). */
interface Step { turn: ChatTurn; versions: ChatTurn[] }

/** The path through the conversation's versions: at each step the chosen one, else the newest. */
export function visiblePath(turns: ChatTurn[], chosen: Record<string, string>): Step[] {
  const kids = new Map<string, ChatTurn[]>();
  for (const t of turns) {
    const k = t.parent ?? '';
    const list = kids.get(k);
    if (list) list.push(t); else kids.set(k, [t]);
  }
  const out: Step[] = [];
  const seen = new Set<string>();
  let at = '';
  while (!seen.has(at)) {
    seen.add(at);
    const list = kids.get(at);
    if (!list?.length) break;
    const pick = list.find((t) => t.key === chosen[at]) ?? list[list.length - 1];
    out.push({ turn: pick, versions: list });
    if (!pick.id) break;                                    // never started on the server: nothing follows it
    at = pick.id;
  }
  return out;
}

interface AskOptions { document_ids?: string[]; focus?: string; retryOf?: ChatTurn }

export default function ChatPage() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const matters = useMatters();
  const { matter, matterId, matters: chosen, matterIds, setMatterId, setMatterIds } = useChatMatter();
  // Someone with a single open matter starts with it chosen (review H1).
  const open = (matters.data ?? []).filter((m) => m.status === 'open');
  useEffect(() => { if (!matterId && matters.data && open.length === 1) setMatterId(open[0].id); }, [matterId, matters.data, open.length]);
  const several = chosen.length > 1;
  const panel = usePanel();
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [picked, setPicked] = useState<Record<string, string>>({});       // parent turn id ('' first) -> version shown
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
  const links = useMatterLinks(several ? null : matter);
  const starters = several ? SEVERAL_MATTERS : ONE_MATTER;
  const busy = turns.some((t) => t.status === 'streaming');
  const path = useMemo(() => visiblePath(turns, picked), [turns, picked]);
  const pathRef = useRef<Step[]>(path);
  pathRef.current = path;

  useEffect(() => {
    setLoadError(null);
    if (!conversationId) { abort.current?.abort(); setTurns([]); setPicked({}); setConvId(undefined); created.current = null; return; }
    if (created.current === conversationId) return;          // our own new conversation: its first turn is already here
    let alive = true;
    counselApi.read(conversationId).then((r) => {
      if (!alive) return;
      setConvId(r.conversation.id);
      setMatterIds(r.conversation.matter_ids?.length ? r.conversation.matter_ids : [r.conversation.matter_id]);
      setTurns(r.turns.map(fromStored));
      setPicked({});
    }).catch((e) => {
      if (!alive) return;
      setTurns([]);
      setLoadError(e instanceof ApiError && e.status === 404 ? 'This conversation is not available.' : 'This conversation could not be opened.');
    });
    return () => { alive = false; };
  }, [conversationId, setMatterIds]);

  useEffect(() => () => abort.current?.abort(), []);

  const lastText = path[path.length - 1]?.turn.text;
  useEffect(() => {
    const el = end.current;
    const scroller = el?.closest('.qc-main') as HTMLElement | null;
    if (!el || !scroller || typeof el.scrollIntoView !== 'function') return;
    const near = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 160;
    if (near) el.scrollIntoView({ block: 'end' });                // never pull a reader away from earlier context
  }, [path.length, lastText]);

  const patch = useCallback((key: string, change: Partial<ChatTurn> | ((t: ChatTurn) => Partial<ChatTurn>)) => {
    setTurns((ts) => ts.map((t) => (t.key === key ? { ...t, ...(typeof change === 'function' ? change(t) : change) } : t)));
  }, []);

  const ask = useCallback(async (text: string, opts: AskOptions = {}) => {
    const question = text.trim();
    if (!question || busy) return;
    if (!matterId) { setPendingAsk(question); return; }
    // Where it goes: another version of a turn (asked again or edited), or after the last turn shown that the server knows.
    let parent: string | null;
    let place: { after_turn_id?: string; retry_of?: string } = {};
    if (opts.retryOf?.id) {
      parent = opts.retryOf.parent ?? null;
      place = { retry_of: opts.retryOf.id };
    } else {
      const last = [...pathRef.current].reverse().find((st) => st.turn.id)?.turn;
      parent = last?.id ?? null;
      if (last?.id) place = { after_turn_id: last.id };
    }
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
    // A question the server never started (refused before it began) is replaced, not kept as a version.
    setTurns((ts) => [...ts.filter((t) => t.id || t.status === 'streaming'), { key, parent, question, status: 'streaming', text: '' }]);
    setPicked((c) => ({ ...c, [parent ?? '']: key }));
    setDraft('');
    setAttached([]);
    const ctrl = new AbortController();
    abort.current = ctrl;
    let terminal = false;
    try {
      await counselApi.ask(id, { text: question, client_request_id: key, ...(docs.length ? { document_ids: docs } : {}),
                                 ...(opts.focus ? { focus_discrepancy_id: opts.focus } : {}), ...place }, (e) => {
        switch (e.event) {
          case 'turn.started': patch(key, { id: e.data?.turn_id, ...(e.data && 'parent_turn_id' in e.data ? { parent: e.data.parent_turn_id ?? null } : {}) }); break;
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

  const rate = useCallback(async (t: ChatTurn, next: { rating: 'up' | 'down' | null; reason?: FeedbackReason; note?: string }) => {
    if (!t.id) return false;
    try {
      const r = await counselApi.feedback(t.id, next);
      patch(t.key, { feedback: r.feedback ?? undefined });
      return true;
    } catch {
      setAnnounce('The rating could not be saved.');
      return false;
    }
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
      matter={several ? null : matter}
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
        <div className="qc-start">
          <div className="qc-hero">
            <h1>How can I help?</h1>
            <p className="qc-context">{several
              ? <>Asking across <b>{chosen.map((m) => m.number).join(', ')}</b>. Each answer says which matter every source comes from{
                new Set(chosen.map((m) => m.client.id)).size > 1 ? '; these belong to different clients, so keep each client\'s information to its own work' : ''}.</>
              : matter
                ? <><b>{matter.number} · {matter.title}</b> · {matter.client.name}</>
                : 'Choose a matter in the question box, or ask and pick one.'}</p>
            {!several && matter && <NextUp matter={matter} />}
          </div>
          {pendingAsk && !matterId && (
            <MatterQuestion matters={(matters.data ?? []).slice(0, 5)} onPick={setMatterId} onCancel={() => setPendingAsk(null)} />
          )}
          {composer}
          <ul className="qc-starters" aria-label="Suggestions">
            {starters.map((st) => (
              <li key={st.label}>
                <button type="button" className="qc-starter" onClick={() => void ask(st.ask)}><st.icon aria-hidden />{st.label}</button>
              </li>
            ))}
          </ul>
          {links.length > 0 && (
            <nav className="qc-start-foot" aria-label="This matter">
              {links.map((l) => <Link key={l.to} to={l.to}>{l.label}</Link>)}
            </nav>
          )}
        </div>
      ) : (
        <>
          <div className="qc-thread" role="group" aria-label="Conversation">
            {path.map(({ turn: t, versions }) => (
              <TurnView key={t.key} t={t} ctx={ctxFor(t)} busy={busy} versions={versions}
                onChoose={(v) => setPicked((c) => ({ ...c, [v.parent ?? '']: v.key }))}
                onRetry={() => void ask(t.question, { retryOf: t })} onEdit={(text) => void ask(text, { retryOf: t })}
                onRate={(next) => rate(t, next)} onStop={() => abort.current?.abort()} />
            ))}
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

const REASONS: { key: FeedbackReason; label: string }[] = [
  { key: 'wrong_source', label: 'Wrong or weak source' }, { key: 'missed', label: 'Missed something' },
  { key: 'not_asked', label: 'Not what I asked' }, { key: 'unclear', label: 'Hard to follow' }, { key: 'other', label: 'Other' },
];

type Rate = (next: { rating: 'up' | 'down' | null; reason?: FeedbackReason; note?: string }) => Promise<boolean>;

/** Copy, ask again, and whether the answer helped (a thumbs down may say why). */
function TurnTools({ t, busy, onRetry, onRate }: { t: ChatTurn; busy: boolean; onRetry: () => void; onRate: Rate }) {
  const [copied, setCopied] = useState<boolean | null>(null);
  const [asking, setAsking] = useState(false);
  const [reason, setReason] = useState<FeedbackReason | null>(null);
  const [note, setNote] = useState('');
  const [said, setSaid] = useState('');
  const rating: Feedback['rating'] | undefined = t.feedback?.rating;
  const copy = async () => {
    const ok = await toClipboard(plainTurn(t.question, t.envelope, t.text));
    setCopied(ok);
    setSaid(ok ? 'Copied to the clipboard.' : 'The answer could not be copied. Select the text instead.');
    if (ok) window.setTimeout(() => setCopied(null), 2000);
  };
  const up = async () => {
    setAsking(false);
    if (await onRate({ rating: rating === 'up' ? null : 'up' })) setSaid(rating === 'up' ? 'Rating removed.' : 'Marked as helpful.');
  };
  const down = async () => {
    if (rating === 'down') { setAsking(false); if (await onRate({ rating: null })) setSaid('Rating removed.'); return; }
    if (await onRate({ rating: 'down' })) { setReason(null); setNote(''); setAsking(true); setSaid('Marked as not helpful.'); }
  };
  const send = async (e: React.FormEvent) => {
    e.preventDefault();
    if (await onRate({ rating: 'down', ...(reason ? { reason } : {}), ...(note.trim() ? { note: note.trim() } : {}) })) {
      setAsking(false);
      setSaid('Thank you. Your rating is saved with this answer.');
    }
  };
  return (
    <>
      <div className="qc-turn-tools" role="group" aria-label="This answer">
        <button type="button" className="qc-icon-btn" onClick={() => void copy()} aria-label={copied ? 'Copied' : 'Copy answer'} title="Copy, with its sources">
          {copied ? <Check aria-hidden /> : <Copy aria-hidden />}
        </button>
        <button type="button" className="qc-icon-btn" onClick={onRetry} disabled={busy} aria-label="Ask again"
          title="Ask again (this answer is kept as a version)"><RefreshCw aria-hidden /></button>
        {t.id && <>
          <button type="button" className="qc-icon-btn" aria-pressed={rating === 'up'} onClick={() => void up()} aria-label="Helpful" title="Helpful">
            <ThumbsUp aria-hidden /></button>
          <button type="button" className="qc-icon-btn" aria-pressed={rating === 'down'} onClick={() => void down()} aria-label="Not helpful" title="Not helpful">
            <ThumbsDown aria-hidden /></button>
        </>}
        <span className="qc-tools-said" aria-live="polite">{said}</span>
      </div>
      {asking && (
        <form className="qc-feedback" onSubmit={(e) => void send(e)} aria-label="What was wrong?">
          <p className="qc-meta" style={{ margin: 0 }}>What was wrong? (optional)</p>
          <div className="qc-chips" role="group" aria-label="Reason">
            {REASONS.map((r) => (
              <button key={r.key} type="button" className="qc-chip" aria-pressed={reason === r.key} onClick={() => setReason(reason === r.key ? null : r.key)}>{r.label}</button>
            ))}
          </div>
          <textarea value={note} maxLength={500} onChange={(e) => setNote(e.target.value)} aria-label="Say more (optional)"
            placeholder="For example: it quotes clause 4.1, but clause 4.2 governs delivery." />
          <div className="qc-actions">
            <button type="submit" className="qc-btn qc-btn-primary">Send</button>
            <button type="button" className="qc-btn" onClick={() => setAsking(false)}>Close</button>
          </div>
        </form>
      )}
    </>
  );
}

/** "‹ 2 / 3 ›": the versions of one question (asked again, or edited). */
function Versions({ t, versions, busy, onChoose }: { t: ChatTurn; versions: ChatTurn[]; busy: boolean; onChoose: (v: ChatTurn) => void }) {
  if (versions.length < 2) return null;
  const i = versions.findIndex((v) => v.key === t.key);
  return (
    <span className="qc-versions" role="group" aria-label={`Version ${i + 1} of ${versions.length}`}>
      <button type="button" className="qc-icon-btn" disabled={busy || i <= 0} onClick={() => onChoose(versions[i - 1])} aria-label="Previous version">
        <ChevronLeft aria-hidden /></button>
      <span aria-hidden>{i + 1} / {versions.length}</span>
      <button type="button" className="qc-icon-btn" disabled={busy || i >= versions.length - 1} onClick={() => onChoose(versions[i + 1])} aria-label="Next version">
        <ChevronRight aria-hidden /></button>
    </span>
  );
}

function TurnView({ t, ctx, busy, versions, onChoose, onRetry, onEdit, onRate, onStop }: {
  t: ChatTurn; ctx: CardContext; busy: boolean; versions: ChatTurn[]; onChoose: (v: ChatTurn) => void; onRetry: () => void;
  onEdit: (text: string) => void; onRate: Rate; onStop: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(t.question);
  const comps = t.envelope?.components ?? (t.status === 'streaming' ? t.partial ?? [] : []);
  const done = t.status === 'completed' || t.status === 'abstained';
  const send = () => { if (text.trim() && !busy) { setEditing(false); onEdit(text); } };
  return (
    <article className="qc-turn" style={{ display: 'grid', gap: 12 }} aria-busy={t.status === 'streaming'}>
      {editing ? (
        <form className="qc-q-edit" onSubmit={(e) => { e.preventDefault(); send(); }}>
          <textarea value={text} maxLength={2000} rows={3} autoFocus aria-label="Edit your question" onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') { setEditing(false); setText(t.question); }
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(); }
            }} />
          <div className="qc-actions" style={{ justifyContent: 'flex-end' }}>
            <button type="button" className="qc-btn" onClick={() => { setEditing(false); setText(t.question); }}>Cancel</button>
            <button type="submit" className="qc-btn qc-btn-primary" disabled={busy || !text.trim()} aria-label="Send the edited question">Send</button>
          </div>
          <p className="qc-meta" style={{ margin: 0, textAlign: 'right' }}>Sending makes a new version of this question. The earlier answer is kept.</p>
        </form>
      ) : (
        <div className="qc-q-row">
          <Versions t={t} versions={versions} busy={busy} onChoose={onChoose} />
          {!busy && t.status !== 'streaming' && (
            <button type="button" className="qc-icon-btn qc-q-edit-btn" onClick={() => { setText(t.question); setEditing(true); }}
              aria-label="Edit question" title="Edit question"><Pencil aria-hidden /></button>
          )}
          <div className="qc-q">{t.question}</div>
        </div>
      )}
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
        {done && <TurnTools t={t} busy={busy} onRetry={onRetry} onRate={onRate} />}
        {t.status === 'failed' && (
          <div className="qc-actions"><p className="qc-error" role="alert" style={{ margin: 0 }}>{t.error}</p>
            <button type="button" className="qc-btn" onClick={onRetry} disabled={busy}>Try again</button></div>
        )}
        {t.status === 'scope_revoked' && <p className="qc-error" role="alert">You no longer have access to this matter, so the answer was stopped.</p>}
        {t.status === 'canceled' && (
          <div className="qc-actions"><p className="qc-meta" style={{ margin: 0 }}>Stopped.</p>
            <button type="button" className="qc-btn" onClick={onRetry} disabled={busy}><RefreshCw aria-hidden />Ask again</button></div>
        )}
      </div>
    </article>
  );
}

/** "Answer only from these documents": up to ten of the matter's documents (the server checks each one again). */
function DocumentPicker({ matterId, attached, onAttach, adding, onError }: {
  matterId: string; attached: Attached[]; onAttach: (a: Attached[]) => void; adding?: string | null; onError: (message: string) => void;
}) {
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
          {adding && (
            <>
              <AddDocument matterId={matterId} matterNumber={adding} onError={(m) => { onError(m); setOpen(false); }}
                onAdded={(d) => { onAttach([...attached.filter((a) => a.id !== d.id), d].slice(0, 10)); setOpen(false); }} />
              <p className="qc-meta" style={{ margin: '0 8px 6px' }}>PDF, Word, text or Markdown. It is added to the matter's documents as a draft.</p>
              <div className="qc-pop-sep" />
            </>
          )}
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

/** The chosen documents under the question box, each saying whether it has been read yet (an added one takes a moment). */
function AttachedChips({ matterId, attached, onDetach, onReading }: {
  matterId: string; attached: Attached[]; onDetach: (id: string) => void; onReading: (reading: string | null) => void;
}) {
  const docs = useDocuments(matterId);
  const state = (id: string) => (docs.data ?? []).find((d) => d.id === id)?.current_version?.index_status;
  const reading = attached.find((a) => ['queued', 'indexing'].includes(state(a.id) ?? '') || (docs.isPending && !docs.data));
  useEffect(() => { onReading(reading ? reading.title : null); }, [reading?.title, onReading]);
  return (
    <>
      {attached.map((a) => {
        const st = state(a.id);
        return (
          <span key={a.id} className="qc-attached" data-state={st === 'failed' ? 'failed' : st === 'queued' || st === 'indexing' ? 'reading' : undefined}>
            {(st === 'queued' || st === 'indexing') && <Loader2 className="qc-spin" aria-hidden />}
            Only from: {a.title}
            {(st === 'queued' || st === 'indexing') && <span className="qc-meta"> · reading…</span>}
            {st === 'failed' && <span> · could not be read</span>}
            <button type="button" className="qc-icon-btn" style={{ width: 22, height: 22 }} onClick={() => onDetach(a.id)} aria-label={`Stop using ${a.title}`}><X aria-hidden /></button>
          </span>
        );
      })}
    </>
  );
}

const CONTRIBUTORS = new Set(['lead', 'attorney', 'paralegal']);

function Composer({ value, onChange, busy, onSend, onStop, placeholder, attached, onAttach, onDetach, matterId, matter, hero, inputRef }: {
  value: string; onChange: (v: string) => void; busy: boolean; onSend: () => void; onStop: () => void; placeholder: string;
  attached: Attached[]; onAttach: (a: Attached[]) => void; onDetach: (id: string) => void; matterId: string | null; matter: Matter | null;
  hero: boolean; inputRef: React.RefObject<HTMLTextAreaElement | null>;
}) {
  const ref = inputRef;
  const me = useSession();
  const where = whereAnswersComeFrom(me);
  const qc = useQueryClient();
  const [reading, setReading] = useState<string | null>(null);
  const [toolError, setToolError] = useState<string | null>(null);
  const [drop, setDrop] = useState(false);
  // Adding a document is for the matter's contributors while it is open (the server checks again); dictation needs voice on.
  const canAdd = !!matterId && !!matter && matter.status === 'open' && CONTRIBUTORS.has(String(matter.my_role ?? ''));
  const canDictate = !!matterId && !!me.features?.conversation;
  const blocked = busy || !!reading;
  const send = () => { if (!blocked) { setToolError(null); onSend(); } };
  const dropped = async (e: React.DragEvent) => {
    e.preventDefault();
    setDrop(false);
    const file = droppedFile(e);
    if (!file || !canAdd || !matterId) return;
    setToolError(null);
    const r = await addDocument(matterId, file);
    if ('error' in r) { setToolError(r.error); return; }
    void qc.invalidateQueries({ queryKey: keys.documents(matterId) });
    onAttach([...attached.filter((a) => a.id !== r.id), r].slice(0, 10));
  };
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(200, el.scrollHeight)}px`;
  }, [value, ref]);
  return (
    <div className="qc-composer-wrap" data-hero={hero ? 'true' : undefined} data-drop={drop ? 'true' : undefined}
      onDragOver={(e) => { if (canAdd && e.dataTransfer?.types?.includes('Files')) { e.preventDefault(); setDrop(true); } }}
      onDragLeave={() => setDrop(false)} onDrop={(e) => void dropped(e)}>
      {drop && matter && <div className="qc-drop" aria-hidden>Drop to add to {matter.number} as a draft</div>}
      <form className="qc-composer" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <textarea ref={ref} rows={hero ? 2 : 1} value={value} maxLength={2000} placeholder={placeholder} aria-label="Ask EchoMind"
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(); } }} />
        <div className="qc-composer-row" style={{ alignItems: 'center' }}>
          <div className="qc-composer-tools">
            {matterId && <DocumentPicker matterId={matterId} attached={attached} onAttach={onAttach} adding={canAdd ? matter!.number : null}
              onError={setToolError} />}
            <MatterPicker variant="composer" />
            {canDictate && <Dictate matterId={matterId!} disabled={busy} onError={setToolError}
              onText={(t) => { onChange(value.trim() ? `${value.trimEnd()} ${t}` : t); ref.current?.focus(); }} />}
          </div>
          {where.chip && <span className="qc-private qc-hide-small" data-external={where.external || undefined}><Lock aria-hidden />{where.chip}</span>}
          {busy
            ? <button type="button" className="qc-send" onClick={onStop} aria-label="Stop the answer"><Square aria-hidden /></button>
            : <button type="submit" className="qc-send" disabled={!value.trim() || !!reading} aria-label="Send"><ArrowUp aria-hidden /></button>}
        </div>
      </form>
      {toolError && <p className="qc-error" role="alert" style={{ margin: '6px 4px 0' }}>{toolError}</p>}
      <div className="qc-composer-meta">
        {matterId ? <AttachedChips matterId={matterId} attached={attached} onDetach={onDetach} onReading={setReading} />
          : attached.map((a) => (
            <span key={a.id} className="qc-attached">Only from: {a.title}
              <button type="button" className="qc-icon-btn" style={{ width: 22, height: 22 }} onClick={() => onDetach(a.id)} aria-label={`Stop using ${a.title}`}><X aria-hidden /></button>
            </span>))}
        {reading && <span role="status">Wait until “{reading}” has been read before asking.</span>}
        <span>Answers come only from the chosen matter's documents and are drafts for a lawyer's review.</span>
      </div>
    </div>
  );
}

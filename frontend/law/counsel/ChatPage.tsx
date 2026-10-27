/**
 * Chat, the home of EchoMind Legal (docs/echomind-legal/02, 04 Journeys B–D). One question at a time about one matter.
 * Text streams as research releases each checked paragraph; citation marks stay unclickable until the final, validated
 * components arrive. Cards act in place; sources open in the right-hand panel without leaving the conversation.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { ArrowUp, Square, X } from 'lucide-react';
import { ApiError } from '../api/client';
import { useMatters } from '../api/queries';
import { counselApi, requestId, type StoredTurn } from './api';
import { ActionButtons, AnswerText, ComponentView, type CardContext } from './cards';
import { useChatMatter, usePanel } from './context';
import { parseEnvelope, parseTaskProgress } from './validate';
import type { ChatTurn, SafeComponent } from './types';

const SUGGESTIONS = [
  'What matters today?',
  'Find conflicting dates or amounts',
  'Summarize this matter from its documents',
  'Which deadlines do the documents set?',
];
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
  const { matter, matterId, setMatterId } = useChatMatter();
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
  const busy = turns.some((t) => t.status === 'streaming');

  useEffect(() => {
    setLoadError(null);
    if (!conversationId) { abort.current?.abort(); setTurns([]); setConvId(undefined); created.current = null; return; }
    if (created.current === conversationId) return;          // our own new conversation: its first turn is already here
    let alive = true;
    counselApi.read(conversationId).then((r) => {
      if (!alive) return;
      setConvId(r.conversation.id);
      setMatterId(r.conversation.matter_id);
      setTurns(r.turns.map(fromStored));
    }).catch((e) => {
      if (!alive) return;
      setTurns([]);
      setLoadError(e instanceof ApiError && e.status === 404 ? 'This conversation is not available.' : 'This conversation could not be opened.');
    });
    return () => { alive = false; };
  }, [conversationId, setMatterId]);

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
        id = (await counselApi.create(matterId)).conversation.id;
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
  }, [busy, matterId, convId, attached, navigate, patch, qc]);

  useEffect(() => {                                             // a question asked before a matter was chosen
    if (pendingAsk && matterId) { const q = pendingAsk; setPendingAsk(null); void ask(q); }
  }, [pendingAsk, matterId, ask]);

  const replace = useCallback((turnKey: string) => (comp: SafeComponent) => {
    patch(turnKey, (t) => (t.envelope ? { envelope: { ...t.envelope, components: t.envelope.components.map((c) => (c.id === comp.id ? comp : c)) } } : {}));
  }, [patch]);

  const ctxFor = (t: ChatTurn): CardContext => ({
    matterId: t.envelope?.matter_id ?? matterId ?? '',
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
  return (
    <div className="qc-chat">
      <div className="qc-sr" aria-live="polite">{announce}</div>
      {loadError && <p className="qc-error" role="alert">{loadError}</p>}
      {start ? (
        <div className="qc-start">
          <h1>How can I help?</h1>
          <p className="qc-context">{matter ? `${matter.number} · ${matter.title}` : 'Choose a matter in the header, or ask and pick one.'}</p>
          <div className="qc-suggestions">
            {SUGGESTIONS.map((s) => <button key={s} type="button" className="qc-suggestion" onClick={() => void ask(s)}>{s}</button>)}
          </div>
        </div>
      ) : (
        <div className="qc-thread" aria-label="Conversation">
          {turns.map((t) => <TurnView key={t.key} t={t} ctx={ctxFor(t)} onRetry={() => void ask(t.question)} onStop={() => abort.current?.abort()} />)}
        </div>
      )}
      {pendingAsk && !matterId && (
        <MatterQuestion matters={(matters.data ?? []).slice(0, 5)} onPick={setMatterId} onCancel={() => setPendingAsk(null)} />
      )}
      <div ref={end} />
      <Composer value={draft} onChange={setDraft} busy={busy} onSend={() => void ask(draft)} onStop={() => abort.current?.abort()}
        placeholder={matter ? `Ask about ${matter.number}…` : 'Ask about a matter…'} attached={attached}
        onDetach={(id) => setAttached((a) => a.filter((x) => x.id !== id))} />
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
  const comps = t.envelope?.components ?? [];
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

function Composer({ value, onChange, busy, onSend, onStop, placeholder, attached, onDetach }: {
  value: string; onChange: (v: string) => void; busy: boolean; onSend: () => void; onStop: () => void; placeholder: string;
  attached: Attached[]; onDetach: (id: string) => void;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(200, el.scrollHeight)}px`;
  }, [value]);
  return (
    <div className="qc-composer-wrap">
      <form className="qc-composer" onSubmit={(e) => { e.preventDefault(); if (!busy) onSend(); }}>
        <textarea ref={ref} rows={1} value={value} maxLength={2000} placeholder={placeholder} aria-label="Ask EchoMind"
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); if (!busy) onSend(); } }} />
        {busy
          ? <button type="button" className="qc-send" onClick={onStop} aria-label="Stop the answer"><Square aria-hidden /></button>
          : <button type="submit" className="qc-send" disabled={!value.trim()} aria-label="Send"><ArrowUp aria-hidden /></button>}
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

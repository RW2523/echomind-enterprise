import React, { useEffect, useState } from 'react';
import { Check } from 'lucide-react';
import { useParams } from 'react-router';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import type { IntakeAnswers, IntakeLinkPage } from '../api/types';
import { ErrorNote, Skeleton } from '../ui/primitives';
import { Conversation } from '../ui/intake';

/** An enquirer's own page (S4.1, D-054), opened from a single-use link with no account: the disclaimer, the
 *  firm's questions one at a time, and later the reply a lawyer approved. Nothing else of the firm is reachable. */
export default function IntakeLink() {
  const { token = '' } = useParams();
  const path = `/api/intake-link/${encodeURIComponent(token)}`;
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ['intake-link', token], queryFn: () => api.get<IntakeLinkPage>(path), retry: false, refetchOnWindowFocus: false });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { document.title = 'Your enquiry'; }, []);
  const set = (page: IntakeLinkPage) => qc.setQueryData(['intake-link', token], page);
  const act = async (fn: () => Promise<IntakeLinkPage>, fallback: string) => {
    setError(''); setBusy(true);
    try { set(await fn()); } catch (e) { setError(e instanceof ApiError ? e.message : fallback); } finally { setBusy(false); }
  };
  let body: React.ReactNode;
  if (q.isPending) body = <><Skeleton height={28} /><Skeleton height={120} /></>;
  else if (q.isError) {
    const status = q.error instanceof ApiError ? q.error.status : 0;
    body = <p role="alert" style={{ margin: 0 }}>{status === 429 ? 'Too many requests from here. Please wait a few minutes and try again.'
      : 'This link is not valid any more. It may have expired or already been used. Please contact the firm for a new one.'}</p>;
  } else {
    const page = q.data;
    if (page.status === 'invited' && !page.disclaimer.accepted) {
      body = (
        <div style={{ display: 'grid', gap: 14 }}>
          <h2 className="lw-h2" style={{ margin: 0 }}>Before we start</h2>
          <blockquote className="lw-intake-disclaimer">{page.disclaimer.text}</blockquote>
          {error && <ErrorNote>{error}</ErrorNote>}
          <div><button type="button" className="lw-btn lw-btn-primary" disabled={busy}
            onClick={() => act(() => api.post<IntakeLinkPage>(`${path}/accept`, { version: page.disclaimer.version }), 'Something went wrong. Please try again.')}><Check />I understand, continue</button></div>
        </div>
      );
    } else if (page.status === 'invited') {
      body = <Conversation script={page.script} busy={busy} error={error}
        onSubmit={(answers: IntakeAnswers) => act(() => api.post<IntakeLinkPage>(`${path}/submit`, { answers }), 'Your answers could not be sent. Please try again.')} />;
    } else {
      body = (
        <div style={{ display: 'grid', gap: 12 }}>
          <h2 className="lw-h2" style={{ margin: 0 }}>Thank you</h2>
          <p style={{ margin: 0 }}>{`We have your answers (reference ${page.reference}). We check every enquiry before we reply; our reply will appear on this page, and we may also contact you directly.`}</p>
          {page.reply && <article className="lw-intake-reply" aria-label="Our reply"><p style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{page.reply.body}</p></article>}
        </div>
      );
    }
  }
  return (
    <main className="lw-intake-page">
      <div className="lw-glass" style={{ borderRadius: 24, padding: 6, width: 'min(640px, 100%)' }}>
        <div className="lw-paper" style={{ padding: 28, display: 'grid', gap: 18 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span className="lw-monogram" aria-hidden="true">EM</span>
            <div><h1 className="lw-h2" style={{ margin: 0 }}>Your enquiry</h1>{q.data && <div className="lw-meta lw-tnum">{`Reference ${q.data.reference}`}</div>}</div>
          </div>
          {body}
        </div>
      </div>
    </main>
  );
}

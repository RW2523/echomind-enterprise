import React, { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { keys } from '../api/queries';
import { ErrorNote, Field } from '../ui/primitives';

/** Password change. `forced`: an administrator set this password, so it must be replaced before
 *  anything else (the server refuses every other request until then — D-021). */
export default function ChangePassword({ forced = false, onDone }: { forced?: boolean; onDone?: () => void }) {
  const qc = useQueryClient();
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (next !== confirm) { setError('The new passwords do not match.'); return; }
    setBusy(true); setError('');
    try {
      await api.post('/api/auth/password', { current_password: current, new_password: next });
      setDone(true); setCurrent(''); setNext(''); setConfirm('');
      await qc.invalidateQueries({ queryKey: keys.me });
      onDone?.();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'The password could not be changed.');
    } finally { setBusy(false); }
  };

  const form = (
    <form onSubmit={submit} className="lw-paper lw-sheet" style={{ display: 'grid', gap: 14, maxWidth: 440 }}>
      <div>
        <h2 className="lw-h2">{forced ? 'Choose your own password' : 'Change password'}</h2>
        <p className="lw-meta" style={{ margin: '6px 0 0', fontSize: 13.5 }}>
          {forced ? 'Your administrator set your current password. Replace it before you continue.'
            : 'Changing your password signs you out on every other device.'} Use at least 12 characters; a few unrelated words work well.
        </p>
      </div>
      <Field label="Current password">{(id) => <input id={id} className="lw-input" type="password" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} />}</Field>
      <Field label="New password">{(id) => <input id={id} className="lw-input" type="password" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} />}</Field>
      <Field label="Repeat new password">{(id) => <input id={id} className="lw-input" type="password" autoComplete="new-password" value={confirm} onChange={(e) => setConfirm(e.target.value)} />}</Field>
      {error && <ErrorNote>{error}</ErrorNote>}
      {done && !forced && <p role="status" className="lw-meta" style={{ color: 'var(--verified)', margin: 0 }}>Password changed.</p>}
      <div><button type="submit" className="lw-btn lw-btn-primary" disabled={busy || !current || !next || !confirm}>{busy ? 'Saving…' : 'Change password'}</button></div>
    </form>
  );
  if (!forced) return form;
  return <div style={{ minHeight: '100dvh', display: 'grid', placeItems: 'center', padding: 16 }}>{form}</div>;
}

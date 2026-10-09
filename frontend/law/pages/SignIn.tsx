import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import QRCode from 'qrcode';
import { api, ApiError } from '../api/client';
import { keys } from '../api/queries';
import { CopyField, ErrorNote, Field } from '../ui/primitives';

type Step = 'password' | 'code' | 'enroll' | 'needs-link' | 'recovery';

/** Only same-app paths are accepted as a destination after sign-in (no open redirect). */
function safeNext(next: string | null): string {
  return next && next.startsWith('/') && !next.startsWith('//') ? next : '/home';
}

/**
 * Sign-in: password, then a code from the person's authenticator app. On first sign-in the account
 * is bound to an authenticator through the one-time enrolment link an administrator issued
 * (/signin?enrol=…); the QR code is drawn locally, nothing leaves the building.
 */
export default function SignIn() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const enrolToken = params.get('enrol') || '';
  const [step, setStep] = useState<Step>('password');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [code, setCode] = useState('');
  const [secret, setSecret] = useState('');
  const [qr, setQr] = useState('');
  const [recovery, setRecovery] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => { document.documentElement.dataset.edition = 'law'; }, []);

  const finish = async () => {
    await qc.invalidateQueries({ queryKey: keys.me });
    navigate(safeNext(params.get('next')), { replace: true });
  };
  const run = async (fn: () => Promise<void>) => {
    setBusy(true); setError('');
    try { await fn(); } catch (err) { setError(err instanceof ApiError ? err.message : 'Sign-in failed.'); }
    setBusy(false);
  };

  const submitPassword = (e: React.FormEvent) => {
    e.preventDefault();
    run(async () => {
      const r = await api.post<{ status: string }>('/api/auth/login', { username, password });
      setPassword('');
      if (r.status === 'ok') return finish();
      if (r.status === 'mfa_required') { setStep('code'); return; }
      if (!enrolToken) { setStep('needs-link'); return; }
      const enrol = await api.post<{ secret: string; otpauth_uri: string }>('/api/auth/mfa/enroll', { enrollment_token: enrolToken });
      setSecret(enrol.secret);
      setQr(await QRCode.toDataURL(enrol.otpauth_uri, { margin: 1, width: 192 }));
      setStep('enroll');
    });
  };

  const submitCode = (e: React.FormEvent) => {
    e.preventDefault();
    run(async () => {
      if (step === 'code') {
        await api.post('/api/auth/mfa/verify', { code });
        return finish();
      }
      const r = await api.post<{ recovery_codes: string[] }>('/api/auth/mfa/confirm', { code, enrollment_token: enrolToken });
      setRecovery(r.recovery_codes);
      setCode('');
      setStep('recovery');
    });
  };

  return (
    <div style={{ minHeight: '100dvh', display: 'grid', placeItems: 'center', padding: 16 }}>
      <div className="lw-glass" style={{ borderRadius: 24, padding: 6, width: 'min(420px, 100%)' }}>
        <div className="lw-paper" style={{ padding: 28, display: 'grid', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span className="lw-monogram" aria-hidden="true">EM</span>
            <div>
              <div style={{ fontWeight: 650, fontSize: 17 }}>EchoMind Law</div>
              <div className="lw-meta">Private legal workspace</div>
            </div>
          </div>

          {step === 'password' && (
            <form onSubmit={submitPassword} style={{ display: 'grid', gap: 14 }}>
              {enrolToken && <p className="lw-meta" style={{ margin: 0, fontSize: 13.5 }}>Sign in with the details your administrator gave you; then you'll set up your authenticator.</p>}
              <Field label="Username">{(id) => <input id={id} className="lw-input" autoComplete="username" autoFocus value={username} onChange={(e) => setUsername(e.target.value)} />}</Field>
              <Field label="Password">{(id) => <input id={id} className="lw-input" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} />}</Field>
              {error && <ErrorNote>{error}</ErrorNote>}
              <button type="submit" className="lw-btn lw-btn-primary" style={{ justifyContent: 'center' }} disabled={busy || !username || !password}>{busy ? 'Checking…' : 'Sign in'}</button>
            </form>
          )}

          {step === 'needs-link' && (
            <div style={{ display: 'grid', gap: 12 }}>
              <h2 className="lw-h2">Set up two-step sign-in</h2>
              <p style={{ margin: 0, color: 'var(--ink-2)' }}>Your account needs an authenticator app. Ask your administrator for your enrolment link, then open it on this device.</p>
              <button type="button" className="lw-btn lw-btn-ghost" onClick={() => setStep('password')}>Back</button>
            </div>
          )}

          {(step === 'code' || step === 'enroll') && (
            <form onSubmit={submitCode} style={{ display: 'grid', gap: 14 }}>
              {step === 'code' ? (
                <p style={{ margin: 0, color: 'var(--ink-2)' }}>Enter the 6-digit code from your authenticator app, or one of your recovery codes.</p>
              ) : (
                <>
                  <h2 className="lw-h2">Set up two-step sign-in</h2>
                  <p style={{ margin: 0, color: 'var(--ink-2)' }}>Scan this code with your authenticator app (or type the key), then enter the 6-digit code it shows.</p>
                  {qr && <img src={qr} alt="Authenticator QR code" width={192} height={192} style={{ justifySelf: 'center', borderRadius: 8, background: '#fff', padding: 6 }} />}
                  <CopyField label="Key" value={secret.replace(/(.{4})/g, '$1 ').trim()} />
                </>
              )}
              <Field label="Authenticator code">{(id) => <input id={id} className="lw-input lw-tnum" autoFocus inputMode="text" autoComplete="one-time-code" value={code} onChange={(e) => setCode(e.target.value)} />}</Field>
              {error && <ErrorNote>{error}</ErrorNote>}
              <button type="submit" className="lw-btn lw-btn-primary" style={{ justifyContent: 'center' }} disabled={busy || !code}>
                {busy ? 'Checking…' : step === 'code' ? 'Verify' : 'Turn on two-step sign-in'}
              </button>
            </form>
          )}

          {step === 'recovery' && (
            <div style={{ display: 'grid', gap: 12 }}>
              <h2 className="lw-h2">Save your recovery codes</h2>
              <p style={{ margin: 0, color: 'var(--ink-2)' }}>Each code works once if you lose your authenticator. They won't be shown again.</p>
              <ul data-testid="recovery-codes" className="lw-code lw-tnum" style={{ listStyle: 'none', margin: 0, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                {recovery.map((c) => <li key={c}>{c}</li>)}
              </ul>
              <button type="button" className="lw-btn lw-btn-primary" style={{ justifyContent: 'center' }} onClick={finish}>I've saved my recovery codes</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

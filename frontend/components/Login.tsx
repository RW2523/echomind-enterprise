import React, { useState } from 'react';
import { login as apiLogin, mfaConfirm, mfaEnroll, mfaVerify, AuthUser } from '../services/backend';
import { resolvePack } from '../packs';

type Step = 'password' | 'code' | 'enroll' | 'recovery';

const inputCls = 'rounded-xl bg-black/30 border border-white/10 px-4 py-3 text-base outline-none focus:border-accent/50';
const buttonCls = 'rounded-xl px-5 py-3 text-sm font-semibold bg-accent/15 text-accent border border-accent/30 hover:bg-accent/25 disabled:opacity-50 transition-colors';

/** Login gate — shown only when the backend reports auth_enabled.
 *  EchoMind Law (S1.1) adds the authenticator steps: a code after the password, or a one-time
 *  authenticator set-up (shared key + recovery codes) on first sign-in. */
const Login: React.FC<{ onSuccess: (u: AuthUser | null) => void }> = ({ onSuccess }) => {
  const pack = resolvePack();
  const [step, setStep] = useState<Step>('password');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [code, setCode] = useState('');
  const [secret, setSecret] = useState('');
  const [recoveryCodes, setRecoveryCodes] = useState<string[]>([]);
  const [pendingUser, setPendingUser] = useState<AuthUser | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const run = async (fn: () => Promise<void>) => {
    if (busy) return;
    setBusy(true);
    setError('');
    try { await fn(); } catch (err: any) { setError(err?.message || 'Sign-in failed'); }
    setBusy(false);
  };

  const submitPassword = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!username || !password) return;
    run(async () => {
      const r = await apiLogin(username, password);
      if (r.status === 'ok') { onSuccess(r.user ?? null); return; }
      setPassword('');
      if (r.status === 'mfa_required') { setStep('code'); return; }
      const enrol = await mfaEnroll();
      setSecret(enrol.secret);
      setStep('enroll');
    });
  };

  const submitCode = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!code) return;
    run(async () => {
      if (step === 'code') {
        onSuccess(await mfaVerify(code));
        return;
      }
      const r = await mfaConfirm(code);
      setPendingUser(r.user);
      setRecoveryCodes(r.recovery_codes);
      setCode('');
      setStep('recovery');
    });
  };

  const groupedSecret = secret.replace(/(.{4})/g, '$1 ').trim();

  return (
    <div className="relative flex h-full w-full items-center justify-center bg-[#05070a] text-slate-200 p-4 overflow-hidden" style={{ height: '100dvh' }}>
      <div className="absolute top-0 right-0 w-[500px] h-[500px] blur-[120px] rounded-full -translate-y-1/2 translate-x-1/2 pointer-events-none" style={{ backgroundColor: 'var(--glow-1)' }} aria-hidden />
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] blur-[100px] rounded-full translate-y-1/2 -translate-x-1/2 pointer-events-none" style={{ backgroundColor: 'var(--glow-2)' }} aria-hidden />
      <form onSubmit={step === 'password' ? submitPassword : submitCode} className="relative z-10 w-full max-w-sm rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-md p-6 sm:p-8 flex flex-col gap-4 shadow-2xl">
        <div className="flex items-center gap-3">
          <svg viewBox="0 0 36 36" className="w-9 h-9 rounded-xl shrink-0" xmlns="http://www.w3.org/2000/svg">
            <rect width="36" height="36" rx="8" fill={pack?.accent ?? '#06b6d4'} />
            <text x="18" y="25" textAnchor="middle" fill="#05070a" fontSize="20" fontFamily="system-ui, sans-serif" fontWeight="bold">E</text>
          </svg>
          <div className="min-w-0">
            <h1 className="text-lg font-bold text-white leading-none truncate">{pack?.name ?? 'EchoMind'}</h1>
            <p className="text-[11px] text-accent/80 uppercase tracking-widest font-semibold mt-1">{pack?.tagline ?? 'by Ajace AI'}</p>
          </div>
        </div>

        {step === 'password' && (
          <>
            <p className="text-sm text-slate-400">Sign in to continue.</p>
            <input autoFocus type="text" autoComplete="username" placeholder="Username" aria-label="Username"
              value={username} onChange={(e) => setUsername(e.target.value)} className={inputCls} />
            <input type="password" autoComplete="current-password" placeholder="Password" aria-label="Password"
              value={password} onChange={(e) => setPassword(e.target.value)} className={inputCls} />
          </>
        )}

        {step === 'code' && (
          <p className="text-sm text-slate-400">Enter the 6-digit code from your authenticator app, or one of your recovery codes.</p>
        )}

        {step === 'enroll' && (
          <>
            <p className="text-sm text-slate-400">
              Set up two-step sign-in. In your authenticator app, add an account with this key
              (time-based), then enter the 6-digit code it shows.
            </p>
            <code data-testid="mfa-secret" className="rounded-xl bg-black/40 border border-white/10 px-4 py-3 text-sm tracking-wider text-slate-100 break-all select-all">{groupedSecret}</code>
          </>
        )}

        {(step === 'code' || step === 'enroll') && (
          <input autoFocus inputMode="text" autoComplete="one-time-code" placeholder="Code" aria-label="Authenticator code"
            value={code} onChange={(e) => setCode(e.target.value)} className={inputCls} />
        )}

        {step === 'recovery' && (
          <>
            <p className="text-sm text-slate-400">
              Two-step sign-in is on. Save these recovery codes somewhere safe: each works once if you
              lose your authenticator. They will not be shown again.
            </p>
            <ul data-testid="recovery-codes" className="grid grid-cols-2 gap-2 rounded-xl bg-black/40 border border-white/10 px-4 py-3 font-mono text-sm text-slate-100">
              {recoveryCodes.map((c) => <li key={c}>{c}</li>)}
            </ul>
          </>
        )}

        {error && <p role="alert" className="text-sm text-red-400">{error}</p>}

        {step === 'recovery' ? (
          <button type="button" className={buttonCls} onClick={() => onSuccess(pendingUser)}>I have saved my recovery codes</button>
        ) : (
          <button type="submit" className={buttonCls}
            disabled={busy || (step === 'password' ? !username || !password : !code)}>
            {busy ? 'Checking…' : step === 'password' ? 'Sign in' : step === 'enroll' ? 'Turn on two-step sign-in' : 'Verify'}
          </button>
        )}
      </form>
    </div>
  );
};

export default Login;

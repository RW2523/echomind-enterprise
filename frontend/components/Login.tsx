import React, { useState } from 'react';
import { login as apiLogin, AuthUser } from '../services/backend';
import { resolvePack } from '../packs';

/** Login gate — shown only when the backend reports auth_enabled. */
const Login: React.FC<{ onSuccess: (u: AuthUser | null) => void }> = ({ onSuccess }) => {
  const pack = resolvePack();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const submit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!username || !password || busy) return;
    setBusy(true);
    setError('');
    try {
      const u = await apiLogin(username, password);
      onSuccess(u);
    } catch (err: any) {
      setError(err?.message || 'Login failed');
      setBusy(false);
    }
  };

  return (
    <div className="relative flex h-full w-full items-center justify-center bg-[#05070a] text-slate-200 p-4 overflow-hidden" style={{ height: '100dvh' }}>
      <div className="absolute top-0 right-0 w-[500px] h-[500px] blur-[120px] rounded-full -translate-y-1/2 translate-x-1/2 pointer-events-none" style={{ backgroundColor: 'var(--glow-1)' }} aria-hidden />
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] blur-[100px] rounded-full translate-y-1/2 -translate-x-1/2 pointer-events-none" style={{ backgroundColor: 'var(--glow-2)' }} aria-hidden />
      <form onSubmit={submit} className="relative z-10 w-full max-w-sm rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-md p-6 sm:p-8 flex flex-col gap-4 shadow-2xl">
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
        <p className="text-sm text-slate-400">Sign in to continue.</p>
        <input
          autoFocus
          type="text"
          autoComplete="username"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="rounded-xl bg-black/30 border border-white/10 px-4 py-3 text-base outline-none focus:border-accent/50"
        />
        <input
          type="password"
          autoComplete="current-password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="rounded-xl bg-black/30 border border-white/10 px-4 py-3 text-base outline-none focus:border-accent/50"
        />
        {error && <p className="text-sm text-red-400">{error}</p>}
        <button
          type="submit"
          disabled={busy || !username || !password}
          className="rounded-xl px-5 py-3 text-sm font-semibold bg-accent/15 text-accent border border-accent/30 hover:bg-accent/25 disabled:opacity-50 transition-colors"
        >
          {busy ? 'Signing in…' : 'Sign in'}
        </button>
      </form>
    </div>
  );
};

export default Login;

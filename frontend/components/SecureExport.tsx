import React, { useState } from 'react';
import { evaluateExport, ExportEvaluation } from '../services/backend';

const RISK_COLOR: Record<string, string> = {
  none: 'text-emerald-400',
  low: 'text-cyan-400',
  medium: 'text-amber-400',
  high: 'text-red-400',
};

/** Secure offline->online export gateway (#5): scan content for sensitive data + get a redacted copy. */
const SecureExport: React.FC = () => {
  const [text, setText] = useState('');
  const [res, setRes] = useState<ExportEvaluation | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    if (!text.trim() || busy) return;
    setBusy(true);
    try { setRes(await evaluateExport(text)); } catch (_) {}
    setBusy(false);
  };

  return (
    <section className="flex flex-col gap-4">
      <div>
        <h3 className="text-lg sm:text-xl font-bold text-white mb-1">Secure Export Gateway</h3>
        <p className="text-xs text-slate-500">Scan content for sensitive data (PII / secrets) and get a redacted copy before it leaves the box. Runs fully offline — nothing is sent anywhere.</p>
      </div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste content to evaluate before export…"
        rows={5}
        className="w-full rounded-xl bg-black/30 border border-white/10 px-4 py-3 text-sm outline-none focus:border-accent/50 resize-y"
      />
      <div className="flex items-center gap-3 flex-wrap">
        <button
          type="button"
          onClick={run}
          disabled={busy || !text.trim()}
          className="rounded-xl px-4 py-2 text-sm font-semibold bg-accent/15 text-accent border border-accent/30 hover:bg-accent/25 disabled:opacity-50 transition-colors"
        >
          {busy ? 'Scanning…' : 'Evaluate'}
        </button>
        {res && (
          <span className={`text-sm font-semibold ${RISK_COLOR[res.risk_level] ?? 'text-slate-400'}`}>
            Risk: {res.risk_level.toUpperCase()} · {res.finding_count} finding{res.finding_count === 1 ? '' : 's'} · {res.safe_to_export ? 'safe to export' : 'redaction recommended'}
          </span>
        )}
      </div>
      {res && res.findings.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {res.findings.map((f, i) => (
            <span key={i} className="text-[11px] px-2 py-1 rounded-lg bg-white/5 border border-white/10 text-slate-300">{f.type}: {f.preview}</span>
          ))}
        </div>
      )}
      {res && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Redacted (safe to export)</span>
            <button type="button" onClick={() => navigator.clipboard?.writeText(res.redacted_text)} className="text-xs text-accent hover:opacity-80 transition-colors">Copy</button>
          </div>
          <pre className="text-sm text-white/90 whitespace-pre-wrap break-words bg-black/30 rounded-xl p-3 border border-white/10">{res.redacted_text}</pre>
        </div>
      )}
    </section>
  );
};

export default SecureExport;

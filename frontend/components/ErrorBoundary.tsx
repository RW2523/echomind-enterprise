import React from 'react';

interface Props {
  children: React.ReactNode;
  /** Short label for the area this boundary protects (e.g. "Document Studio"). */
  label?: string;
}

interface State {
  error: Error | null;
  componentStack: string;
}

/**
 * Catches render/runtime errors in its subtree so a single component fault can
 * never blank the entire app. It shows the actual error (so it is diagnosable
 * instead of a white screen) and best-effort reports it to the backend, which
 * logs it for server-side debugging.
 */
export default class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null, componentStack: '' };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    const componentStack = info?.componentStack || '';
    this.setState({ componentStack });
    // Always log to the browser console.
    // eslint-disable-next-line no-console
    console.error('[EchoMind] UI crash in', this.props.label || 'app', error, componentStack);
    // Best-effort report so the error is visible in backend logs even if the
    // user cannot read the browser console (e.g. on a remote/deployed instance).
    try {
      fetch('/api/client-error', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          where: this.props.label || 'app',
          message: String(error?.message || error),
          stack: String(error?.stack || ''),
          componentStack: String(componentStack),
          url: typeof location !== 'undefined' ? location.href : '',
          ua: typeof navigator !== 'undefined' ? navigator.userAgent : '',
        }),
        keepalive: true,
      }).catch(() => {});
    } catch {
      /* never let reporting throw */
    }
  }

  private handleDismiss = () => this.setState({ error: null, componentStack: '' });

  private handleReload = () => {
    if (typeof location !== 'undefined') location.reload();
  };

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="h-full min-h-0 w-full flex items-center justify-center p-6">
        <div className="max-w-lg w-full rounded-2xl border border-red-500/30 bg-red-500/[0.06] p-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-9 h-9 rounded-xl bg-red-500/20 border border-red-500/30 flex items-center justify-center shrink-0">
              <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m0 3.75h.01M10.34 3.94l-7.2 12.45A1.5 1.5 0 004.45 18.6h15.1a1.5 1.5 0 001.31-2.21l-7.2-12.45a1.5 1.5 0 00-2.62 0z" />
              </svg>
            </div>
            <div className="min-w-0">
              <div className="text-sm font-semibold text-white">
                {this.props.label ? `${this.props.label} hit an error` : 'Something went wrong'}
              </div>
              <div className="text-xs text-slate-400">The rest of the app is still working.</div>
            </div>
          </div>

          <pre className="text-[11px] leading-relaxed text-red-200/90 bg-black/30 border border-white/10 rounded-lg p-3 overflow-auto max-h-40 whitespace-pre-wrap break-words">
            {String(error.message || error)}
          </pre>

          <div className="mt-4 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={this.handleDismiss}
              className="rounded-xl px-3.5 py-1.5 text-xs font-semibold bg-white/10 text-slate-200 hover:bg-white/15 border border-white/10 transition-colors"
            >
              Try again
            </button>
            <button
              type="button"
              onClick={this.handleReload}
              className="rounded-xl px-3.5 py-1.5 text-xs font-semibold bg-cyan-500/20 text-cyan-200 hover:bg-cyan-500/30 border border-cyan-500/30 transition-colors"
            >
              Reload page
            </button>
          </div>
        </div>
      </div>
    );
  }
}

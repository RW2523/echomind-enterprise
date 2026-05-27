import React, { useState, useCallback, useEffect, useRef } from 'react';
import type { BoardroomSession, DiarizedSegment, MeetingReport } from '../types';
import { analyseBoardroomSession, getBoardroomSession, boardroomExportUrl } from '../services/backend';
import { ICONS } from '../constants';

// Speaker colours (cycling)
const SPEAKER_COLOURS = [
  { bg: 'bg-cyan-500/10 border-cyan-500/30', label: 'text-cyan-400', dot: 'bg-cyan-400' },
  { bg: 'bg-violet-500/10 border-violet-500/30', label: 'text-violet-400', dot: 'bg-violet-400' },
  { bg: 'bg-emerald-500/10 border-emerald-500/30', label: 'text-emerald-400', dot: 'bg-emerald-400' },
  { bg: 'bg-amber-500/10 border-amber-500/30', label: 'text-amber-400', dot: 'bg-amber-400' },
];

function getSpeakerColour(speaker: string): typeof SPEAKER_COLOURS[0] {
  const idx = parseInt(speaker.replace(/\D/g, '') || '1', 10) - 1;
  return SPEAKER_COLOURS[idx % SPEAKER_COLOURS.length];
}

function formatTime(sec?: number): string {
  if (sec == null) return '';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

interface BoardroomViewProps {
  session: BoardroomSession;
  onSessionUpdate: (s: BoardroomSession) => void;
  onClose: () => void;
}

const BoardroomView: React.FC<BoardroomViewProps> = ({ session, onSessionUpdate, onClose }) => {
  const [analysing, setAnalysing] = useState(false);
  const [activeTab, setActiveTab] = useState<'transcript' | 'report'>('transcript');
  const [exportLoading, setExportLoading] = useState<'pdf' | 'pptx' | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const pollStatus = useCallback((sid: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await getBoardroomSession(sid);
        onSessionUpdate(s);
        if (['transcribed', 'analysed', 'error'].includes(s.status)) {
          clearInterval(pollRef.current!);
          pollRef.current = null;
        }
      } catch {}
    }, 3000);
  }, [onSessionUpdate]);

  useEffect(() => {
    if (session.status === 'processing') {
      pollStatus(session.id);
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [session.id, session.status, pollStatus]);

  const handleAnalyse = useCallback(async () => {
    setAnalysing(true);
    try {
      await analyseBoardroomSession(session.id);
      onSessionUpdate({ ...session, status: 'analysing' });
      pollStatus(session.id);
    } catch (e) {
      console.error('Analyse failed:', e);
    } finally {
      setAnalysing(false);
    }
  }, [session, onSessionUpdate, pollStatus]);

  const handleExport = useCallback(async (format: 'pdf' | 'pptx') => {
    setExportLoading(format);
    try {
      const url = boardroomExportUrl(session.id, format);
      const a = document.createElement('a');
      a.href = url;
      a.download = `boardroom_${session.id.slice(0, 8)}.${format}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (e) {
      console.error('Export failed:', e);
    } finally {
      setExportLoading(null);
    }
  }, [session.id]);

  const segments: DiarizedSegment[] = session.diarized_segments || [];
  const report: MeetingReport | null = session.report || null;

  return (
    <div className="h-full flex flex-col rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
      {/* Header */}
      <div className="shrink-0 flex items-center gap-3 px-4 py-3 border-b border-white/10 bg-black/10">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-xl bg-violet-500/20 border border-violet-500/30 flex items-center justify-center">
            <svg className="w-4 h-4 text-violet-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z"/>
            </svg>
          </div>
          <div>
            <div className="text-sm font-semibold text-white">Boardroom</div>
            <div className="text-xs text-slate-400">Session {session.id.slice(0, 8)}…</div>
          </div>
        </div>

        {/* Status badge */}
        <div className="ml-2">
          {session.status === 'processing' && (
            <span className="inline-flex items-center gap-1.5 text-xs text-amber-400 bg-amber-500/10 border border-amber-500/30 rounded-full px-2.5 py-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
              Transcribing…
            </span>
          )}
          {session.status === 'analysing' && (
            <span className="inline-flex items-center gap-1.5 text-xs text-cyan-400 bg-cyan-500/10 border border-cyan-500/30 rounded-full px-2.5 py-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
              Analysing…
            </span>
          )}
          {session.status === 'transcribed' && (
            <span className="inline-flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 rounded-full px-2.5 py-0.5">
              ✓ Transcribed
            </span>
          )}
          {session.status === 'analysed' && (
            <span className="inline-flex items-center gap-1.5 text-xs text-violet-400 bg-violet-500/10 border border-violet-500/30 rounded-full px-2.5 py-0.5">
              ✓ Report Ready
            </span>
          )}
          {session.status === 'error' && (
            <span className="text-xs text-red-400">Error</span>
          )}
        </div>

        <div className="ml-auto flex items-center gap-2">
          {/* Analyse button */}
          {(session.status === 'transcribed' || (session.status === 'analysed' && !report)) && (
            <button
              type="button"
              onClick={handleAnalyse}
              disabled={analysing}
              className="rounded-xl px-3 py-1.5 text-xs font-semibold bg-violet-500/20 text-violet-400 border border-violet-500/30 hover:bg-violet-500/30 disabled:opacity-50 transition-colors"
            >
              {analysing ? 'Analysing…' : 'Analyse Meeting'}
            </button>
          )}

          {/* Export buttons */}
          {report && (
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => handleExport('pdf')}
                disabled={exportLoading !== null}
                className="rounded-xl px-3 py-1.5 text-xs font-semibold bg-white/10 text-slate-300 hover:bg-white/15 border border-white/10 disabled:opacity-50 transition-colors"
              >
                {exportLoading === 'pdf' ? '…' : 'PDF'}
              </button>
              <button
                type="button"
                onClick={() => handleExport('pptx')}
                disabled={exportLoading !== null}
                className="rounded-xl px-3 py-1.5 text-xs font-semibold bg-white/10 text-slate-300 hover:bg-white/15 border border-white/10 disabled:opacity-50 transition-colors"
              >
                {exportLoading === 'pptx' ? '…' : 'PPT'}
              </button>
            </div>
          )}

          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
            aria-label="Close boardroom view"
          >
            <ICONS.Close className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Tabs */}
      {segments.length > 0 && (
        <div className="shrink-0 flex gap-1 px-4 pt-2">
          <button
            type="button"
            onClick={() => setActiveTab('transcript')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
              activeTab === 'transcript'
                ? 'bg-white/10 text-white'
                : 'text-slate-400 hover:text-white hover:bg-white/5'
            }`}
          >
            Transcript
          </button>
          {report && (
            <button
              type="button"
              onClick={() => setActiveTab('report')}
              className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
                activeTab === 'report'
                  ? 'bg-white/10 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}
            >
              AI Report
            </button>
          )}
        </div>
      )}

      {/* Body */}
      <div className="flex-1 min-h-0 overflow-auto p-4">
        {/* Loading / empty states */}
        {session.status === 'recording' && (
          <div className="h-full flex flex-col items-center justify-center gap-3 text-center">
            <div className="w-12 h-12 rounded-2xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
              <svg className="w-6 h-6 text-violet-400 animate-pulse" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              </svg>
            </div>
            <p className="text-sm text-slate-400">Recording in Boardroom Mode…</p>
            <p className="text-xs text-slate-500">Stop the session to upload and transcribe audio.</p>
          </div>
        )}

        {(session.status === 'processing' || session.status === 'analysing') && (
          <div className="h-full flex flex-col items-center justify-center gap-3 text-center">
            <div className="flex gap-1">
              {[0, 1, 2].map((i) => (
                <div key={i} className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
              ))}
            </div>
            <p className="text-sm text-slate-400">
              {session.status === 'processing' ? 'Transcribing audio with VibeVoice-ASR…' : 'Analysing meeting with AI…'}
            </p>
          </div>
        )}

        {/* Transcript tab */}
        {activeTab === 'transcript' && segments.length > 0 && (
          <div className="space-y-3">
            {segments.map((seg, i) => {
              const col = getSpeakerColour(seg.speaker);
              return (
                <div key={i} className={`rounded-xl border p-3.5 ${col.bg}`}>
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className={`w-2 h-2 rounded-full shrink-0 ${col.dot}`} />
                    <span className={`text-xs font-bold ${col.label}`}>{seg.speaker}</span>
                    {(seg.start_time != null) && (
                      <span className="ml-auto text-[10px] text-slate-500">{formatTime(seg.start_time)}</span>
                    )}
                  </div>
                  <p className="text-sm text-slate-200 leading-relaxed pl-4">{seg.text}</p>
                </div>
              );
            })}
          </div>
        )}

        {/* Report tab */}
        {activeTab === 'report' && report && (
          <div className="space-y-6">
            {report.overall_sentiment && (
              <div className="inline-flex items-center gap-2 text-xs font-medium rounded-full px-3 py-1.5 bg-white/5 border border-white/10 text-slate-300">
                Sentiment: <span className="text-white font-semibold capitalize">{report.overall_sentiment}</span>
              </div>
            )}

            {report.executive_summary && (
              <ReportSection title="Executive Summary" icon="📋">
                <p className="text-sm text-slate-200 leading-relaxed">{report.executive_summary}</p>
              </ReportSection>
            )}

            {report.key_topics && report.key_topics.length > 0 && (
              <ReportSection title="Key Topics" icon="💡">
                <div className="flex flex-wrap gap-2">
                  {report.key_topics.map((t, i) => (
                    <span key={i} className="text-xs rounded-lg bg-white/10 border border-white/10 px-2.5 py-1 text-slate-200">{t}</span>
                  ))}
                </div>
              </ReportSection>
            )}

            {report.speakers && report.speakers.length > 0 && (
              <ReportSection title="Speaker Breakdown" icon="👥">
                <div className="space-y-3">
                  {report.speakers.map((spk, i) => {
                    const col = getSpeakerColour(spk.speaker);
                    return (
                      <div key={i} className={`rounded-xl border p-3 ${col.bg}`}>
                        <div className={`text-xs font-bold ${col.label} mb-1`}>{spk.speaker}</div>
                        <p className="text-sm text-slate-200 leading-relaxed">{spk.summary}</p>
                        {spk.key_points && spk.key_points.length > 0 && (
                          <ul className="mt-2 space-y-1">
                            {spk.key_points.map((pt, j) => (
                              <li key={j} className="text-xs text-slate-300 flex gap-2">
                                <span className="text-slate-500">•</span>
                                {pt}
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    );
                  })}
                </div>
              </ReportSection>
            )}

            {report.rag_verified_facts && report.rag_verified_facts.length > 0 && (
              <ReportSection title="RAG-Verified Facts" icon="✅">
                <ul className="space-y-2">
                  {report.rag_verified_facts.map((f, i) => (
                    <li key={i} className="flex gap-2 text-sm text-emerald-300">
                      <span className="shrink-0 text-emerald-500">✓</span>
                      {f}
                    </li>
                  ))}
                </ul>
              </ReportSection>
            )}

            {report.contradictions && report.contradictions.length > 0 && (
              <ReportSection title="Contradictions / Risks" icon="⚠️">
                <ul className="space-y-2">
                  {report.contradictions.map((c, i) => (
                    <li key={i} className="flex gap-2 text-sm text-amber-300">
                      <span className="shrink-0">⚠</span>
                      {c}
                    </li>
                  ))}
                </ul>
              </ReportSection>
            )}

            {report.recommendations && report.recommendations.length > 0 && (
              <ReportSection title="Recommendations" icon="→">
                <ul className="space-y-2">
                  {report.recommendations.map((r, i) => (
                    <li key={i} className="flex gap-2 text-sm text-cyan-200">
                      <span className="shrink-0 text-cyan-400">→</span>
                      {r}
                    </li>
                  ))}
                </ul>
              </ReportSection>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const ReportSection: React.FC<{ title: string; icon: string; children: React.ReactNode }> = ({ title, icon, children }) => (
  <div>
    <div className="flex items-center gap-2 mb-2.5">
      <span className="text-base">{icon}</span>
      <h3 className="text-sm font-semibold text-white">{title}</h3>
    </div>
    <div className="pl-1">{children}</div>
  </div>
);

export default BoardroomView;

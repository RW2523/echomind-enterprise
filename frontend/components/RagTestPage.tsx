import React, { useState, useCallback, useRef } from 'react';
import { ICONS } from '../constants';
import { ragDebug, RagDebugResult, getDataPreview, DataPreview, chunkPreview, ChunkPreviewResult, storeTranscript } from '../services/backend';
import { AppSettings } from '../types';

/** Sample transcripts with different dates for Transcript E2E test. echodate is ISO datetime. */
const SAMPLE_TRANSCRIPTS: { echodate: string; name: string; rawText: string; tags: string[] }[] = [
  {
    echodate: '2025-02-18T10:00:00.000Z',
    name: 'Planning kickoff',
    tags: ['planning', 'Q1', 'goals'],
    rawText: `Meeting: Q1 planning kickoff.
Attendees: Alex, Sam, Jordan.
Decisions: We will ship the new dashboard by March 15. Marketing to run the campaign from March 1. Engineering to focus on performance and accessibility.
Action items: Alex to send design specs by Feb 22. Sam to book the launch review for March 10.`,
  },
  {
    echodate: '2025-02-19T14:30:00.000Z',
    name: 'Budget review',
    tags: ['budget', 'finance', 'approvals'],
    rawText: `Budget review call.
Approved: Additional 20k for cloud infrastructure. 15k for contractor support in April.
Deferred: New hire for support team – revisit in April.
Summary: We are on track for Q1. Next review scheduled for March 5.`,
  },
  {
    echodate: '2025-02-20T09:15:00.000Z',
    name: 'Product sync',
    tags: ['product', 'roadmap', 'feedback'],
    rawText: `Product sync with customer success.
Top feedback: Users want bulk export and better filters. We will add both to the April release.
Roadmap: Beta for new API in March. GA in April. Documentation sprint next week.`,
  },
  {
    echodate: '2025-02-21T11:00:00.000Z',
    name: 'Standup week summary',
    tags: ['standup', 'blockers', 'progress'],
    rawText: `Week summary from standups.
Done: Auth refactor merged. Onboarding flow updated. Docs for API v2 published.
Blockers: Waiting on legal for terms of service. No blockers for engineering.
Next week: Focus on performance and starting the bulk export feature.`,
  },
  {
    echodate: '2025-02-22T16:00:00.000Z',
    name: 'Retrospective',
    tags: ['retro', 'improvements', 'team'],
    rawText: `Sprint retrospective.
What went well: Clear priorities. Good collaboration with design.
What to improve: Earlier dependency checks. Shorter refinement sessions.
Action: We will try async refinement and a short Monday dependency review.`,
  },
];

interface RagTestPageProps {
  settings: AppSettings;
}

export default function RagTestPage({ settings }: RagTestPageProps) {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RagDebugResult | null>(null);
  const [dataPreview, setDataPreview] = useState<DataPreview | null>(null);
  const [chunkingExpanded, setChunkingExpanded] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [chunkPreviewResult, setChunkPreviewResult] = useState<ChunkPreviewResult | null>(null);
  const [chunkPreviewLoading, setChunkPreviewLoading] = useState(false);
  const [chunkPreviewError, setChunkPreviewError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [transcriptSamplesLoading, setTranscriptSamplesLoading] = useState(false);
  const [transcriptSamplesError, setTranscriptSamplesError] = useState<string | null>(null);
  const [transcriptSamplesAdded, setTranscriptSamplesAdded] = useState(0);
  const [manualContent, setManualContent] = useState('');
  const [manualDate, setManualDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [manualTags, setManualTags] = useState('');
  const [manualName, setManualName] = useState('');
  const [manualAdding, setManualAdding] = useState(false);
  const [manualError, setManualError] = useState<string | null>(null);
  const [manualSuccess, setManualSuccess] = useState<string | null>(null);

  const PRESET_TAGS = ['planning', 'budget', 'product', 'meeting', 'action-items', 'retro', 'standup', 'decisions'];

  const addSampleTranscripts = useCallback(async () => {
    setTranscriptSamplesError(null);
    setTranscriptSamplesLoading(true);
    let added = 0;
    try {
      for (const s of SAMPLE_TRANSCRIPTS) {
        await storeTranscript(s.rawText, { echodate: s.echodate, name: s.name, tags: s.tags });
        added += 1;
      }
      setTranscriptSamplesAdded(added);
    } catch (e) {
      setTranscriptSamplesError(e instanceof Error ? e.message : 'Failed to add sample transcripts');
    } finally {
      setTranscriptSamplesLoading(false);
    }
  }, []);

  const addManualTranscript = useCallback(async () => {
    const text = manualContent.trim();
    if (!text) {
      setManualError('Enter transcript content.');
      return;
    }
    setManualError(null);
    setManualSuccess(null);
    setManualAdding(true);
    try {
      const tagsList = manualTags
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean);
      const echodate = `${manualDate}T12:00:00.000Z`;
      await storeTranscript(text, {
        echodate,
        name: manualName.trim() || undefined,
        tags: tagsList.length > 0 ? tagsList : undefined,
      });
      setManualSuccess('Transcript added.');
      setManualContent('');
      setManualName('');
      setManualTags('');
    } catch (e) {
      setManualError(e instanceof Error ? e.message : 'Failed to add transcript');
    } finally {
      setManualAdding(false);
    }
  }, [manualContent, manualDate, manualTags, manualName]);

  const addPresetTag = useCallback((tag: string) => {
    setManualTags((prev) => {
      const list = prev.split(',').map((t) => t.trim()).filter(Boolean);
      if (list.includes(tag)) return prev;
      return list.concat(tag).join(', ');
    });
  }, []);

  const runRag = useCallback(async () => {
    const q = question.trim();
    if (!q) {
      setError('Enter a question.');
      return;
    }
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const res = await ragDebug(q, settings.advancedRag);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'RAG debug failed');
    } finally {
      setLoading(false);
    }
  }, [question, settings.advancedRag]);

  const loadDataPreview = useCallback(async () => {
    if (dataPreview !== null) {
      setChunkingExpanded((v) => !v);
      return;
    }
    setPreviewLoading(true);
    try {
      const p = await getDataPreview();
      setDataPreview(p);
      setChunkingExpanded(true);
    } catch {
      setError('Failed to load data preview.');
    } finally {
      setPreviewLoading(false);
    }
  }, [dataPreview]);

  const runChunkPreview = useCallback(async () => {
    const input = fileInputRef.current;
    if (!input?.files?.length) {
      setChunkPreviewError('Choose a PDF or document first.');
      return;
    }
    const file = input.files[0];
    setChunkPreviewError(null);
    setChunkPreviewResult(null);
    setChunkPreviewLoading(true);
    try {
      const res = await chunkPreview(file);
      setChunkPreviewResult(res);
    } catch (e) {
      setChunkPreviewError(e instanceof Error ? e.message : 'Chunk preview failed');
    } finally {
      setChunkPreviewLoading(false);
    }
  }, []);

  return (
    <div className="flex flex-col gap-6 max-w-4xl mx-auto">
      <div>
        <h1 className="text-xl font-semibold text-white flex items-center gap-2">
          <ICONS.Search className="w-6 h-6 text-cyan-400" />
          RAG Test — End-to-end API flow
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Ask a question and see the context retrieved from embeddings and the final answer.
        </p>
      </div>

      {/* Chunking preview: upload PDF/document and see how it is chunked */}
      <div className="rounded-xl border border-white/10 bg-white/5 p-4">
        <h2 className="text-base font-semibold text-white flex items-center gap-2 mb-2">
          <ICONS.File className="w-5 h-5 text-cyan-400" />
          Chunking preview
        </h2>
        <p className="text-slate-400 text-sm mb-3">
          Upload a PDF (or DOCX/PPTX/TXT) to see how it is parsed and split into chunks. Only &quot;embed&quot; chunks are passed to the embedding model and index; section headers are not embedded.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.pptx,.txt"
            className="text-slate-400 text-sm file:mr-2 file:py-2 file:px-3 file:rounded-lg file:border-0 file:bg-slate-700 file:text-cyan-300 file:font-medium"
          />
          <button
            type="button"
            onClick={runChunkPreview}
            disabled={chunkPreviewLoading}
            className="px-4 py-2 rounded-lg bg-violet-500/20 text-violet-300 border border-violet-500/30 hover:bg-violet-500/30 disabled:opacity-50 font-medium"
          >
            {chunkPreviewLoading ? 'Chunking…' : 'Preview chunking'}
          </button>
        </div>
        {chunkPreviewError && (
          <p className="text-red-300 text-sm mt-2">{chunkPreviewError}</p>
        )}
        {chunkPreviewResult && (
          <div className="mt-4 space-y-3 border-t border-white/5 pt-4">
            <div className="flex flex-wrap gap-4 text-sm">
              <span className="text-slate-400">File: <span className="text-white">{chunkPreviewResult.filename}</span></span>
              <span className="text-slate-400">Type: <span className="text-white">{chunkPreviewResult.filetype}</span></span>
              <span className="text-slate-400">Detected: <span className="text-cyan-400 font-mono">{chunkPreviewResult.doc_type}</span></span>
              <span className="text-slate-400">Extracted: <span className="text-white">{chunkPreviewResult.extracted_length.toLocaleString()} chars</span></span>
              <span className="text-slate-400">Chunks: <span className="text-white">{chunkPreviewResult.total_chunks}</span> total, <span className="text-cyan-400">{chunkPreviewResult.embed_count}</span> sent to embedding</span>
            </div>
            <ul className="space-y-2 max-h-[400px] overflow-y-auto">
              {chunkPreviewResult.chunks.map((c) => (
                <li key={c.chunk_index} className="rounded-lg bg-slate-800/60 border border-white/5 p-3 text-left">
                  <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400 mb-1">
                    <span className="font-mono">#{c.chunk_index}</span>
                    <span className={c.is_parent ? 'text-amber-400' : 'text-cyan-400'}>
                      {c.is_parent ? 'Section header (not embedded)' : 'Embed chunk'}
                    </span>
                    {c.section && <span>• {c.section}</span>}
                    <span>• {c.char_count} chars</span>
                  </div>
                  <p className="text-slate-200 text-sm whitespace-pre-wrap break-words">{c.text.length > 500 ? c.text.slice(0, 500) + '…' : c.text}</p>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Transcript E2E: add sample transcripts with different dates, then ask for summary */}
      <div className="rounded-xl border border-white/10 bg-white/5 p-4">
        <h2 className="text-base font-semibold text-white flex items-center gap-2 mb-2">
          <ICONS.Transcript className="w-5 h-5 text-cyan-400" />
          Transcript E2E test
        </h2>
        <p className="text-slate-400 text-sm mb-3">
          Add sample transcripts (planning, budget, product, standup, retro) with dates Feb 18–22. Then use the question box below to ask for a summary or &quot;What did we discuss last week?&quot;
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={addSampleTranscripts}
            disabled={transcriptSamplesLoading}
            className="px-4 py-2 rounded-lg bg-violet-500/20 text-violet-300 border border-violet-500/30 hover:bg-violet-500/30 disabled:opacity-50 font-medium"
          >
            {transcriptSamplesLoading ? 'Adding…' : 'Add sample transcripts to knowledge base'}
          </button>
          {transcriptSamplesAdded > 0 && (
            <span className="text-cyan-400 text-sm">Added {transcriptSamplesAdded} transcript(s).</span>
          )}
        </div>
        {transcriptSamplesError && (
          <p className="text-red-300 text-sm mt-2">{transcriptSamplesError}</p>
        )}
        <details className="mt-3 text-sm">
          <summary className="cursor-pointer text-slate-400 hover:text-white">Sample content (dates and topics)</summary>
          <ul className="mt-2 space-y-2 text-slate-400">
            {SAMPLE_TRANSCRIPTS.map((s, i) => (
              <li key={i}>
                <span className="text-cyan-400 font-mono">{s.echodate.slice(0, 10)}</span> — {s.name} ({s.tags.join(', ')})
              </li>
            ))}
          </ul>
        </details>

        <div className="mt-5 pt-4 border-t border-white/5">
          <h3 className="text-sm font-medium text-slate-300 mb-3">Add transcript manually</h3>
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-slate-500 mb-1">Content</label>
              <textarea
                value={manualContent}
                onChange={(e) => setManualContent(e.target.value)}
                placeholder="Paste or type transcript text…"
                rows={4}
                className="w-full rounded-lg bg-slate-800/80 border border-white/10 px-3 py-2 text-white placeholder-slate-500 focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/30 outline-none text-sm resize-y min-h-[80px]"
              />
            </div>
            <div className="flex flex-wrap gap-4">
              <div className="flex-1 min-w-[140px]">
                <label className="block text-xs text-slate-500 mb-1">Date</label>
                <input
                  type="date"
                  value={manualDate}
                  onChange={(e) => setManualDate(e.target.value)}
                  className="w-full rounded-lg bg-slate-800/80 border border-white/10 px-3 py-2 text-white focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/30 outline-none text-sm"
                />
              </div>
              <div className="flex-1 min-w-[140px]">
                <label className="block text-xs text-slate-500 mb-1">Name (optional)</label>
                <input
                  type="text"
                  value={manualName}
                  onChange={(e) => setManualName(e.target.value)}
                  placeholder="e.g. Team standup"
                  className="w-full rounded-lg bg-slate-800/80 border border-white/10 px-3 py-2 text-white placeholder-slate-500 focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/30 outline-none text-sm"
                />
              </div>
            </div>
            <div>
              <label className="block text-xs text-slate-500 mb-1">Tags (comma-separated or click below)</label>
              <input
                type="text"
                value={manualTags}
                onChange={(e) => setManualTags(e.target.value)}
                placeholder="e.g. meeting, planning, Q1"
                className="w-full rounded-lg bg-slate-800/80 border border-white/10 px-3 py-2 text-white placeholder-slate-500 focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/30 outline-none text-sm"
              />
              <div className="flex flex-wrap gap-1.5 mt-1.5">
                {PRESET_TAGS.map((tag) => (
                  <button
                    key={tag}
                    type="button"
                    onClick={() => addPresetTag(tag)}
                    className="px-2 py-1 rounded-md text-xs border border-white/10 bg-slate-800/60 text-slate-400 hover:border-cyan-500/30 hover:text-cyan-400 transition-colors"
                  >
                    + {tag}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              <button
                type="button"
                onClick={addManualTranscript}
                disabled={manualAdding}
                className="px-4 py-2 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/30 disabled:opacity-50 font-medium text-sm"
              >
                {manualAdding ? 'Adding…' : 'Add transcript'}
              </button>
              {manualSuccess && <span className="text-cyan-400 text-sm">{manualSuccess}</span>}
              {manualError && <span className="text-red-300 text-sm">{manualError}</span>}
            </div>
          </div>
        </div>

        {/* How time/date is considered and how to test it */}
        <details className="mt-5 pt-4 border-t border-white/5 text-sm">
          <summary className="cursor-pointer font-medium text-slate-300 hover:text-white">
            How time/date in transcripts is considered and tested
          </summary>
          <div className="mt-3 space-y-3 text-slate-400">
            <p><strong className="text-slate-200">Stored time (echodate)</strong> — When you add a transcript (sample or manual), the date you choose is sent as <code className="text-cyan-400/90">echodate</code> (ISO). It is stored in the DB and in the RAG index metadata for every chunk from that transcript. Filtering uses this date.</p>
            <p><strong className="text-slate-200">Single reference time</strong> — For each question the backend fixes one &quot;now&quot; (<code className="text-cyan-400/90">reference_ts</code>) for the whole request. All time-based filters use that same instant, so &quot;last 24hrs&quot; and &quot;yesterday&quot; are consistent.</p>
            <p><strong className="text-slate-200">How the question is parsed</strong> — The question is scanned and only matching transcript chunks are kept (all times stored and compared in <strong>UTC with seconds</strong>):</p>
            <ul className="list-disc list-inside space-y-1 ml-2">
              <li><strong className="text-slate-300">Specific time or range</strong> — &quot;at 2pm&quot;, &quot;at 14:00 on Feb 20&quot;, &quot;between 2pm and 3pm&quot;, &quot;from 14:00 to 15:00 on Feb 20&quot;. Interpreted in the query timezone (default UTC), converted to UTC, then only chunks whose <code className="text-cyan-400/90">echodate</code> falls in that window are kept.</li>
              <li><strong className="text-slate-300">Specific date</strong> — &quot;today&quot;, &quot;yesterday&quot;, or a calendar date (e.g. &quot;Feb 20&quot;, &quot;2025-02-20&quot;). Keeps only chunks whose transcript <code className="text-cyan-400/90">echodate</code> falls on that day.</li>
              <li><strong className="text-slate-300">Last N transcripts</strong> — &quot;last 2 transcripts&quot;, &quot;pick last 2&quot;, &quot;summary of last 3&quot;. Keeps only chunks from the N most recent transcript documents.</li>
              <li><strong className="text-slate-300">Time window</strong> — &quot;last 24hrs&quot;, &quot;48hrs&quot;, &quot;last 5 mins&quot;. Keeps only chunks whose <code className="text-cyan-400/90">echodate</code> ≥ (reference_ts − window).</li>
              <li><strong className="text-slate-300">Latest/recent</strong> (no number) — &quot;recent transcript&quot;, &quot;latest transcript&quot;. Keeps only the single most recent transcript.</li>
            </ul>
            <p><strong className="text-slate-200">How to test</strong> — After adding the sample transcripts (Feb 18–22, times at 10:00 / 14:30 / 09:15 / 11:00 / 16:00 UTC), try:</p>
            <ul className="list-disc list-inside space-y-1 ml-2">
              <li> &quot;Transcript at 14:30 on Feb 19&quot; or &quot;What was said at 2:30pm on 2025-02-19?&quot; (UTC) → only Budget review.</li>
              <li> &quot;Between 09:00 and 10:00 on Feb 20&quot; → Product sync (09:15 UTC).</li>
              <li> &quot;Summarize transcripts from February 20&quot; → only Product sync (Feb 20).</li>
              <li> &quot;Last 2 transcripts&quot; or &quot;Pick last 2 transcripts&quot; → Standup (Feb 21) + Retro (Feb 22).</li>
              <li> &quot;Last 24hrs&quot; / &quot;48hrs&quot; → transcripts whose echodate is within that window from the request time.</li>
            </ul>
            <p className="text-slate-500 text-xs">Check the <strong>Intent</strong> and <strong>Retrieved context</strong> in the result: intent should be <code className="text-cyan-400/90">transcript</code>, and the chunks listed should match the date/window you asked for.</p>
          </div>
        </details>

        <p className="text-slate-500 text-xs mt-3">
          Try: &quot;Summarize my transcripts&quot; · &quot;What did we decide about the budget?&quot; · &quot;What are the action items from last week?&quot; · &quot;Last 2 transcripts&quot; · &quot;Transcripts from Feb 20&quot;
        </p>
      </div>

      <div className="rounded-xl border border-white/10 bg-white/5 p-4">
        <label className="block text-sm font-medium text-slate-300 mb-2">Question</label>
        <div className="flex gap-2 flex-wrap">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && runRag()}
            placeholder="e.g. What did we decide about the budget?"
            className="flex-1 min-w-[200px] rounded-lg bg-slate-800/80 border border-white/10 px-3 py-2 text-white placeholder-slate-500 focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/30 outline-none"
          />
          <button
            type="button"
            onClick={runRag}
            disabled={loading}
            className="px-4 py-2 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/30 disabled:opacity-50 font-medium"
          >
            {loading ? 'Running…' : 'Run RAG'}
          </button>
        </div>
        <p className="text-xs text-slate-500 mt-2">
          Mode: {settings.advancedRag ? 'Advanced RAG (single-query retrieval)' : 'Full RAG (intent + query expansion)'}
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 text-red-300 px-4 py-2 text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          <div className="rounded-xl border border-white/10 bg-white/5 p-4">
            <h2 className="text-sm font-semibold text-cyan-400 mb-2">Intent</h2>
            <p className="text-white font-mono text-sm">{result.intent}</p>
            {result.message && (
              <p className="text-slate-400 text-xs mt-2">{result.message}</p>
            )}
          </div>

          {result.chunks.length > 0 && (
            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <h2 className="text-sm font-semibold text-cyan-400 mb-3">Retrieved context ({result.chunks.length} chunks)</h2>
              <ul className="space-y-3">
                {result.chunks.map((c, i) => (
                  <li key={i} className="rounded-lg bg-slate-800/60 border border-white/5 p-3 text-left">
                    <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400 mb-1">
                      <span>Score: {c.score}</span>
                      {c.filename && <span>• {c.filename}</span>}
                      {c.chunk_index != null && <span>• chunk #{c.chunk_index}</span>}
                    </div>
                    <p className="text-slate-200 text-sm whitespace-pre-wrap break-words">{c.text_preview}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="rounded-xl border border-white/10 bg-white/5 p-4">
            <h2 className="text-sm font-semibold text-cyan-400 mb-2">Answer</h2>
            <p className="text-white whitespace-pre-wrap">{result.answer || '—'}</p>
          </div>
        </div>
      )}

      <div className="rounded-xl border border-white/10 bg-white/5 overflow-hidden">
        <button
          type="button"
          onClick={loadDataPreview}
          disabled={previewLoading}
          className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-white/5 transition-colors"
        >
          <span className="font-medium text-slate-200">How data is chunked and added</span>
          {previewLoading ? (
            <span className="text-slate-400 text-sm">Loading…</span>
          ) : (
            <span className="text-cyan-400 text-sm">{chunkingExpanded ? 'Collapse' : 'Expand'}</span>
          )}
        </button>
        {chunkingExpanded && (
          <div className="px-4 pb-4 pt-0 border-t border-white/5 space-y-4">
            <div className="text-sm text-slate-300 space-y-2 pt-3">
              <p><strong className="text-slate-200">Documents & transcripts</strong> are parsed (PDF/DOCX/PPTX/text), then split into <strong>chunks</strong> by type (paragraphs, lists, tables, code). Only &quot;embed&quot; chunks are sent to the embedding model and stored in FAISS + BM25 + SQLite.</p>
              <p><strong className="text-slate-200">Upload flow:</strong> file → parse → chunk_document → embed (Ollama) → FAISS + SQLite + BM25.</p>
              <p><strong className="text-slate-200">Transcript flow:</strong> store transcript → add_document(transcript_…) → same chunking + embed + store. Documents and transcripts share one index.</p>
            </div>
            {dataPreview && (
              <div className="space-y-3">
                <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Current index</p>
                <p className="text-slate-400 text-sm">
                  {dataPreview.documents.length} document(s), {dataPreview.chunks.length} chunk(s), {dataPreview.transcripts.length} transcript(s).
                </p>
                {dataPreview.chunks.length > 0 && (
                  <details className="text-sm">
                    <summary className="cursor-pointer text-cyan-400 hover:underline">Show chunk preview</summary>
                    <ul className="mt-2 space-y-1 max-h-48 overflow-y-auto">
                      {dataPreview.chunks.slice(0, 20).map((ch) => (
                        <li key={ch.id} className="text-slate-400 truncate">
                          doc {ch.doc_id} chunk #{ch.chunk_index}: {(ch.text_preview || '').slice(0, 60)}…
                        </li>
                      ))}
                      {dataPreview.chunks.length > 20 && (
                        <li className="text-slate-500">… and {dataPreview.chunks.length - 20} more</li>
                      )}
                    </ul>
                  </details>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

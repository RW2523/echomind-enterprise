import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { ICONS } from '../constants';
import { API_BASE } from '../services/backend';
import type { AppSettings } from '../types';

// ── Types (mirror the backend docgen v2 contract) ──────────────────────────────

interface TemplateMeta {
  id: string;
  name: string;
  icon: string;
  persona: string;
  description: string;
  sections: string[];
  default_doc_type: string;
  theme?: string;            // theme NAME (midnight | executive | evergreen | counsel | spotlight)
  supports_images?: boolean; // can include generated image blocks / a cover hero
  custom?: boolean;          // an uploaded custom template
}

interface DocListItem {
  id: string;
  filename: string;
  filetype: string;
  created_at: string;
}

interface ChatListItem {
  id: string;
  title: string;
  created_at: string;
}

interface JobListItem {
  id: string;
  title: string;
  template_id: string;
  status: JobStatus;
  created_at: string;
}

interface ImageStatus {
  backend: string;            // comfyui | nim | none
  enabled: boolean;
  url?: string;
  has_key?: boolean;
  checkpoint?: string;
}

type JobStatus = 'queued' | 'planning' | 'writing' | 'assembling' | 'done' | 'error';

// Document model block types (see backend/app/docgen/models.py).
type Block =
  | { type: 'heading'; level: number; text: string }
  | { type: 'paragraph'; text: string }
  | { type: 'bullets'; items: string[]; ordered?: boolean }
  | { type: 'table'; columns: string[]; rows: string[][]; caption?: string }
  | { type: 'flow'; title?: string; steps: string[] }
  | { type: 'callout'; style: 'info' | 'warning' | 'principle' | 'success'; title?: string; text: string }
  | { type: 'image'; prompt?: string; caption?: string; alt?: string; data_url?: string }
  | { type: 'divider' }
  | { type: 'pagebreak' }
  | { type: string; [k: string]: unknown }; // tolerate unknown types

interface DocumentModel {
  title: string;
  subtitle?: string;
  org?: string;
  doc_type?: string;
  template_id?: string;
  persona?: string;
  theme?: string;
  cover_image_data_url?: string;
  blocks: Block[];
}

interface JobDetail {
  id: string;
  title: string;
  template_id: string;
  persona?: string;
  status: JobStatus;
  stage?: string;
  document?: DocumentModel | null;
  error?: string | null;
}

type SourceMode = 'chat' | 'kb' | 'brief';

interface DocumentStudioProps {
  settings?: AppSettings | null;
}

// ── Theme name -> a small accent color for the gallery dot ──────────────────────

const THEME_DOT: Record<string, string> = {
  midnight: '#22d3ee',   // cyan
  executive: '#818cf8',  // indigo
  evergreen: '#2dd4bf',  // teal
  counsel: '#f59e0b',    // amber
  spotlight: '#f472b6',  // pink
};

function themeDot(theme?: string): string {
  return THEME_DOT[(theme || '').toLowerCase()] || '#64748b';
}

// ── Small fetch helpers (self-contained; do NOT touch backend.ts) ───────────────

async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, { signal, cache: 'no-store' });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `${path} failed: ${r.status}`);
  }
  return (await r.json()) as T;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `${path} failed: ${r.status}`);
  }
  return (await r.json()) as T;
}

// ── Status / stage helpers ──────────────────────────────────────────────────────

const ACTIVE_STATUSES: JobStatus[] = ['queued', 'planning', 'writing', 'assembling'];

function isActive(status?: JobStatus): boolean {
  return !!status && ACTIVE_STATUSES.includes(status);
}

const STATUS_META: Record<JobStatus, { label: string; cls: string }> = {
  queued: { label: 'Queued', cls: 'text-slate-300 bg-white/10 border-white/15' },
  planning: { label: 'Planning', cls: 'text-cyan-300 bg-cyan-500/10 border-cyan-500/30' },
  writing: { label: 'Writing', cls: 'text-cyan-300 bg-cyan-500/10 border-cyan-500/30' },
  assembling: { label: 'Assembling', cls: 'text-violet-300 bg-violet-500/10 border-violet-500/30' },
  done: { label: 'Ready', cls: 'text-emerald-300 bg-emerald-500/10 border-emerald-500/30' },
  error: { label: 'Error', cls: 'text-red-300 bg-red-500/10 border-red-500/30' },
};

function StatusBadge({ status }: { status: JobStatus }) {
  const m = STATUS_META[status] ?? STATUS_META.queued;
  return (
    <span className={`inline-flex items-center gap-1.5 text-[11px] font-semibold rounded-full px-2.5 py-0.5 border ${m.cls}`}>
      {isActive(status) && <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />}
      {status === 'done' && <span aria-hidden>✓</span>}
      {m.label}
    </span>
  );
}

// ── Callout style map ───────────────────────────────────────────────────────────

const CALLOUT_STYLES: Record<string, { box: string; title: string; icon: string }> = {
  info: { box: 'border-cyan-500/30 bg-cyan-500/[0.07]', title: 'text-cyan-300', icon: 'ℹ' },
  warning: { box: 'border-amber-500/30 bg-amber-500/[0.07]', title: 'text-amber-300', icon: '⚠' },
  principle: { box: 'border-violet-500/30 bg-violet-500/[0.07]', title: 'text-violet-300', icon: '◆' },
  success: { box: 'border-emerald-500/30 bg-emerald-500/[0.07]', title: 'text-emerald-300', icon: '✓' },
};

// ── Image preview (block / cover) with graceful placeholder ─────────────────────

function ImageView({ dataUrl, caption, alt }: { dataUrl?: string; caption?: string; alt?: string }) {
  const [broken, setBroken] = useState(false);
  const label = caption || alt || 'Image';
  const showImg = !!dataUrl && !broken;
  return (
    <figure className="mb-4">
      {showImg ? (
        <img
          src={dataUrl}
          alt={alt || caption || 'Document image'}
          onError={() => setBroken(true)}
          className="w-full rounded-xl border border-white/10 bg-black/30 object-contain max-h-[420px]"
        />
      ) : (
        <div className="w-full rounded-xl border border-dashed border-cyan-500/30 bg-white/[0.03] flex flex-col items-center justify-center gap-2 py-12 px-4 text-center">
          <span className="text-3xl text-cyan-400/70" aria-hidden>▦</span>
          <span className="text-xs text-slate-400 max-w-sm">{label}</span>
        </div>
      )}
      {caption && <figcaption className="mt-1.5 text-center text-[11px] text-slate-500 italic">{caption}</figcaption>}
    </figure>
  );
}

// ── Block preview renderer ──────────────────────────────────────────────────────

function BlockView({ block }: { block: Block }) {
  switch (block.type) {
    case 'heading': {
      const lvl = Math.max(1, Math.min(3, Number((block as any).level) || 2));
      const text = String((block as any).text ?? '');
      if (lvl === 1) return <h2 className="text-xl font-bold text-white mt-6 mb-2 first:mt-0">{text}</h2>;
      if (lvl === 2) return <h3 className="text-base font-bold text-white/95 mt-5 mb-1.5">{text}</h3>;
      return <h4 className="text-sm font-semibold text-white/85 mt-4 mb-1">{text}</h4>;
    }
    case 'paragraph':
      return <p className="text-sm text-slate-200/90 leading-relaxed mb-3 whitespace-pre-wrap">{String((block as any).text ?? '')}</p>;
    case 'bullets': {
      const items: string[] = Array.isArray((block as any).items) ? (block as any).items : [];
      const ordered = !!(block as any).ordered;
      const ListTag = ordered ? 'ol' : 'ul';
      return (
        <ListTag className={`${ordered ? 'list-decimal' : 'list-disc'} list-outside pl-5 mb-3 space-y-1 text-sm text-slate-200/90`}>
          {items.map((it, i) => <li key={i} className="leading-relaxed">{it}</li>)}
        </ListTag>
      );
    }
    case 'table': {
      const cols: string[] = Array.isArray((block as any).columns) ? (block as any).columns : [];
      const rows: string[][] = Array.isArray((block as any).rows) ? (block as any).rows : [];
      const caption = (block as any).caption as string | undefined;
      return (
        <div className="mb-4 overflow-x-auto rounded-xl border border-white/10">
          <table className="w-full text-sm border-collapse">
            {cols.length > 0 && (
              <thead>
                <tr className="bg-white/[0.06]">
                  {cols.map((c, i) => (
                    <th key={i} className="text-left font-semibold text-white/90 px-3 py-2 border-b border-white/10 whitespace-nowrap">{c}</th>
                  ))}
                </tr>
              </thead>
            )}
            <tbody>
              {rows.map((r, ri) => (
                <tr key={ri} className="odd:bg-white/[0.02]">
                  {(Array.isArray(r) ? r : [r]).map((cell, ci) => (
                    <td key={ci} className="align-top text-slate-200/85 px-3 py-2 border-b border-white/[0.06]">{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {caption && <div className="px-3 py-1.5 text-[11px] text-slate-500 italic border-t border-white/[0.06]">{caption}</div>}
        </div>
      );
    }
    case 'flow': {
      const steps: string[] = Array.isArray((block as any).steps) ? (block as any).steps : [];
      const title = (block as any).title as string | undefined;
      return (
        <div className="mb-4">
          {title && <div className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">{title}</div>}
          <div className="flex flex-wrap items-center gap-2">
            {steps.map((s, i) => (
              <React.Fragment key={i}>
                <span className="inline-flex items-center rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-1.5 text-xs font-medium text-cyan-200">
                  {s}
                </span>
                {i < steps.length - 1 && <span className="text-slate-500 select-none" aria-hidden>→</span>}
              </React.Fragment>
            ))}
          </div>
        </div>
      );
    }
    case 'callout': {
      const style = String((block as any).style ?? 'info');
      const s = CALLOUT_STYLES[style] ?? CALLOUT_STYLES.info;
      const title = (block as any).title as string | undefined;
      const text = String((block as any).text ?? '');
      return (
        <div className={`mb-4 rounded-xl border px-4 py-3 ${s.box}`}>
          <div className={`flex items-center gap-2 text-xs font-bold ${s.title} mb-1`}>
            <span aria-hidden>{s.icon}</span>
            <span>{title || style.charAt(0).toUpperCase() + style.slice(1)}</span>
          </div>
          {text && <p className="text-sm text-slate-200/90 leading-relaxed whitespace-pre-wrap">{text}</p>}
        </div>
      );
    }
    case 'image':
      return (
        <ImageView
          dataUrl={(block as any).data_url as string | undefined}
          caption={(block as any).caption as string | undefined}
          alt={((block as any).alt || (block as any).prompt) as string | undefined}
        />
      );
    case 'divider':
      return <hr className="border-white/10 my-5" />;
    case 'pagebreak':
      return (
        <div className="my-5 flex items-center gap-2 text-[10px] uppercase tracking-widest text-slate-600">
          <span className="flex-1 border-t border-dashed border-white/15" />
          Page break
          <span className="flex-1 border-t border-dashed border-white/15" />
        </div>
      );
    default: {
      // Unknown block type: fall back to any text it carries.
      const text = (block as any).text;
      return text ? <p className="text-sm text-slate-300 mb-3 whitespace-pre-wrap">{String(text)}</p> : null;
    }
  }
}

function DocumentPreview({ doc }: { doc: DocumentModel }) {
  return (
    <article className="rounded-2xl border border-white/10 bg-[#080b14] p-5 sm:p-7">
      {doc.cover_image_data_url && (
        <ImageView dataUrl={doc.cover_image_data_url} alt={`${doc.title} cover`} />
      )}
      <header className="mb-5 pb-4 border-b border-white/10">
        <h1 className="text-2xl font-bold text-white leading-tight">{doc.title}</h1>
        {doc.subtitle && <p className="text-sm text-slate-400 mt-1.5">{doc.subtitle}</p>}
        <div className="flex flex-wrap items-center gap-2 mt-3">
          {doc.doc_type && <span className="text-[11px] rounded-md bg-white/[0.06] border border-white/10 px-2 py-0.5 text-slate-300">{doc.doc_type}</span>}
          {doc.org && <span className="text-[11px] text-slate-500">{doc.org}</span>}
          {doc.persona && <span className="text-[11px] rounded-md bg-cyan-500/15 border border-cyan-500/25 px-2 py-0.5 text-cyan-300">{doc.persona}</span>}
          {doc.theme && (
            <span className="inline-flex items-center gap-1.5 text-[11px] rounded-md bg-white/[0.04] border border-white/10 px-2 py-0.5 text-slate-400">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: themeDot(doc.theme) }} />
              {doc.theme}
            </span>
          )}
        </div>
      </header>
      <div>
        {doc.blocks.map((b, i) => <BlockView key={i} block={b} />)}
        {doc.blocks.length === 0 && <p className="text-sm text-slate-500 italic">This document has no content blocks.</p>}
      </div>
    </article>
  );
}

// ── Main component ──────────────────────────────────────────────────────────────

const DocumentStudio: React.FC<DocumentStudioProps> = ({ settings: _settings }) => {
  // Catalog data
  const [templates, setTemplates] = useState<TemplateMeta[]>([]);
  const [templatesError, setTemplatesError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocListItem[]>([]);
  const [chats, setChats] = useState<ChatListItem[]>([]);
  const [recentJobs, setRecentJobs] = useState<JobListItem[]>([]);
  const [imageStatus, setImageStatus] = useState<ImageStatus | null>(null);

  // Builder selections
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  const [mode, setMode] = useState<SourceMode>('brief');
  const [chatId, setChatId] = useState<string>('');
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [title, setTitle] = useState('');
  const [org, setOrg] = useState('');
  const [brief, setBrief] = useState('');
  const [withImages, setWithImages] = useState(false);
  const [customOpen, setCustomOpen] = useState(false);
  const [customTemplate, setCustomTemplate] = useState('');

  // Custom template upload
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Generation / job state
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<JobDetail | null>(null);
  const [exportLoading, setExportLoading] = useState<'pdf' | 'pptx' | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const loadTemplates = useCallback((signal?: AbortSignal) => {
    return getJSON<{ templates: TemplateMeta[] }>('/api/docgen/templates', signal)
      .then(res => { if (mountedRef.current) { setTemplates(res.templates || []); setTemplatesError(null); } })
      .catch(e => { if (mountedRef.current && e?.name !== 'AbortError') setTemplatesError((e as Error).message || 'Failed to load templates'); });
  }, []);

  const loadRecentJobs = useCallback((signal?: AbortSignal) => {
    getJSON<{ jobs: JobListItem[] }>('/api/docgen/jobs', signal)
      .then(res => { if (mountedRef.current) setRecentJobs(res.jobs || []); })
      .catch(() => {});
  }, []);

  // ── Load catalogs on mount ──
  useEffect(() => {
    mountedRef.current = true;
    const ctrl = new AbortController();

    loadTemplates(ctrl.signal);

    getJSON<{ documents: DocListItem[] }>('/api/docs/list', ctrl.signal)
      .then(res => { if (mountedRef.current) setDocuments(res.documents || []); })
      .catch(() => {});

    getJSON<{ chats: ChatListItem[] }>('/api/chat/list', ctrl.signal)
      .then(res => { if (mountedRef.current) setChats(res.chats || []); })
      .catch(() => {});

    getJSON<ImageStatus>('/api/docgen/image-status', ctrl.signal)
      .then(res => {
        if (!mountedRef.current) return;
        setImageStatus(res);
        // Default "Generate with images" ON when a backend is available, so generated
        // documents come out illustrated without the user having to hunt for the toggle.
        if (res?.enabled) setWithImages(true);
      })
      .catch(() => {});

    loadRecentJobs(ctrl.signal);

    return () => {
      mountedRef.current = false;
      ctrl.abort();
      stopPolling();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Polling for the active job ──
  const beginPolling = useCallback((jobId: string) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const detail = await getJSON<JobDetail>(`/api/docgen/jobs/${encodeURIComponent(jobId)}`);
        if (!mountedRef.current) return;
        setActiveJob(detail);
        if (detail.status === 'done' || detail.status === 'error') {
          stopPolling();
          setGenerating(false);
          loadRecentJobs();
        }
      } catch {
        // transient errors: keep polling
      }
    }, 2000);
  }, [stopPolling, loadRecentJobs]);

  const selectedTemplate = useMemo(
    () => templates.find(t => t.id === selectedTemplateId) || null,
    [templates, selectedTemplateId],
  );

  const imagesEnabled = !!imageStatus?.enabled;

  const toggleDoc = useCallback((id: string) => {
    if (!id) return;
    setSelectedDocIds(prev => (prev.includes(id) ? prev.filter(d => d !== id) : [...prev, id]));
  }, []);

  const canGenerate = useMemo(() => {
    if (!selectedTemplateId || generating) return false;
    if (mode === 'chat') return !!chatId;
    if (mode === 'kb') return selectedDocIds.length > 0;
    if (mode === 'brief') return brief.trim().length > 0;
    return false;
  }, [selectedTemplateId, generating, mode, chatId, selectedDocIds, brief]);

  // ── Custom template upload ──
  const handleUploadClick = useCallback(() => {
    setUploadError(null);
    fileInputRef.current?.click();
  }, []);

  const handleFileSelected = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    // Reset the input so selecting the same file again still fires onChange.
    e.target.value = '';
    if (!file) return;
    setUploadError(null);
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const r = await fetch(`${API_BASE}/api/docgen/templates/upload`, { method: 'POST', body: fd });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error((err as { detail?: string }).detail || `upload failed: ${r.status}`);
      }
      const res = (await r.json()) as { template: TemplateMeta };
      await loadTemplates();
      if (mountedRef.current && res.template?.id) {
        setSelectedTemplateId(res.template.id);
      }
    } catch (err) {
      if (mountedRef.current) setUploadError((err as Error).message || 'Template upload failed');
    } finally {
      if (mountedRef.current) setUploading(false);
    }
  }, [loadTemplates]);

  const handleGenerate = useCallback(async () => {
    if (!selectedTemplateId || !canGenerate) return;
    setGenError(null);
    setGenerating(true);
    setActiveJob(null);
    stopPolling();
    try {
      const body: Record<string, unknown> = {
        template_id: selectedTemplateId,
        mode,
        title: title.trim() || undefined,
        org: org.trim() || undefined,
        brief: brief.trim() || undefined,
      };
      if (mode === 'chat') body.chat_id = chatId;
      if (mode === 'kb') body.doc_ids = selectedDocIds;
      if (customOpen && customTemplate.trim()) body.custom_template = customTemplate.trim();
      if (withImages) body.with_images = true;

      const res = await postJSON<{ job_id: string; status: JobStatus }>('/api/docgen/generate', body);
      if (!mountedRef.current) return;
      setActiveJob({
        id: res.job_id,
        title: title.trim() || selectedTemplate?.name || 'Document',
        template_id: selectedTemplateId,
        status: res.status || 'queued',
      });
      beginPolling(res.job_id);
    } catch (e) {
      if (!mountedRef.current) return;
      setGenError((e as Error).message || 'Generation failed');
      setGenerating(false);
    }
  }, [selectedTemplateId, canGenerate, mode, title, org, brief, chatId, selectedDocIds, customOpen, customTemplate, withImages, selectedTemplate, beginPolling, stopPolling]);

  const handleOpenJob = useCallback(async (jobId: string) => {
    setGenError(null);
    stopPolling();
    try {
      const detail = await getJSON<JobDetail>(`/api/docgen/jobs/${encodeURIComponent(jobId)}`);
      if (!mountedRef.current) return;
      setActiveJob(detail);
      if (isActive(detail.status)) {
        setGenerating(true);
        beginPolling(jobId);
      } else {
        setGenerating(false);
      }
    } catch (e) {
      if (mountedRef.current) setGenError((e as Error).message || 'Failed to open document');
    }
  }, [beginPolling, stopPolling]);

  const handleDeleteJob = useCallback(async (jobId: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    if (!window.confirm('Delete this generated document? This cannot be undone.')) return;
    try {
      const r = await fetch(`${API_BASE}/api/docgen/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error((err as { detail?: string }).detail || `delete failed: ${r.status}`);
      }
      if (!mountedRef.current) return;
      setRecentJobs(prev => prev.filter(j => j.id !== jobId));
      setActiveJob(prev => (prev?.id === jobId ? null : prev));
    } catch (err) {
      alert((err as Error).message || 'Failed to delete');
    }
  }, []);

  const handleExport = useCallback((format: 'pdf' | 'pptx') => {
    if (!activeJob) return;
    setExportLoading(format);
    try {
      window.open(`${API_BASE}/api/docgen/jobs/${encodeURIComponent(activeJob.id)}/export?format=${format}`, '_blank', 'noopener');
    } finally {
      // brief visual feedback; download happens in a new tab
      setTimeout(() => { if (mountedRef.current) setExportLoading(null); }, 800);
    }
  }, [activeJob]);

  const resetBuilder = useCallback(() => {
    stopPolling();
    setGenerating(false);
    setActiveJob(null);
    setGenError(null);
  }, [stopPolling]);

  // ── Render: active job preview takes over the main panel ──
  const showPreview = !!activeJob;

  return (
    <div className="h-full min-h-0 flex flex-col rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
      {/* Header */}
      <div className="shrink-0 flex items-center gap-3 px-4 py-3 border-b border-white/10 bg-black/10">
        <div className="w-8 h-8 rounded-xl bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center shrink-0">
          <ICONS.File className="w-4 h-4 text-cyan-400" />
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold text-white">Document Studio</div>
          <div className="text-xs text-slate-400 truncate">
            {showPreview ? activeJob?.title : 'Generate structured documents from chats, files, or a brief'}
          </div>
        </div>
        {showPreview && (
          <button
            type="button"
            onClick={resetBuilder}
            className="ml-auto rounded-xl px-3 py-1.5 text-xs font-semibold bg-white/10 text-slate-300 hover:bg-white/15 border border-white/10 transition-colors"
          >
            New Document
          </button>
        )}
      </div>

      <div className="flex-1 min-h-0 overflow-auto">
        {showPreview ? (
          <PreviewPanel
            job={activeJob!}
            exportLoading={exportLoading}
            onExport={handleExport}
          />
        ) : (
          <div className="p-4 sm:p-5 space-y-6">
            {/* 1. Template gallery */}
            <section>
              <div className="flex items-center justify-between mb-3">
                <SectionLabel step={1} title="Choose a template" />
                <div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".json,.yaml,.yml,.docx,.pptx,.md,.txt"
                    className="hidden"
                    onChange={handleFileSelected}
                  />
                  <button
                    type="button"
                    onClick={handleUploadClick}
                    disabled={uploading}
                    className="inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-semibold bg-white/10 text-slate-200 hover:bg-white/15 border border-white/10 disabled:opacity-50 transition-colors"
                  >
                    <ICONS.Upload className="w-3.5 h-3.5" />
                    {uploading ? 'Uploading…' : 'Upload template'}
                  </button>
                </div>
              </div>
              {uploadError && <div className="text-xs text-red-400 mb-2">{uploadError}</div>}
              {templatesError && <div className="text-xs text-red-400 mb-2">{templatesError}</div>}
              {templates.length === 0 && !templatesError && (
                <div className="text-xs text-slate-500 py-6 text-center">Loading templates…</div>
              )}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {templates.map(t => {
                  const active = t.id === selectedTemplateId;
                  return (
                    <button
                      key={t.id}
                      type="button"
                      onClick={() => setSelectedTemplateId(t.id)}
                      className={`text-left rounded-2xl border p-4 transition-colors ${
                        active
                          ? 'border-cyan-500/50 bg-cyan-500/[0.08] shadow-[0_0_18px_-6px_rgba(34,211,238,0.4)]'
                          : 'border-white/10 bg-white/[0.03] hover:bg-white/[0.06] hover:border-white/20'
                      }`}
                    >
                      <div className="flex items-start gap-2.5">
                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${active ? 'bg-cyan-500/25' : 'bg-white/[0.06]'}`}>
                          <ICONS.File className={`w-4 h-4 ${active ? 'text-cyan-300' : 'text-slate-400'}`} />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-1.5">
                            <span
                              className="w-2 h-2 rounded-full shrink-0"
                              style={{ backgroundColor: themeDot(t.theme) }}
                              title={`Theme: ${t.theme || 'default'}`}
                            />
                            <div className="text-sm font-semibold text-white truncate">{t.name}</div>
                          </div>
                          <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
                            <span className="inline-block text-[10px] rounded-md bg-cyan-500/15 border border-cyan-500/25 px-1.5 py-0.5 text-cyan-300">{t.persona}</span>
                            {t.supports_images && (
                              <span className="inline-flex items-center gap-1 text-[10px] rounded-md bg-violet-500/15 border border-violet-500/25 px-1.5 py-0.5 text-violet-300">
                                <span aria-hidden>▦</span> images
                              </span>
                            )}
                            {t.custom && (
                              <span className="inline-block text-[10px] rounded-md bg-amber-500/15 border border-amber-500/25 px-1.5 py-0.5 text-amber-300">Custom</span>
                            )}
                          </div>
                        </div>
                        {active && (
                          <span className="shrink-0 w-5 h-5 rounded-full bg-cyan-500/30 flex items-center justify-center">
                            <svg className="w-3 h-3 text-cyan-200" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-slate-400 leading-relaxed mt-2.5 line-clamp-3">{t.description}</p>
                    </button>
                  );
                })}
              </div>
            </section>

            {/* 2. Source mode */}
            <section className={selectedTemplateId ? '' : 'opacity-50 pointer-events-none'}>
              <SectionLabel step={2} title="Pick a source" />
              <div className="inline-flex rounded-xl bg-white/5 p-0.5 border border-white/10 mb-3">
                {([
                  { key: 'chat', label: 'From Chat' },
                  { key: 'kb', label: 'From Documents' },
                  { key: 'brief', label: 'From Brief' },
                ] as { key: SourceMode; label: string }[]).map(m => (
                  <button
                    key={m.key}
                    type="button"
                    onClick={() => setMode(m.key)}
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                      mode === m.key ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>

              {mode === 'chat' && (
                <div>
                  {chats.length === 0 ? (
                    <p className="text-xs text-slate-500 py-2">No chats found. Start a conversation in Knowledge Chat first.</p>
                  ) : (
                    <select
                      value={chatId}
                      onChange={e => setChatId(e.target.value)}
                      className="w-full rounded-xl bg-black/30 border border-white/10 px-3 py-2.5 text-sm text-white outline-none focus:border-white/30"
                    >
                      <option value="">Select a chat…</option>
                      {chats.map(c => (
                        <option key={c.id} value={c.id}>{c.title || 'Untitled chat'}{c.created_at ? `  ·  ${c.created_at.slice(0, 10)}` : ''}</option>
                      ))}
                    </select>
                  )}
                </div>
              )}

              {mode === 'kb' && (
                <div className="rounded-xl border border-white/10 bg-black/20 max-h-56 overflow-auto">
                  {documents.length === 0 ? (
                    <p className="text-xs text-slate-500 p-4 text-center">No documents uploaded yet.</p>
                  ) : (
                    <ul className="divide-y divide-white/[0.06]">
                      {documents.filter(Boolean).map((d, i) => {
                        const id = String(d.id ?? '');
                        const checked = selectedDocIds.includes(id);
                        const name = String(d.filename ?? 'Untitled document');
                        return (
                          <li key={id || i}>
                            <button
                              type="button"
                              role="checkbox"
                              aria-checked={checked}
                              onClick={() => toggleDoc(id)}
                              className="w-full flex items-center gap-3 px-3 py-2.5 cursor-pointer hover:bg-white/[0.04] text-left"
                            >
                              <span className={`shrink-0 w-4 h-4 rounded border flex items-center justify-center ${checked ? 'bg-cyan-500/40 border-cyan-400/50' : 'border-white/20 bg-white/[0.04]'}`}>
                                {checked && <svg className="w-2.5 h-2.5 text-cyan-100" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>}
                              </span>
                              <span className="flex-1 text-xs text-white/90 truncate" title={name}>{name}</span>
                              <span className="text-[10px] text-slate-500 shrink-0 uppercase">{String(d.filetype ?? '')}</span>
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                  {documents.length > 0 && (
                    <div className="px-3 py-1.5 border-t border-white/[0.06] text-[10px] text-slate-500">{selectedDocIds.length} selected</div>
                  )}
                </div>
              )}

              {mode === 'brief' && (
                <p className="text-xs text-slate-500 py-1">Describe what you want below. The generator will write the document from your instructions alone.</p>
              )}
            </section>

            {/* 3. Details */}
            <section className={selectedTemplateId ? '' : 'opacity-50 pointer-events-none'}>
              <SectionLabel step={3} title="Details" />
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-3">
                <input
                  type="text"
                  value={title}
                  onChange={e => setTitle(e.target.value)}
                  placeholder="Document title (optional)"
                  className="rounded-xl bg-black/30 border border-white/10 px-3 py-2.5 text-sm text-white outline-none focus:border-white/30"
                />
                <input
                  type="text"
                  value={org}
                  onChange={e => setOrg(e.target.value)}
                  placeholder="Organisation / footer (optional)"
                  className="rounded-xl bg-black/30 border border-white/10 px-3 py-2.5 text-sm text-white outline-none focus:border-white/30"
                />
              </div>
              <textarea
                value={brief}
                onChange={e => setBrief(e.target.value)}
                rows={mode === 'brief' ? 5 : 3}
                placeholder={mode === 'brief'
                  ? 'Describe the document you want to generate — topic, audience, key points to cover…'
                  : 'Optional brief / extra instructions to steer the generation…'}
                className="w-full rounded-xl bg-black/30 border border-white/10 px-3 py-2.5 text-sm text-white outline-none focus:border-white/30 resize-y"
              />

              {/* Generate with images toggle */}
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  role="switch"
                  aria-checked={withImages}
                  onClick={() => setWithImages(v => !v)}
                  className={`inline-flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-medium border transition-colors ${
                    withImages
                      ? 'bg-violet-500/15 border-violet-500/30 text-violet-200'
                      : 'bg-white/5 border-white/10 text-slate-300 hover:bg-white/[0.08]'
                  }`}
                >
                  <span className={`relative w-8 h-4 rounded-full transition-colors ${withImages ? 'bg-violet-500/60' : 'bg-white/15'}`}>
                    <span className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${withImages ? 'translate-x-4' : 'translate-x-0.5'}`} />
                  </span>
                  Generate with images
                </button>
                {!imagesEnabled && (
                  <span className="text-[11px] text-slate-500 max-w-md leading-snug">
                    Image generation backend not configured (set DOCGEN_IMAGE_BACKEND); placeholders will be used.
                  </span>
                )}
                {imagesEnabled && imageStatus?.backend && (
                  <span className="text-[11px] text-emerald-400/80">Image backend: {imageStatus.backend}</span>
                )}
              </div>

              {/* Optional free-text custom template */}
              <button
                type="button"
                onClick={() => setCustomOpen(o => !o)}
                className="mt-3 inline-flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-cyan-300 transition-colors"
              >
                <span className={`transition-transform ${customOpen ? 'rotate-90' : ''}`}>▸</span>
                Advanced: custom template (optional)
              </button>
              {customOpen && (
                <textarea
                  value={customTemplate}
                  onChange={e => setCustomTemplate(e.target.value)}
                  rows={4}
                  placeholder="Paste a custom template definition or section blueprint to override the selected template…"
                  className="mt-2 w-full rounded-xl bg-black/30 border border-white/10 px-3 py-2.5 text-xs font-mono text-white outline-none focus:border-white/30 resize-y"
                />
              )}
            </section>

            {/* Generate */}
            <section className="pt-1">
              {genError && <div className="text-xs text-red-400 mb-2">{genError}</div>}
              <button
                type="button"
                onClick={handleGenerate}
                disabled={!canGenerate}
                className="w-full sm:w-auto rounded-xl px-6 py-3 text-sm font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {generating ? 'Generating…' : 'Generate Document'}
              </button>
              {!selectedTemplateId && <p className="text-[11px] text-slate-500 mt-2">Select a template to begin.</p>}
            </section>

            {/* 5. Recent documents */}
            <section className="pt-2 border-t border-white/10">
              <div className="flex items-center justify-between mb-2.5">
                <SectionLabel title="Recent Documents" />
                <button type="button" onClick={() => loadRecentJobs()} className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors">Refresh</button>
              </div>
              {recentJobs.length === 0 ? (
                <p className="text-xs text-slate-500 py-3 text-center">No documents generated yet.</p>
              ) : (
                <ul className="space-y-1.5">
                  {recentJobs.map(j => (
                    <li key={j.id}>
                      <button
                        type="button"
                        onClick={() => handleOpenJob(j.id)}
                        className="w-full text-left flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.03] hover:bg-white/[0.06] px-3 py-2.5 transition-colors group"
                      >
                        <div className="w-7 h-7 rounded-lg bg-white/[0.06] flex items-center justify-center shrink-0">
                          <ICONS.File className="w-3.5 h-3.5 text-slate-400" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-xs font-medium text-white/90 truncate">{j.title || 'Untitled'}</div>
                          <div className="text-[10px] text-slate-500 truncate">{j.template_id}{j.created_at ? `  ·  ${j.created_at.slice(0, 10)}` : ''}</div>
                        </div>
                        <StatusBadge status={j.status} />
                        <span
                          role="button"
                          tabIndex={0}
                          onClick={(e) => handleDeleteJob(j.id, e)}
                          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleDeleteJob(j.id); } }}
                          className="shrink-0 p-1.5 rounded text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                          title="Delete document"
                        >
                          <ICONS.Trash className="w-3.5 h-3.5" />
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </div>
        )}
      </div>
    </div>
  );
};

// ── Preview / progress panel ────────────────────────────────────────────────────

const STAGE_ORDER: JobStatus[] = ['queued', 'planning', 'writing', 'assembling', 'done'];

function PreviewPanel({
  job,
  exportLoading,
  onExport,
}: {
  job: JobDetail;
  exportLoading: 'pdf' | 'pptx' | null;
  onExport: (format: 'pdf' | 'pptx') => void;
}) {
  if (job.status === 'error') {
    return (
      <div className="p-6">
        <div className="rounded-2xl border border-red-500/30 bg-red-500/[0.07] p-5">
          <div className="flex items-center gap-2 text-sm font-semibold text-red-300 mb-2">
            <span aria-hidden>⚠</span> Generation failed
          </div>
          <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">{job.error || 'An unknown error occurred.'}</p>
        </div>
      </div>
    );
  }

  if (job.status !== 'done' || !job.document) {
    const stageIdx = Math.max(0, STAGE_ORDER.indexOf(job.status));
    return (
      <div className="p-6">
        <div className="max-w-md mx-auto text-center py-10">
          <div className="flex justify-center gap-1.5 mb-5">
            {[0, 1, 2].map(i => (
              <span key={i} className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
            ))}
          </div>
          <p className="text-sm font-semibold text-white capitalize mb-1">{STATUS_META[job.status]?.label ?? job.status}…</p>
          <p className="text-xs text-slate-500 mb-6">{job.stage || 'Working through the document pipeline.'}</p>
          {/* Stage stepper */}
          <div className="flex items-center justify-center gap-1.5">
            {STAGE_ORDER.slice(0, 4).map((s, i) => (
              <React.Fragment key={s}>
                <span className={`text-[10px] px-2 py-1 rounded-full border ${i <= stageIdx ? 'border-cyan-500/40 bg-cyan-500/10 text-cyan-300' : 'border-white/10 text-slate-600'}`}>
                  {STATUS_META[s]?.label ?? s}
                </span>
                {i < 3 && <span className={`text-xs ${i < stageIdx ? 'text-cyan-400' : 'text-slate-700'}`}>→</span>}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // status === 'done' with a document
  return (
    <div className="p-4 sm:p-5">
      <div className="flex items-center justify-end gap-2 mb-4">
        <button
          type="button"
          onClick={() => onExport('pdf')}
          disabled={exportLoading !== null}
          className="rounded-xl px-4 py-2 text-xs font-semibold bg-white/10 text-slate-200 hover:bg-white/15 border border-white/10 disabled:opacity-50 transition-colors"
        >
          {exportLoading === 'pdf' ? 'Exporting…' : 'Export PDF'}
        </button>
        <button
          type="button"
          onClick={() => onExport('pptx')}
          disabled={exportLoading !== null}
          className="rounded-xl px-4 py-2 text-xs font-semibold bg-white/10 text-slate-200 hover:bg-white/15 border border-white/10 disabled:opacity-50 transition-colors"
        >
          {exportLoading === 'pptx' ? 'Exporting…' : 'Export PPTX'}
        </button>
      </div>
      <DocumentPreview doc={job.document} />
    </div>
  );
}

// ── Tiny presentational helper ──────────────────────────────────────────────────

function SectionLabel({ step, title }: { step?: number; title: string }) {
  return (
    <div className="flex items-center gap-2 mb-3">
      {step != null && (
        <span className="w-5 h-5 rounded-full bg-cyan-500/20 border border-cyan-500/30 text-cyan-300 text-[10px] font-bold flex items-center justify-center shrink-0">{step}</span>
      )}
      <h3 className="text-sm font-semibold text-white">{title}</h3>
    </div>
  );
}

export default DocumentStudio;

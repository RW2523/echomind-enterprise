import React, { useState } from 'react';
import { ICONS } from '../constants';
import { uploadDocument } from '../services/backend';

interface UploaderProps {
  onComplete: () => void;
  compact?: boolean;
}

const Uploader: React.FC<UploaderProps> = ({ onComplete, compact = false }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [status, setStatus] = useState<string>('');
  const [error, setError] = useState<string>('');

  const onPick = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setIsUploading(true);
    setStatus('Uploading & indexing…');
    setError('');
    try {
      const res = await uploadDocument(f);
      setStatus(res.ok ? `Indexed ${res.chunks ?? 0} chunks.` : 'Upload failed.');
      onComplete();
    } catch (err: any) {
      setError(err?.message || 'Upload failed.');
      setStatus('');
    } finally {
      setIsUploading(false);
      e.target.value = '';
    }
  };

  return (
    <div className={`rounded-xl border border-white/10 bg-white/5 ${compact ? 'p-3' : 'p-4'}`}>
      <div className="flex items-center gap-3">
        <div className={`shrink-0 flex items-center justify-center w-10 h-10 rounded-lg bg-black/20 ${isUploading ? 'opacity-90' : 'opacity-80'}`}>
          {isUploading ? (
            <svg className="animate-spin h-5 w-5 text-cyan-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" aria-hidden="true">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          ) : (
            <ICONS.Upload className="w-5 h-5 text-white/90" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold">Upload</div>
          <div className="text-xs opacity-70 mt-0.5">
            {isUploading ? status : 'PDF, DOCX, PPTX'}
          </div>
        </div>
        <label className={`shrink-0 cursor-pointer rounded-lg px-4 py-2 text-sm font-medium bg-white/10 hover:bg-white/15 transition-colors ${isUploading ? 'opacity-60 pointer-events-none' : ''}`}>
          {isUploading ? 'Indexing…' : 'Choose file'}
          <input type="file" className="hidden" accept=".pdf,.docx,.pptx" onChange={onPick} />
        </label>
      </div>
      {error && <div className="mt-2 text-xs text-red-400">{error}</div>}
      {status && !isUploading && !error && <div className="mt-2 text-xs text-emerald-400/90">{status}</div>}
    </div>
  );
};

export default Uploader;

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

  const onPick = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setIsUploading(true);
    setStatus('Uploading & indexing...');
    try {
      const res = await uploadDocument(f);
      setStatus(res.ok ? `Indexed ${res.chunks ?? 0} chunks.` : 'Upload failed.');
      onComplete();
    } catch (err: any) {
      setStatus(err?.message || 'Upload failed.');
    } finally {
      setIsUploading(false);
      e.target.value = '';
    }
  };

  return (
    <div className={`rounded-2xl border border-white/10 bg-white/5 ${compact ? 'p-3' : 'p-5'}`}>
      <div className="flex items-center gap-3">
        <div className="opacity-80">{ICONS.upload}</div>
        <div className="flex-1">
          <div className="text-sm font-semibold">Upload documents</div>
          <div className="text-xs opacity-70">PDF / DOCX / PPTX. They will be indexed into enterprise RAG.</div>
        </div>
        <label className={`cursor-pointer rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 hover:bg-white/15 ${isUploading ? 'opacity-60 pointer-events-none' : ''}`}>
          {isUploading ? 'Uploading...' : 'Choose file'}
          <input type="file" className="hidden" onChange={onPick} />
        </label>
      </div>
      {status && <div className="mt-3 text-xs opacity-80">{status}</div>}
    </div>
  );
};

export default Uploader;

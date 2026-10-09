import React, { useEffect, useState } from 'react';
import { AlertTriangle, Check, Plus } from 'lucide-react';
import Toolbar from '../ui/Toolbar';
import { applyPreferences, setReducedTransparency, setThemeMode } from '../app/preferences';
import { CopyField, EmptyState, ErrorNote, Field, MatterStatusPill, SensitivityPill, Skeleton, StatusPill } from '../ui/primitives';

/** /__design (development builds only): every component and state, for visual review. */
export default function DesignCatalogue() {
  const [dark, setDark] = useState(false);
  const [reduced, setReduced] = useState(false);
  useEffect(() => { document.documentElement.dataset.edition = 'law'; applyPreferences('light', false); }, []);
  return (
    <div style={{ minHeight: '100dvh' }}>
      <Toolbar title="Design catalogue" crumb="Dev">
        <button type="button" className="lw-btn" onClick={() => { setThemeMode(dark ? 'light' : 'dark'); setDark(!dark); }}>{dark ? 'Light' : 'Dark'}</button>
        <button type="button" className="lw-btn" onClick={() => { setReducedTransparency(!reduced); setReduced(!reduced); }}>{reduced ? 'Normal transparency' : 'Reduce transparency'}</button>
        <button type="button" className="lw-btn lw-btn-primary"><Plus />Primary action</button>
      </Toolbar>
      <div className="lw-content" style={{ maxWidth: 960 }}>
        <section className="lw-paper lw-sheet" style={{ display: 'grid', gap: 12 }}>
          <div className="lw-eyebrow">Status pills</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <StatusPill tone="verified" icon={<Check />}>Supported</StatusPill>
            <StatusPill tone="attention" icon={<AlertTriangle />}>Needs review</StatusPill>
            <StatusPill tone="conflict">Conflicting sources</StatusPill>
            <StatusPill tone="privileged">Privileged</StatusPill>
            <StatusPill tone="hold">On hold</StatusPill>
            <StatusPill tone="neutral">Unreviewed</StatusPill>
            <MatterStatusPill value="open" /><MatterStatusPill value="closed" />
            <SensitivityPill value="standard" /><SensitivityPill value="restricted" /><SensitivityPill value="highly_restricted" />
          </div>
          <div className="lw-eyebrow">Buttons</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button type="button" className="lw-btn lw-btn-primary">Primary</button>
            <button type="button" className="lw-btn">Neutral</button>
            <button type="button" className="lw-btn lw-btn-ghost">Ghost</button>
            <button type="button" className="lw-btn lw-btn-danger">Remove</button>
            <button type="button" className="lw-btn" disabled>Disabled</button>
          </div>
          <div className="lw-eyebrow">Fields</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <Field label="Matter number" hint="A hint">{(id) => <input id={id} className="lw-input" placeholder="M-101" />}</Field>
            <CopyField label="Key" value="ABCD EFGH IJKL MNOP" />
          </div>
          <ErrorNote>An honest, specific error message.</ErrorNote>
          <div className="lw-eyebrow">Loading</div>
          <Skeleton /><Skeleton width="60%" />
          <div className="lw-eyebrow">Document text</div>
          <p className="lw-serif" style={{ fontSize: 16.5, lineHeight: '27px', margin: 0, maxWidth: '68ch' }}>
            Provider shall provide no less than thirty (30) days' prior written notice of any Change of Control to Client in accordance with Clause 22.
          </p>
        </section>
        <EmptyState title="No documents in this matter yet" action={<button type="button" className="lw-btn lw-btn-primary">Import</button>}>
          Empty states say what to do next.
        </EmptyState>
      </div>
    </div>
  );
}

import React from 'react';

/** Floating glass toolbar: the page title on the left, at most one tinted primary action on the right. */
export default function Toolbar({ title, crumb, children }: { title: string; crumb?: string; children?: React.ReactNode }) {
  return (
    <div className="lw-toolbar-wrap">
      <div className="lw-toolbar">
        <div className="lw-glass lw-cluster">
          <h1 className="lw-title-cap" style={{ margin: 0 }}>
            {crumb && <span className="lw-crumb">{crumb}</span>}
            <span>{title}</span>
          </h1>
        </div>
        {children && <div className="lw-glass lw-cluster">{children}</div>}
      </div>
    </div>
  );
}

import React from 'react';
import { Link } from 'react-router';
import Toolbar from '../ui/Toolbar';

/** Same page for "does not exist" and "no access" — never reveals which (THREAT_MODEL T-01/T-04). */
export default function NotAvailable() {
  return (
    <>
      <Toolbar title="Not available" />
      <div className="lw-content">
        <div className="lw-paper lw-sheet" style={{ maxWidth: 640 }}>
          <h2 className="lw-h2">This item isn't available.</h2>
          <p style={{ color: 'var(--ink-2)', margin: '8px 0 16px' }}>
            It may not exist, or you may not have access. If you think you should, ask the lead on the matter.
          </p>
          <Link to="/matters" className="lw-btn lw-btn-ghost">Go to your matters</Link>
        </div>
      </div>
    </>
  );
}

import React, { useEffect } from 'react';
import { Outlet, useMatch } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { Siren } from 'lucide-react';
import { api } from '../api/client';
import { keys, useBreakGlass, useMatters } from '../api/queries';
import type { Matter } from '../api/types';
import NotAvailable from '../pages/NotAvailable';
import { Skeleton } from '../ui/primitives';

export const MatterContext = React.createContext<Matter | null>(null);
export const useCurrentMatter = () => React.useContext(MatterContext);
export const useMatter = (): Matter => {
  const m = useCurrentMatter();
  if (!m) throw new Error('useMatter outside a matter route');
  return m;
};

/**
 * The matter in the URL (/m/:matterNo/...), looked up by number in the person's own list of
 * visible matters. A hidden matter and one that does not exist therefore look the same, with no
 * extra request that could tell them apart. Leaving a matter drops its cached data.
 */
export function MatterProvider({ children }: { children: React.ReactNode }) {
  const match = useMatch('/m/:matterNo/*');
  const matters = useMatters();
  const qc = useQueryClient();
  const number = match?.params.matterNo;
  const matter = (number && matters.data?.find((m) => m.number === number)) || null;
  const matterId = matter?.id;
  useEffect(() => () => {
    if (matterId) qc.removeQueries({ queryKey: keys.matter(matterId) });
  }, [matterId, qc]);
  return <MatterContext.Provider value={matter}>{children}</MatterContext.Provider>;
}

/** Route element for /m/:matterNo: waits for the list, then shows the matter or "not available". */
export function MatterLayout() {
  const matters = useMatters();
  const matter = useCurrentMatter();
  if (matters.isPending) return <div className="lw-content" aria-busy="true"><Skeleton height={48} /><Skeleton height={200} /></div>;
  if (!matter) return <NotAvailable />;
  return <>{matter.my_role === 'break_glass' && <BreakGlassBanner number={matter.number} />}<Outlet /></>;
}

/** Shown on every page of a matter opened through emergency access. */
function BreakGlassBanner({ number }: { number: string }) {
  const grants = useBreakGlass(true);
  const qc = useQueryClient();
  const grant = grants.data?.requests.find((g) => g.status === 'active' && g.matter_number === number);
  const end = async () => {
    if (!grant) return;
    await api.post(`/api/legal/break-glass/${encodeURIComponent(grant.id)}/end`);
    await qc.invalidateQueries({ queryKey: keys.matters });
  };
  return (
    <div role="status" className="lw-paper" style={{ margin: '12px 16px 0', padding: '10px 16px', display: 'flex', gap: 12, alignItems: 'center',
      background: 'color-mix(in srgb, var(--conflict) 9%, var(--paper))', boxShadow: 'none', border: '1px solid color-mix(in srgb, var(--conflict) 30%, transparent)' }}>
      <Siren size={18} style={{ color: 'var(--conflict)', flex: 'none' }} />
      <span style={{ flex: 1 }}><b>Emergency access.</b> Read only{grant?.expires_at ? `, until ${new Date(grant.expires_at).toLocaleTimeString([], { timeStyle: 'short' })}` : ''}. You are not a member of this matter.</span>
      {grant && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={end}>End now</button>}
    </div>
  );
}

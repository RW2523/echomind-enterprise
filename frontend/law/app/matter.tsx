import React, { useEffect } from 'react';
import { Outlet, useMatch } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { keys, useMatters } from '../api/queries';
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
  return <Outlet />;
}

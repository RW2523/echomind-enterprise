import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client';
import type { AuditPage, AuditVerification, BreakGlassRequest, Client, DocVersion, LawDocument, DirectoryMatter, FirmUser, Matter, Membership, Me, SessionInfo, Wall } from './types';

/** Query keys. Everything about one matter lives under ['matter', id] so it can be dropped in one
 *  call when the person leaves that matter or loses access (no stale content in memory). */
export const keys = {
  me: ['me'] as const,
  matters: ['matters'] as const,
  clients: ['clients'] as const,
  matter: (id: string) => ['matter', id] as const,
  memberships: (id: string) => ['matter', id, 'memberships'] as const,
  users: ['admin', 'users'] as const,
  sessions: ['me', 'sessions'] as const,
  walls: ['admin', 'walls'] as const,
  breakGlass: ['admin', 'break-glass'] as const,
  audit: ['admin', 'audit'] as const,
  activity: (id: string) => ['matter', id, 'activity'] as const,
  documents: (id: string) => ['matter', id, 'documents'] as const,
  document: (id: string, docId: string) => ['matter', id, 'documents', docId] as const,
};

export const useMe = () => useQuery({ queryKey: keys.me, queryFn: () => api.get<Me>('/api/auth/me'), retry: false });
export const useMatters = () =>
  useQuery({ queryKey: keys.matters, queryFn: async () => (await api.get<{ matters: Matter[] }>('/api/legal/matters')).matters });
export const useClients = () =>
  useQuery({ queryKey: keys.clients, queryFn: async () => (await api.get<{ clients: Client[] }>('/api/legal/clients')).clients });
export const useMemberships = (matterId: string) =>
  useQuery({
    queryKey: keys.memberships(matterId),
    queryFn: async () => (await api.get<{ memberships: Membership[] }>(`/api/legal/matters/${encodeURIComponent(matterId)}/memberships`)).memberships,
  });
export const useFirmUsers = (enabled: boolean) =>
  useQuery({ queryKey: keys.users, enabled, queryFn: async () => (await api.get<{ users: FirmUser[] }>('/api/legal/users')).users });
export const useMySessions = () =>
  useQuery({ queryKey: keys.sessions, queryFn: async () => (await api.get<{ sessions: SessionInfo[] }>('/api/auth/sessions')).sessions });

/** Mutation that refreshes the given keys afterwards. */
export function useApiMutation<TVars, TResult>(fn: (v: TVars) => Promise<TResult>, invalidate: readonly (readonly unknown[])[]) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => Promise.all(invalidate.map((k) => qc.invalidateQueries({ queryKey: k as unknown[] }))),
  });
}
export const useWalls = (enabled: boolean) =>
  useQuery({ queryKey: keys.walls, enabled, queryFn: async () => (await api.get<{ walls: Wall[] }>('/api/legal/walls')).walls });
export const useMatterDirectory = (q: string, enabled: boolean) =>
  useQuery({ queryKey: ['admin', 'directory', q], enabled,
    queryFn: async () => (await api.get<{ matters: DirectoryMatter[] }>(`/api/legal/walls/matter-directory?q=${encodeURIComponent(q)}`)).matters });
export const useBreakGlass = (enabled: boolean) =>
  useQuery({ queryKey: keys.breakGlass, enabled, refetchInterval: 30_000,
    queryFn: () => api.get<{ requests: BreakGlassRequest[]; can_approve: boolean }>('/api/legal/break-glass') });

export interface AuditFilters { action?: string; actor?: string; outcome?: string }
const qs = (o: Record<string, string | number | undefined>) =>
  Object.entries(o).filter(([, v]) => v !== undefined && v !== '').map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`).join('&');

/** The firm-wide trail, newest first, a page at a time. Not refetched behind the reader's back. */
export const useAuditTrail = (filters: AuditFilters, enabled: boolean) =>
  useInfiniteQuery({
    queryKey: [...keys.audit, filters], enabled, refetchOnWindowFocus: false, initialPageParam: undefined as number | undefined,
    queryFn: ({ pageParam }) => api.get<AuditPage>(`/api/legal/audit?${qs({ ...filters, before: pageParam })}`),
    getNextPageParam: (last) => last.next_before ?? undefined,
  });
export const useMatterActivity = (matterId: string) =>
  useInfiniteQuery({
    queryKey: keys.activity(matterId), refetchOnWindowFocus: false, initialPageParam: undefined as number | undefined,
    queryFn: ({ pageParam }) => api.get<AuditPage>(`/api/legal/matters/${encodeURIComponent(matterId)}/activity?${qs({ before: pageParam })}`),
    getNextPageParam: (last) => last.next_before ?? undefined,
  });
export const verifyAuditTrail = () => api.get<AuditVerification>('/api/legal/audit/verify');

const docsBase = (matterId: string) => `/api/legal/matters/${encodeURIComponent(matterId)}/documents`;
const busy = (d: LawDocument) => d.current_version?.index_status === 'queued' || d.current_version?.index_status === 'indexing';

/** The matter's library; polls while anything is still being read. */
export const useDocuments = (matterId: string) =>
  useQuery({
    queryKey: keys.documents(matterId),
    queryFn: async () => (await api.get<{ documents: LawDocument[] }>(docsBase(matterId))).documents,
    refetchInterval: (q) => (q.state.data?.some(busy) ? 2000 : false),
  });
/** Opening a document's details is a recorded read. */
export const useDocument = (matterId: string, docId: string | null) =>
  useQuery({
    queryKey: keys.document(matterId, docId ?? ''), enabled: !!docId, refetchOnWindowFocus: false,
    queryFn: async () => {
      const [d, v] = await Promise.all([
        api.get<{ document: LawDocument }>(`${docsBase(matterId)}/${encodeURIComponent(docId!)}`),
        api.get<{ versions: DocVersion[] }>(`${docsBase(matterId)}/${encodeURIComponent(docId!)}/versions`),
      ]);
      return { ...d.document, versions: v.versions };
    },
  });
export const documentsPath = docsBase;

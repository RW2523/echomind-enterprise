import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client';
import type { AuditPage, AuditVerification, BreakGlassRequest, Client, Connector, ConnectorLink, Deletion, LawExport, DocVersion, Hold, LabelState, LawDocument, MeetingDetail, ReaderData, RetentionPolicy, RetentionRun, MeetingSummary, ResearchTurn, ComparisonSummary, Segment, DirectoryMatter, FirmUser, Matter, Membership, Me, SessionInfo, Wall } from './types';

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
  research: (id: string) => ['matter', id, 'research'] as const,
  comparisons: (id: string) => ['matter', id, 'comparisons'] as const,
  meetings: (id: string) => ['matter', id, 'meetings'] as const,
  meeting: (id: string, meetingId: string) => ['matter', id, 'meetings', meetingId] as const,
  labels: (id: string, type: string, objectId: string) => ['matter', id, 'labels', type, objectId] as const,
  matterDeletions: (id: string) => ['matter', id, 'deletions'] as const,
  deletions: ['records', 'deletions'] as const,
  holds: ['records', 'holds'] as const,
  retention: ['records', 'retention'] as const,
  exports: (id: string) => ['matter', id, 'exports'] as const,
  export: (id: string, exportId: string) => ['matter', id, 'exports', exportId] as const,
  connectors: ['admin', 'connectors'] as const,
  links: (id: string) => ['matter', id, 'connector-links'] as const,
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
/** A document's versions (not a recorded read: numbers, files and statuses only). */
export const useVersions = (matterId: string, docId: string | null) =>
  useQuery({
    queryKey: [...keys.document(matterId, docId ?? ''), 'versions'] as const, enabled: !!docId,
    queryFn: async () => (await api.get<{ versions: DocVersion[] }>(`${docsBase(matterId)}/${encodeURIComponent(docId!)}/versions`)).versions,
  });


/** A version's text and what reading it found (S2.1). Opening it is a recorded read. */
export const useReader = (matterId: string, docId: string, versionId: string | null) =>
  useQuery({
    queryKey: [...keys.document(matterId, docId), 'reader', versionId ?? ''] as const, enabled: !!versionId,
    refetchOnWindowFocus: false, retry: false, staleTime: Infinity,
    queryFn: async () => (await api.get<{ reader: ReaderData }>(
      `${docsBase(matterId)}/${encodeURIComponent(docId)}/versions/${encodeURIComponent(versionId!)}/reader`)).reader,
  });

/** The caller's own earlier questions in this matter. */
export const useResearchHistory = (matterId: string) =>
  useQuery({ queryKey: keys.research(matterId), refetchOnWindowFocus: false,
    queryFn: async () => (await api.get<{ turns: ResearchTurn[] }>(`/api/legal/matters/${encodeURIComponent(matterId)}/research`)).turns });

export const comparePath = (matterId: string) => `/api/legal/matters/${encodeURIComponent(matterId)}/compare`;
export const useComparisons = (matterId: string) =>
  useQuery({ queryKey: keys.comparisons(matterId), refetchOnWindowFocus: false,
    queryFn: async () => (await api.get<{ comparisons: ComparisonSummary[] }>(comparePath(matterId))).comparisons });

export const meetingsPath = (matterId: string) => `/api/legal/matters/${encodeURIComponent(matterId)}/meetings`;
export const useMeetings = (matterId: string) =>
  useQuery({ queryKey: keys.meetings(matterId), queryFn: async () => (await api.get<{ meetings: MeetingSummary[] }>(meetingsPath(matterId))).meetings });
/** A meeting with its participants' consent and its capture timeline. */
export const useMeeting = (matterId: string, meetingId: string) =>
  useQuery({
    queryKey: keys.meeting(matterId, meetingId), refetchOnWindowFocus: false,
    queryFn: async () => {
      const base = `${meetingsPath(matterId)}/${encodeURIComponent(meetingId)}`;
      const [m, t] = await Promise.all([api.get<{ meeting: MeetingDetail }>(base), api.get<{ segments: Segment[] }>(`${base}/transcript`)]);
      return { ...m.meeting, transcript: t.segments };
    },
  });

const legal = (p: string) => `/api/legal${p}`;
export const labelsPath = (matterId: string, type: string, objectId: string) =>
  legal(`/matters/${encodeURIComponent(matterId)}/labels/${type}/${encodeURIComponent(objectId)}`);
export const useLabels = (matterId: string, type: 'document' | 'meeting', objectId: string) =>
  useQuery({ queryKey: keys.labels(matterId, type, objectId), refetchOnWindowFocus: false, queryFn: () => api.get<LabelState>(labelsPath(matterId, type, objectId)) });
/** Deletions waiting for approval in one matter (for those who may request and restore them). */
export const useMatterDeletions = (matterId: string, enabled: boolean) =>
  useQuery({ queryKey: keys.matterDeletions(matterId), enabled,
    queryFn: async () => (await api.get<{ deletions: Deletion[] }>(legal(`/matters/${encodeURIComponent(matterId)}/deletions`))).deletions });
export const useDeletions = (enabled: boolean) =>
  useQuery({ queryKey: keys.deletions, enabled, queryFn: async () => (await api.get<{ deletions: Deletion[] }>(legal('/deletions'))).deletions });
export const useHolds = (enabled: boolean) =>
  useQuery({ queryKey: keys.holds, enabled, queryFn: () => api.get<{ holds: Hold[]; can_request: boolean }>(legal('/holds')) });
export const useRetention = (enabled: boolean) =>
  useQuery({ queryKey: keys.retention, enabled,
    queryFn: () => api.get<{ policies: RetentionPolicy[]; runs: RetentionRun[]; apply_enabled: boolean }>(legal('/retention')) });

export const exportsPath = (matterId: string) => `/api/legal/matters/${encodeURIComponent(matterId)}/exports`;
export const useExports = (matterId: string) =>
  useQuery({ queryKey: keys.exports(matterId), queryFn: async () => (await api.get<{ exports: LawExport[] }>(exportsPath(matterId))).exports });
export const useExport = (matterId: string, exportId: string) =>
  useQuery({ queryKey: keys.export(matterId, exportId), refetchOnWindowFocus: false,
    queryFn: async () => (await api.get<{ export: LawExport }>(`${exportsPath(matterId)}/${encodeURIComponent(exportId)}`)).export });
export const useExportPreview = (matterId: string, exportId: string, planKey: string) =>
  useQuery({ queryKey: [...keys.export(matterId, exportId), 'preview', planKey], refetchOnWindowFocus: false,
    queryFn: () => api.get<{ pages: number }>(`${exportsPath(matterId)}/${encodeURIComponent(exportId)}/preview`) });

export const useConnectors = (enabled: boolean) =>
  useQuery({ queryKey: keys.connectors, enabled,
    queryFn: () => api.get<{ connectors: Connector[]; providers: { kind: string; name: string; available: boolean; note?: string }[] }>('/api/legal/connectors') });
export const useConnectorLinks = (matterId: string) =>
  useQuery({ queryKey: keys.links(matterId),
    queryFn: () => api.get<{ links: ConnectorLink[]; connectors: { id: string; name: string; kind: string }[] }>(`/api/legal/matters/${encodeURIComponent(matterId)}/connector-links`) });

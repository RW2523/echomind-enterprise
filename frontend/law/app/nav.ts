/**
 * Where a person can go (D-065): one list for the sidebar, the command palette and the matter's tiles, so the
 * three never disagree. An item a person cannot use is left out, never shown disabled.
 */
import type { LucideIcon } from 'lucide-react';
import {
  Archive, BookMarked, BookOpenText, Briefcase, CalendarCheck, CalendarClock, ClipboardCheck, ClipboardList, FileOutput, FilePen,
  Files, FileText, FolderSync, Gauge, GitCompareArrows, History, Inbox, Landmark, LayoutTemplate, ListCollapse, ListTodo, MessageSquare, Mic, ScrollText,
  ShieldOff, Siren, Timer, UserCog, Users, Users2,
} from 'lucide-react';
import type { Matter, Me } from '../api/types';
import { hasFirmRole } from './session';

export interface NavEntry { key: string; to: string; label: string; icon: LucideIcon; about?: string; hue?: number }

export function firmEntries(me: Me): NavEntry[] {
  const lawyer = hasFirmRole(me, 'managing_partner', 'attorney', 'paralegal');
  const list: (NavEntry | false)[] = [
    { key: 'chat', to: '/chat', label: 'Chat', icon: MessageSquare },
    { key: 'matters', to: '/matters', label: 'Matters', icon: Briefcase },
    { key: 'drafts', to: '/drafts', label: 'Drafts', icon: ClipboardCheck },
    lawyer && { key: 'templates', to: '/templates', label: 'Templates', icon: LayoutTemplate },
    lawyer && { key: 'playbooks', to: '/playbooks', label: 'Playbooks', icon: BookMarked },
    lawyer && { key: 'checklists', to: '/checklists', label: 'Checklists', icon: ListTodo },
    !!me.features?.intake && hasFirmRole(me, 'intake', 'conflicts', 'paralegal', 'attorney', 'managing_partner', 'records_admin')
      && { key: 'enquiries', to: '/enquiries', label: 'Enquiries', icon: Inbox },
    !!me.features?.time_entries && { key: 'time', to: '/time', label: 'Time', icon: Timer },
    !!me.features?.regulatory && hasFirmRole(me, 'managing_partner', 'attorney', 'paralegal', 'reviewer')
      && { key: 'regulatory', to: '/regulatory', label: 'Regulatory updates', icon: Landmark },
    hasFirmRole(me, 'firm_admin', 'managing_partner') && { key: 'users', to: '/admin/users', label: 'Users & roles', icon: UserCog },
    hasFirmRole(me, 'firm_admin') && { key: 'connectors', to: '/admin/connectors', label: 'Connectors', icon: FolderSync },
    hasFirmRole(me, 'managing_partner') && { key: 'walls', to: '/admin/walls', label: 'Ethical walls', icon: ShieldOff },
    hasFirmRole(me, 'firm_admin', 'managing_partner') && { key: 'emergency', to: '/admin/emergency-access', label: 'Emergency access', icon: Siren },
    hasFirmRole(me, 'records_admin', 'managing_partner') && { key: 'records', to: '/admin/records', label: 'Records', icon: Archive },
    hasFirmRole(me, 'firm_admin', 'managing_partner', 'records_admin') && { key: 'audit', to: '/admin/audit', label: 'Audit trail', icon: ScrollText },
    hasFirmRole(me, 'firm_admin', 'managing_partner', 'records_admin') && { key: 'evaluations', to: '/admin/evaluations', label: 'Evaluations', icon: Gauge },
  ];
  return list.filter(Boolean) as NavEntry[];
}

export function matterEntries(me: Me, matter: Pick<Matter, 'number' | 'my_role'>): NavEntry[] {
  const base = `/m/${encodeURIComponent(matter.number)}`;
  const list: (NavEntry | false)[] = [
    { key: 'overview', to: `${base}/overview`, label: 'Overview', icon: FileText, about: 'The matter, its parties and status.', hue: 220 },
    { key: 'documents', to: `${base}/documents`, label: 'Documents', icon: Files, about: 'Import, read and search the documents.', hue: 205 },
    { key: 'research', to: `${base}/research`, label: 'Research', icon: BookOpenText, about: 'Ask the documents; every statement quoted.', hue: 232 },
    !!me.features?.conversation && { key: 'conversation', to: `${base}/conversation`, label: 'Conversation', icon: Mic,
      about: 'Speak a question, hear the answer; follow-ups remembered.', hue: 262 },
    { key: 'compare', to: `${base}/compare`, label: 'Compare', icon: GitCompareArrows, about: 'Two versions clause by clause; conflicts.', hue: 190 },
    { key: 'drafting', to: `${base}/drafting`, label: 'Drafting', icon: FilePen, about: 'Letters and agreements from firm templates.', hue: 278 },
    { key: 'obligations', to: `${base}/obligations`, label: 'Obligations', icon: CalendarCheck, about: 'Deadlines and duties from the documents.', hue: 150 },
    { key: 'diligence', to: `${base}/diligence`, label: 'Due diligence', icon: ClipboardList, about: 'A checklist run across a data room.', hue: 168 },
    !!me.features?.summaries && { key: 'summaries', to: `${base}/summaries`, label: 'Summaries', icon: ListCollapse,
      about: 'Key points of a document, each cited.', hue: 32 },
    !!me.features?.panel && { key: 'panel', to: `${base}/panel`, label: 'Review panel', icon: Users2,
      about: 'Four perspectives on one question; a partner approves.', hue: 300 },
    !!me.features?.regulatory && { key: 'regulatory', to: `${base}/regulatory`, label: 'Regulatory', icon: Landmark,
      about: 'New legislation and rules that may concern it.', hue: 120 },
    { key: 'meetings', to: `${base}/meetings`, label: 'Meetings', icon: CalendarClock,
      about: 'Record with consent: live transcript, Silent Assist, minutes.', hue: 352 },
    ['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') && { key: 'exports', to: `${base}/exports`, label: 'Exports', icon: FileOutput,
      about: 'Release documents with redaction and approval.', hue: 14 },
    { key: 'members', to: `${base}/members`, label: 'Members', icon: Users, about: 'Who works on this matter.', hue: 240 },
    matter.my_role === 'lead' && { key: 'activity', to: `${base}/activity`, label: 'Activity', icon: History, about: 'Everything that happened here.', hue: 210 },
  ];
  return list.filter(Boolean) as NavEntry[];
}

/** A stable hue for a name, for avatars. */
export function hueOf(text: string): number {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) % 360;
  return h;
}

export function initials(text: string): string {
  const words = text.replace(/[^\p{L}\p{N}\s-]/gu, ' ').split(/\s+/).filter(Boolean);
  return ((words[0]?.[0] ?? '') + (words[1]?.[0] ?? '')).toUpperCase() || '—';
}

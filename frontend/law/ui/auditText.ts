import type { AuditEvent } from '../api/types';
import { roleLabel } from './primitives';

/**
 * Plain-language wording for audit events: "alice added carol as Paralegal", not "membership.add".
 * The raw action stays available (title attribute, exports) for anyone who needs it.
 */
const str = (v: unknown) => (typeof v === 'string' ? v : '');
const list = (v: unknown) => (Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : []);

export function matterLabel(e: AuditEvent): string {
  if (!e.matter) return '';
  return e.matter.number ?? 'a matter';
}

const REASONS: Record<string, string> = {
  not_found: 'not visible to them',
  forbidden: 'not permitted',
  matter_closed: 'the matter is closed',
  bad_password: 'wrong password',
  unknown_user: 'unknown username',
  locked: 'account locked',
  locked_out: 'too many attempts; account locked',
  disabled: 'account disabled',
  bad_code: 'code not accepted',
  csrf_failed: 'security check failed',
  break_glass: 'emergency access',
};

export function reasonLabel(reason?: string | null): string {
  if (!reason) return '';
  if (REASONS[reason]) return REASONS[reason];
  const http = /^http_(\d{3})$/.exec(reason);
  if (http) return http[1] === '422' ? 'invalid input' : http[1] === '409' ? 'conflicting change' : `refused (${http[1]})`;
  return reason.replace(/_/g, ' ');
}

/** What happened, as the rest of a sentence that starts with the actor's name. */
export function describe(e: AuditEvent): string {
  const d = e.details ?? {};
  const m = matterLabel(e);
  const who = str(d.username);
  const refused = e.outcome !== 'allow';
  switch (e.action) {
    case 'auth.login':
      if (refused) return 'could not sign in';
      return str(d.status) === 'ok' ? 'signed in' : 'entered their password';
    case 'auth.mfa.verify': return refused ? 'gave an authenticator code that was not accepted' : 'completed two-step sign-in';
    case 'auth.mfa.enroll': return refused ? 'could not start authenticator set-up' : 'started authenticator set-up';
    case 'auth.mfa.confirm': return refused ? 'could not finish authenticator set-up' : 'set up an authenticator';
    case 'auth.logout': return 'signed out';
    case 'auth.password.change': return refused ? 'could not change their password' : 'changed their password';
    case 'auth.obo.issue': return 'connected voice on their own behalf';
    case 'session.revoke': return 'ended a session';
    case 'user.create': return `created the account ${who}${str(d.role) && str(d.role) !== 'user' ? ` (${roleLabel(str(d.role))})` : ''}`;
    case 'user.delete': return `deleted the account ${who}`;
    case 'user.enrollment_link': return `issued an enrolment link for ${who}`;
    case 'user.mfa_reset': return `reset the authenticator of ${who}`;
    case 'user.sessions_revoke': return `signed ${who} out everywhere`;
    case 'user.disable': return `disabled ${who}`;
    case 'user.enable': return `re-enabled ${who}`;
    case 'roles.set': {
      const add = list(d.added).map(roleLabel), rem = list(d.removed).map(roleLabel);
      const parts = [add.length ? `added ${add.join(', ')}` : '', rem.length ? `removed ${rem.join(', ')}` : ''].filter(Boolean);
      return `changed the firm roles of ${who}${parts.length ? ` (${parts.join('; ')})` : ''}`;
    }
    case 'client.create': return `added the client ${str(d.code) || ''}`.trim();
    case 'client.update': return 'updated a client';
    case 'matter.create': return refused ? 'could not open a new matter' : `opened ${m || 'a new matter'}`;
    case 'matter.read': return refused ? `tried to view ${m || 'a matter'}` : `viewed ${m || 'a matter'}`;
    case 'matter.update': {
      if (str(d.status) === 'closed') return `closed ${m}`;
      if (str(d.status) === 'open') return `reopened ${m}`;
      if (str(d.status) === 'archived') return `archived ${m}`;
      const fields = list(d.fields).filter((f) => f !== 'status').join(', ');
      return `${refused ? 'tried to update' : 'updated'} ${m}${fields ? ` (${fields.replace(/_/g, ' ')})` : ''}`;
    }
    case 'matter.activity.read': return refused ? `tried to view the activity of ${m}` : `viewed the activity of ${m}`;
    case 'membership.add': return `${refused ? 'tried to add' : 'added'} ${who || 'a member'} to ${m}${str(d.role) ? ` as ${roleLabel(str(d.role))}` : ''}`;
    case 'membership.change': return `${refused ? 'tried to change' : 'changed'} ${who ? `${who}’s` : 'a'} membership of ${m}${str(d.role) ? ` to ${roleLabel(str(d.role))}` : ''}`;
    case 'membership.revoke': return `${refused ? 'tried to remove' : 'removed'} a member from ${m}`;
    case 'wall.request': return refused ? 'could not propose an ethical wall' : 'proposed an ethical wall';
    case 'wall.approve': return 'approved an ethical wall';
    case 'wall.reject': return 'rejected a proposed ethical wall';
    case 'wall.release.request': return 'asked to release an ethical wall';
    case 'wall.release.approve': return 'released an ethical wall';
    case 'wall.exception.request': return `asked for a wall exception for ${who}${m ? ` on ${m}` : ''}`;
    case 'wall.exception.approve': return `approved a wall exception${m ? ` on ${m}` : ''}`;
    case 'wall.exception.revoke': return `revoked a wall exception${m ? ` on ${m}` : ''}`;
    case 'breakglass.request': return `asked for emergency access to ${str(d.number) || 'a matter'}`;
    case 'breakglass.approve': return `approved emergency access${m ? ` to ${m}` : ''}`;
    case 'breakglass.reject': return 'rejected an emergency access request';
    case 'breakglass.end': return `ended emergency access${m ? ` to ${m}` : ''}`;
    case 'audit.read': return refused ? 'tried to view the audit trail' : 'viewed the audit trail';
    case 'audit.verify': return `verified the audit trail${str(d.status) === 'broken' ? ' and found it broken' : ''}`;
    case 'audit.export': return `exported the audit trail${str(d.format) ? ` (${str(d.format).toUpperCase()})` : ''}`;
    case 'request.refused': return 'had a request refused';
    case 'api.request': return `${str(d.method)} ${str(d.route)}`.trim() || 'made a request';
    default:
      return `${refused ? 'was refused' : 'performed'} ${e.action}${m ? ` on ${m}` : ''}`;
  }
}

export function actorName(e: AuditEvent): string {
  return e.actor?.username ?? (e.action === 'auth.login' ? 'Someone' : 'System');
}

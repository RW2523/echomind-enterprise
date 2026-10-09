/** Types for the EchoMind Law API (backend/app/legal/*, backend/app/api/routes/auth.py). */
export type MatterRole = 'lead' | 'attorney' | 'paralegal' | 'reviewer' | 'read_only';
export type FirmRole = 'firm_admin' | 'managing_partner' | 'attorney' | 'paralegal' | 'reviewer' | 'records_admin' | 'read_only';
export type Sensitivity = 'standard' | 'restricted' | 'highly_restricted';
export type MatterStatus = 'open' | 'closed' | 'archived';

export interface Me {
  user: { id: string; username: string; role: string; roles: FirmRole[] };
  session: { auth: string; mfa_enrolled: boolean; must_change_password: boolean; idle_expires_at: number; abs_expires_at: number };
}
export interface Client { id: string; name: string; code?: string | null; status: string; version: number; }
export interface Matter {
  id: string; number: string; title: string; practice_area?: string | null; jurisdiction?: string | null;
  status: MatterStatus; sensitivity: Sensitivity; version: number; created_at: string; updated_at?: string | null;
  client: { id: string; name: string; code?: string | null }; my_role?: MatterRole | 'break_glass' | null;
}
export interface Membership {
  id: string; user_id: string; username: string; display_name: string; role: MatterRole; can_export: boolean;
  valid_from: string; valid_to?: string | null; granted_at: string;
}
export interface FirmUser { username: string; display_name: string; status: string; roles: FirmRole[]; mfa_enabled: boolean; }
export interface SessionInfo { id: string; created_at: string; last_seen_at?: string; ip?: string; user_agent?: string; mfa_verified: boolean; current: boolean; }
export interface EnrollmentLink { token: string; expires_at: number; path: string; }

export interface Wall {
  id: string; name: string; rationale: string;
  status: 'requested' | 'active' | 'release_requested' | 'released' | 'rejected';
  requested_by: string; requested_at: string; approved_by?: string | null; approved_at?: string | null;
  release_requested_by?: string | null; released_by?: string | null; released_at?: string | null;
  screened: string[]; protected: { type: 'matter' | 'client'; id: string; label: string }[];
  exceptions: { id: string; username: string; matter_id: string; reason: string; expires_at: string; approved: boolean; revoked: boolean; requested_by: string }[];
}
export interface DirectoryMatter { id: string; number: string; title: string; client: { id: string; name: string } }
export interface BreakGlassRequest {
  id: string; matter_number: string; status: 'requested' | 'active' | 'rejected' | 'ended'; minutes: number; reason: string;
  requested_at: string; expires_at?: string | null; requester?: string; decided_by?: string | null; decision_note?: string | null;
  matter?: { number: string; title: string; client: string; sensitivity: string } | null;
}

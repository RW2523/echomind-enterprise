# 06 — Data model, access control and source provenance
## Key entities
`firms(id,name,retention_policy_id)`
`users(id,firm_id,status,auth_provider)`
`roles(id,firm_id,name)` + `user_role_assignments`
`matters(id,firm_id,name,type,jurisdiction,status,ethical_wall_id)`
`matter_memberships(matter_id,user_id,role,permissions,active)`
`clients(id,firm_id,metadata)` + `matter_clients`
`documents(id,firm_id,matter_id,hash_sha256,storage_key,mime,version,origin,privilege_label,processing_status,created_by)`
`document_pages(id,document_id,page_no,text,ocr_quality)`
`chunks(id,firm_id,matter_id,document_id,page_from,page_to,start_offset,end_offset,text_hash,index_version,embedding)`
`citations(id,chunk_id,document_id,exact_text,start_offset,end_offset,page,verification_state)`
`people(id,firm_id,display_name)`
`assertions(id,matter_id,subject,predicate,object_text,claimed_at,valid_from,valid_to,epistemic_state,source_citation_id)`
`relationships(id,matter_id,source_entity,target_entity,type,source_citation_id,valid_from,valid_to)`
`events(id,matter_id,time_start,time_end,time_certainty,description,source_citation_id,status)`
`findings(id,matter_id,kind,status,explanation,created_at)` + `finding_sources`
`conversations(id,matter_id,owner_id)` + `messages` + `ui_component_states`
`drafts(id,matter_id,type,status,reviewer_id)` + `draft_versions` + `draft_source_links`
`approval_requests(id,matter_id,action,policy,requester_id,reviewer_id,status)`
`recording_sessions(id,matter_id,consent_status,start,end,retention_class)`
`transcript_segments(id,session_id,speaker_label,t_start,t_end,text,confidence)`
`jobs(id,matter_id,kind,priority,state,progress,created_by)`
`audit_events(id,firm_id,matter_id,actor,action,target_id,result,request_id,time)`

## Do not flatten everything into embeddings
Original files remain authoritative. Chunks are search accelerators; assertions and relationships are *claims supported by evidence* rather than indisputable facts. A relationship edge without citation should be rejected from verified graph displays. Track legal authority version/jurisdiction separately from party statements.

## Suggested SQL example (adapt existing migrations)
```sql
CREATE TABLE matters (
 id uuid PRIMARY KEY, firm_id uuid NOT NULL, name text NOT NULL,
 ethical_wall_id uuid, jurisdiction text, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE documents (
 id uuid PRIMARY KEY, firm_id uuid NOT NULL, matter_id uuid NOT NULL REFERENCES matters(id),
 sha256 char(64) NOT NULL, storage_key text NOT NULL, version int NOT NULL DEFAULT 1,
 privilege_label text NOT NULL DEFAULT 'confidential', created_at timestamptz DEFAULT now()
);
CREATE INDEX documents_matter_idx ON documents (firm_id,matter_id);
```
In production also add database RLS keyed to vetted session context as defense in depth; **do not rely on RLS alone** when background jobs or pooled connections are involved. Use service authorization before every action.

## Policy model
Subject: identity, role, firm, matter memberships, ethical wall grants, administrative capabilities. Object: owner firm, matter, privilege level, sensitivity, status. Action: read, upload, annotate, create draft, approve, export, manage users, record, delete. Context: session, local network policy, ethical wall, audit reason, legal hold.

Default deny: if `firm` mismatch, no matter grant, wall restriction, revoked membership, locked matter, source no longer available, or unclear access. Firm admin does **not** automatically have access to privileged matter content.

## Retrieval / graph permission contract
All vector queries use prefilter e.g. `{firm_id: F, matter_id: M, document_id: {$in: authorized_ids}}` before top-K. Do not retrieve global top-K and filter afterward; this harms both confidentiality and recall. Graph expansion checks **each node and edge source**. Search caches, rerank traces, autocomplete and chat summaries also require scope checks.

## Citation validation
Store file hash, document version, page and exact match offsets. Reopen original extracted representation; independently match quote (allow normalized whitespace with tracked normalization), verify snippet belongs to authorized version, and log `verified/unverified`. Never present arbitrary model citation IDs as proof.

## Temporal model
`assertions` preserve claim time (when statement made), event validity time (when fact purportedly true), superseding record, and provenance. Example: contract due May 15, later amendment due May 20; both stay in history and latest effective version is derived from signed evidence.

## Retention / deletion
Matter-specific retention, legal hold overrides, encrypted disposal workflow, search index and cache invalidation, backups retention/cryptographic erasure process documented. Audit deletion separately; never silently delete evidence under hold.

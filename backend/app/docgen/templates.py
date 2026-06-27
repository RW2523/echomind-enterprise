"""
Document Studio template registry.

Each template defines, for the multi-agent generation pipeline:
  - id / name / description / icon
  - persona            : the EchoMind persona this template is tuned for
  - default_org        : default organisation/footer label
  - default_doc_type   : sub-title label (e.g. "Technical Documentation")
  - system_guidance    : voice/style instructions for every LLM call
  - section_blueprint   : ordered list of sections the planner should produce; each entry
                         guides what content + block kinds (prose, table, flow, callout,
                         status-codes) that section should contain.
  - appendix           : optional special closing section (e.g. status-codes table)

Templates are intentionally data-only so the pipeline + frontend can both consume them.
"""
from __future__ import annotations

from typing import Any, Dict, List

# kinds the planner may put in a section (maps to models.py block types + "status_codes").
# "image" is only honoured by the writer when image generation is enabled (see pipeline.py).
SECTION_KINDS = ("prose", "table", "flow", "bullets", "callout", "image", "status_codes", "toc")

TEMPLATES: Dict[str, Dict[str, Any]] = {
    # ── Vertical-pack templates (health / law / meetings / retail / bank) ────────────
    "clinical_note": {
        "id": "clinical_note",
        "name": "Clinical Visit Note (SOAP)",
        "icon": "File",
        "persona": "Clinical Assistant",
        "description": "Structured SOAP visit note — Subjective, Objective, Assessment, Plan — from a visit summary or transcript. For clinical documentation support.",
        "default_org": "EchoMind Health",
        "default_doc_type": "Clinical Documentation",
        "theme": "midnight",
        "system_guidance": (
            "Write as a careful clinician documenting an encounter. Be concise, factual, and structured. "
            "Use only information present in the source/brief; never invent vitals, doses, history, or findings. "
            "This is documentation support, not medical advice; include a one-line disclaimer."
        ),
        "section_blueprint": [
            {"key": "subjective", "title": "Subjective", "kinds": ["prose"], "guidance": "Chief complaint, history of present illness, relevant history from the source."},
            {"key": "objective", "title": "Objective", "kinds": ["table", "prose"], "guidance": "Vitals as a table; exam findings and any labs as prose/table. Only what the source states."},
            {"key": "assessment", "title": "Assessment", "kinds": ["bullets", "prose"], "guidance": "Problem list with brief clinical reasoning, grounded in the source."},
            {"key": "plan", "title": "Plan", "kinds": ["bullets"], "guidance": "Per-problem plan: treatment, follow-up, referrals, patient instructions."},
        ],
    },
    "contract_review": {
        "id": "contract_review",
        "name": "Contract Review Memo",
        "icon": "File",
        "persona": "Lawyer",
        "description": "Structured review of an agreement: key terms, risk flags, and recommended positions. Informational analysis, not legal advice.",
        "default_org": "EchoMind Law",
        "default_doc_type": "Legal Analysis",
        "theme": "midnight",
        "system_guidance": (
            "Write as an experienced legal associate reviewing a contract. Be precise and cite clauses by number/heading "
            "when present in the source. Do not invent terms. Always add: 'Informational analysis, not legal advice.'"
        ),
        "section_blueprint": [
            {"key": "summary", "title": "Summary & Parties", "kinds": ["prose"], "guidance": "Parties, purpose, term, and overall posture of the agreement."},
            {"key": "key_terms", "title": "Key Terms", "kinds": ["table"], "guidance": "Table: Term | Provision | Clause ref — fees, IP, confidentiality, liability cap, termination, governing law."},
            {"key": "risk_flags", "title": "Risk Flags", "kinds": ["bullets", "callout"], "guidance": "Risky/missing/one-sided clauses, each with why it matters. Use a callout for the single highest risk."},
            {"key": "recommendations", "title": "Recommended Positions", "kinds": ["table"], "guidance": "Table: Issue | Standard fallback position | Priority."},
        ],
    },
    "meeting_minutes": {
        "id": "meeting_minutes",
        "name": "Meeting Minutes & Actions",
        "icon": "File",
        "persona": "Meeting Facilitator",
        "description": "Board/meeting minutes from a transcript or notes: attendees, decisions, and an action-item table with owners and due dates.",
        "default_org": "EchoMind",
        "default_doc_type": "Meeting Minutes",
        "theme": "midnight",
        "system_guidance": (
            "Write crisp, neutral meeting minutes. Capture only what the source states. Make decisions and action items "
            "unambiguous, with owners and due dates where present."
        ),
        "section_blueprint": [
            {"key": "overview", "title": "Overview", "kinds": ["prose", "table"], "guidance": "Date, attendees, and agenda. Use a small table for attendees/roles if available."},
            {"key": "discussion", "title": "Discussion", "kinds": ["bullets"], "guidance": "Key discussion points per agenda item, concise bullets."},
            {"key": "decisions", "title": "Decisions", "kinds": ["bullets", "callout"], "guidance": "Explicit decisions made. Callout the most consequential one."},
            {"key": "actions", "title": "Action Items", "kinds": ["table"], "guidance": "Table: Action | Owner | Due date | Status."},
        ],
    },
    "loan_suitability": {
        "id": "loan_suitability",
        "name": "Loan Suitability Summary",
        "icon": "File",
        "persona": "Banking Advisor",
        "description": "Banking product/loan summary with required disclosures and a suitability check. Grounded in your bank's materials; not personalized investment advice.",
        "default_org": "EchoMind Bank",
        "default_doc_type": "Banking Summary",
        "theme": "midnight",
        "system_guidance": (
            "Write as a compliance-minded banking advisor. State product terms exactly from the source, surface required "
            "disclosures, and never invent rates/fees. Add: 'Not personalized investment advice.'"
        ),
        "section_blueprint": [
            {"key": "product", "title": "Product & Terms", "kinds": ["table"], "guidance": "Table: Item | Value — rate/APR/APY, fees, term, eligibility from the source."},
            {"key": "suitability", "title": "Suitability Assessment", "kinds": ["bullets", "prose"], "guidance": "Fit against the stated customer need/profile, grounded in the source."},
            {"key": "disclosures", "title": "Required Disclosures", "kinds": ["bullets", "callout"], "guidance": "KYC/AML and product disclosures that must be presented. Callout any mandatory one."},
            {"key": "next_steps", "title": "Next Steps", "kinds": ["bullets"], "guidance": "Documentation needed and the path to proceed."},
        ],
    },
    "product_quote": {
        "id": "product_quote",
        "name": "Product / Vehicle Quote Sheet",
        "icon": "File",
        "persona": "Retail Advisor",
        "description": "Customer-facing quote/spec sheet: products or vehicles with specs, pricing, warranty, and financing options from your catalog.",
        "default_org": "EchoMind Retail",
        "default_doc_type": "Sales Quote",
        "theme": "midnight",
        "system_guidance": (
            "Write a clear, honest, customer-facing quote. Use only catalog/source facts; never invent prices or specs. "
            "Be helpful, no pressure tactics."
        ),
        "section_blueprint": [
            {"key": "selection", "title": "Selection", "kinds": ["table"], "guidance": "Table of chosen product(s)/vehicle(s): Item | Key specs | Price."},
            {"key": "warranty", "title": "Warranty & Coverage", "kinds": ["bullets"], "guidance": "Warranty terms and coverage from the source."},
            {"key": "financing", "title": "Financing Options", "kinds": ["table"], "guidance": "Table: Option | Term | APR | Est. payment, from the source."},
            {"key": "summary", "title": "Quote Summary", "kinds": ["prose", "callout"], "guidance": "Total and what's included; callout the headline offer."},
        ],
    },
    "technical_document": {
        "id": "technical_document",
        "name": "Technical Document",
        "icon": "File",
        "persona": "AI Expert & Manager",
        "description": (
            "Precise engineering / system-design document: purpose, control principles, "
            "numbered sections with process flows + activity tables, and a status-codes appendix. "
            "Ideal for software flow reports, SOPs-as-spec, and EPC/PMC technical documentation."
        ),
        "default_org": "EchoMind",
        "default_doc_type": "Technical Documentation",
        "theme": "midnight",
        "cover_image": "a clean modern technical/engineering hero illustration, abstract circuit and flowchart motifs, blue and cyan, professional, minimal",
        "system_guidance": (
            "Write as a senior systems/engineering author. Be precise, structured, and factual. "
            "Prefer numbered sections, process flows (ordered steps), and two-column activity tables "
            "(e.g. 'Technical Activity' vs 'Commercial / System Activity', or 'Step' vs 'System Action'). "
            "Use exact terminology. Do not invent facts beyond the provided source/brief; where the source "
            "is thin, write reasonable, clearly-generic engineering guidance."
        ),
        "section_blueprint": [
            {"key": "purpose", "title": "Purpose & Scope", "kinds": ["prose", "callout"],
             "guidance": "One paragraph on what this document defines and who it's for. Add a 'principle' callout for any single guiding rule."},
            {"key": "principles", "title": "Control Principles", "kinds": ["table"],
             "guidance": "A 2-column table 'Principle' vs 'Meaning' capturing the core control/design principles."},
            {"key": "overview_flow", "title": "Overall Flow", "kinds": ["flow", "prose", "table"],
             "guidance": "Describe the end-to-end flow as an ordered 'flow' block (Stage1 -> Stage2 -> ...), a short intro paragraph, and a 'Core Object/Purpose' table."},
            {"key": "subflows", "title": "Detailed Sub-Flows", "kinds": ["flow", "table", "prose"],
             "guidance": "For each major sub-process, produce a numbered heading, an ordered flow block, and an activity table (e.g. Stage | Technical Activity | Commercial/System Activity, or Step | System Action | Output). Create one sub-section per distinct sub-process found in the source/brief."},
            {"key": "data_model", "title": "Core Objects / Data Model", "kinds": ["table", "prose"],
             "guidance": "A table of the core software objects/entities and their purpose, plus how they link (single source of truth, traceability)."},
        ],
        "appendix": {"key": "status_codes", "title": "Status Codes & Meaning", "kinds": ["status_codes", "table"],
                     "guidance": "A 'Code' vs 'Meaning' table of the status/state codes used across the flows."},
    },
    "business_report": {
        "id": "business_report",
        "name": "Business / Executive Report",
        "icon": "File",
        "persona": "Financial Advisor",
        "description": "Executive-ready report: executive summary, key findings, analysis, risks, and recommendations with supporting tables.",
        "default_org": "EchoMind",
        "default_doc_type": "Business Report",
        "theme": "executive",
        "cover_image": "a polished corporate abstract background, indigo gradient, subtle data and growth motifs, professional, minimal",
        "system_guidance": (
            "Write for executives: lead with the bottom line, be concise and quantitative where possible, "
            "use clear headings, bullet key points, and tables for comparisons. Ground claims in the source."
        ),
        "section_blueprint": [
            {"key": "exec_summary", "title": "Executive Summary", "kinds": ["prose", "bullets"],
             "guidance": "2-4 sentence summary then 3-6 bullet key takeaways."},
            {"key": "background", "title": "Background & Context", "kinds": ["prose"], "guidance": "Context and objectives."},
            {"key": "findings", "title": "Key Findings", "kinds": ["bullets", "table", "prose"],
             "guidance": "The main findings, with a comparison/data table where useful."},
            {"key": "analysis", "title": "Analysis", "kinds": ["prose", "table"], "guidance": "Interpret the findings; trade-offs and drivers."},
            {"key": "risks", "title": "Risks & Considerations", "kinds": ["bullets", "callout"], "guidance": "Risks and mitigations; a 'warning' callout for the top risk."},
            {"key": "recommendations", "title": "Recommendations", "kinds": ["bullets"], "guidance": "Prioritised, actionable recommendations."},
        ],
        "appendix": None,
    },
    "sop_process": {
        "id": "sop_process",
        "name": "SOP / Process Document",
        "icon": "File",
        "persona": "General Assistant",
        "description": "Standard Operating Procedure: purpose, scope, roles & responsibilities, step-by-step procedure, and references.",
        "default_org": "EchoMind",
        "default_doc_type": "Standard Operating Procedure",
        "theme": "evergreen",
        "cover_image": "a clean process and workflow abstract illustration, teal and emerald, gears and checklist motifs, minimal",
        "system_guidance": (
            "Write a clear SOP. Use imperative, numbered steps; define roles; call out safety/compliance notes. "
            "Be unambiguous and operational."
        ),
        "section_blueprint": [
            {"key": "purpose", "title": "Purpose & Scope", "kinds": ["prose"], "guidance": "What the procedure covers and its boundaries."},
            {"key": "roles", "title": "Roles & Responsibilities", "kinds": ["table"], "guidance": "Role vs Responsibility table."},
            {"key": "prereqs", "title": "Prerequisites", "kinds": ["bullets"], "guidance": "Required inputs, tools, approvals."},
            {"key": "procedure", "title": "Procedure", "kinds": ["flow", "bullets", "prose"],
             "guidance": "The main step-by-step procedure as an ordered flow and/or numbered steps; one sub-section per phase if multi-phase."},
            {"key": "controls", "title": "Controls & Exceptions", "kinds": ["bullets", "callout"], "guidance": "Quality controls, exception handling; a 'warning' callout for critical do-nots."},
        ],
        "appendix": None,
    },
    "training_guide": {
        "id": "training_guide",
        "name": "Training / Learning Guide",
        "icon": "File",
        "persona": "Teacher / Professor",
        "description": "Structured learning material: objectives, modules with explanations & examples, key takeaways, and review questions.",
        "default_org": "EchoMind",
        "default_doc_type": "Learning Guide",
        "theme": "evergreen",
        "cover_image": "a friendly modern education abstract illustration, teal, lightbulb and book motifs, minimal, professional",
        "system_guidance": (
            "Write as an expert teacher. Explain clearly with analogies and worked examples, scale complexity, "
            "and reinforce with takeaways and review questions. Encouraging, precise tone."
        ),
        "section_blueprint": [
            {"key": "objectives", "title": "Learning Objectives", "kinds": ["bullets"], "guidance": "What the learner will be able to do."},
            {"key": "overview", "title": "Overview", "kinds": ["prose"], "guidance": "Big-picture framing of the topic."},
            {"key": "modules", "title": "Modules", "kinds": ["prose", "bullets", "callout", "table"],
             "guidance": "One sub-section per module/concept: explanation, an example (callout), and key points."},
            {"key": "takeaways", "title": "Key Takeaways", "kinds": ["bullets"], "guidance": "The most important points to remember."},
            {"key": "review", "title": "Review Questions", "kinds": ["bullets"], "guidance": "5-8 review/practice questions."},
        ],
        "appendix": None,
    },
    "legal_brief": {
        "id": "legal_brief",
        "name": "Legal / Compliance Brief",
        "icon": "File",
        "persona": "Lawyer",
        "description": "Structured legal/compliance analysis using IRAC, citing clauses/regulations, with obligations, risks, and a disclaimer.",
        "default_org": "EchoMind",
        "default_doc_type": "Legal & Compliance Brief",
        "theme": "counsel",
        "cover_image": "an elegant legal abstract background, warm amber on dark, scales of justice and document motifs, refined, minimal",
        "system_guidance": (
            "Write as a careful legal analyst. Use IRAC (Issue, Rule, Application, Conclusion). Cite clauses, "
            "regulations, or document sections precisely from the source; never invent references. "
            "Always include the disclaimer: 'This is informational legal analysis, not formal legal advice. "
            "Consult a licensed attorney for binding decisions.'"
        ),
        "section_blueprint": [
            {"key": "summary", "title": "Summary", "kinds": ["prose"], "guidance": "Issue and bottom-line conclusion."},
            {"key": "issues", "title": "Issues", "kinds": ["bullets"], "guidance": "The legal/compliance questions at stake."},
            {"key": "analysis", "title": "Analysis (IRAC)", "kinds": ["prose", "table"], "guidance": "Per issue: Rule, Application, Conclusion; cite sources."},
            {"key": "obligations", "title": "Obligations & Requirements", "kinds": ["table", "bullets"], "guidance": "Obligation vs Source/Clause table."},
            {"key": "risks", "title": "Risks", "kinds": ["bullets", "callout"], "guidance": "Compliance risks; a 'warning' callout for the highest-severity risk."},
            {"key": "disclaimer", "title": "Disclaimer", "kinds": ["callout"], "guidance": "An 'info' callout with the standard legal disclaimer."},
        ],
        "appendix": None,
    },
    "meeting_summary": {
        "id": "meeting_summary",
        "name": "Meeting / Conversation Summary",
        "icon": "File",
        "persona": "General Assistant",
        "description": "Turn a chat or meeting transcript into a clean summary: overview, decisions, action items, and open questions.",
        "default_org": "EchoMind",
        "default_doc_type": "Meeting Summary",
        "theme": "midnight",
        "cover_image": "a clean abstract collaboration motif, slate and cyan, conversation and decision iconography, minimal",
        "system_guidance": (
            "Summarise a conversation faithfully and concisely. Separate decisions from discussion, extract "
            "concrete action items with owners if mentioned, and list open questions. Do not invent attendees or decisions."
        ),
        "section_blueprint": [
            {"key": "overview", "title": "Overview", "kinds": ["prose"], "guidance": "What the conversation was about."},
            {"key": "key_points", "title": "Key Points Discussed", "kinds": ["bullets"], "guidance": "Main discussion points."},
            {"key": "decisions", "title": "Decisions", "kinds": ["bullets"], "guidance": "Decisions made (or 'None recorded')."},
            {"key": "actions", "title": "Action Items", "kinds": ["table", "bullets"], "guidance": "Action | Owner | Due (when known)."},
            {"key": "open", "title": "Open Questions", "kinds": ["bullets"], "guidance": "Unresolved questions / follow-ups."},
        ],
        "appendix": None,
    },
    "pitch_deck": {
        "id": "pitch_deck",
        "name": "Pitch Deck",
        "icon": "File",
        "persona": "Financial Advisor",
        "description": "Investor/sales pitch deck: problem, solution, market, product, business model, traction, team, and the ask. Best exported as PPTX.",
        "default_org": "EchoMind",
        "default_doc_type": "Pitch Deck",
        "theme": "spotlight",
        "cover_image": "a bold modern startup hero image, dramatic dark background with magenta accent, abstract rocket/growth motif, cinematic, professional",
        "system_guidance": (
            "Write a compelling, punchy investor pitch. One idea per section, concrete and quantitative, "
            "confident but credible. Short headline-style lines, not paragraphs. Each section should read like a slide."
        ),
        "section_blueprint": [
            {"key": "problem", "title": "The Problem", "kinds": ["prose", "bullets", "image"], "guidance": "The pain, made vivid; a supporting illustration."},
            {"key": "solution", "title": "Our Solution", "kinds": ["prose", "bullets", "image"], "guidance": "What we do and why it's 10x better."},
            {"key": "market", "title": "Market Opportunity", "kinds": ["bullets", "table"], "guidance": "TAM/SAM/SOM and why now."},
            {"key": "product", "title": "Product", "kinds": ["bullets", "image"], "guidance": "Key capabilities; a product illustration."},
            {"key": "model", "title": "Business Model", "kinds": ["bullets", "table"], "guidance": "How we make money."},
            {"key": "traction", "title": "Traction", "kinds": ["bullets", "table"], "guidance": "Metrics, milestones, momentum."},
            {"key": "team", "title": "Team", "kinds": ["bullets"], "guidance": "Why this team wins."},
            {"key": "ask", "title": "The Ask", "kinds": ["prose", "bullets"], "guidance": "Raise amount and use of funds."},
        ],
        "appendix": None,
    },
    "whitepaper": {
        "id": "whitepaper",
        "name": "Whitepaper / Research Report",
        "icon": "File",
        "persona": "AI Expert & Manager",
        "description": "In-depth whitepaper: abstract, introduction, background, approach/architecture, evaluation, discussion, and conclusion with references.",
        "default_org": "EchoMind",
        "default_doc_type": "Whitepaper",
        "theme": "midnight",
        "cover_image": "an elegant research whitepaper hero, abstract data/network visualization, deep blue and cyan, sophisticated, minimal",
        "system_guidance": (
            "Write as a rigorous technical author. Be analytical and well-structured: abstract, motivation, "
            "method, evidence, and limitations. Use precise language, diagrams (as flows), and comparison tables. "
            "Cite sources from the provided material; do not fabricate references."
        ),
        "section_blueprint": [
            {"key": "abstract", "title": "Abstract", "kinds": ["prose"], "guidance": "150-word abstract."},
            {"key": "intro", "title": "Introduction", "kinds": ["prose", "image"], "guidance": "Problem, motivation, contributions; a concept illustration."},
            {"key": "background", "title": "Background & Related Work", "kinds": ["prose", "bullets"], "guidance": "Prior art and context."},
            {"key": "approach", "title": "Approach / Architecture", "kinds": ["prose", "flow", "table", "image"], "guidance": "The method/architecture as prose + a flow diagram + an architecture illustration; one sub-section per major component."},
            {"key": "evaluation", "title": "Evaluation", "kinds": ["prose", "table"], "guidance": "Results and analysis with tables."},
            {"key": "discussion", "title": "Discussion & Limitations", "kinds": ["prose", "bullets"], "guidance": "Implications and limits."},
            {"key": "conclusion", "title": "Conclusion", "kinds": ["prose"], "guidance": "Summary and future work."},
        ],
        "appendix": None,
    },
    "prd": {
        "id": "prd",
        "name": "Product Requirements (PRD)",
        "icon": "File",
        "persona": "AI Expert & Manager",
        "description": "Product Requirements Document: overview, goals & non-goals, users & use cases, requirements, UX flows, metrics, and milestones.",
        "default_org": "EchoMind",
        "default_doc_type": "Product Requirements Document",
        "theme": "executive",
        "cover_image": "a clean product design hero, indigo gradient, abstract app/wireframe motifs, modern, minimal",
        "system_guidance": (
            "Write a crisp PRD for a product/engineering team. Be specific and testable: numbered requirements, "
            "clear goals/non-goals, user stories, acceptance criteria, and measurable success metrics."
        ),
        "section_blueprint": [
            {"key": "overview", "title": "Overview", "kinds": ["prose"], "guidance": "Problem and proposed product in brief."},
            {"key": "goals", "title": "Goals & Non-Goals", "kinds": ["bullets", "table"], "guidance": "Goals vs Non-Goals."},
            {"key": "users", "title": "Users & Use Cases", "kinds": ["bullets", "table"], "guidance": "Personas and user stories."},
            {"key": "requirements", "title": "Requirements", "kinds": ["table", "bullets"], "guidance": "Numbered functional requirements with priority; a Requirement | Priority | Acceptance table; one sub-section per area."},
            {"key": "ux", "title": "UX & Flows", "kinds": ["flow", "image", "prose"], "guidance": "Key user flows as flow blocks + a UI concept illustration."},
            {"key": "metrics", "title": "Success Metrics", "kinds": ["bullets", "table"], "guidance": "Measurable success criteria."},
            {"key": "milestones", "title": "Milestones", "kinds": ["table"], "guidance": "Milestone | Scope | Target table."},
        ],
        "appendix": None,
    },
    "project_proposal": {
        "id": "project_proposal",
        "name": "Project Proposal",
        "icon": "File",
        "persona": "Financial Advisor",
        "description": "Persuasive project proposal: summary, objectives, scope, approach, timeline, budget, risks, and expected outcomes.",
        "default_org": "EchoMind",
        "default_doc_type": "Project Proposal",
        "theme": "executive",
        "cover_image": "a professional proposal hero, indigo and white, abstract blueprint/plan motifs, corporate, minimal",
        "system_guidance": (
            "Write a persuasive yet credible proposal. Lead with value, be concrete about scope, timeline, and "
            "budget, and pre-empt risks. Use tables for timeline and budget."
        ),
        "section_blueprint": [
            {"key": "summary", "title": "Executive Summary", "kinds": ["prose"], "guidance": "The proposal in brief and its value."},
            {"key": "objectives", "title": "Objectives", "kinds": ["bullets"], "guidance": "What success looks like."},
            {"key": "scope", "title": "Scope & Deliverables", "kinds": ["bullets", "table"], "guidance": "In-scope deliverables (and out-of-scope)."},
            {"key": "approach", "title": "Approach", "kinds": ["prose", "flow"], "guidance": "Methodology as prose + a phased flow."},
            {"key": "timeline", "title": "Timeline", "kinds": ["table"], "guidance": "Phase | Activities | Duration table."},
            {"key": "budget", "title": "Budget", "kinds": ["table", "prose"], "guidance": "Line-item budget table."},
            {"key": "risks", "title": "Risks & Mitigation", "kinds": ["table", "callout"], "guidance": "Risk | Impact | Mitigation; a warning callout for the top risk."},
            {"key": "outcomes", "title": "Expected Outcomes", "kinds": ["bullets"], "guidance": "Benefits and ROI."},
        ],
        "appendix": None,
    },
    "case_study": {
        "id": "case_study",
        "name": "Case Study",
        "icon": "File",
        "persona": "General Assistant",
        "description": "Marketing-grade case study: background, challenge, solution, results (with metrics), and a closing quote/CTA.",
        "default_org": "EchoMind",
        "default_doc_type": "Case Study",
        "theme": "spotlight",
        "cover_image": "a striking case-study hero image, bold dark background with magenta accent, success/results motif, editorial, professional",
        "system_guidance": (
            "Write an engaging, results-driven case study. Tell a story: situation -> challenge -> solution -> "
            "measurable results. Use concrete numbers and a confident, polished marketing tone."
        ),
        "section_blueprint": [
            {"key": "snapshot", "title": "Snapshot", "kinds": ["bullets", "table"], "guidance": "At-a-glance: client, industry, key result metrics."},
            {"key": "background", "title": "Background", "kinds": ["prose", "image"], "guidance": "The client and context; a contextual image."},
            {"key": "challenge", "title": "The Challenge", "kinds": ["prose", "bullets"], "guidance": "The problem they faced."},
            {"key": "solution", "title": "The Solution", "kinds": ["prose", "flow", "image"], "guidance": "What was done, as prose + flow + an illustration."},
            {"key": "results", "title": "Results", "kinds": ["bullets", "table", "callout"], "guidance": "Quantified outcomes; a success callout with the headline metric."},
            {"key": "cta", "title": "Conclusion", "kinds": ["prose"], "guidance": "Closing statement / call to action."},
        ],
        "appendix": None,
    },
    "architecture_doc": {
        "id": "architecture_doc",
        "name": "System Architecture Document",
        "icon": "File",
        "persona": "AI Expert & Manager",
        "description": "Software architecture doc: context, requirements, architecture overview, components, data flow, deployment, and trade-offs.",
        "default_org": "EchoMind",
        "default_doc_type": "Architecture Document",
        "theme": "midnight",
        "cover_image": "a sophisticated system architecture hero, abstract nodes and connections, deep blue and cyan, technical, minimal",
        "system_guidance": (
            "Write a clear software architecture document for engineers. Use C4-style thinking (context, containers, "
            "components), data-flow diagrams (as flows), component tables, and explicit trade-off discussion."
        ),
        "section_blueprint": [
            {"key": "context", "title": "Context & Goals", "kinds": ["prose", "image"], "guidance": "System purpose, drivers, constraints; a context diagram illustration."},
            {"key": "requirements", "title": "Requirements & Constraints", "kinds": ["bullets", "table"], "guidance": "Functional + non-functional (quality attributes)."},
            {"key": "overview", "title": "Architecture Overview", "kinds": ["prose", "flow", "image"], "guidance": "High-level architecture as prose + a flow + an architecture diagram illustration."},
            {"key": "components", "title": "Components", "kinds": ["table", "prose"], "guidance": "Component | Responsibility | Tech table; one sub-section per major component."},
            {"key": "dataflow", "title": "Data Flow", "kinds": ["flow", "prose"], "guidance": "Key data flows as flow blocks."},
            {"key": "deployment", "title": "Deployment", "kinds": ["prose", "flow", "table"], "guidance": "Runtime/deployment topology."},
            {"key": "tradeoffs", "title": "Trade-offs & Decisions", "kinds": ["table", "callout"], "guidance": "Decision | Options | Choice & Rationale table."},
        ],
        "appendix": None,
    },
    "marketing_book": {
        "id": "marketing_book",
        "name": "Marketing Book / Go-to-Market Playbook",
        "icon": "File",
        "persona": "Marketing Strategist & Brand Storyteller",
        "description": "A book-length, board-ready marketing playbook that aligns brand, demand, and growth in one authoritative handbook. It moves from market and audience analysis to positioning and value proposition, brand story, messaging pillars, the full awareness-to-advocacy funnel, the customer journey, channel and content strategy, a repeatable campaign engine, high-leverage growth tactics, a KPI dashboard, and a 90-day action plan. Built for go-to-market teams who need a single source of truth that creative, demand-gen, and leadership can all run on.",
        "default_org": "EchoMind",
        "default_doc_type": "Marketing Playbook",
        "theme": "spotlight",
        "cover_image": "Premium marketing handbook cover, cinematic and editorial. Deep near-black background washed with vivid magenta and electric-violet gradient light. A single bold upward growth curve sweeps from lower-left to upper-right, formed from an abstract audience-journey funnel that narrows and then blooms into a burst of advocacy sparks at its peak. Subtle line-art iconography woven into the composition: a megaphone, a target ring, a speech bubble, a rising bar trio. Soft volumetric glow, fine grain, generous negative space for a title, no text, no logos, high-contrast, sophisticated, confident, minimal.",
        "system_guidance": "Write as a world-class marketing strategist and brand storyteller producing a premium playbook a real team will run on, not a slide deck of platitudes. Lead with the customer's desire and the value they get; the brand is the guide, never the hero. Be punchy, benefit-led, and specific: use real audience language, vivid named examples, actual channels, and concrete target numbers instead of buzzword padding. Make every claim falsifiable and every recommendation actionable. Use confident, headline-style lines for pillars and principles; flow blocks for the funnel, journey, and campaign run-of-show; and crisp, scannable tables for personas, channels, content, and KPIs. Callouts should carry the one idea worth remembering per section. Maintain a consistent, ownable voice throughout and resolve the document into a plan someone can execute on Monday.",
        "section_blueprint": [
                {"key": "exec_overview", "title": "Marketing Strategy at a Glance", "kinds": ["prose", "bullets", "callout"], "guidance": "Open with a 2-3 sentence strategic narrative (who we serve, the change we create for them, the business outcome we drive). Then produce 4-6 bullets, one per strategic pillar, each a confident headline-style line. Close with a 'principle' callout stating the single North Star belief that governs every marketing decision in this book."},
                {"key": "market_audience", "title": "Market & Audience Analysis", "kinds": ["prose", "table", "image"], "guidance": "Write one short prose paragraph on market size, key trends, and the competitive set (name 2-3 alternatives). Produce a persona table with columns Segment | Needs & Pains | Buying Triggers | Where They Spend Attention, covering 2-4 segments in real audience language. Add one hero image visualizing the target audience or market landscape."},
                {"key": "positioning", "title": "Positioning & Value Proposition", "kinds": ["prose", "table", "callout"], "guidance": "State one sharp positioning statement using the form: For [audience] who [need], [brand] is the [category] that [unique benefit], unlike [alternative]. Produce a value-prop table with columns For (audience) | We (benefit delivered) | Unlike (alternative & why we win). Add a 'highlight' callout with the single one-line value proposition a customer would repeat to a peer."},
                {"key": "brand_story", "title": "Brand Story & Narrative", "kinds": ["prose", "image", "callout"], "guidance": "Tell the brand story with the customer as hero and the brand as guide, in 2-3 vivid paragraphs: the world before, the tension that demands change, and the transformation we make possible. Include one evocative brand image. Close with a 'highlight' callout containing the brand promise or tagline."},
                {"key": "messaging_pillars", "title": "Messaging Pillars & Voice", "kinds": ["table", "bullets", "callout"], "guidance": "Produce a messaging table with columns Pillar | Core Message | Proof Points | Primary Audience, defining 3-5 pillars. Add 4-6 bullets of tone-of-voice do's and don'ts. Close with a 'note' callout listing the words we own and the words we never use."},
                {"key": "funnel", "title": "The Funnel: Awareness to Advocacy", "kinds": ["flow", "prose", "table"], "guidance": "Open with one short intro paragraph framing the funnel as a system, not a metaphor. Produce an ordered flow block: Awareness -> Consideration -> Conversion -> Retention -> Advocacy. Then a stage table with columns Stage | Customer Mindset | Our Goal | Key Tactics | Success Metric for all five stages."},
                {"key": "customer_journey", "title": "Customer Journey & Touchpoints", "kinds": ["flow", "prose", "image"], "guidance": "Produce a flow block of real, ordered touchpoints (e.g. first ad -> landing page -> nurture email -> demo -> onboarding -> referral). Write one paragraph on the emotional arc across the journey and the top 2-3 friction points to remove. Include one journey-map image."},
                {"key": "channel_strategy", "title": "Channel Strategy", "kinds": ["table", "prose", "callout"], "guidance": "Produce a channel table with columns Channel | Funnel Stage | Role | Owner | Budget Weight | Primary KPI, covering paid, owned, and earned channels. Add one prose paragraph on channel philosophy (where we go all-in versus where we run small tests). Close with a 'principle' callout naming the 1-2 hero channels we commit to disproportionately."},
                {"key": "campaign_playbook", "title": "Campaign Playbook", "kinds": ["flow", "table", "prose"], "guidance": "Produce a repeatable campaign run-of-show as a flow block: Brief -> Concept -> Build -> Launch -> Optimize -> Report. Then a table of 2-3 flagship campaign concepts with columns Campaign | Objective | Audience | Hook | Channels | Offer. Add one short prose paragraph on the rule that makes a campaign greenlit versus killed. Keep it operational and reusable."},
                {"key": "content_strategy", "title": "Content Strategy", "kinds": ["prose", "table", "bullets"], "guidance": "Write one prose paragraph on the editorial point of view and the create-once, repurpose-everywhere model. Produce a content table with columns Theme | Format | Funnel Stage | Cadence | Channel. Add bullets separating evergreen content priorities from campaign-driven content priorities."},
                {"key": "growth_acquisition", "title": "Growth & Acquisition Tactics", "kinds": ["bullets", "table", "callout"], "guidance": "List 5-7 high-leverage acquisition and retention tactics as bullets (e.g. referral loops, SEO, partnerships, lifecycle email, paid scaling). Produce an experiment table with columns Hypothesis | Tactic | Expected Lift | Effort. Close with a 'warning' callout on tactics to avoid or that conflict with the brand."},
                {"key": "metrics_kpis", "title": "Metrics & KPI Dashboard", "kinds": ["table", "prose", "callout"], "guidance": "Produce a KPI table with columns Metric | Definition | Target | Owner | Cadence, spanning awareness, acquisition, conversion, retention, and revenue/ROI (include CAC, LTV, ROAS, and MQL->SQL conversion). Add one prose paragraph naming the single true-north metric and why. Close with a 'note' callout defining the reporting rhythm."},
                {"key": "action_plan_90", "title": "90-Day Action Plan", "kinds": ["table", "flow", "callout"], "guidance": "Produce a phased plan table with columns Phase (Days 1-30 / 31-60 / 61-90) | Focus | Key Initiatives | Owner | Success Criteria. Add a flow block of the three phases in order. Close with a 'highlight' callout naming the first quick win to ship in week one."},
            ],
        "appendix": None,
        "layout": "document",
    },
    "flyer": {
        "id": "flyer",
        "name": "Promotional Flyer",
        "icon": "File",
        "persona": "Creative Marketer",
        "description": "A single-page, high-impact promotional flyer built to stop the scroll and drive one action. Leads with a magnetic hook headline, stacks 3-5 benefit-first selling points, and closes with a bold offer and clear call-to-action. Every word earns its place: no paragraphs, no filler, just persuasion at a glance.",
        "default_org": "EchoMind",
        "default_doc_type": "Promotional Flyer",
        "theme": "spotlight",
        "cover_image": "A high-energy promotional poster hero, viewed head-on with a strong central focal point and generous negative space for a headline. Bold dynamic diagonal shapes sweep across the frame under a dramatic spotlight beam, vivid magenta and electric neon accents glowing against deep matte black. Crisp high-contrast lighting, subtle lens glow, confident and modern advertising aesthetic, clean and minimal, premium studio finish, no text.",
        "system_guidance": "You are a bold creative marketer writing flyer copy, not prose. Every line is short, punchy, and benefit-led: lead with what the reader gains, never with features or process. Use vivid, concrete, sensory language and an energetic, confident voice. Cut every hedge, qualifier, and generic phrase (\"high quality,\" \"world class,\" \"we strive to\"). Write to one reader about one offer, and aim the entire page at a single, unmistakable action. If a word does not sell, delete it.",
        "section_blueprint": [
                {"key": "hook", "title": "Headline & Hook", "kinds": ["prose", "image"], "guidance": "Produce exactly two blocks. First, a prose block holding ONLY 1-2 ultra-short tagline-style lines (no paragraphs, ~3-9 words each): the lead line names the audience and the single biggest benefit; the optional second line is a sharper restatement or a curiosity teaser. Second, one image block: a hero visual that reinforces the offer (the product, the result, or the feeling), described as a concrete art-direction prompt, not a caption."},
                {"key": "offer", "title": "Benefits & Call to Action", "kinds": ["bullets", "callout"], "guidance": "Produce exactly two blocks. First, a bullets block of 3-5 selling points, each a benefit-first fragment of 3-7 words with no full sentences and no end punctuation (e.g. \"Save 10 hours a week\"). Second, a callout block that states the offer (discount, deadline, freebie, or guarantee), gives one action-verb CTA (\"Call,\" \"Book,\" \"Claim,\" \"Visit\"), and lists contact details (phone, website, or address). Keep the callout to 2-3 tight lines."},
            ],
        "appendix": None,
        "layout": "flyer",
    },
    "brand_book": {
        "id": "brand_book",
        "name": "Brand Book / Brand Guidelines",
        "icon": "File",
        "persona": "Brand Director",
        "description": "The definitive guardian of brand identity: a premium, editorial brand book covering the brand story and mission, core values, market positioning and personality, a tone-of-voice system with do/don't direction, a full visual identity system (logo, color, typography, layout), imagery and art direction, signature messaging and taglines, accessibility and co-branding rules, real-world application examples, and brand governance. Built to inspire the boardroom and instruct the design desk — so every team, in every market, ships unmistakably on-brand.",
        "default_org": "EchoMind",
        "default_doc_type": "Brand Guidelines",
        "theme": "spotlight",
        "cover_image": "A premium editorial brand-guidelines cover, shot like the opening spread of a high-fashion lookbook. Deep matte-black canvas with a single vivid magenta accent cutting through. Composed identity motifs arranged on an invisible grid: an oversized wordmark fragment, a horizontal color-swatch strip, a serif-and-sans type specimen (\"Aa\"), and a circular logo-construction grid with faint guideline rings. Generous negative space, dramatic raking sidelight, crisp print-grade typography, restrained palette of black, white and one bold accent. Sophisticated, minimal, confident — no clutter, no stock-photo gloss, no people.",
        "system_guidance": "Write as a Brand Director authoring the company's definitive brand book — a document that must impress a CMO and arm a junior designer in the same read. Hold two registers at once: aspirational narrative that makes the brand feel inevitable, and exacting, do-this-not-that rules a stranger could execute without asking a question. Every sentence should be ownable to this brand specifically — kill generic filler, hedge words, and anything a competitor could paste into their own book unchanged. Lead with conviction, name real choices (and the choices rejected), and make abstractions concrete: turn values into behaviours, voice into example sentences, visuals into hex codes and pixel rules. Keep the brand's own voice alive throughout — polished, deliberate, human — so the book demonstrates the standard it sets. Prefer specific nouns over adjectives, active sentences over passive, and one strong line over three soft ones.",
        "section_blueprint": [
                {"key": "contents", "title": "How to Use This Book", "kinds": ["prose", "toc", "callout"], "guidance": "Open with a 2-3 sentence prose orientation: who this book is for, how it's organized, and the standard it upholds. Add a 'toc' navigating every following section. Close with a 'note' callout giving the one rule for using the book (e.g. 'When in doubt, choose the option that protects clarity and consistency') plus where to get assets and approvals."},
                {"key": "brand_story", "title": "Brand Story & Mission", "kinds": ["prose", "callout", "image"], "guidance": "Write an evocative 2-3 paragraph narrative of origin, purpose, and the change the brand exists to make — specific to this company, not a category. Add a 'mission' callout stating the mission and vision as two memorable one-liners. Include an editorial hero image capturing the brand's spirit and emotional register."},
                {"key": "brand_values", "title": "Brand Values", "kinds": ["bullets", "table"], "guidance": "Name 4-6 core values. In a bullet list, give each value a one-line manifesto in the brand's voice. Then build a 'Value | What It Means | How We Live It (behaviour)' table that makes each value observable and testable, so a team can tell whether they upheld it."},
                {"key": "positioning", "title": "Positioning & Personality", "kinds": ["prose", "table", "callout"], "guidance": "In prose, state the positioning precisely: target audience, competitive category, the single differentiator, and the reason to believe it. Add a 'Brand Personality' table mapping traits to expression ('Trait | We Are | We Are Not'). Close with a one-sentence positioning statement as a 'mission' callout, formatted 'For [audience], [brand] is the [category] that [differentiator], because [reason to believe].'"},
                {"key": "tone_of_voice", "title": "Tone of Voice", "kinds": ["prose", "table", "callout"], "guidance": "Describe the voice in prose via 3-4 named voice principles (e.g. 'Plain, not dumbed-down'). Add a do/don't table ('Do | Don't') with real rewritten example sentences, not abstractions. Provide a second short table or row covering how voice flexes by context (e.g. 'Context | Dial Up | Dial Down' for marketing vs. support vs. legal). End with a 'tip' callout giving one quick on-brand phrasing rule."},
                {"key": "visual_identity", "title": "Logo System", "kinds": ["prose", "image", "callout"], "guidance": "Cover the logo as a system. Prose on the primary logo, accepted variations (lockups, monochrome, icon-only), minimum size, and required clear-space (stated as a unit of the mark). Include a logo-construction/clear-space specimen image. Add a 'warning' callout listing specific misuses to never commit (stretch, recolor, add effects, place on busy backgrounds, rotate)."},
                {"key": "color", "title": "Color Palette", "kinds": ["prose", "table", "callout"], "guidance": "In prose, explain the palette logic (what the primary and accent colors signify and where each dominates). Build a color table with 'Color | HEX | RGB | CMYK | Usage / Ratio' covering primary, secondary, accent, and neutral colors. Add a 'note' callout stating the dominant-to-accent usage ratio and one rule for accessible color pairings."},
                {"key": "typography", "title": "Typography", "kinds": ["prose", "table", "callout"], "guidance": "In prose, introduce the typeface(s) and the personality they carry. Build a type-scale table ('Style | Typeface | Weight | Size / Line-height | Use') for headline, subhead, body, and caption. Add a 'tip' callout with hierarchy and pairing rules, web-safe fallbacks, and the one typographic habit that keeps layouts on-brand."},
                {"key": "imagery_style", "title": "Imagery & Art Direction", "kinds": ["prose", "image", "bullets"], "guidance": "In prose, define the photographic and graphic style: mood, lighting, composition, subject, color grade, and any illustration or iconography style. Include 1-2 reference images that exemplify the look (one 'hero', one 'supporting'). Add a bullet checklist of imagery do's and don'ts that an art buyer or designer can filter assets against."},
                {"key": "messaging", "title": "Messaging & Taglines", "kinds": ["callout", "table", "bullets"], "guidance": "Lead with the primary tagline as a large 'mission' callout. Add an approved-copy table ('Context | Approved Message') covering the one-liner, elevator pitch (25-50 words), and boilerplate (50-100 words). Finish with a bullet list of supporting taglines, key phrases, and proof points to reuse — plus any words or claims to avoid."},
                {"key": "accessibility", "title": "Accessibility & Co-Branding", "kinds": ["prose", "table", "callout"], "guidance": "In prose, state the accessibility commitment (target standard, e.g. WCAG AA) and the co-branding stance. Add a table for color-contrast minimums ('Pairing | Ratio | Pass/Fail') and/or partner-logo rules ('Scenario | Rule'). Close with a 'warning' callout on what must never be compromised (legibility, contrast, logo integrity, partner equity)."},
                {"key": "applications", "title": "Application Examples", "kinds": ["image", "prose", "bullets"], "guidance": "Show the identity in the wild across 3-4 touchpoints (e.g. business card, social post, packaging, web header, email signature). Provide a mockup image per example with a one-line prose caption naming what makes it on-brand. End with a bullet quick-reference of cross-application rules teams can apply to any new format."},
                {"key": "governance", "title": "Brand Governance & Contacts", "kinds": ["prose", "bullets", "callout"], "guidance": "In prose, explain how the brand is maintained: who owns it, how to request assets or approvals, and how exceptions are handled. Add a bullets block of asset locations, request channels, and review cadence. Close with a 'note' callout listing the brand team contact and the document version/date."},
            ],
        "appendix": None,
        "layout": "document",
    },
    "campaign_plan": {
        "id": "campaign_plan",
        "name": "Marketing Campaign Plan",
        "icon": "File",
        "persona": "Campaign Manager",
        "description": "A launch-ready, end-to-end marketing campaign plan: executive summary, objectives, market context, audience segments, a sharp big idea, on-message copy, an explicit channel and media mix, a phased timeline, a budgeted spend plan, measurable KPIs, a risk and contingency playbook, and a sign-off page. Built for teams that need to ship, not deliberate.",
        "default_org": "EchoMind",
        "default_doc_type": "Marketing Campaign Plan",
        "theme": "executive",
        "cover_image": "A premium, executive marketing campaign hero illustration. Deep indigo-to-violet gradient field with generous negative space at the top for a title. Centered composition built from clean geometric motifs: interlocking channel rings, a concentric target with a bright focal core, and a single confident ascending growth arrow tracing a launch trajectory into the upper third. Minimal flat shapes, crisp edges, soft directional studio lighting, restrained metallic-violet accents, no text, no logos, no clutter. Modern, polished, boardroom-grade.",
        "system_guidance": "Write as the campaign lead briefing a launch team and its executive sponsor. Be crisp, decisive, and execution-focused: every sentence should drive toward action, ownership, dates, or a measurable outcome. Lead with the audience and the benefit, then ground every claim in concrete numbers, named channels, and real dates. Name an owner wherever work is described. Cut hedging, throat-clearing, and generic marketing fluff (\"synergy,\" \"best-in-class,\" \"game-changing\"). Prefer scannable tables, flows, and tight callouts a team can act on the same day over long paragraphs. Keep the executive summary skimmable for a sponsor who reads nothing else; keep the working sections detailed enough to brief a media buyer or copywriter without a meeting. Sentence case throughout; no exclamation marks.",
        "section_blueprint": [
                {"key": "executive_summary", "title": "Executive Summary", "kinds": ["prose", "callout"], "guidance": "Write one tight paragraph (4-6 sentences) a sponsor can read in 30 seconds: what the campaign is, who it targets, the single business outcome it drives, the run window, and the headline budget. Follow with a callout stating the one-line bottom line — the primary goal, the total budget, and the launch date — so the ask is unmissable."},
                {"key": "objectives_goals", "title": "Campaign Objectives & Goals", "kinds": ["prose", "callout"], "guidance": "Open with one paragraph stating the campaign's purpose, the business outcome it serves, and the exact window it runs in. Then add a callout naming the single primary goal as a SMART target (specific metric, baseline, target number, deadline). If there are secondary goals, list at most two beneath it — do not dilute the primary."},
                {"key": "market_context", "title": "Market & Competitive Context", "kinds": ["prose", "table"], "guidance": "Write one short paragraph on the market moment and the insight the campaign exploits (timing, shift, unmet need). Then add a table with columns Competitor / Alternative, Their Position, Their Weakness, and Our Wedge, covering 3-4 rivals so the team knows exactly where we win and how to position against them."},
                {"key": "audience_segments", "title": "Target Audience & Segments", "kinds": ["prose", "table"], "guidance": "Lead with one framing sentence on who matters most and why. Then a table with columns Segment, Profile / Pain Point, Core Motivation, Where to Reach Them, and Priority (High/Med/Low). Cover 3-5 distinct segments so writers and media buyers know exactly who they are targeting and on which channels."},
                {"key": "big_idea", "title": "Big Idea / Creative Concept", "kinds": ["prose", "image", "callout"], "guidance": "Articulate the central creative concept in one vivid paragraph: the human insight it is built on, the creative device, and why it will cut through. Add an image prompt visualizing the hero creative or key visual (subject, mood, palette, composition). Close with a callout stating the campaign tagline or rallying line exactly as it would appear in market."},
                {"key": "messaging", "title": "Messaging & Key Narrative", "kinds": ["bullets", "table"], "guidance": "Provide a bulleted message hierarchy: one headline promise, then 3-4 supporting proof points, each tied to evidence. Follow with a table mapping each Segment to its tailored Message Angle, Primary CTA, and Tone so copy stays on-message and on-voice per audience."},
                {"key": "channel_media_plan", "title": "Channel & Media Plan", "kinds": ["prose", "table"], "guidance": "Open with one sentence on the channel strategy logic (which channel carries the launch vs. supports it). Then a table with columns Channel, Type (Paid/Owned/Earned), Role / Objective, Format & Assets, Owner, and Cadence. Include paid, owned, and earned channels so the mix is explicit, balanced, and assignable to a named owner."},
                {"key": "timeline_phases", "title": "Timeline & Phases", "kinds": ["flow", "prose"], "guidance": "Render the campaign as a flow of sequential phases (e.g., Tease → Launch → Sustain → Wrap), each step naming its date range, the 2-3 key activities, the owner, and the milestone that gates moving to the next phase. Add a short note on critical dependencies and the single hard deadline that cannot slip."},
                {"key": "budget_allocation", "title": "Budget Allocation", "kinds": ["table", "callout"], "guidance": "Present a table with columns Line Item / Channel, Allocation, % of Total, and Expected Output (the concrete deliverable or result the spend buys). Ensure the percentages sum to 100. Add a callout stating the total budget, the reserved contingency amount and percentage, and who controls the contingency."},
                {"key": "kpis_measurement", "title": "KPIs & Measurement", "kinds": ["prose", "table"], "guidance": "Open with one sentence on how success will be judged and reviewed. Then a table with columns KPI, Definition, Target, Source / Tool, and Reporting Cadence. Span awareness, engagement, and conversion, and tie at least one KPI directly back to the primary goal from the objectives section."},
                {"key": "risks_contingencies", "title": "Risks & Contingencies", "kinds": ["table", "callout"], "guidance": "Provide a table with columns Risk, Likelihood (H/M/L), Impact (H/M/L), Owner, and Mitigation / Contingency, covering 4-6 realistic threats across budget, timing, creative, channel, and market. Close with a callout naming the escalation owner, the go/no-go trigger, and the date that decision is made."},
                {"key": "approvals_ownership", "title": "Approvals & Ownership", "kinds": ["table", "callout"], "guidance": "Add a RACI-style table with columns Workstream, Owner, Approver, and Status (Approved / Pending / Blocked) covering creative, media, budget, and measurement. Close with a callout naming the single accountable campaign lead, the sign-off deadline, and what is blocked until approval is granted."},
            ],
        "appendix": None,
        "layout": "document",
    },
    "social_playbook": {
        "id": "social_playbook",
        "name": "Social Media Playbook",
        "icon": "File",
        "persona": "Social Media Strategist",
        "description": "A premium, ready-to-run social media playbook that turns brand ambition into daily action: positioning and audience, per-platform strategy, content pillars, tone and voice, a seven-day content calendar, engagement and community rules, influencer and UGC partnerships, campaign and moment planning, crisis response, and a metrics dashboard. Built for marketers who need to ship great social work fast and prove its impact.",
        "default_org": "EchoMind",
        "default_doc_type": "Social Media Playbook",
        "theme": "spotlight",
        "cover_image": "A bold, modern social-media hero illustration for a premium brand playbook cover. A stylized smartphone tilted at a dynamic angle, its screen erupting with floating feed cards, Reels frames, and chat bubbles that lift off into 3D space. Orbiting the device: crisp like, share, comment, save, and play-button icons rendered as glowing glass chips, with subtle motion trails and a rising engagement graph weaving through the composition. Deep charcoal-to-near-black gradient background with vivid magenta and electric pink accent lighting, a single cool cyan highlight for contrast. Clean negative space at top for a title, confident minimal geometry, soft depth-of-field, premium editorial vector style, no text, no logos.",
        "system_guidance": "Write as a sharp, in-the-feed social strategist briefing a brand team that expects results. Voice: confident, modern, and benefit-led — every line should help the reader post, engage, or measure better, never restate theory. Be tactical and specific: name formats (Reels, carousels, threads, Stories), cadences (posts per week), best-post times, and metrics instead of adjectives. Favor punchy, scannable copy — short paragraphs, parallel bullets, and concrete example posts the team could publish tomorrow. Keep recommendations platform-native and current, and tie creative back to goals so the playbook reads like a system, not a wish list. Default to the brand's own product, audience, and category in every example so the document feels custom-built, not generic. Make it the kind of document a CMO would approve and a junior creator could execute from on day one.",
        "section_blueprint": [
                {"key": "positioning_audience", "title": "Positioning & Audience", "kinds": ["prose", "bullets", "callout"], "guidance": "Open with one tight paragraph (prose) framing the brand's social mission, its distinct point of view in the category, and what winning looks like this quarter. Add a bullets block of 3-5 SMART goals (e.g. follower growth %, engagement rate, qualified DM/lead volume, share of voice) each with a number and a deadline. Add a second bullets block sketching 2-3 audience personas: who they are, what they scroll for, and the platforms where they actually live. Close with a callout naming the single most important objective the whole playbook serves."},
                {"key": "platform_strategy", "title": "Platform Strategy", "kinds": ["prose", "table", "callout"], "guidance": "Lead with a one-paragraph rationale (prose) for the channel mix and where the brand will go deep versus maintain. Add a table with columns Platform | Primary Purpose | Hero Format | Posting Cadence | Success Metric, one row per active channel (e.g. Instagram, TikTok, LinkedIn, X, YouTube). Be specific on formats (Reels, carousels, threads, Shorts) and concrete weekly cadence. Close with a callout flagging the one platform to prioritize this quarter and why."},
                {"key": "content_pillars", "title": "Content Pillars", "kinds": ["bullets", "image"], "guidance": "Define 3-5 content pillars (e.g. Educate, Entertain, Inspire, Promote, Community) as a bullets block. For each pillar give a one-line description, a ready-to-shoot example post, and a target mix percentage that sums to 100. Add an image block visualizing the pillar mix as a balanced content wheel or stacked bar so the ratio is instantly readable."},
                {"key": "tone_voice", "title": "Tone & Voice", "kinds": ["prose", "table", "callout"], "guidance": "Describe the brand's social personality in one vivid paragraph (prose) — how it would talk if it walked into a group chat. Add a table with columns We Sound Like | We Don't Sound Like, with 4-6 contrasting rows to make voice tangible. Close with a callout of quick do's and don'ts covering emoji use, slang, capitalization, and reply style."},
                {"key": "content_calendar", "title": "Weekly Content Calendar", "kinds": ["table", "prose"], "guidance": "Provide a weekly calendar table with columns Day | Platform | Pillar | Post Idea | Format | Best Time, covering Monday through Sunday with realistic, platform-appropriate ideas mapped back to the content pillars. Follow with a short prose note on the batching and scheduling workflow: when to plan, shoot, schedule, and who owns approvals."},
                {"key": "engagement_community", "title": "Engagement & Community Guidelines", "kinds": ["callout", "bullets", "table"], "guidance": "Use two or three callout blocks to set hard community rules: first-response time target, how to handle negative or troll comments, and the escalation trigger and owner. Add a bullets block of proactive engagement tactics (commenting on adjacent accounts, DM outreach, UGC reshares, recurring community rituals). Add a table with columns Scenario | Response Approach | Owner | SLA covering common situations (praise, complaint, question, crisis-adjacent)."},
                {"key": "influencer_ugc", "title": "Influencer & UGC Approach", "kinds": ["prose", "table", "bullets"], "guidance": "Explain the creator and partnership strategy in one paragraph (prose), including how partners are sourced and vetted. Add a table with columns Tier | Example Reach | Best For | Typical Ask | Comp Model covering nano, micro, and macro creators. Add a bullets block of UGC tactics — branded hashtags, reshare-rights workflow, incentives, and a simple ask script — for turning customers into advocates."},
                {"key": "campaigns_moments", "title": "Campaigns & Cultural Moments", "kinds": ["prose", "table", "callout"], "guidance": "Open with one paragraph (prose) on how always-on content and tentpole campaigns work together across the year. Add a table with columns Moment/Campaign | Window | Objective | Hero Content | Channels mapping 4-6 key launches, seasonal beats, or cultural moments the brand will own. Close with a callout on the reactive/newsjacking rule: what the brand will and will not jump on, and who approves it."},
                {"key": "crisis_response", "title": "Crisis & Issues Response", "kinds": ["callout", "bullets", "table"], "guidance": "Lead with a callout defining what counts as a social crisis versus routine negativity, so the team escalates correctly. Add a bullets block of the first-hour response steps (pause scheduled posts, acknowledge, route internally, draft holding statement). Add a table with columns Severity | Example | Action | Approver | Response Time so anyone on call knows exactly what to do under pressure."},
                {"key": "metrics_dashboard", "title": "Metrics Dashboard", "kinds": ["table", "callout"], "guidance": "Build a metrics table with columns Metric | What It Tells Us | Target | Review Cadence covering reach/impressions, engagement rate, follower growth, saves/shares, click-through, and conversions or qualified leads. Close with a callout naming the single north-star metric, the reporting rhythm (weekly check / monthly report), and who the report goes to."},
            ],
        "appendix": {"key": "channel_specs", "title": "Channel Specs & Asset Cheat Sheet", "kinds": ["table", "bullets"], "guidance": "A quick-reference table with columns Platform | Image/Video Spec | Caption Length | Hashtag Tip summarizing current best-practice asset dimensions, aspect ratios, and limits per channel. Follow with a short bullets block of evergreen posting reminders (alt text, captions/subtitles, link placement, first-comment strategy, accessibility) the team can sanity-check before every post."},
        "layout": "document",
    },
    "gtm_launch": {
        "id": "gtm_launch",
        "name": "Product Launch / Go-to-Market Plan",
        "icon": "File",
        "persona": "Product Marketing Manager",
        "description": "A decision-ready go-to-market launch plan that takes a product from positioning to market with a single cross-functional source of truth. Covers the launch thesis and objectives, market and competitive landscape, positioning and messaging pillars, target segments and personas, pricing and packaging, a phased launch timeline, channel and enablement strategy, the cross-functional team and accountabilities, success metrics, and risk mitigation. Concrete, opinionated, and built so sales, product, demand gen, and leadership can execute directly from the page.",
        "default_org": "EchoMind",
        "default_doc_type": "Go-to-Market Launch Plan",
        "theme": "executive",
        "cover_image": "A premium, energetic go-to-market launch hero illustration: an ascending trajectory line that arcs upward and resolves into a market growth curve, set against an abstract geometric grid and subtle data-point constellation. Deep indigo and corporate navy base with a confident bright accent gradient (electric blue to warm signal-amber) lighting the trajectory. Clean, modern, minimal, high-end editorial feel with generous negative space and a sense of forward momentum. Flat vector aesthetic, no photorealism, no text, no logos, no literal product mockups.",
        "system_guidance": "Write as a seasoned Product Marketing Manager owning a high-stakes, cross-functional launch. Be strategic, decisive, and specific: name the buyer, the value, and the proof in every section, never generic fluff or buzzwords. Lead with customer benefit and competitive differentiation, then earn every claim with a proof point, number, or named example. Use confident active voice and avoid hedging. Quantify wherever possible (dates, dollars, percentages, owners). Write so that a sales rep, a demand-gen lead, a product manager, and an executive could each pick up the document and know exactly what to do, by when, and how success will be measured.",
        "section_blueprint": [
                {"key": "launch_summary", "title": "Launch Summary", "kinds": ["toc", "prose", "callout"], "guidance": "Open with a one-line table of contents listing the sections that follow so readers can navigate. Then write one tight paragraph (4-6 sentences) covering what is launching, who it is for, why now, and the single headline outcome we expect. Close with a callout box stating the launch date, the launch tier (tier-1 major / tier-2 standard / tier-3 minor), the launch lead, and the one primary goal in a single sentence."},
                {"key": "strategy_objectives", "title": "Launch Strategy & Objectives", "kinds": ["prose", "bullets"], "guidance": "Write one short paragraph articulating the strategic thesis of this launch: the market bet we are making and how this launch advances company priorities. Then a bullet list of 3-5 SMART launch objectives, each phrased as a measurable outcome with a target and timeframe (e.g. 'Generate $X in influenced pipeline within 90 days'). Keep objectives outcome-oriented, not activity lists."},
                {"key": "market_landscape", "title": "Market & Competitive Landscape", "kinds": ["prose", "table"], "guidance": "Write a short paragraph (3-4 sentences) framing market size, the dominant trend, and the specific gap we exploit. Then a competitive table with columns Competitor, Their Strength, Their Weakness, Our Edge, covering 3-5 named or archetypal rivals. Make every 'Our Edge' cell a sharp, defensible differentiator, not a parity claim."},
                {"key": "positioning_messaging", "title": "Positioning & Messaging", "kinds": ["callout", "table"], "guidance": "Open with a callout containing the one-sentence positioning statement in the form: For [audience] who [need], [product] is the [category] that [benefit], unlike [alternative]. Follow with a messaging pillars table with columns Pillar, Core Message, Proof Point, covering 3-4 pillars. Each Core Message should be customer-facing copy; each Proof Point must be a concrete fact, metric, feature, or customer outcome."},
                {"key": "target_segments", "title": "Target Segments & Personas", "kinds": ["prose", "table", "bullets"], "guidance": "Write one paragraph naming the primary versus secondary segments and the rationale for sequencing them in that order. Then a persona table with columns Persona, Role & Goals, Pain, Buying Trigger, covering 2-3 personas. Close with a short bullet list of disqualifiers that explicitly states who this is NOT for, to keep targeting sharp."},
                {"key": "pricing_packaging", "title": "Pricing & Packaging", "kinds": ["prose", "table", "callout"], "guidance": "Write a paragraph on the pricing strategy and rationale: the value metric, the pricing model, and the anchor that frames perceived value. Then a packaging table with columns Tier, Price, Who It's For, Key Features / Limits, covering each offered tier. Close with a short callout describing any launch promotion or introductory offer and its expiry, or state explicitly that there is none."},
                {"key": "launch_plan_timeline", "title": "Launch Plan & Timeline", "kinds": ["flow", "table"], "guidance": "Begin with a flow diagram showing the launch phases in sequence: Pre-Launch -> Soft Launch -> Launch Day -> Post-Launch / Sustain. Follow with a phases table with columns Phase, Timeframe, Key Activities, Owner, Exit Criteria. Make each Exit Criteria a binary, checkable gate so phase readiness is unambiguous."},
                {"key": "channel_enablement", "title": "Channel & Enablement Plan", "kinds": ["table", "bullets"], "guidance": "Build a channel table with columns Channel, Audience, Asset / Play, Owner, KPI, spanning the relevant mix (web, email, paid, PR, social, events, partners). Then a bullet list of sales and CS enablement deliverables (pitch deck, battlecard, demo script, FAQ, pricing sheet, objection handling), each with a named owner and a readiness date that precedes launch day."},
                {"key": "team_raci", "title": "Cross-Functional Team & Ownership", "kinds": ["prose", "table"], "guidance": "Write one sentence naming the launch lead and the executive sponsor. Then a RACI-style ownership table with columns Workstream, Responsible, Accountable, Consulted, Informed, covering the core workstreams (product, marketing, sales, customer success, support, legal/finance). Ensure every workstream has exactly one Accountable owner."},
                {"key": "success_metrics", "title": "Success Metrics", "kinds": ["prose", "table"], "guidance": "Write one sentence naming the single north-star metric for this launch and why it matters. Then a metrics table with columns Metric, Baseline, Target, Timeframe, Owner, covering the funnel across awareness, pipeline/adoption, and revenue. Every Target must be a specific number, and every Metric must have a named owner."},
                {"key": "risks_mitigation", "title": "Risks & Mitigation", "kinds": ["table", "callout"], "guidance": "Build a risk table with columns Risk, Likelihood, Impact, Mitigation, Owner, covering the top 4-6 launch risks (technical readiness, competitive response, messaging miss, adoption shortfall, pricing pushback). Use a consistent Likelihood/Impact scale (e.g. High/Medium/Low). Close with a callout stating the go/no-go decision criteria, the decision owner, and the decision date."},
            ],
        "appendix": {"key": "launch_assets_checklist", "title": "Launch Readiness Checklist", "kinds": ["table", "bullets"], "guidance": "Build a readiness checklist table with columns Asset / Deliverable, Owner, Status, Due Date, covering every cross-functional dependency (messaging assets, web/product changes, sales enablement, PR, legal sign-off, pricing in systems). Use a consistent Status scale (e.g. Not Started / In Progress / Ready). Close with a short bullet list of the final go-live gate items that must all be green before launch is approved."},
        "layout": "document",
    },
}

DEFAULT_TEMPLATE_ID = "technical_document"


def list_templates() -> List[Dict[str, Any]]:
    """Public template metadata for the frontend gallery (no internal prompt guidance)."""
    out = []
    for t in TEMPLATES.values():
        blueprint = t["section_blueprint"]
        supports_images = any("image" in (s.get("kinds") or []) for s in blueprint) or bool(t.get("cover_image"))
        out.append({
            "id": t["id"],
            "name": t["name"],
            "icon": t.get("icon", "File"),
            "persona": t["persona"],
            "description": t["description"],
            "sections": [s["title"] for s in blueprint] + ([t["appendix"]["title"]] if t.get("appendix") else []),
            "default_doc_type": t["default_doc_type"],
            "theme": t.get("theme", "midnight"),
            "supports_images": supports_images,
        })
    return out


def get_template(template_id: str) -> Dict[str, Any]:
    return TEMPLATES.get(template_id) or TEMPLATES[DEFAULT_TEMPLATE_ID]

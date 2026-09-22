"""Voice tool routing: the model decides whether a turn needs the organisation's own
material instead of a keyword heuristic guessing it.

The tools are offered on every conversational turn (their block is byte-stable so the
TRT-LLM prefix cache absorbs it). The model either answers directly — that stream IS the
reply, zero extra latency — or emits ONE tool call, which the session turns into the
existing grounded backend path (/api/chat/ask-voice-stream) with matching source options.
The tool's short ``query``/``identifier`` argument doubles as the topic of the spoken
hold phrase ("Let me check the refund policy."), which is why it is asked to be short."""
from __future__ import annotations

import re
from typing import Dict, Optional

# One-tool-per-turn, short arguments (a 2-6 word query streams in ~0.5 s; copying the
# whole user sentence took ~0.9 s in measurement).
VOICE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the organisation's own uploaded documents: policies, regulations, procedures, "
                "contracts, agreements, fee schedules, product terms, protocols, manuals, reports. Use it "
                "for ANY question whose answer should come from those materials rather than general "
                "knowledge, including when the user cites a section, paragraph, volume, chapter or page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short topic phrase, 2 to 6 words, naming what to look up "
                                       "(e.g. 'refund policy cancellations', 'notice period clause').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_transcripts",
            "description": (
                "Search saved meeting and call transcripts: what was said, discussed, decided or promised "
                "earlier, by whom, and when. Use it for questions about previous meetings, calls or "
                "conversations that are not part of the current chat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short topic phrase, 2 to 6 words (e.g. 'vendor contract decision').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_record",
            "description": (
                "Pull up the record for a specific identifier or named party: customer, client, account, "
                "policy, order, ticket, case or matter number, or a person's or company's name. Use it "
                "when the user gives an ID or asks to open, pull up, look up or check a specific record."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "The identifier or name exactly as given (e.g. '48213', 'case 4471', 'Ms Patel').",
                    }
                },
                "required": ["identifier"],
            },
        },
    },
]

TOOL_NAMES = tuple(t["function"]["name"] for t in VOICE_TOOLS)

# Appended to the system prompt on tool-routed turns. Kept byte-stable.
ROUTER_RULES = (
    "KNOWLEDGE RULE (highest priority): you have no built-in knowledge of this organisation. Whenever the "
    "user asks about its policies, regulations, procedures, contracts, fees, products, protocols, meetings, "
    "cases, accounts, or names a person, record or identifier, CALL ONE TOOL FIRST and do not guess. If you "
    "are unsure whether the organisation's materials cover it, call search_knowledge_base. Never reply with "
    "'I'll look that up', 'let me check', 'one moment' or 'I don't have that information' - those are what "
    "the tools are for; call the tool instead of saying it. Never say you could not find something "
    "unless a tool has already returned nothing, and a lookup that found nothing earlier says "
    "nothing about a new topic - call a tool again. Answer directly, with no tool, only for "
    "greetings, small talk, thanks, general knowledge, and follow-ups that need no new information. Never "
    "mention tools, searching or documents in a direct answer.\n"
)


def source_options_for(tool: str) -> Dict[str, bool]:
    """Backend retrieval scope for each tool (the backend defaults to all three)."""
    if tool == "search_transcripts":
        return {"document": False, "transcript": True, "general": True}
    if tool == "lookup_record":
        return {"document": True, "transcript": True, "general": False}
    return {"document": True, "transcript": False, "general": True}


def tool_topic_arg(name: str, arguments: Dict) -> Optional[str]:
    """The argument that names the topic: query for searches, identifier for records."""
    if not isinstance(arguments, dict):
        return None
    for key in (("identifier", "query") if name == "lookup_record" else ("query", "identifier", "topic", "q")):
        v = arguments.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


_SECTION_REF_RE = re.compile(r"\b(?:section|paragraph|vol(?:ume)?\.?|chapter|page)\s*[0-9]", re.I)


def cites_a_reference(text: str) -> bool:
    """User named a section/volume/page: always grounded, even if the model tried to answer directly."""
    return bool(_SECTION_REF_RE.search(text or ""))

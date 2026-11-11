"""About a hundred phrasings, half that must reach the documents and half that must not (2026-11-12, evaluation fix list
items 5 and 10). Keyword rules decide which pipeline answers a chat message; these keep them from drifting: "What are the
payment obligations of the Customer?" returned the task register and never read the documents, "Write a Python script…"
opened the template-draft card, "Thanks, that helps!" was searched and then withdrawn."""
from __future__ import annotations

import pytest

from app.legal.counsel import intent

TO_THE_DOCUMENTS = [
    "What are the payment obligations of the Customer?",
    "What obligations does Harrowgate have under the MSA?",
    "What are the Supplier's obligations?",
    "What are the confidentiality obligations?",
    "Any obligations on the Customer to pay interest?",
    "What deadlines does the lease set?",
    "Which deadlines apply to the Tenant?",
    "What notice deadlines does the agreement give?",
    "What are the insurance obligations of the Provider?",
    "What reporting obligations are there for Apex?",
    "What post-termination obligations survive?",
    "List the data protection obligations of the processor",
    "What tasks does the Consultant have to perform?",
    "What is the liability cap?",
    "How can the Master Services Agreement be terminated?",
    "Who are the parties to the supply agreement?",
    "When does the term end?",
    "What is the notice period for termination for convenience?",
    "Is there a late payment interest rate?",
    "Prepare a summary of the payment terms",
    "Write down the notice period",
    "Write out the termination clause",
    "What does clause 12.3 say?",
    "Who signed the delivery note?",
    "When did Witness B say the delivery took place?",
    "What did Harrowgate propose for the renewal?",
    "Did the board approve it?",
    "On what conditions?",
    "What is the monthly storage fee now?",
    "Did Amendment No. 1 change the liability cap?",
    "The cap is USD 500,000, correct?",
    "cap?",
    "How much has Apex offered to settle?",
    "What are the service levels?",
    "What happens if they are missed?",
    "Which units were rejected and why?",
    "What did Simone Achterberg learn from her phone call?",
    "Who must keep the roof in repair?",
    "Who pays the buildings insurance?",
    "Can the Recipient disclose information to its employees?",
    "What governing law applies to the NDA?",
    "Where should notices to the Provider be sent?",
    "What fees are payable each month?",
    "If delivery was 23 days late, what liquidated damages are due?",
    "What is 28 July plus 45 days under the notice clause?",
    "¿Cuál es el límite de responsabilidad de cada parte?",
    "Quelle est la durée du contrat ?",
    "What did the advice memo recommend about the cap?",
    "Is that consistent with the board minutes?",
    "What date is the acceptance certificate dated?",
]

NOT_THE_DOCUMENTS = [
    ("What are the open obligations?", "tasks"),
    ("List upcoming deadlines", "tasks"),
    ("Show my tasks", "tasks"),
    ("Any overdue deadlines?", "tasks"),
    ("Which obligations are overdue?", "tasks"),
    ("What are the obligations?", "tasks"),
    ("What deadlines are coming up this week?", "tasks"),
    ("Show the obligations register", "tasks"),
    ("What are my open tasks for Monday?", "tasks"),
    ("What is due tomorrow?", "ask_fact"),         # no register word for "what": the documents are read (kept as it was)
    ("Draft a letter before claim to Apex", "draft"),
    ("Write a reply to Harrowgate", "draft"),
    ("Please write an email to the client about the renewal", "draft"),
    ("Compose a notice of termination", "draft"),
    ("Prepare the board minutes", "draft"),
    ("Draft a client update", "draft"),
    ("Hi", "smalltalk"),
    ("Hello, can you hear me?", "smalltalk"),
    ("Hola", "smalltalk"),
    ("Bonjour", "smalltalk"),
    ("Good morning", "smalltalk"),
    ("Thanks", "smalltalk"),
    ("Thanks, that helps!", "smalltalk"),
    ("That helps a lot, thank you", "smalltalk"),
    ("Gracias", "smalltalk"),
    ("Merci beaucoup", "smalltalk"),
    ("Cheers", "smalltalk"),
    ("Great, got it", "smalltalk"),
    ("OK thanks", "smalltalk"),
    ("That's all for now", "smalltalk"),
    ("Goodbye", "smalltalk"),
    ("What can you do?", "smalltalk"),
    ("Who are you?", "smalltalk"),
    ("Which model are you?", "smalltalk"),
    ("What model is this?", "smalltalk"),
    ("Are you ChatGPT?", "smalltalk"),
    ("Who made you?", "smalltalk"),
    ("What is today's date?", "smalltalk"),
    ("What day is it today?", "smalltalk"),
    ("What did I ask you first?", "smalltalk"),
    ("What was my last question?", "smalltalk"),
    ("What have we discussed so far?", "smalltalk"),
    ("Write a Python script to add up the fees", "smalltalk"),
    ("Write me a poem about the merger", "smalltalk"),
    ("Tell me a joke", "smalltalk"),
    ("What's the weather in London today?", "smalltalk"),
    ("Find conflicting dates or amounts", "compare"),
    ("Give me a chronology of the delivery, from dispatch to acceptance", "chronology"),
    ("What's the latest on this case?", "case_status"),
    ("Which documents are available?", "documents"),
]


@pytest.mark.parametrize("text", TO_THE_DOCUMENTS)
def test_a_question_about_what_the_documents_say_is_researched(text):
    assert intent.classify(text) == "ask_fact", text


@pytest.mark.parametrize("text,want", NOT_THE_DOCUMENTS)
def test_a_request_for_a_register_a_draft_or_small_talk_is_not_researched(text, want):
    assert intent.classify(text) == want, text


def test_the_two_halves_are_about_the_same_size():
    assert 45 <= len(TO_THE_DOCUMENTS) <= 55 and 45 <= len(NOT_THE_DOCUMENTS) <= 55


def test_each_aside_has_its_own_kind():
    assert intent.aside("Which model are you?") == "model"
    assert intent.aside("What is today's date?") == "today"
    assert intent.aside("What did I ask you first?") == "memory"
    assert intent.aside("Write a Python script to add up the fees") == "offtopic"
    assert intent.aside("What does clause 4 say about the date?") is None

"""
EchoMind: tests for ConversationMemory and intent routing.
Run from voice/ with: python -m pytest tests/test_echo_memory.py -v
Or: PYTHONPATH=app pytest tests/test_echo_memory.py -v
"""
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from conversation_memory import ConversationMemory, MemoryEntry
from echo_commands import parse_and_route, strip_wake_word


def test_memory_add_and_query_last():
    mem = ConversationMemory(window_minutes=60.0)
    mem.add_text("Hello world", speaker="user")
    mem.add_text("I said something important", speaker="user")
    entries = mem.query_last(1)
    assert len(entries) == 2
    assert entries[0].text == "Hello world"
    assert entries[1].text == "I said something important"


def test_memory_query_topic():
    mem = ConversationMemory(window_minutes=60.0)
    mem.add_text("We discussed the budget yesterday", speaker="user")
    mem.add_text("The weather is nice", speaker="user")
    mem.add_text("Budget was approved", speaker="assistant")
    found = mem.query_topic("budget")
    assert len(found) == 2
    texts = [e.text for e in found]
    assert "budget" in " ".join(texts).lower()


def test_memory_summarize_last():
    mem = ConversationMemory(window_minutes=60.0)
    mem.add_text("First utterance", speaker="user")
    mem.add_text("Second utterance", speaker="assistant")
    s = mem.summarize_last(1)
    assert "First utterance" in s
    assert "Second utterance" in s
    assert "User" in s or "user" in s
    assert "Assistant" in s or "assistant" in s


def test_strip_wake_word():
    assert strip_wake_word("EchoMind, what did I say?", "EchoMind") == "what did I say?"
    assert strip_wake_word("echomind  hello", "EchoMind") == "hello"
    assert strip_wake_word("Something else", "EchoMind") == "Something else"


def test_intent_set_assistant_name():
    profile = {"assistant_name": "EchoMind", "wake_word": "EchoMind", "user_name": "", "timezone": "America/New_York", "location": ""}
    handled, response, extra = parse_and_route("Your name is Watson", profile, "", False, [])
    assert handled is True
    assert extra.get("set_assistant_name") == "watson"
    assert response and "watson" in response.lower()


def test_intent_start_listening():
    profile = {"assistant_name": "EchoMind", "wake_word": "EchoMind", "user_name": "", "timezone": "America/New_York", "location": ""}
    handled, response, extra = parse_and_route("Listen to conversation", profile, "", False, [])
    assert handled is True
    assert extra.get("set_listen_only") is True


def test_intent_fact_check():
    profile = {"assistant_name": "EchoMind", "wake_word": "EchoMind", "user_name": "", "timezone": "America/New_York", "location": ""}
    handled, response, extra = parse_and_route("Fact check that", profile, "", False, [])
    assert handled is True
    assert extra.get("fact_check") is True
    assert response is None


def test_intent_memory_query_minutes():
    profile = {"assistant_name": "EchoMind", "wake_word": "EchoMind", "user_name": "", "timezone": "America/New_York", "location": ""}
    handled, response, extra = parse_and_route("What did I say in the last 5 minutes?", profile, "", False, [])
    assert handled is True
    assert extra.get("memory_query_type") == "recap"
    assert extra.get("memory_query_minutes") == 5.0


def test_intent_clear_memory():
    profile = {"assistant_name": "EchoMind", "wake_word": "EchoMind", "user_name": "", "timezone": "America/New_York", "location": ""}
    handled, response, extra = parse_and_route("Clear memory", profile, "", False, [])
    assert handled is True
    assert extra.get("clear_memory") is True

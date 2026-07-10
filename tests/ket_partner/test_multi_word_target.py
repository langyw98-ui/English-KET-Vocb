import re

import pytest

from flow.ket_partner.multi_word_target import (
    build_target_pattern,
    find_placeholder,
    has_placeholder,
    target_in_sentence,
)


@pytest.mark.parametrize("target,expected", [
    ("give somebody a call", True),
    ("give somebody a ring", True),
    ("tell someone a story", True),
    ("buy something", True),
    ("alarm clock", False),
    ("CD player", False),
    ("cat", False),
    ("", False),
    (None, False),
])
def test_has_placeholder(target, expected):
    assert has_placeholder(target) is expected


@pytest.mark.parametrize("sentence,should_match", [
    ("I give my mom a call every night.", True),
    ("He gives him a call.", True),            # verb inflection (gives)
    ("She is giving the teacher a call.", True),   # verb inflection (giving)
    ("She gives the tall man a call.", True),  # 3-word substitution: limit
    ("They give the very tall man a call.", False),  # 4 words: over limit
    ("I give a call.", False),                 # missing substitution
    ("I give the dog a walk.", False),         # wrong literal word
    ("I gave my mom a call.", False),          # irregular past tense not covered
    ("Please call my mom.", False),            # missing phrase structure
])
def test_build_target_pattern_placeholder_phrase(sentence, should_match):
    pat = build_target_pattern("give somebody a call")
    assert bool(pat.search(sentence)) is should_match


@pytest.mark.parametrize("sentence,should_match", [
    ("The alarm clock rings.", True),
    ("The Alarm Clock rings.", True),          # case-insensitive
    ("I set the alarm clock.", True),
    ("The clock alarms me.", False),           # different word order
    ("alarm clocks ring.", True),              # substring match — existing behavior
])
def test_build_target_pattern_literal_phrase(sentence, should_match):
    pat = build_target_pattern("alarm clock")
    assert bool(pat.search(sentence)) is should_match


@pytest.mark.parametrize("target,sentence,expected", [
    ("alarm clock", "the alarm clock rings", True),
    ("alarm clock", "the clock alarms", False),
    ("give somebody a call", "I give him a call", True),
    ("give somebody a call", "I give a call", False),
    ("cat", "the cat sleeps", True),
])
def test_target_in_sentence(target, sentence, expected):
    assert target_in_sentence(target, sentence) is expected


def test_find_placeholder_returns_first_match():
    assert find_placeholder("give somebody a call") == "somebody"
    assert find_placeholder("tell someone something") == "someone"
    assert find_placeholder("alarm clock") == ""
    assert find_placeholder("") == ""
    assert find_placeholder(None) == ""

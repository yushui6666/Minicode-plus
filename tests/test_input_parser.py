"""Regression tests for the bracketed-paste path of parse_input_chunk.

Pasting text into the TUI crashed with
``TextEvent.__init__() missing 1 required positional argument: 'meta'``
because the paste branch was the only TextEvent construction site that
omitted ``meta``. Existing coverage never hit this branch: the hypothesis
fuzzer almost never synthesizes a complete ESC[200~ ... ESC[201~ pair, and
the ported paste test feeds bare CRLF text without the markers.
"""
from __future__ import annotations

from minicode.tui.input_parser import TextEvent, parse_input_chunk


def test_bracketed_paste_produces_meta_false_text_event():
    chunk = "\x1b[200~pasted text\x1b[201~"
    result = parse_input_chunk(chunk)
    events = result.events
    remaining = result.rest

    assert remaining == ""
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, TextEvent)
    assert event.text == "pasted text"
    assert event.ctrl is False
    assert event.meta is False


def test_bracketed_paste_strips_unprintable_control_chars():
    chunk = "\x1b[200~a\x07b\x00c\td\ne\x1b[201~"
    events = parse_input_chunk(chunk).events

    # \x07 and \x00 are stripped; tab and newline survive
    assert [e.text for e in events] == ["abc\td\ne"]


def test_partial_paste_waits_for_end_marker():
    chunk = "\x1b[200~not finished yet"
    result = parse_input_chunk(chunk)
    events = result.events
    remaining = result.rest

    assert events == []
    # The opening marker is consumed; the partial text stays buffered as
    # the remainder until the end marker arrives in a later chunk.
    assert remaining == "not finished yet"


def test_regular_typing_still_works():
    result = parse_input_chunk("hi")
    events = result.events
    remaining = result.rest

    assert [e.text for e in events] == ["h", "i"]
    assert remaining == ""


def test_paste_after_typed_text_keeps_both():
    chunk = "ab\x1b[200~XY\x1b[201~c"
    result = parse_input_chunk(chunk)
    events = result.events
    remaining = result.rest

    assert remaining == ""
    texts = [getattr(e, "text", "") for e in events]
    assert texts == ["a", "b", "XY", "c"]

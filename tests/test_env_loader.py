"""Tests for the project ``.env`` loader in minicode.config.

The loader fills missing process env vars from a project .env file so
the documented ``cp .env.example .env`` flow works without dotenv.
"""
from __future__ import annotations

from pathlib import Path

from minicode.config import _load_env_file


def _write(tmp_path: Path, text: str) -> Path:
    env = tmp_path / ".env"
    env.write_text(text, encoding="utf-8")
    return env


def test_parses_basic_pairs_comments_and_quotes(tmp_path):
    lines = [
        "# comment line",
        "",
        "PLAIN=value",
        "QUOTED=\"hello world\"",
        "SINGLE='single'",
        "export EXPORTED=yes",
        "SPACED =  padded  ",
    ]
    values = _load_env_file(_write(tmp_path, "\n".join(lines)))
    assert values["PLAIN"] == "value"
    assert values["QUOTED"] == "hello world"
    assert values["SINGLE"] == "single"
    assert values["EXPORTED"] == "yes"
    assert values["SPACED"] == "padded"


def test_first_duplicate_key_wins(tmp_path):
    env = _write(tmp_path, "KEY=first\nKEY=second")
    assert _load_env_file(env)["KEY"] == "first"


def test_missing_file_returns_empty(tmp_path):
    assert _load_env_file(tmp_path / "nope.env") == {}


def test_malformed_lines_skipped(tmp_path):
    env = _write(tmp_path, "NO_EQUALS_SIGN\n=A\nOK=1")
    assert _load_env_file(env) == {"OK": "1"}

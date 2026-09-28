"""The README's numbers are produced by the code, and this file proves it.

The quickstart block is executed; every `print(...)  # expected` line must print the
expected text. The demo output and the CLI transcripts pasted in the README must match
what the commands print today.
"""

from __future__ import annotations

import contextlib
import io
import re
import runpy
import sys
from pathlib import Path

import pytest

from creditrisk.cli import main

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def blocks(lang: str) -> list[str]:
    return re.findall(rf"```{lang}\n(.*?)```", README, flags=re.S)


def section(title: str) -> str:
    start = README.index(title)
    end = README.find("\n## ", start + len(title))
    return README[start : end if end != -1 else None]


def test_the_quickstart_runs_and_prints_what_the_comments_say(tmp_path, monkeypatch):
    quick = next(b for b in blocks("python") if "fit_scorecard(" in b and "print(" in b)
    expected = [
        line.split("#", 1)[1].strip()
        for line in quick.splitlines()
        if line.startswith("print(") and "#" in line
    ]
    assert len(expected) == 6
    monkeypatch.chdir(tmp_path)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(quick, "README-quickstart", "exec"), {})
    assert out.getvalue().splitlines() == expected


def test_the_keeping_block_runs(tmp_path, monkeypatch):
    quick = next(b for b in blocks("python") if "fit_scorecard(" in b and "print(" in b)
    keep = next(b for b in blocks("python") if "card.save(" in b)
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(quick + keep, "README-keeping", "exec"), {})
    assert (tmp_path / "card.json").exists()


def test_the_demo_output_in_the_readme_is_current(monkeypatch):
    pasted = next(b for b in re.findall(r"```\n(.*?)```", section("## Input / Output"), re.S))
    monkeypatch.setattr(sys, "path", list(sys.path))  # demo.py inserts src/
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        runpy.run_path(str(ROOT / "demo.py"), run_name="__main__")
    assert out.getvalue() == pasted


def test_the_cli_transcripts_in_the_readme_are_current(tmp_path, monkeypatch, capsys):
    transcript = re.findall(r"```\n(\$ creditrisk.*?)```", section("## Command line"), re.S)[0]
    monkeypatch.chdir(tmp_path)
    assert main(["sample-data", "loans.csv"]) == 0
    capsys.readouterr()
    (tmp_path / "applicant.json").write_text(
        '{"income": 34, "age": 29, "employment": "self-employed"}', encoding="utf-8"
    )
    actual = []
    for chunk in transcript.split("$ ")[1:]:
        command, _, shown = chunk.partition("\n")
        if not command.startswith("creditrisk "):
            continue
        assert main(command.split()[1:]) == 0
        assert capsys.readouterr().out.strip() == shown.strip(), command
        actual.append(command)
    assert len(actual) == 2


def test_the_stated_test_count_is_real(request):
    """Checked against the collected session, so the README cannot drift."""
    stated = {int(n) for n in re.findall(r"(\d+) tests", README)}
    assert len(stated) == 1, f"README states different test counts: {stated}"
    config = request.config
    if any(".py" in a or "::" in a for a in config.args) or config.option.keyword:
        pytest.skip("only meaningful when the whole suite is collected")
    assert request.session.testscollected == stated.pop()

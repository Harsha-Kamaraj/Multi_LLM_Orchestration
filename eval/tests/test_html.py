"""The HTML report must be self-contained, theme-complete, and honest.

Three classes of check. **Self-containment**: no network reference of any kind,
because a report that needs a CDN stops rendering the day the CDN moves — and
this file is meant to survive being emailed and opened in five years.
**Theme completeness**: every colour token defined on bare `:root`, which is the
classic unreadable-report bug — a token defined only inside a media query
renders one theme's text on the other theme's ground. **Honesty**: the numbers
in the page are the numbers in `results.json`.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest

from eval.html import render
from eval.leakage import AuditReport, Finding
from eval.policies import standard_baselines
from eval.report import build

LAMS = (0.0, 0.05, 0.2)

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
         "meta", "param", "source", "track", "wbr"}


@pytest.fixture(scope="module")
def report(store):
    return build(store, standard_baselines(), lams=LAMS, n_resamples=200, seed=4)


@pytest.fixture(scope="module")
def page(report):
    return render(report)


# --- self-containment ------------------------------------------------------

def test_no_external_references(page):
    """No CDN, no webfont, no remote image. The file must render offline."""
    for pattern in ("http://", "https://", "//cdn", "@import", "<script"):
        assert pattern not in page, f"external reference {pattern!r}"


def test_no_remote_font_or_stylesheet(page):
    assert "<link" not in page
    assert "fonts.googleapis" not in page


def test_document_is_well_formed(page):
    """Unclosed tags are the bug that hides between source and output."""

    class Checker(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack: list[str] = []
            self.errors: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag not in _VOID:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in _VOID:
                return
            if not self.stack:
                self.errors.append(f"stray </{tag}>")
            elif self.stack[-1] != tag:
                self.errors.append(f"</{tag}> closes <{self.stack[-1]}>")
            else:
                self.stack.pop()

    checker = Checker()
    checker.feed(page)
    assert checker.errors == []
    assert checker.stack == [], f"unclosed: {checker.stack}"


# --- theme completeness ----------------------------------------------------

def test_every_token_is_defined_on_bare_root(page):
    """The classic unreadable-report bug: a colour whose only definition sits
    inside a media query never applies in the un-stamped default state, so the
    page renders one theme's text on the other theme's ground."""
    used = set(re.findall(r"var\((--[a-z-]+)\)", page))
    bare = page.split(":root {", 1)[1].split("}", 1)[0]
    defined = set(re.findall(r"(--[a-z-]+)\s*:", bare))
    assert used - defined == set(), f"undefined in bare :root: {sorted(used - defined)}"


def test_all_three_theme_states_are_handled(page):
    """system-default, explicit dark, explicit light."""
    assert "prefers-color-scheme: dark" in page
    assert ':root:not([data-theme="light"])' in page
    assert ':root[data-theme="dark"]' in page


def test_body_paints_its_own_background(page):
    """A transparent body silently borrows the host's ground."""
    body = page.split("body {", 1)[1].split("}", 1)[0]
    assert "background: var(--paper)" in body


def test_wide_tables_scroll_inside_their_container(page):
    """The page body must never scroll sideways."""
    assert "overflow-x: auto" in page
    assert page.count('class="scroll"') >= 1


# --- honesty ---------------------------------------------------------------

def test_every_policy_appears(page, report):
    for name in report.payload["policies"]:
        assert name in page


def test_the_reference_is_marked_not_merely_captioned(page, report):
    """Burying the reference in prose is how a reader comes away thinking the
    policy was compared against the easy baselines."""
    assert 'class="ref"' in page
    assert report.payload["reference_policy"] in page


def test_every_comparison_renders_an_interval(page, report):
    """No bare means — the project's own merge blocker, checked in the output
    a human actually reads."""
    assert page.count("<svg") >= len(report.payload["comparisons"])


def test_accuracy_values_match_the_payload(page, report):
    for name, summary in report.payload["policies"].items():
        assert f"{summary['accuracy'] * 100:.2f}%" in page, name


def test_interval_bars_share_one_scale(report):
    """Rescaling each row independently would make a tiny effect look large."""
    from eval.html import _interval_bar

    wide = _interval_bar(-0.5, 0.5, 0.0, scale=1.0)
    narrow = _interval_bar(-0.05, 0.05, 0.0, scale=1.0)
    assert wide != narrow, "the same scale must render different widths"


def test_an_interval_crossing_zero_is_drawn_neutrally():
    from eval.html import _interval_bar

    crosses = _interval_bar(-0.02, 0.03, 0.005, scale=0.1)
    clears = _interval_bar(0.01, 0.03, 0.02, scale=0.1)
    assert "var(--zero)" in crosses
    assert "var(--good)" in clears


def test_taxonomy_categories_are_rendered_with_their_meaning(page, report):
    taxonomy = report.payload["failure_taxonomy"]
    for name, count in taxonomy["counts"].items():
        if count:
            assert name in page, name
            assert taxonomy["meaning"][name][:24] in page, name


def test_warnings_render_above_the_numbers(store):
    """A blocked audit changes what every figure below it means, so it cannot
    sit in a footer."""
    audit = AuditReport([Finding("canary", False, "canary present")])
    page = render(build(store, standard_baselines(), lams=LAMS,
                        n_resamples=100, audit=audit))
    assert "not publishable" in page
    assert page.index("not publishable") < page.index("At a glance")


def test_content_is_escaped(store):
    """Task ids and error classes reach this page from model output."""
    report = build(store, standard_baselines(), lams=LAMS, n_resamples=100)
    report.payload["run_id"] = '<script>alert("x")</script>'
    page = render(report)
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_write_emits_html_beside_the_json(tmp_path, report):
    report.write(tmp_path / "out")
    html_path = tmp_path / "out" / "report.html"
    assert html_path.exists()
    assert "<!doctype html>" in html_path.read_text(encoding="utf-8")


def test_manifest_records_the_html(tmp_path, report):
    import json

    report.write(tmp_path / "out")
    manifest = json.loads((tmp_path / "out" / "_MANIFEST.json").read_text())
    assert manifest["html"] == "report.html"

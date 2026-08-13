"""Reproducibility is a test, not a promise.

The golden-run gate only works if the report is genuinely byte-stable. These
tests attack that: same inputs twice, different insertion orders, NaN handling,
and a deliberately altered result that must be caught.
"""

from __future__ import annotations

import json

import pytest

from eval.leakage import AuditReport, Finding
from eval.policies import standard_baselines
from eval.report import (
    PRECISION,
    build,
    compare_to_golden,
    dumps,
    format_table,
)

LAMS = (0.0, 0.05, 0.2)


@pytest.fixture(scope="module")
def report(store):
    return build(store, standard_baselines(), lams=LAMS, n_resamples=300, seed=7)


def test_report_is_byte_identical_across_runs(store):
    """The whole regression gate rests on this."""
    a = build(store, standard_baselines(), lams=LAMS, n_resamples=300, seed=7)
    b = build(store, standard_baselines(), lams=LAMS, n_resamples=300, seed=7)
    assert a.to_json() == b.to_json()


def test_a_different_seed_changes_the_report(store):
    """If it did not, the seed is not reaching the bootstrap and 'reproducible'
    would mean 'constant'."""
    a = build(store, standard_baselines(), lams=LAMS, n_resamples=300, seed=1)
    b = build(store, standard_baselines(), lams=LAMS, n_resamples=300, seed=2)
    assert a.to_json() != b.to_json()


def test_serialization_is_key_order_independent():
    """A dict iteration-order change must not masquerade as a result change."""
    assert dumps({"a": 1, "b": 2}) == dumps({"b": 2, "a": 1})


def test_floats_are_rounded_to_fixed_precision():
    out = json.loads(dumps({"x": 0.1234567891234}))
    assert out["x"] == round(0.1234567891234, PRECISION)


def test_non_finite_becomes_null_not_nan():
    """JSON has no NaN. Emitting one produces a file that is not valid JSON and
    that some parsers accept anyway — the worst combination."""
    text = dumps({"x": float("nan"), "y": float("inf")})
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"x": None, "y": None}


def test_report_contains_every_policy_and_comparison(report):
    payload = report.payload
    assert set(payload["policies"]) == set(standard_baselines())
    expected = set(standard_baselines()) - {payload["reference_policy"]}
    assert set(payload["comparisons"]) == expected


def test_every_comparison_carries_an_interval(report):
    """No bare means — a merge blocker in CONTRIBUTING.md, so it is a test."""
    for name, cmp in report.payload["comparisons"].items():
        ci = cmp["accuracy_difference"]
        assert {"point", "low", "high", "level", "method"} <= set(ci), name
        assert ci["low"] <= ci["point"] <= ci["high"], name


def test_multiplicity_correction_is_applied(report):
    for name, cmp in report.payload["comparisons"].items():
        mc = cmp["mcnemar"]
        assert "p_adjusted" in mc, name
        assert mc["p_adjusted"] >= mc["p_value"] - 1e-12, name


def test_bootstrap_settings_are_recorded(report):
    """A bootstrap without a recorded seed is not reproducible, and 'we ran it
    again and got roughly the same thing' is not the claim being made."""
    b = report.payload["bootstrap"]
    assert b["n_resamples"] == 300 and b["seed"] == 7 and b["level"] == 0.95


def test_missing_reference_policy_is_refused(store):
    """Comparing only against the easy baselines answers an easier question."""
    policies = {k: v for k, v in standard_baselines().items()
                if k != "verifier_gated_cascade"}
    with pytest.raises(KeyError, match="reference policy"):
        build(store, policies, lams=LAMS, n_resamples=100)


def test_headroom_is_in_the_report(report):
    """Reported before any policy is judged, so a small gap is not misread as a
    weak policy when it is a saturated problem."""
    assert "escalation_helps" in report.payload["headroom"]


def test_blocked_audit_marks_the_report_unpublishable(store):
    audit = AuditReport([Finding("canary", False, "canary present")])
    report = build(store, standard_baselines(), lams=LAMS,
                   n_resamples=100, audit=audit)
    assert report.payload["leakage_audit"]["blocked"] is True
    assert any("LEAKAGE AUDIT BLOCKED" in w for w in report.warnings)


# --- the golden gate -------------------------------------------------------

def test_golden_matches_itself(tmp_path, report):
    golden = tmp_path / "results.json"
    golden.write_text(report.to_json(), encoding="utf-8")
    assert compare_to_golden(report, golden) == []


def test_golden_catches_a_moved_number(tmp_path, report):
    """The bug this gate exists for: a statistic moves, every test still passes
    because the tests were written against the new behaviour."""
    payload = json.loads(report.to_json())
    payload["policies"]["always_small"]["accuracy"] += 0.01
    golden = tmp_path / "results.json"
    golden.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")

    diffs = compare_to_golden(report, golden)
    assert diffs
    assert any("always_small" in d and "accuracy" in d for d in diffs)


def test_golden_catches_an_added_key(tmp_path, report):
    payload = json.loads(report.to_json())
    del payload["headroom"]
    golden = tmp_path / "results.json"
    golden.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    assert any("headroom" in d and "added" in d
               for d in compare_to_golden(report, golden))


def test_missing_golden_is_an_explicit_error(tmp_path, report):
    with pytest.raises(FileNotFoundError, match="regression gate does not exist"):
        compare_to_golden(report, tmp_path / "absent.json")


# --- sealing ---------------------------------------------------------------

def test_write_seals_with_a_manifest(tmp_path, report):
    """A directory is invalid until the manifest lands; readers skip unsealed
    ones so a half-written report is never read as complete."""
    path = report.write(tmp_path / "out")
    manifest = json.loads((tmp_path / "out" / "_MANIFEST.json").read_text())
    assert manifest["run_id"] == report.run_id
    assert manifest["precision"] == PRECISION
    assert manifest["sha256"]
    assert path.exists()


def test_manifest_checksum_matches_the_file(tmp_path, report):
    import hashlib

    path = report.write(tmp_path / "out")
    manifest = json.loads((tmp_path / "out" / "_MANIFEST.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_table_renders_every_policy_with_an_interval(report):
    text = format_table(report)
    for name in report.payload["policies"]:
        assert name in text
    assert "reference" in text
    assert text.count("[") >= len(report.payload["comparisons"])

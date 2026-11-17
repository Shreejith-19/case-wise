from isr.pipeline import rule_abstain_reason
from isr.query.rules import ParsedQuery

AB = {"missing_provision": True, "unresolved_law": True}


def _p(acts=(), unresolved=False):
    p = ParsedQuery(text="q", acts=[{"title": "x", "year": None, "document_ids": list(acts), "how": "title"}] if acts else [])
    p.named_law_unresolved = unresolved
    return p


def test_missing_provision_needs_resolved_act_named_number_and_no_hit():
    assert rule_abstain_reason(_p(["d1"]), ["32"], [], AB) == "missing_provision"
    assert rule_abstain_reason(_p(["d1"]), ["32"], [7], AB) is None      # provision exists
    assert rule_abstain_reason(_p(["d1"]), [], [], AB) is None           # no number named
    assert rule_abstain_reason(_p([]), ["32"], [], AB) is None           # Act not resolved


def test_unresolved_law_and_switches():
    assert rule_abstain_reason(_p([], unresolved=True), [], [], AB) == "unresolved_law"
    assert rule_abstain_reason(_p([], unresolved=True), [], [], {}) is None
    assert rule_abstain_reason(_p(["d1"]), ["32"], [], {"unresolved_law": True}) is None

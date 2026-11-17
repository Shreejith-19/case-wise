from isr.xref import extract


def nums(text, target=None):
    return sorted(n for k, n, t in extract(text) if target is None or t == target)


def test_single_and_lists():
    assert nums("as provided in section 7 of this Act") == ["7"]
    assert nums("under sections 4 and 5, or section 9A") == ["4", "5", "9A"]


def test_subsection_markers_are_not_numbers():
    assert nums("referred to in sub-section (2) of section 12") == ["12"]
    assert nums("section 12 (3) (a)") == ["12"]


def test_ranges_expand_small_only():
    assert nums("sections 10 to 12") == ["10", "11", "12"]
    assert nums("sections 1 to 200") == ["1", "200"]


def test_external_and_parent_targets():
    assert nums("section 302 of the Indian Penal Code, 1860", "external") == ["302"]
    assert nums("section 8 of the Act", "parent") == ["8"]
    assert nums("section 8 of the said Act", "parent") == ["8"]
    assert nums("section 3 of this Act", "self") == ["3"]


def test_rules_and_articles():
    assert [(k, n) for k, n, _ in extract("in the manner specified in rule 5")] == [("rule", "5")]
    assert [(k, n) for k, n, _ in extract("subject to article 21")] == [("article", "21")]

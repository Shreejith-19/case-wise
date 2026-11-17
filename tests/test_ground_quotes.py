from isr.eval.kit import kit_score
from isr.llm.stages import ground_quotes

SRC = ("The mark shall consist of the figure of the rising sun, and the words \"Hkkjrh; mRikn\" "
       "printed below the design, and the period of six months after he had ceased to be subject to this Act.")


def _all_grounded(text: str, sources: list[str]) -> bool:
    from benchmark.terms import normalise
    hay = " ".join(normalise(s) for s in sources)
    return all(normalise(q) in hay for q in kit_score.quotes(text))


def test_real_quote_kept():
    t = ground_quotes('Within "six months after he had ceased to be subject" of release.', [SRC])
    assert kit_score.quotes(t) == ["six months after he had ceased to be subject"]


def test_fake_quote_demoted():
    t = ground_quotes('It says "a period of ten years from the date" here.', [SRC])
    assert kit_score.quotes(t) == [] and "'a period of ten years from the date'" in t


def test_short_quotes_cannot_pair_across():
    # the scorer skips short pairs and would read the text *between* two short quotes as one quote
    t = ground_quotes('The "sun" figure, the figure of the rising sun, and the words "Bharat" appear.', [SRC])
    assert _all_grounded(t, [SRC])


def test_no_cited_sources_means_no_quotes():
    t = ground_quotes('He has "six months after he had ceased to be subject" to act.', [])
    assert kit_score.quotes(t) == []


def test_unpaired_mark_removed():
    t = ground_quotes('He has "six months after he had ceased to be subject" and "dangling', [SRC])
    assert '"' not in t

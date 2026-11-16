"""Budget packer: ranked units -> ranked character spans within a budget.

Budget accounting mirrors the official scorer (``within_budget``): only characters not
already returned for that document count. Rules:
* one unit per dedup class (copies of the same text never spend budget twice);
* a unit whose characters are already all returned is skipped;
* when the next unit does not fit, emit the best window of the remaining size inside it
  (centred on its highest-ranked sub-unit, if one was retrieved) — never worse than the
  oracle's truncation once the right provision is found (plan.md Phase 1 results);
* stop when fewer than ``min_piece`` characters remain.
"""

from __future__ import annotations

from typing import Optional, Sequence

from isr.store import UnitStore

Span = tuple[str, int, int]
MIN_PIECE = 200


def _novel(taken: list[tuple[int, int]], s: int, e: int) -> int:
    return (e - s) - sum(max(0, min(e, b) - max(s, a)) for a, b in taken)


def pack(ranked: Sequence[int], budget: int, store: UnitStore, *,
         min_piece: int = MIN_PIECE, member_of: Optional[dict[int, int]] = None) -> list[Span]:
    """``member_of`` optionally maps class_id -> the member unit to emit (e.g. jurisdiction swap)."""
    rank_of: dict[int, int] = {}
    for r, uid in enumerate(ranked):
        rank_of.setdefault(int(uid), r)
    best_child: dict[int, int] = {}            # provision uid -> best-ranked retrieved sub-unit
    for uid in ranked:
        uid = int(uid)
        if store.kind[uid] == "subunit":
            best_child.setdefault(int(store.parent[uid]), uid)

    spans: list[Span] = []
    taken: dict[str, list[tuple[int, int]]] = {}
    seen: set[int] = set()
    used = 0
    for uid in ranked:
        uid = int(uid)
        cid = int(store.class_id[uid])
        if cid in seen:
            continue
        seen.add(cid)
        if member_of and cid in member_of:
            uid = member_of[cid]
        did, s, e = store.span(uid)
        t = taken.setdefault(did, [])
        novel = _novel(t, s, e)
        if novel <= 0:
            continue
        room = budget - used
        if novel <= room:
            spans.append((did, s, e))
            t.append((s, e))
            used += novel
        else:
            if room < min_piece:
                break
            focus = best_child.get(uid) if store.kind[uid] == "provision" else None
            if focus is not None:
                fs, fe = int(store.start[focus]), int(store.end[focus])
                ws = max(s, min(fs - max(0, (room - (fe - fs)) // 2), e - room))
            else:
                ws = s
            spans.append((did, ws, ws + room))
            used = budget
            break
        if used >= budget:
            break
    return spans

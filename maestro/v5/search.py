"""The search stage's beam search — pure structure (the builder renders and measures).

Stage 1: every table in the chunk is rendered as an osc-1 probe and listened to (the full
audition). Stage 2: the B closest singles are beamed into ORIENTED pairs — (a, b) and (b, a)
are both rendered, because osc 1 leads and the order is part of the answer. Depth stops at
pairs; assembling a third oscillator is the judge's recombo step over the pooled
configurations.

The search ranks only against the target — never against the GT. The GT tables of a chunk
matter only to the local gate: one of the returned top-k configurations must contain every
GT table in the chunk (in relative oscillator order), else the judge can never reassemble
the answer. If the search can't produce one, the oracle rescues it into the reply (the
sample stays honest: the search heard what was there, and the pool still reaches the judge).
"""
from __future__ import annotations

BEAM = 8          # singles that survive to the pair stage
TOP_K = 3         # configurations returned per chunk
CHUNK = 32        # tables per search subagent
BATCH = 10        # render commands per bash step


def chunks(window_len: int, size: int = CHUNK) -> list[tuple[int, int]]:
    """[start, end) index pairs splitting a window into contiguous chunks of ~`size` (the number of
    chunks is rounded, so the sizes differ by at most one and there is no degenerate tail chunk)."""
    k = max(1, -(-window_len // size))
    base, r = divmod(window_len, k)
    sizes = [base + 1] * r + [base] * (k - r)
    out, start = [], 0
    for w in sizes:
        out.append((start, start + w))
        start += w
    return out


def oriented_pairs(entries: list[dict]) -> list[tuple[dict, dict]]:
    """All ordered (a, b) with a != b — both orientations: (a,b) leads on osc 1, (b,a) on osc 2."""
    return [(a, b) for a in entries for b in entries if a["id"] != b["id"]]


def beam(singles_ranked: list[dict], n: int = BEAM) -> list[dict]:
    """The top singles of a distance ranking (best first)."""
    return singles_ranked[:n]


def contains_ordered(ids: list[str], need: list[str]) -> bool:
    """True if `ids` contains every id in `need`, in `need`'s relative order (as a subsequence)."""
    i = 0
    for x in ids:
        if i < len(need) and x == need[i]:
            i += 1
    return i == len(need)


def gate_missing(configs: list[list[dict]], need: list[str]) -> bool:
    """The local gate: does any returned configuration contain all of the chunk's GT tables,
    in relative order? (need = the chunk's GT table ids, in the plan's oscillator order.)"""
    if not need:
        return False
    return not any(contains_ordered([e["id"] for e in c], need) for c in configs)


def pair_label(a: dict, b: dict) -> str:
    return f"{a['id']}_{b['id']}"


def widen_pairs(missed: list[dict], top: list[dict], rendered: set[str]) -> list[tuple[dict, dict]]:
    """Targeted widen: instead of re-beaming to B+4 (which renders ~76 new pairs), render only the
    pairs that involve a GT single that just missed the beam (both orientations, vs the top-B singles)
    — ~2B renders, which keeps a widened chunk under the training sequence budget."""
    out = []
    for g in missed:
        for x in top:
            if x["id"] == g["id"]:
                continue
            for p in ((g, x), (x, g)):
                if pair_label(*p) not in rendered:
                    out.append(p)
    for i in range(len(missed)):  # two GT tables both missing: pair them with each other
        for j in range(i + 1, len(missed)):
            for p in ((missed[i], missed[j]), (missed[j], missed[i])):
                if pair_label(*p) not in rendered:
                    out.append(p)
    return out


def combo_lines(combos: list[tuple[str, list[dict]]], out_dir: str, init: str, set_cli: str,
                wt_cli: str, render_cli: str) -> list[str]:
    """One render command per combo: cp init, turn on the extra oscs, swap the tables (osc 1..n),
    fix silent-at-frame-0 tables to frame 128, render through the session notes. `combos` is
    (label, entries in oscillator order)."""
    lines = [f"mkdir -p {out_dir}"]
    for label, entries in combos:
        p = f"{out_dir}/{label}"
        parts = [f"cp {init} {p}.json"]
        for n, e in enumerate(entries, 1):
            if n > 1:
                parts.append(f"{set_cli} {p}.json osc_{n}_on=on >/dev/null")
            frame = f" && {set_cli} {p}.json osc_{n}_wave_frame=128 >/dev/null" if e.get("silent_frame0") else ""
            parts.append(f"{wt_cli} set {p}.json --osc {n} {e['id']} >/dev/null{frame}")
        parts.append(f"{render_cli} {p}.json notes.json {p}.wav")
        lines.append(" && ".join(parts))
    return lines


def batched(lines: list[str], batch: int = BATCH) -> list[str]:
    """Split render lines into bash steps of at most `batch` commands (one step per batch)."""
    out = []
    for i in range(0, len(lines), batch):
        out.append("\n".join(lines[i:i + batch]))
    return out

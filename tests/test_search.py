"""maestro.v5.search: the beam-search structure (pure — no containers): chunking, oriented pairs, the local
gate (every GT table of a chunk must reach the judge inside one returned configuration), combo render lines."""
from maestro.v5 import search

W = 499


def test_chunks_cover_the_window_contiguously():
    for n in (1, 8, 31, 32, 33, 63, 64, 321, W):
        spans = search.chunks(n)
        assert spans[0][0] == 0 and spans[-1][1] == n
        assert sum(b - a for a, b in spans) == n
        sizes = {b - a for a, b in spans}
        assert max(sizes) - min(sizes) <= 1          # even out: sizes differ by at most one
        assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))
        if n >= search.CHUNK:
            assert min(sizes) >= search.CHUNK // 2   # no degenerate tail chunks


def test_chunk_count():
    assert len(search.chunks(W)) == 16               # ~32 each
    assert len(search.chunks(321)) == 11             # 29/30 each (fixed-32 left a 1-table tail)
    assert len(search.chunks(32)) == 1
    assert len(search.chunks(64)) == 2


def test_oriented_pairs_both_orders_no_self():
    es = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    p = search.oriented_pairs(es)
    assert {(x["id"], y["id"]) for x, y in p} == {
        ("a", "b"), ("a", "c"), ("b", "a"), ("b", "c"), ("c", "a"), ("c", "b")}


def test_beam_takes_the_head():
    es = [{"id": str(i)} for i in range(10)]
    assert [e["id"] for e in search.beam(es, 8)] == ["0", "1", "2", "3", "4", "5", "6", "7"]
    assert [e["id"] for e in search.beam(es, 12)] == [str(i) for i in range(10)]


def test_contains_ordered_is_a_subsequence_check():
    assert search.contains_ordered(["a", "b"], ["a", "b"])
    assert search.contains_ordered(["x", "a", "b", "y"], ["a", "b"])
    assert search.contains_ordered(["a", "x"], ["a"])
    assert not search.contains_ordered(["b", "a"], ["a", "b"])   # wrong relative order
    assert not search.contains_ordered(["a", "c"], ["a", "b"])   # missing
    assert not search.contains_ordered([], ["a"])


def test_gate_missing():
    a, b, c, d = [{"id": i} for i in "abcd"]
    assert not search.gate_missing([[c]], [])                     # no GT in the chunk: vacuous
    assert not search.gate_missing([[a, b]], ["a", "b"])
    assert not search.gate_missing([[c, a]], ["a"])               # one GT table: any orientation passes
    assert not search.gate_missing([[c, d], [a, b]], ["a", "b"])  # somewhere in the top-k
    assert search.gate_missing([[c, d]], ["a"])
    assert search.gate_missing([[b, a]], ["a", "b"])              # relative order broken
    assert search.gate_missing([], ["a"])


def test_widen_pairs_targeted():
    g = {"id": "g"}                                # GT single that just missed the beam
    t = [{"id": f"t{i}"} for i in range(4)]        # top-B singles
    extra = search.widen_pairs([g], t, set())
    assert {search.pair_label(*p) for p in extra} == {f"g_t{i}" for i in range(4)} | \
                                                   {f"t{i}_g" for i in range(4)}
    # already-rendered pairs are skipped
    extra2 = search.widen_pairs([g], t, {"g_t0"})
    assert all(search.pair_label(*p) != "g_t0" for p in extra2) and len(extra2) == 7


def test_widen_pairs_two_missed():
    g1, g2 = {"id": "a"}, {"id": "b"}
    t = [{"id": "x"}, {"id": "y"}]
    extra = search.widen_pairs([g1, g2], t, set())
    labels = {search.pair_label(*p) for p in extra}
    assert {"a_b", "b_a"} <= labels                # the two GT tables are paired with each other
    assert len(extra) == 10                        # 2 missed × 2 top × 2 orders + 2 GT×GT


def test_batch_impression():
    from maestro.v5.agents.searcher import _batch_impression
    a, b, q = ({"id": x} for x in ("a", "b", "q"))
    dist = {"a": 5.0, "b": 3.0, "q": 9.0}
    line = _batch_impression([a, b, q], dist, {"q"}, None, lambda it: it["id"])
    assert line == ("b is the closest of that batch, but still not the target. "
                    "q renders near-silent on this patch; dropping it.")
    assert _batch_impression([a], dist, set(), b, lambda it: it["id"]) == "Nothing in that batch beats b."
    assert _batch_impression([q], dist, {"q"}, None, lambda it: it["id"]) == \
        "q renders near-silent on this patch; dropping it."


def test_combo_lines_two_osc():
    e1 = {"id": "wt001_a"}
    e2 = {"id": "wt002_b", "silent_frame0": True}
    lines = search.combo_lines([("c1", [e1, e2])], "judge", "IN", "SET", "WT", "REND")
    assert lines[0] == "mkdir -p judge"
    cmd = lines[1]
    assert cmd.startswith("cp IN judge/c1.json && ")
    assert "SET judge/c1.json osc_2_on=on >/dev/null" in cmd
    assert "WT set judge/c1.json --osc 1 wt001_a >/dev/null" in cmd
    assert "WT set judge/c1.json --osc 2 wt002_b >/dev/null && SET judge/c1.json osc_2_wave_frame=128 >/dev/null" in cmd
    assert cmd.endswith(" && REND judge/c1.json notes.json judge/c1.wav")


def test_combo_lines_single_osc():
    e = {"id": "wt007_a"}
    l = search.combo_lines([("s", [e])], "p", "IN", "SET", "WT", "REND")
    assert l[1] == "cp IN p/s.json && WT set p/s.json --osc 1 wt007_a >/dev/null && REND p/s.json notes.json p/s.wav"


def test_batched():
    assert search.batched([], 10) == []
    out = search.batched(["x"] * 25, 10)
    assert len(out) == 3
    assert out[0] == "\n".join(["x"] * 10) and out[2] == "x\nx\nx\nx\nx"

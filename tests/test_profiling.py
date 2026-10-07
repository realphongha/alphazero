import time

from alphazero.profiling import LatencyProfiler


def test_track_accumulates():
    p = LatencyProfiler("t")
    for _ in range(5):
        with p.track("a.b"):
            time.sleep(0.001)
    snap = p.snapshot()
    assert snap["a.b"]["count"] == 5
    assert snap["a.b"]["total_s"] > 0


def test_record_and_rank():
    p = LatencyProfiler("t")
    p.record("x", 0.5)
    p.record("y", 0.2)
    top = p.rank("total")[0]
    assert top[0] == "x"


def test_report_and_group():
    p = LatencyProfiler("t")
    p.record("mcts.select", 0.3)
    p.report_lines = p.report()
    assert "mcts.select" in p.report_lines
    grp = p.group_prefix(1)
    assert "mcts" in grp


def test_disabled_is_noop():
    p = LatencyProfiler("t")
    p.enabled = False
    with p.track("z"):
        time.sleep(0.001)
    p.record("z2", 1.0)
    assert "z" not in p.snapshot()
    assert "z2" not in p.snapshot()

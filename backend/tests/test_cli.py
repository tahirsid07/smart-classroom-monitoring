import io

from classroom_monitor.cli import monitor_quit_requested


def test_monitor_quit_requested_reads_q_from_stdin(monkeypatch):
    stdin = io.StringIO("q")
    monkeypatch.setattr("sys.stdin", stdin)

    assert monitor_quit_requested() is True

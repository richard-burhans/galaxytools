"""A failed lastz command must make the tool exit nonzero.

⛔ THE REGRESSION THIS PINS. `run_command` reports a failed lastz by RETURNING a message,
not by raising. The caller used to test only future state:

    if not f.done() or f.cancelled() or f.exception() is not None:
        found_falures = True

None of those is ever true for a worker that returned normally, so the message was printed
to stderr and `found_falures` stayed False. The tool exited 0, and because the wrapper sets
`detect_errors="exit_code"` Galaxy marked the job green and handed back a silently
incomplete alignment -- exactly the case where one chunk of a whole-genome run is missing.

These drive `collect_failures` with hand-built futures: no processes, no tarball, no lastz,
and no GPU, so the check runs anywhere.
"""
import concurrent.futures
import importlib.util
import pathlib

import pytest

MODULE = (
    pathlib.Path(__file__).resolve().parent.parent
    / "tools" / "batched_lastz" / "run_lastz_tarball.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("run_lastz_tarball", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _future(result=None, exception=None, cancelled=False):
    f: "concurrent.futures.Future" = concurrent.futures.Future()
    if cancelled:
        f.cancel()
        return f
    f.set_running_or_notify_cancel()
    if exception is not None:
        f.set_exception(exception)
    else:
        f.set_result(result if result is not None else [])
    return f


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_a_returned_failure_is_collected(mod):
    """The regression itself: a worker that RETURNS a failure must be counted."""
    futures = [_future(result=["command failed (rc=1): lastz ..."])]
    assert mod.collect_failures(futures) == ["command failed (rc=1): lastz ..."]


def test_clean_run_reports_nothing(mod):
    """And the converse, or the check would pass by always failing."""
    assert mod.collect_failures([_future(result=[]), _future(result=[])]) == []


def test_every_failure_is_reported_not_just_the_first(mod):
    """A worker drains its queue, so one worker can carry several failures.

    ⚠ The old code returned on the first failure, which also abandoned that worker's
    sentinel -- another worker then consumed it and exited early, leaving queued commands
    silently unrun.
    """
    futures = [_future(result=["first", "second"]), _future(result=["third"])]
    assert sorted(mod.collect_failures(futures)) == ["first", "second", "third"]


def test_a_raising_worker_is_still_a_failure(mod):
    """The state checks the old code relied on were not wrong, only insufficient."""
    failures = mod.collect_failures([_future(exception=RuntimeError("boom"))])
    assert len(failures) == 1
    assert "boom" in failures[0]


def test_a_cancelled_worker_is_a_failure(mod):
    assert mod.collect_failures([_future(cancelled=True)]) == ["worker was cancelled"]

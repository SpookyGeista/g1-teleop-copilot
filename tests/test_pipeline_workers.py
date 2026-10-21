"""Real CPU threads, coordinated by events instead of scheduling sleeps."""
import gc
import threading
import weakref

import pytest

from teleop_assist.pipeline_workers import (
    BoundedArchiveWorker, LatestBundleWorker, PipelineWorkerError,
)


TIMEOUT = 3.


def await_event(event):
    assert event.wait(TIMEOUT), 'worker/event did not complete within test bound'


def conservation(worker):
    stats = worker.statistics()
    assert stats['captured'] == sum(stats[k] for k in ('processed', 'failed', 'dropped', 'pending', 'running'))
    assert stats['pending'] <= stats['capacity']
    return stats


def test_latest_replaces_only_pending_and_close_drains_latest_without_parallel_callbacks():
    entered = threading.Event()
    release_first = threading.Event()
    entered_latest = threading.Event()
    release_latest = threading.Event()
    closed = threading.Event()
    seen = []

    def consume(bundle):
        seen.append(bundle)
        if bundle == 1:
            entered.set()
            await_event(release_first)
        else:
            entered_latest.set()
            await_event(release_latest)

    worker = LatestBundleWorker(consume)
    closer = None
    try:
        assert worker.submit(1)
        await_event(entered)
        assert worker.submit(2)
        assert worker.submit(3)
        assert seen == [1]
        stats = conservation(worker)
        assert (stats['captured'], stats['dropped'], stats['pending'], stats['processed']) == (3, 1, 1, 0)
        assert stats['running']
        closer = threading.Thread(target=lambda: (worker.close(), closed.set()))
        closer.start()
        release_first.set()
        await_event(entered_latest)
        assert seen == [1, 3]
        assert not closed.is_set()
        assert worker.statistics()['closing']
        with pytest.raises(RuntimeError, match='closing or closed'):
            worker.submit(4)
        release_latest.set()
        await_event(closed)
        closer.join(TIMEOUT)
        assert not closer.is_alive()
        assert conservation(worker)['processed'] == 2
        assert worker.statistics()['closed']
        worker.close()
        worker.check()
    finally:
        release_first.set()
        release_latest.set()
        worker.close()
        if closer is not None:
            closer.join(TIMEOUT)


def test_archive_fifo_nonblocking_full_rejection_and_shutdown_drain():
    entered = threading.Event()
    release = threading.Event()
    offered = threading.Event()
    seen, accepted = [], []

    def consume(bundle):
        seen.append(bundle)
        if bundle == 0:
            entered.set()
            await_event(release)

    worker = BoundedArchiveWorker(consume, capacity=2)
    producer = None
    try:
        assert worker.submit(0)
        await_event(entered)

        def produce():
            accepted.extend(worker.submit(i) for i in range(1, 6))
            offered.set()

        producer = threading.Thread(target=produce)
        producer.start()
        await_event(offered)  # Must complete while the consumer is blocked.
        producer.join(TIMEOUT)
        assert accepted == [True, True, False, False, False]
        assert seen == [0]
        stats = conservation(worker)
        assert (stats['captured'], stats['dropped'], stats['pending']) == (6, 3, 2)
        release.set()
        worker.close()
        assert seen == [0, 1, 2]
        assert conservation(worker)['processed'] == 3
        with pytest.raises(RuntimeError, match='closing or closed'):
            worker.submit(6)
    finally:
        release.set()
        worker.close()
        if producer is not None:
            producer.join(TIMEOUT)


@pytest.mark.parametrize('factory', [LatestBundleWorker, BoundedArchiveWorker])
def test_error_is_sticky_visible_and_pending_work_still_drains(factory):
    entered = threading.Event()
    fail_now = threading.Event()
    second_entered = threading.Event()
    finish_second = threading.Event()
    seen = []

    def consume(bundle):
        seen.append(bundle)
        if bundle == 'bad':
            entered.set()
            await_event(fail_now)
            raise ValueError('deliberate archive/processing failure')
        second_entered.set()
        await_event(finish_second)

    worker = factory(consume)
    try:
        worker.submit('bad')
        await_event(entered)
        worker.submit('accepted-before-failure')
        fail_now.set()
        await_event(second_entered)
        with pytest.raises(PipelineWorkerError, match='builtins.ValueError') as info:
            worker.check()
        assert 'consume' in info.value.failure.traceback
        assert info.value.failure.message == 'deliberate archive/processing failure'
        with pytest.raises(PipelineWorkerError):
            worker.submit('must-not-be-accepted')
        assert conservation(worker)['failed'] == 1
        finish_second.set()
        with pytest.raises(PipelineWorkerError):
            worker.close()
        assert seen == ['bad', 'accepted-before-failure']
        stats = conservation(worker)
        assert (stats['processed'], stats['failed'], stats['dropped']) == (1, 1, 0)
        assert stats['closed'] and not stats['running'] and stats['pending'] == 0
        with pytest.raises(PipelineWorkerError):
            worker.check()
        with pytest.raises(PipelineWorkerError):
            worker.close()
    finally:
        fail_now.set()
        finish_second.set()
        try:
            worker.close()
        except PipelineWorkerError:
            pass


class Payload:
    def __init__(self, index):
        self.index = index
        self.storage = bytearray(128 * 1024)


@pytest.mark.parametrize('factory,capacity', [(LatestBundleWorker, 1), (BoundedArchiveWorker, 4)])
def test_payload_reference_bound_and_no_retention_after_shutdown(factory, capacity):
    entered = threading.Event()
    release = threading.Event()
    references = weakref.WeakValueDictionary()

    def consume(bundle):
        if bundle.index == 0:
            entered.set()
            await_event(release)

    worker = factory(consume)
    try:
        first = Payload(0)
        references[0] = first
        worker.submit(first)
        del first
        await_event(entered)
        for index in range(1, 101):
            bundle = Payload(index)
            references[index] = bundle
            worker.submit(bundle)
        del bundle
        gc.collect()
        assert len(references) == capacity + 1
        assert set(references) == ({0, 100} if capacity == 1 else {0, 1, 2, 3, 4})
        stats = conservation(worker)
        assert stats['maximum_pending'] == capacity
        assert stats['dropped'] == 100-capacity
        release.set()
        worker.close()
        gc.collect()
        assert len(references) == 0
        assert conservation(worker)['processed'] == capacity+1
    finally:
        release.set()
        worker.close()


def test_failure_traceback_does_not_retain_bundle_and_baseexception_is_visible():
    def consume(bundle):
        assert bundle.storage
        raise SystemExit('callback attempted to leave')

    worker = LatestBundleWorker(consume)
    bundle = Payload(0)
    reference = weakref.ref(bundle)
    worker.submit(bundle)
    del bundle
    with pytest.raises(PipelineWorkerError, match='builtins.SystemExit'):
        worker.close()
    gc.collect()
    assert reference() is None
    assert conservation(worker)['failed'] == 1


def test_broken_exception_message_does_not_silently_terminate_worker():
    class BrokenMessageError(Exception):
        def __str__(self):
            raise RuntimeError('failure while describing failure')

    def consume(bundle):
        raise BrokenMessageError()

    worker = LatestBundleWorker(consume)
    worker.submit(None)
    with pytest.raises(PipelineWorkerError, match='BrokenMessageError') as info:
        worker.close()
    assert 'could not be formatted' in info.value.failure.message
    assert conservation(worker)['failed'] == 1


@pytest.mark.parametrize('factory', [LatestBundleWorker, BoundedArchiveWorker])
def test_empty_shutdown_idempotent_and_none_is_a_valid_bundle(factory):
    empty = factory(lambda bundle: None)
    empty.close()
    empty.close()
    assert conservation(empty)['captured'] == 0
    seen = []
    worker = factory(seen.append)
    worker.submit(None)
    worker.close()
    assert seen == [None]
    assert conservation(worker)['processed'] == 1


@pytest.mark.parametrize('capacity,error', [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)])
def test_archive_capacity_validation(capacity, error):
    with pytest.raises(error, match='positive integer'):
        BoundedArchiveWorker(lambda bundle: None, capacity=capacity)


def test_callback_must_be_callable():
    with pytest.raises(TypeError, match='callable'):
        LatestBundleWorker(None)

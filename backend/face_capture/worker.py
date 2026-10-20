import threading
import time
import logging
from sqlalchemy.exc import SQLAlchemyError
from .push import deliver_once
from .signals import capture_ready, delivery_ready
from .streams import shutdown
from .sampling import claim_frame, process_frame, fail_frame, release_frame, cleanup_frame, reconcile_frames

log = logging.getLogger(__name__)


def capture_once(factory, stop_event=None):
    """Only a queued alarm-frame signal can run camera face detection."""
    from .detector import detect
    claimed = claim_frame(factory)
    if claimed is None:
        return False
    sample, token = claimed
    try:
        if process_frame(factory, claimed, detect, stop_event):
            delivery_ready.set()
    except InterruptedError:
        release_frame(factory, sample['id'], token)
    except SQLAlchemyError:
        # A lost/uncertain commit must keep the immutable PNG for lease recovery.
        log.warning('Face sample database unavailable; lease recovery retained')
    except Exception as exc:
        fail_frame(factory, sample['id'], token, type(exc).__name__)
    cleanup_frame(factory, sample['id'])
    return True


class Workers:
    def __init__(self, factory):
        self.factory = factory
        self.stop_event = threading.Event()
        self.threads = []

    def loop(self, operation):
        signal = capture_ready if operation is capture_once else delivery_ready
        last_cleanup = 0
        while not self.stop_event.is_set():
            signal.clear()
            try:
                if operation is capture_once and time.monotonic() - last_cleanup > 10:
                    reconcile_frames(self.factory)
                    last_cleanup = time.monotonic()
                worked = operation(self.factory, self.stop_event) if operation is capture_once else operation(self.factory)
            except Exception as exc:
                log.warning('Face worker error: %s', type(exc).__name__)
                worked = False
            signal.wait(.01 if worked else 1)

    def start(self):
        for operation in (capture_once, deliver_once):
            thread = threading.Thread(target=self.loop, args=(operation,), daemon=True, name='face-' + operation.__name__)
            thread.start()
            self.threads.append(thread)

    def stop(self):
        self.stop_event.set()
        capture_ready.set()
        delivery_ready.set()
        for thread in self.threads:
            thread.join(timeout=20)
        shutdown()

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier
from app.db.base import utcnow
from app.db.models import ResourceLease
from app.domains import lease_service as leases


def test_racing_first_and_expired_lease(sessions):
    for index in range(20):
        resource = f"test:{index}"
        if index % 2:
            with sessions() as db:
                db.add(ResourceLease(resource=resource, holder="expired", generation=1,
                    expires_at=utcnow() - timedelta(seconds=1)))
                db.commit()
        barrier = Barrier(2)
        def acquire(holder):
            with sessions() as db:
                barrier.wait(timeout=10)
                return leases.acquire_lease(db, resource, holder=holder)
        with ThreadPoolExecutor(2) as pool:
            tokens = list(pool.map(acquire, ["a", "b"]))
        assert sum(token is not None for token in tokens) == 1


def test_fencing_survives_release_and_reacquire(sessions):
    with sessions() as db:
        first = leases.acquire_lease(db, "gpu:0", holder="a")
        assert leases.release_lease(db, "gpu:0", holder="a", generation=first)
        second = leases.acquire_lease(db, "gpu:0", holder="a")
        assert second > first
        assert not leases.renew_lease(db, "gpu:0", holder="a", generation=first)
        assert not leases.release_lease(db, "gpu:0", holder="a", generation=first)
        assert leases.renew_lease(db, "gpu:0", holder="a", generation=second)


def test_expired_gpu_is_quarantined(sessions):
    with sessions() as db:
        first = leases.acquire_lease(db, "gpu:0", holder="old")
        row = db.get(ResourceLease, "gpu:0")
        row.expires_at = utcnow() - timedelta(seconds=1)
        db.commit()
        assert leases.acquire_lease(db, "gpu:0", holder="new") is None
        assert not leases.renew_lease(db, "gpu:0", holder="old", generation=first)
        assert leases.release_lease(db, "gpu:0", holder="old", generation=first)
        assert leases.acquire_lease(db, "gpu:0", holder="new") > first

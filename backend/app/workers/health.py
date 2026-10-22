"""Container-local role heartbeat healthcheck; does not initialize CUDA."""
import socket
import sys
from app.db.base import utcnow
from app.db.models import ResourceLease
from app.db.session import get_session_factory


def main():
    role = sys.argv[1]
    if role not in {"trainer", "dataset", "inference"}:
        return 2
    prefix = f"worker:{role}:{role}-{socket.gethostname()[:16]}-"
    with get_session_factory()() as db:
        alive = db.query(ResourceLease).filter(ResourceLease.resource.startswith(prefix),
            ResourceLease.holder != "", ResourceLease.expires_at > utcnow()).count()
    return 0 if alive else 1


if __name__ == "__main__":
    raise SystemExit(main())

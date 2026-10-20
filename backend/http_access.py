"""Pure HTTP access rules used by the API middleware and security tests."""

from typing import FrozenSet, Tuple


PUBLIC_EXACT_ROUTES: FrozenSet[Tuple[str, str]] = frozenset(
    {
        ("GET", "/"),
        ("GET", "/health"),
        ("GET", "/api/health"),
        ("GET", "/health/live"),
        ("GET", "/api/health/live"),
        ("GET", "/health/ready"),
        ("GET", "/api/health/ready"),
        # MediaMTX posts here without an application login token; the route
        # validates its own scoped media token or loopback publisher identity.
        ("POST", "/monitoring/webrtc-auth"),
        ("POST", "/api/monitoring/webrtc-auth"),
    }
)

PUBLIC_PATH_PREFIXES = (
    "/auth",
    "/api/auth",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
    "/static",
    "/api/static",
    "/image",
    "/api/image",
    "/behavior/event-media",
    "/api/behavior/event-media",
)

PUBLIC_METHOD_PATH_PREFIXES = (
    ("GET", "/api/behavior/event-recording"),
    ("GET", "/behavior/event-recording"),
    ("GET", "/api/integration/alarm-media"),
    ("GET", "/integration/alarm-media"),
)


def is_public_http_route(method: str, path: str) -> bool:
    """Return True only for explicitly public method/path combinations."""
    normalized_method = str(method or "GET").upper()
    normalized_path = str(path or "/")
    if (normalized_method, normalized_path) in PUBLIC_EXACT_ROUTES:
        return True
    if any(
        normalized_method == allowed_method
        and (normalized_path == prefix or normalized_path.startswith(prefix + "/"))
        for allowed_method, prefix in PUBLIC_METHOD_PATH_PREFIXES
    ):
        return True
    return any(
        normalized_path == prefix or normalized_path.startswith(prefix + "/")
        for prefix in PUBLIC_PATH_PREFIXES
    )

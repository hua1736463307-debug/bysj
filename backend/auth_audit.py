"""Authentication session/audit consistency checks."""
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from database import OnlineSession, OperationLog, SessionLocal


REVOCATION_REASONS = ("manual_logout", "forced_logout", "session_revoked")


def audit_session_consistency(
    db: Session,
    *,
    since_hours: int = 24,
    limit: int = 100,
) -> Dict[str, Any]:
    """Find session transitions that do not have a matching authentication audit row."""
    now = datetime.now()
    cutoff = now - timedelta(hours=max(1, int(since_hours or 24)))
    safe_limit = max(1, min(int(limit or 100), 1000))
    missing_audit = []
    state_gaps = []
    checked = 0
    missing_audit_total = 0

    sessions = (
        db.query(OnlineSession)
        .filter(
            (OnlineSession.revoked_at >= cutoff)
            | (OnlineSession.expires_at >= cutoff)
        )
        .order_by(OnlineSession.id.asc())
        .all()
    )
    for session in sessions:
        transition_time: Optional[datetime] = None
        reasons = ()
        if session.status == "revoked":
            transition_time = session.revoked_at
            reasons = REVOCATION_REASONS
            if not session.revoked_at:
                state_gaps.append({
                    "sessionId": session.session_id,
                    "username": session.username,
                    "status": session.status,
                    "gap": "revoked_without_timestamp",
                })
        elif session.status == "expired":
            transition_time = session.expires_at
            reasons = ("token_expired",)
        elif session.status == "active" and session.expires_at <= now:
            state_gaps.append({
                "sessionId": session.session_id,
                "username": session.username,
                "status": session.status,
                "gap": "active_after_expiry",
            })
            continue
        else:
            continue

        if not transition_time or transition_time < cutoff:
            continue
        checked += 1
        has_audit = (
            db.query(OperationLog.id)
            .filter(
                OperationLog.title == "认证",
                OperationLog.session_id == session.session_id,
                OperationLog.reason.in_(reasons),
            )
            .first()
        )
        if not has_audit:
            missing_audit_total += 1
            if len(missing_audit) < safe_limit:
                missing_audit.append({
                    "sessionId": session.session_id,
                    "username": session.username,
                    "status": session.status,
                    "transitionTime": transition_time.isoformat(),
                    "expectedReasons": list(reasons),
                })

    return {
        "checked": checked,
        "missingAudit": missing_audit,
        "stateGaps": state_gaps[:safe_limit],
        "missingAuditTotal": missing_audit_total,
        "stateGapTotal": len(state_gaps),
        "sinceHours": max(1, int(since_hours or 24)),
    }


def run_session_consistency_check(*, since_hours: int = 24, limit: int = 100) -> Dict[str, Any]:
    """Run a check with an isolated database session for background tasks."""
    db = SessionLocal()
    try:
        return audit_session_consistency(db, since_hours=since_hours, limit=limit)
    finally:
        db.close()

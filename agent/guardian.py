"""guardian.request_review — record a REVIEW action. No pixels stored."""

from __future__ import annotations

from vision_core.contracts import ActionType, ProtectionAction, TrackedRegion, new_trace_id


def request_review(
    *,
    reason_code: str,
    policy_version: str,
    region: TrackedRegion | None = None,
    trace_id: str | None = None,
) -> ProtectionAction:
    """Stamp a REVIEW row when policy wants a grown-up look."""
    return ProtectionAction(
        action=ActionType.REVIEW,
        region=region,
        reason_code=reason_code or "GUARDIAN_REVIEW",
        policy_version=policy_version,
        trace_id=trace_id or new_trace_id(),
    )

"""Allowlist for the evaluation API.

Only ``fixtures/safe/*`` ids are legal. Remote URLs, uploads, and path
traversal are rejected in this module before any video is opened.
"""

from __future__ import annotations

import json
from pathlib import Path

VISION_ROOT = Path(__file__).resolve().parents[1]
SAFE_FIXTURES_DIR = VISION_ROOT / "fixtures" / "safe"
POLICIES_DIR = VISION_ROOT / "policies"

# Hard belt: even if someone drops a file elsewhere, only this folder counts.
ALLOWED_FIXTURE_ROOT = SAFE_FIXTURES_DIR.resolve()
ALLOWED_POLICY_ROOT = POLICIES_DIR.resolve()

_REMOTE_PREFIXES = (
    "http://",
    "https://",
    "s3://",
    "s3a://",
    "file://",
    "ftp://",
    "ftps://",
    "data:",
)

FORBIDDEN_BODY_KEYS = frozenset(
    {
        "url",
        "uri",
        "href",
        "image",
        "image_url",
        "image_uri",
        "video_url",
        "video_uri",
        "upload",
        "file",
        "files",
        "screenshot",
        "screenshots",
        "s3_uri",
        "s3_url",
        "source_url",
        "media_url",
    }
)


class CatalogError(ValueError):
    """Allowlist / request-shape rejection."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _looks_remote(value: str) -> bool:
    lowered = value.strip().lower()
    return any(lowered.startswith(prefix) for prefix in _REMOTE_PREFIXES)


def reject_remote_or_path(value: str, *, field: str) -> None:
    text = (value or "").strip()
    if not text:
        raise CatalogError(f"{field} is required")
    if _looks_remote(text):
        raise CatalogError(f"{field} must be an allowlisted id, not a URL")
    if any(token in text for token in ("..", "/", "\\", ":", "@")):
        raise CatalogError(f"{field} must be an allowlisted id, not a path")


def reject_upload_or_url_body(payload: dict) -> None:
    """Refuse any evaluation/perceive body that tries to sneak in a URL or file."""
    if not isinstance(payload, dict):
        raise CatalogError("body must be a JSON object")
    extra = FORBIDDEN_BODY_KEYS.intersection(payload)
    if extra:
        keys = ", ".join(sorted(extra))
        raise CatalogError(
            f"URL/upload fields are not allowed ({keys}). Use an allowlisted fixture_id."
        )
    for key, value in payload.items():
        if isinstance(value, str) and _looks_remote(value):
            raise CatalogError(f"{key} must not be a URL")


def _gt_id(path: Path) -> str:
    name = path.name
    suffix = ".gt.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def allowlisted_fixture_ids() -> frozenset[str]:
    """Ids of files that live in ``fixtures/safe/`` only."""
    ids: set[str] = set()
    if not SAFE_FIXTURES_DIR.is_dir():
        return frozenset()
    for mp4 in SAFE_FIXTURES_DIR.glob("*.mp4"):
        ids.add(mp4.stem)
    for gt in SAFE_FIXTURES_DIR.glob("*.gt.json"):
        ids.add(_gt_id(gt))
    return frozenset(ids)


def allowlisted_policy_ids() -> frozenset[str]:
    if not POLICIES_DIR.is_dir():
        return frozenset()
    # live_*.json are desktop protection modes, not fixture-eval presets.
    return frozenset(
        path.stem for path in POLICIES_DIR.glob("*.json") if not path.stem.startswith("live_")
    )


def list_fixtures() -> list[dict]:
    items: list[dict] = []
    for fixture_id in sorted(allowlisted_fixture_ids()):
        video = SAFE_FIXTURES_DIR / f"{fixture_id}.mp4"
        gt = SAFE_FIXTURES_DIR / f"{fixture_id}.gt.json"
        items.append(
            {
                "fixture_id": fixture_id,
                "has_video": video.is_file(),
                "has_gt": gt.is_file(),
                "allowlisted": True,
                "root": "fixtures/safe",
            }
        )
    return items


def list_policies() -> list[dict]:
    items: list[dict] = []
    for policy_id in sorted(allowlisted_policy_ids()):
        path = POLICIES_DIR / f"{policy_id}.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        items.append(
            {
                "policy_id": str(data.get("name") or policy_id),
                "version": data.get("version"),
                "description": str(data.get("description") or ""),
            }
        )
    return items


def resolve_fixture(fixture_id: str) -> tuple[Path, Path]:
    reject_remote_or_path(fixture_id, field="fixture_id")
    if fixture_id not in allowlisted_fixture_ids():
        raise CatalogError("unknown fixture_id")
    video = (SAFE_FIXTURES_DIR / f"{fixture_id}.mp4").resolve()
    gt = (SAFE_FIXTURES_DIR / f"{fixture_id}.gt.json").resolve()
    root = str(ALLOWED_FIXTURE_ROOT)
    if not str(video).startswith(root) or not str(gt).startswith(root):
        raise CatalogError("unknown fixture_id")
    if not video.is_file():
        raise CatalogError("allowlisted fixture video is missing", status_code=404)
    if not gt.is_file():
        raise CatalogError("allowlisted fixture ground truth is missing", status_code=404)
    return video, gt


def resolve_allowlisted_relpath(relpath: str) -> tuple[str, Path, Path]:
    """Accept ``fixtures/safe/<id>.mp4`` only. Never a URL or absolute path."""
    text = (relpath or "").strip()
    if not text:
        raise CatalogError("path is required")
    if _looks_remote(text):
        raise CatalogError("path must be an allowlisted fixture path, not a URL")
    if text.startswith("/") or text.startswith("\\") or ".." in Path(text).parts:
        raise CatalogError("path must be a relative allowlisted fixture path")
    candidate = (VISION_ROOT / text).resolve()
    try:
        candidate.relative_to(ALLOWED_FIXTURE_ROOT)
    except ValueError as exc:
        raise CatalogError("path is outside fixtures/safe") from exc
    if candidate.suffix.lower() != ".mp4":
        raise CatalogError("path must be an allowlisted fixture mp4")
    fixture_id = candidate.stem
    video, gt = resolve_fixture(fixture_id)
    if candidate != video:
        raise CatalogError("path does not match an allowlisted fixture")
    return fixture_id, video, gt


def resolve_policy(policy_id: str) -> Path:
    reject_remote_or_path(policy_id, field="policy_id")
    if policy_id not in allowlisted_policy_ids():
        raise CatalogError("unknown policy_id")
    path = (POLICIES_DIR / f"{policy_id}.json").resolve()
    if not str(path).startswith(str(ALLOWED_POLICY_ROOT)) or not path.is_file():
        raise CatalogError("unknown policy_id")
    return path

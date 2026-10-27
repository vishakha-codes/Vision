#!/usr/bin/env python3
"""Block AWS account ids, ARNs, IPs, resource ids and keys from entering git.

    python3 scripts/leakcheck.py            # staged changes (the pre-commit hook)
    python3 scripts/leakcheck.py --all      # every tracked file in the work tree
    python3 scripts/leakcheck.py --history  # every blob in every commit

Generic patterns catch any account's ARNs, resource ids and keys. The literal
list (this account's ids, IPs, user names) is built at run time from local,
untracked files only: ``aws.local.json``, the two ``outputs.json`` files,
``~/.screen-guardian/cloud.json`` and ``.leakcheck.local`` (one literal per
line). Matches are reported by rule name, never printed.

Install as a hook: ``ln -sf ../../scripts/leakcheck.py .git/hooks/pre-commit``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

VISION = Path(__file__).resolve().parents[1]
LOCAL_JSON = [
    VISION / "aws.local.json",
    VISION / "infra" / "outputs.json",
    VISION / "kids_chat" / "infra" / "outputs.json",
    Path.home() / ".screen-guardian" / "cloud.json",
]
LOCAL_LIST = VISION / ".leakcheck.local"
SELF = "scripts/leakcheck.py"

GENERIC = {
    "arn-with-account": re.compile(r"arn:aws[a-z-]*:[a-z0-9-]+:[a-z0-9-]*:\d{12}:"),
    "access-key-id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "private-key": re.compile("-----BEGIN [A-Z ]*" + "PRIVATE KEY-----"),
    "secret-key-assignment": re.compile(r"aws_secret_access_key\s*[=:]\s*\S{20,}", re.I),
    "ec2-resource-id": re.compile(r"\b(?:i|sg|subnet|vpc|vol|eni|igw|rtb|eipalloc)-[0-9a-f]{17}\b"),
    "agentcore-gateway-url": re.compile(r"[a-z0-9-]+-[a-z0-9]{10}\.gateway\.bedrock-agentcore\."),
}
IPV4 = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
# Loopback, wildcard, link-local metadata, RFC 5737 documentation ranges, test
# fixtures, and the opencv-python version string.
IP_OK = re.compile(r"^(?:0\.0\.0\.0|127\.0\.0\.1|169\.254\.169\.254|1\.2\.3\.4|192\.0\.2\.\d+|198\.51\.100\.\d+|203\.0\.113\.\d+|5\.0\.0\.93)$")
ID_TOKEN = re.compile(r"\b\d{12}\b|\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[a-z0-9_]+-[A-Za-z0-9]{10}\b")


def _strings(v):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from ((k, s) for _, s in _strings(x)) if not isinstance(x, str) else [(k, x)]
    elif isinstance(v, list):
        for x in v:
            yield from _strings(x)
    elif isinstance(v, str):
        yield "", v


def local_literals() -> set[str]:
    out: set[str] = set()
    for p in LOCAL_JSON:
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        for key, s in _strings(data):
            out.update(ID_TOKEN.findall(s))
            if key.endswith("id") and re.fullmatch(r"[A-Za-z0-9_-]{8,}", s):
                out.add(s)
    try:
        out.update(ln.strip() for ln in LOCAL_LIST.read_text().splitlines()
                   if ln.strip() and not ln.startswith("#"))
    except OSError:
        pass
    return {s for s in out if len(s) >= 6}


def scan(text: str, literals: set[str]) -> list[tuple[int, str]]:
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        for name, rx in GENERIC.items():
            if rx.search(line):
                hits.append((n, name))
        if any(not IP_OK.match(ip) for ip in IPV4.findall(line)):
            hits.append((n, "ipv4-address"))
        for s in literals:
            if s in line:
                hits.append((n, "local-literal"))
                break
    return hits


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=VISION, capture_output=True, text=True, check=True).stdout


def blob(spec: str) -> str | None:
    r = subprocess.run(["git", "show", spec], cwd=VISION, capture_output=True)
    return None if r.returncode or b"\0" in r.stdout[:8000] else r.stdout.decode("utf-8", "replace")


def main(argv: list[str]) -> int:
    literals = local_literals()
    mode = argv[1] if len(argv) > 1 else "--staged"
    found: list[str] = []
    if mode == "--history":
        for rev in git("rev-list", "--all").split():
            for path in git("ls-tree", "-r", "--name-only", rev).splitlines():
                if path == SELF:
                    continue
                text = blob(f"{rev}:{path}")
                for n, name in scan(text or "", literals):
                    found.append(f"{rev[:7]}:{path}:{n}: {name}")
    else:
        if mode == "--all":
            paths, spec = git("ls-files").splitlines(), None
        else:
            paths, spec = git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines(), ":"
        for path in paths:
            if path == SELF:
                continue
            if spec is None:
                try:
                    raw = (VISION / path).read_bytes()
                except OSError:
                    continue
                text = None if b"\0" in raw[:8000] else raw.decode("utf-8", "replace")
            else:
                text = blob(f":{path}")
            for n, name in scan(text or "", literals):
                found.append(f"{path}:{n}: {name}")
    for f in sorted(set(found)):
        print(f)
    if found:
        print(f"leakcheck: {len(set(found))} hit(s); mask them (see .gitignore'd aws.local.json for real ids)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

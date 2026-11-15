#!/usr/bin/env python3
"""Check public handover files using only the Python standard library.

Run from any directory. --online additionally checks anonymous GitHub visibility
and exact upstream commit IDs. No credentials, installs or workloads are used.
This checks references and packaging, not experimental or payload correctness.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html import unescape
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote
from urllib.request import Request, urlopen


PUBLIC_REPOSITORIES = {
    "arise-initiative/robosuite", "ascend/pytorch",
    "awomo-westlakedi/awomo-0.5", "xiaomirobotics/xiaomi-robotics-1",
    "daweidavidwang/ecr-xr1-ascend-guide", "google-deepmind/mujoco",
    "loop-wam/loopwam", "modelscope/modelscope_hub",
    "robocasa-benchmark/leaderboard", "robocasa/robocasa",
}
REPOSITORY_PATTERNS = (
    r"(?<![\w.-])(?:www\.)?github\.com[:/]([\w.-]+/[\w.-]+)",
    r"(?<![\w.-])(?:raw\.githubusercontent|codeload\.github)\.com/([\w.-]+/[\w.-]+)",
    r"(?<![\w.-])api\.github\.com/repos/([\w.-]+/[\w.-]+)",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def repository_references(text):
    # Inspect copied URLs inside JSON, HTML and encoded command arguments too.
    # Decode as text only; never evaluate code or require repository credentials.
    for _ in range(8):
        normalized = re.sub(r"\\u([0-9a-fA-F]{4})",
                            lambda match: chr(int(match.group(1), 16)), text)
        normalized = unescape(unquote(normalized)).replace(r"\/", "/")
        if normalized == text:
            break
        text = normalized
    return {name.removesuffix(".git").casefold()
            for pattern in REPOSITORY_PATTERNS
            for name in re.findall(pattern, text, re.I)}


def read_public(url):
    request = Request(url, headers={"User-Agent": "WAM-public-handover-check"})
    with urlopen(request, timeout=25) as response:
        require(response.status == 200, "Public URL did not return HTTP 200")
        return response.read(4_000_000).decode("utf-8"), response.url


def check_public_repository(repository):
    body, _ = read_public("https://github.com/" + repository)
    name = re.search(r'<meta name="octolytics-dimension-repository_nwo" content="([^"]+)"', body)
    visibility = re.search(r'<meta name="octolytics-dimension-repository_public" content="([^"]+)"', body)
    require(name is not None and name.group(1).casefold() == repository,
            "Anonymous repository identity check failed")
    require(visibility is not None and visibility.group(1) == "true",
            "Anonymous repository public-visibility check failed")
    return {"repository": repository, "public": True}


def check_public_commit(pin):
    url = pin["repository"] + "/commit/" + pin["commit"]
    body, actual_url = read_public(url)
    oid = re.search(r'"commitRoute":\s*\{\s*"commit":\s*\{\s*"oid"\s*:\s*"([0-9a-f]{40})"', body)
    require(actual_url.rstrip("/") == url and oid is not None
            and oid.group(1) == pin["commit"], "Anonymous commit identity check failed")
    return {"role": pin["role"], "commit": pin["commit"], "public": True}


def check(root, online=False):
    root = root.resolve()
    checksum_file = root / "PUBLIC_SHA256SUMS"
    files = {"PUBLIC_SHA256SUMS": checksum_file.read_bytes()}
    for line in files["PUBLIC_SHA256SUMS"].decode().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "Malformed checksum record")
        digest, name = match.groups()
        path = root / name
        require(name not in files and not Path(name).is_absolute()
                and ".." not in Path(name).parts and path.resolve().is_relative_to(root),
                "Duplicate or unsafe checksum path")
        payload = path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == digest,
                "Checksum mismatch: " + name)
        files[name] = payload
    if (root / ".git").exists():
        tracked = set(subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=root).decode().split("\0")[:-1])
        require(tracked == set(files), "Checksum list must cover every tracked file")

    references = set()
    links = python_blocks = json_files = 0
    for name, payload in files.items():
        body = payload.decode("utf-8")
        refs = repository_references(body)
        require(refs <= PUBLIC_REPOSITORIES,
                "Unreviewed GitHub repository reference in " + name)
        references.update(refs)
        if name.endswith(".json"):
            json.loads(body)
            json_files += 1
        if name == "WAM_HANDOFF.md" or name.startswith("wam/") and name.endswith(".md"):
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", body):
                if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) or target.startswith("#"):
                    continue
                path = (root / name).parent / unquote(target.split("#", 1)[0])
                require(path.is_file() and path.resolve().is_relative_to(root),
                        "Missing or external local file link in " + name)
                links += 1
            for code in re.findall(r"^```python\n(.*?)^```", body, re.M | re.S):
                compile(code, name + ":inline", "exec")
                python_blocks += 1

    pins = json.loads(files["wam/pins/source-pins.json"])["components"]
    for pin in pins:
        require(pin["repository"].startswith("https://github.com/")
                and re.fullmatch(r"[0-9a-f]{40}", pin["commit"]) is not None,
                "Expected an exact public upstream source pin")
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "result": "passed",
        "scope": "Documentation references and packaging only; no model or payload validation",
        "files_scanned": len(files), "checksums_verified": len(files) - 1,
        "github_reference_allowlist": "passed", "local_wam_file_links": links,
        "inline_python_blocks_compiled": python_blocks, "json_files_parsed": json_files,
        "online": online,
        "review_required": "Manually review bare repository names, branches, internal paths and missing implementation details; URL matching cannot prove their absence",
    }
    if online:
        with ThreadPoolExecutor(max_workers=4) as pool:
            result["public_repositories"] = list(pool.map(check_public_repository, sorted(references)))
            result["public_source_commits"] = list(pool.map(check_public_commit, pins))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    try:
        result = check(Path(__file__).resolve().parent.parent, args.online)
    except Exception as exc:
        print(json.dumps({"result": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

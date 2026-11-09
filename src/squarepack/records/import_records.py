"""Download, parse, verify, and store published square-in-square records (section 5)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import numpy as np

from squarepack.records.svg_to_squares import svg_to_packing
from squarepack.store import BestStore, Candidate, load_config

DEFAULT_INDEX_URL = "https://kingbird.myphotos.cc/packing/squares_in_squares.html"
DEFAULT_USER_AGENT = "squarepack-phase2/0.1 (+local research cache; 1 req/s)"
MIN_REQUEST_INTERVAL_S = 1.0


@dataclass
class IndexEntry:
    n: int
    svg_href: str
    L_published: float | None
    finder_hint: str | None


class RateLimitedFetcher:
    """HTTP GET with robots check and >= 1 request/second spacing."""

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        min_interval_s: float = MIN_REQUEST_INTERVAL_S,
    ) -> None:
        self.user_agent = user_agent
        self.min_interval_s = min_interval_s
        self._last_request_t = 0.0
        self._robots: dict[str, RobotFileParser] = {}

    def _robots_for(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            rp = RobotFileParser()
            robots_url = origin + "/robots.txt"
            try:
                self._wait()
                req = urllib.request.Request(
                    robots_url, headers={"User-Agent": self.user_agent}
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                rp.parse(text.splitlines())
            except Exception:
                # If robots.txt cannot be fetched, allow only paths clearly not disallowed
                # by the known Ellsworth robots (packing/ is allowed). Fail closed on
                # unknown hosts by disallowing everything except when parse failed empty.
                rp.parse([])
            self._robots[origin] = rp
        return self._robots[origin]

    def allowed(self, url: str) -> bool:
        rp = self._robots_for(url)
        # RobotFileParser.can_fetch with empty rules returns True.
        return bool(rp.can_fetch(self.user_agent, url))

    def _wait(self) -> None:
        now = time.monotonic()
        delay = self.min_interval_s - (now - self._last_request_t)
        if delay > 0:
            time.sleep(delay)

    def get_bytes(self, url: str) -> bytes:
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows fetching {url}")
        self._wait()
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        self._last_request_t = time.monotonic()
        return data

    def get_text(self, url: str, encoding: str = "utf-8") -> str:
        return self.get_bytes(url).decode(encoding, errors="replace")


def _parse_published_L(tex: str) -> float | None:
    """Extract a float side length from an HTML/TeX ``$s = ...$`` fragment."""
    tex = tex.strip()
    # Prefer explicit numeric form after \\Nn{...} or trailing decimal.
    m = re.search(r"\\Nn\{([0-9.+-eE]+)\}", tex)
    if m:
        return float(m.group(1))
    m = re.search(r"=\s*([0-9]+\.[0-9]+)\s*$", tex)
    if m:
        return float(m.group(1))
    m = re.search(r"^([0-9]+(?:\.[0-9]+)?)$", tex)
    if m:
        return float(m.group(1))
    # Bare integer-ish expressions like "2" already handled; give up otherwise.
    m = re.search(r"([0-9]+\.[0-9]{6,})", tex)
    if m:
        return float(m.group(1))
    try:
        return float(tex)
    except ValueError:
        return None


def parse_index_html(html: str, base_url: str) -> list[IndexEntry]:
    """Parse the Ellsworth index page for primary (n, svg, L) boxes."""
    entries: list[IndexEntry] = []
    # Each record box: <div class="box">...<font>+3">N<br><a href="square-N.svg">... $s = ...$
    pattern = re.compile(
        r'<div class="box"><font size="\+3">(\d+)<br><a href="([^"]+\.svg)"'
        r".*?"
        r"\$s\s*=\s*([^$]+)\$",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(html):
        n = int(m.group(1))
        href = urljoin(base_url, m.group(2))
        L = _parse_published_L(m.group(3))
        # Finder hint: often the first "Found by X" / "Proved" text in comments of SVG later.
        entries.append(IndexEntry(n=n, svg_href=href, L_published=L, finder_hint=None))

    # Deduplicate by n keeping first (page order = current record display).
    by_n: dict[int, IndexEntry] = {}
    for e in entries:
        by_n.setdefault(e.n, e)
    return [by_n[k] for k in sorted(by_n)]


def _extract_comment_attribution(svg_text: str) -> dict[str, str | None]:
    """Pull a rough finder/date from the leading SVG comment block."""
    m = re.search(r"<!--(.*?)-->", svg_text, re.DOTALL)
    if not m:
        return {"finder": None, "date": None}
    comment = m.group(1)
    finder = None
    date = None
    m_found = re.search(
        r"(?:Found by|First found by|Packing with this side length first found by)\s+([^\n(]+?)(?:\s+\(|\s+in\s+|\s*$)",
        comment,
        re.IGNORECASE,
    )
    if m_found:
        finder = m_found.group(1).strip().rstrip(".")
    m_date = re.search(r"\bin\s+(early\s+)?(19\d{2}|20\d{2})\b", comment)
    if m_date:
        date = m_date.group(2)
    if finder is None and re.search(r"\bTrivial\b", comment):
        finder = "trivial"
    return {"finder": finder, "date": date}


def snapshot_dir(root: Path, snapshot_time: str) -> Path:
    d = root / snapshot_time
    d.mkdir(parents=True, exist_ok=True)
    (d / "svg").mkdir(exist_ok=True)
    return d


def import_records(
    *,
    repo_root: Path,
    index_url: str = DEFAULT_INDEX_URL,
    n_min: int = 1,
    n_max: int = 1000,
    promote: bool = True,
    fetcher: RateLimitedFetcher | None = None,
    store: BestStore | None = None,
    snapshot_time: str | None = None,
    parse_precision: int = 50,
    delta_max: float = 1e-4,
) -> dict[str, Any]:
    """Full section-5 import pipeline. Returns a summary dict."""
    config = load_config(repo_root / "config.yaml")
    data_root = repo_root / config.get("paths", {}).get(
        "data_records", "data/records_snapshot"
    )
    data_root.mkdir(parents=True, exist_ok=True)

    if snapshot_time is None:
        snapshot_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap = snapshot_dir(data_root, snapshot_time)
    fetcher = fetcher or RateLimitedFetcher()
    store = store or BestStore(config=config)
    # Imports may need a larger dilation than constructed grids.
    store.verify_kwargs = dict(store.verify_kwargs)
    store.verify_kwargs["delta_max"] = float(
        max(float(store.verify_kwargs.get("delta_max", 1e-6)), delta_max)
    )

    failures_path = repo_root / "data" / "import_failures.csv"
    snapshot_csv = repo_root / "data" / "records_snapshot.csv"

    report: dict[str, Any] = {
        "snapshot_time": snapshot_time,
        "index_url": index_url,
        "imported": 0,
        "parsed": 0,
        "verified": 0,
        "promoted": 0,
        "failures": 0,
        "within_1e9": 0,
        "entries": [],
    }

    # 1. Fetch and cache index page.
    index_path = snap / "squares_in_squares.html"
    try:
        html = fetcher.get_text(index_url)
        index_path.write_text(html, encoding="utf-8")
    except Exception as exc:
        report["fatal"] = f"failed to fetch index: {exc}"
        _write_failures(failures_path, [{"n": "", "svg": index_url, "reason": str(exc)}])
        _write_snapshot_csv(snapshot_csv, [])
        (snap / "import_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    entries = [
        e for e in parse_index_html(html, index_url) if n_min <= e.n <= n_max
    ]
    (snap / "index_entries.json").write_text(
        json.dumps([e.__dict__ for e in entries], indent=2) + "\n", encoding="utf-8"
    )

    failure_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []

    for entry in entries:
        row: dict[str, Any] = {
            "n": entry.n,
            "L_published": entry.L_published,
            "L_verified": None,
            "verified_ok": False,
            "svg": entry.svg_href,
            "delta": None,
        }
        svg_name = Path(urlparse(entry.svg_href).path).name
        svg_path = snap / "svg" / svg_name
        try:
            if not svg_path.is_file():
                data = fetcher.get_bytes(entry.svg_href)
                svg_path.write_bytes(data)
            svg_text = svg_path.read_text(encoding="utf-8", errors="replace")
            report["imported"] += 1
        except Exception as exc:
            report["failures"] += 1
            failure_rows.append(
                {"n": entry.n, "svg": entry.svg_href, "reason": f"download: {exc}"}
            )
            snapshot_rows.append(row)
            continue

        # Prefer published L from SVG entity if HTML parse missed it.
        try:
            squares, L_pub_svg, meta = svg_to_packing(
                svg_path, precision=parse_precision
            )
            report["parsed"] += 1
        except Exception as exc:
            report["failures"] += 1
            failure_rows.append(
                {"n": entry.n, "svg": svg_name, "reason": f"parse: {exc}"}
            )
            snapshot_rows.append(row)
            continue

        if squares.shape[0] != entry.n:
            report["failures"] += 1
            failure_rows.append(
                {
                    "n": entry.n,
                    "svg": svg_name,
                    "reason": f"parse count mismatch: got {squares.shape[0]} squares",
                }
            )
            snapshot_rows.append(row)
            continue

        L_published = entry.L_published if entry.L_published is not None else L_pub_svg
        row["L_published"] = L_published
        attr = _extract_comment_attribution(svg_text)

        if not promote:
            snapshot_rows.append(row)
            continue

        cand = Candidate(
            n=entry.n,
            squares=squares,
            origin="imported",
            method="ellsworth_svg",
            seed=None,
            source={
                "finder": attr["finder"],
                "date": attr["date"],
                "url": entry.svg_href,
                "snapshot_time": snapshot_time,
                "L_published": L_published,
                "svg_file": svg_name,
            },
        )
        ok, vreport, record = store.promote(cand)
        if not vreport.ok or vreport.L is None:
            # Verification failed entirely.
            report["failures"] += 1
            failure_rows.append(
                {
                    "n": entry.n,
                    "svg": svg_name,
                    "reason": f"verify: {vreport.reason}",
                }
            )
            snapshot_rows.append(row)
            continue

        # Verified even if not promoted (already had equal/better L).
        report["verified"] += 1
        row["verified_ok"] = True
        row["L_verified"] = float(vreport.L)
        row["delta"] = float(vreport.delta)
        if L_published is not None and abs(float(vreport.L) - float(L_published)) <= 1e-9:
            report["within_1e9"] += 1
        if ok:
            report["promoted"] += 1
        elif record is not None and "not improved" in vreport.reason:
            # Keep a verified imported seed without clobbering a better best.
            _write_candidate_seed(repo_root, record)
        snapshot_rows.append(row)
        report["entries"].append(row)

    _write_failures(failures_path, failure_rows)
    _write_snapshot_csv(snapshot_csv, snapshot_rows)
    (snap / "import_report.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "entries"}, indent=2) + "\n",
        encoding="utf-8",
    )
    # Also copy summary CSVs into the timestamped snapshot.
    if failures_path.is_file():
        (snap / "import_failures.csv").write_text(
            failures_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if snapshot_csv.is_file():
        (snap / "records_snapshot.csv").write_text(
            snapshot_csv.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return report


def _write_candidate_seed(repo_root: Path, record: dict[str, Any]) -> None:
    """Save a verified imported packing under results/candidates when not best."""
    out_dir = repo_root / "results" / "candidates" / "imported"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(record["n"])
    path = out_dir / f"n{n:04d}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _write_failures(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "svg", "reason"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_snapshot_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["n", "L_published", "L_verified", "verified_ok"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "n": r.get("n"),
                    "L_published": r.get("L_published"),
                    "L_verified": r.get("L_verified"),
                    "verified_ok": r.get("verified_ok"),
                }
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=1000)
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--delta-max", type=float, default=1e-4)
    args = parser.parse_args(argv)

    report = import_records(
        repo_root=args.repo_root.resolve(),
        index_url=args.index_url,
        n_min=args.n_min,
        n_max=args.n_max,
        promote=not args.no_promote,
        delta_max=args.delta_max,
    )
    print(
        f"imported={report['imported']} parsed={report['parsed']} "
        f"verified={report['verified']} promoted={report['promoted']} "
        f"failures={report['failures']} within_1e9={report['within_1e9']}"
    )
    if report.get("fatal"):
        print("FATAL:", report["fatal"], file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Check that every paper_id in a CSV produced by a deep-research run exists.

Deep-research tools fabricate references. This script resolves each `paper_id`
column entry against arXiv (ids like 2103.09815 or 2103.09815v2) or Crossref
(DOIs) and prints the resolved title next to the title the tool claimed, so a
mismatch is visible at a glance.

Usage:
    .venv/bin/python docs/acl/literature/verify_references.py screening.csv evidence.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
DOI = re.compile(r"^10\.\d{4,9}/\S+$")


STOPWORDS = frozenset({"a", "an", "the", "of", "for", "in", "on", "and", "via", "with", "to", "by"})
TITLE_JACCARD_THRESHOLD = 0.6  # two safe-RL titles sharing only the generic words score ~0.5


def content_words(title: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", title.lower()) if word not in STOPWORDS}


@dataclass(frozen=True)
class Resolution:
    paper_id: str
    claimed_title: str
    resolved_title: str | None

    @property
    def exists(self) -> bool:
        return self.resolved_title is not None

    @property
    def titles_agree(self) -> bool:
        """Jaccard similarity of the two titles' content words is above the threshold.

        An id that exists but names a different paper is the failure a human misses.
        Generic words (benchmarking, deep, reinforcement, learning) are shared by
        many titles; Jaccard also counts the words the resolved title has that the
        claim lacks, so two unrelated safe-RL papers do not pass on those alone.
        Requires the claimed title to be the full title, not an acronym.
        """
        if self.resolved_title is None:
            return False
        claimed = content_words(self.claimed_title)
        resolved = content_words(self.resolved_title)
        if not claimed:
            return True
        return len(claimed & resolved) / len(claimed | resolved) >= TITLE_JACCARD_THRESHOLD


THROTTLED = frozenset({429, 503})


def fetch_with_retry(url: str, attempts: int = 4) -> str:
    """arXiv and Semantic Scholar throttle bursts (429/503); back off and retry before giving up."""
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(url, timeout=60).read().decode()
        except urllib.error.HTTPError as error:
            if error.code not in THROTTLED or attempt == attempts - 1:
                raise
            time.sleep(5.0 * (attempt + 1))
    raise AssertionError("unreachable")


def resolve_arxiv(identifier: str) -> str | None:
    """Title of the arXiv paper, via Semantic Scholar (lenient rate limit), falling back to arXiv."""
    bare = ARXIV_ID.match(identifier).group(1)
    try:
        body = fetch_with_retry(f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{bare}?fields=title")
        return json.loads(body)["title"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        if error.code not in THROTTLED:
            raise
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": bare})
    body = fetch_with_retry(url)
    titles = re.findall(r"<title>(.*?)</title>", body, re.S)
    if len(titles) < 2:
        return None
    return re.sub(r"\s+", " ", titles[1]).strip()


def resolve_doi(identifier: str) -> str | None:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(identifier, safe="")
    try:
        body = fetch_with_retry(url)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    titles = json.loads(body)["message"].get("title", [])
    return titles[0] if titles else None


def resolve(paper_id: str) -> str | None:
    if ARXIV_ID.match(paper_id):
        return resolve_arxiv(paper_id)
    if DOI.match(paper_id):
        return resolve_doi(paper_id)
    raise ValueError(f"paper_id {paper_id!r} is neither an arXiv id nor a DOI")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "paper_id" not in rows[0]:
        raise ValueError(f"{path} has no paper_id column; columns are {list(rows[0])}")
    return rows


def main(paths: list[str]) -> int:
    seen: dict[str, Resolution] = {}
    for path in map(Path, paths):
        for row in read_rows(path):
            paper_id = row["paper_id"].strip()
            if paper_id in seen:
                continue
            try:
                resolved = resolve(paper_id)
            except ValueError as error:
                print(f"MALFORMED  {error}")
                seen[paper_id] = Resolution(paper_id, row.get("title", ""), None)
                continue
            seen[paper_id] = Resolution(paper_id, row.get("title", ""), resolved)
            time.sleep(3.0)  # arXiv asks for ≥ 3 s between requests

    missing = [r for r in seen.values() if not r.exists]
    mismatched = [r for r in seen.values() if r.exists and not r.titles_agree]
    for resolution in seen.values():
        if not resolution.exists:
            mark = "MISSING "
        elif not resolution.titles_agree:
            mark = "MISMATCH"
        else:
            mark = "ok      "
        print(f"{mark} {resolution.paper_id:<28} claimed: {resolution.claimed_title[:60]}")
        if resolution.exists:
            print(f"{'':37}resolved: {resolution.resolved_title[:60]}")
    print(
        f"\n{len(seen)} ids checked, {len(missing)} missing or malformed, "
        f"{len(mismatched)} resolve to a differently titled paper."
    )
    return 1 if missing or mismatched else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1:]))

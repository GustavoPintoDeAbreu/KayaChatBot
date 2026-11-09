#!/usr/bin/env python3
"""Dry run for the audit's dependency bumps: does each fixed version exist?

For every exact pin in EXACT_PINS this reads https://pypi.org/project/<name>/<version>/
and checks the HTTP status (200 = the audit version exists on PyPI, 404 = it does not)
plus one markup-light check (the version string in the body and the page being at
least MIN_REAL_PAGE_BYTES). This deliberately deviates from the audit plan's dry-run
spec ('assert nothing about page markup — only HTTP status'): the live run showed the
sandbox tunnel serving ~3 KB interstitial pages with status 200 for pandas,
transformers and ctranslate2, so status alone reported three pins as 'exists' that
were never verified. A 200 without the marker is reported as unverified instead.
For each cuNNN PyTorch index it reads the index page and prints how many wheel links
it lists plus the torch wheels it names, so a reviewer can see real index data rather
than an error page.

Writes nothing and sends no message. Exit 0 only when all five PyPI statuses are
200 and at least one probed index hosts a torch-2.10.0 wheel; otherwise exit
non-zero with the reason on stderr.
"""

import re
import sys
import urllib.request

EXACT_PINS = {
    "ctranslate2": "4.8.5",
    "fastapi": "0.162.2",
    "pandas": "3.1.1",
    "sentencepiece": "0.2.3",
    "transformers": "5.22.2",
}

TORCH_INDEXES = ("cu124", "cu126", "cu128")
WANTED_TORCH = "2.10.0"

# Real PyPI release pages are hundreds of KB; the observed tunnel interstitials
# were ~3 KB served with status 200, so a page below this floor cannot be one.
MIN_REAL_PAGE_BYTES = 10_000


def fetch(url: str):
    """GET a URL once and return (status, body bytes); network errors report as status None."""
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "kaya-audit-pin-check", "Accept": "text/html"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, b""
    except (urllib.error.URLError, OSError, TimeoutError) as failure:
        print(f"GET {url} failed: {failure}", file=sys.stderr)
        return None, b""


def main() -> int:
    failures = []

    for name in sorted(EXACT_PINS):
        version = EXACT_PINS[name]
        url = f"https://pypi.org/project/{name}/{version}/"
        status, body = fetch(url)
        print(f"{name}=={version}: HTTP {status}, {len(body)} bytes")
        if status != 200:
            failures.append(f"{name}=={version} is not on PyPI (HTTP {status})")
        elif version.encode() not in body or len(body) < MIN_REAL_PAGE_BYTES:
            failures.append(
                f"{name}=={version}: HTTP 200 but the body carries no version marker "
                f"or is smaller than {MIN_REAL_PAGE_BYTES} bytes ({len(body)}), "
                "so it is likely an interstitial page, not a release page — unverified"
            )

    torch_found_on = []
    unreachable_indexes = []
    for index in TORCH_INDEXES:
        url = f"https://download.pytorch.org/whl/{index}"
        status, body = fetch(url)
        if status is None:
            print(f"index {index}: unreachable (not evidence of absence)")
            unreachable_indexes.append(index)
            continue
        if status != 200:
            print(f"index {index}: absent (HTTP {status})")
            continue
        page = body.decode("utf-8", errors="ignore")
        hrefs = re.findall(r'href="([^"]+)"', page)
        wheel_links = [href for href in hrefs if href.endswith(".whl") or "whl/" in href]
        torch_wheels = [href for href in hrefs if re.search(r"torch-[0-9]", href)]
        print(f"index {index}: {len(wheel_links)} wheel links, torch wheels: {torch_wheels[:5]}")
        if any(f"torch-{WANTED_TORCH}" in href for href in torch_wheels):
            torch_found_on.append(index)

    if not torch_found_on:
        reason = f"no probed index lists a torch-{WANTED_TORCH} wheel"
        if unreachable_indexes:
            reason += (
                " — and these indexes were unreachable, so their absence is unverified: "
                + ", ".join(unreachable_indexes)
            )
        failures.append(reason)

    if failures:
        for reason in failures:
            print(reason, file=sys.stderr)
        return 1
    print(f"torch {WANTED_TORCH} found on: {', '.join(torch_found_on)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

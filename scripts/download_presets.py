"""
Bulk-download Vital synth presets from presetshare.com.

Usage:
    python scripts/download_presets.py \
        --cookies cookies.txt \
        --out-dir ~/Downloads/vital_presets \
        --start-page 1 \
        --delay 0.5 \
        --max-pages 0   # 0 = no limit

One-time setup:
    1. Log into presetshare.com in Chrome/Firefox
    2. Install "Get cookies.txt LOCALLY" browser extension
    3. Export cookies for presetshare.com → save as cookies.txt in the project root
"""

import argparse
import http.cookiejar
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://presetshare.com"
LISTING_URL = BASE_URL + "/presets?instrument=2&page={page}"
PRESET_URL = BASE_URL + "/{preset_id}"
DOWNLOAD_URL = BASE_URL + "/download/index?id={key}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": BASE_URL,
}


# ---------------------------------------------------------------------------
# Cookie loading
# ---------------------------------------------------------------------------

def load_session(cookies_path: str) -> requests.Session:
    jar = http.cookiejar.MozillaCookieJar(cookies_path)
    jar.load(ignore_discard=True, ignore_expires=True)
    session = requests.Session()
    session.cookies = jar
    session.headers.update(HEADERS)
    return session


# ---------------------------------------------------------------------------
# Listing page parsing
# ---------------------------------------------------------------------------

def get_preset_ids_from_page(session: requests.Session, page: int) -> list[str]:
    """Return list of preset IDs (e.g. 'p19785') from one listing page."""
    url = LISTING_URL.format(page=page)
    resp = _request_with_retry(session, "GET", url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    ids: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        # Match /pNNNNN or pNNNNN patterns
        m = re.fullmatch(r"/?p(\d+)", href.strip("/").split("?")[0])
        if m:
            pid = f"p{m.group(1)}"
            if pid not in seen:
                seen.add(pid)
                ids.append(pid)

    return ids


def has_next_page(session: requests.Session, page: int) -> bool:
    """Return True if the listing page has a 'next' pagination link."""
    url = LISTING_URL.format(page=page)
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Common pagination patterns: rel="next", text "Next", aria-label "Next page"
    for a in soup.find_all("a", href=True):
        rel = a.get("rel", [])
        if "next" in rel:
            return True
        text = a.get_text(strip=True).lower()
        aria = (a.get("aria-label") or "").lower()
        if text in ("next", "»", "›", "next →") or "next" in aria:
            return True

    return False


# ---------------------------------------------------------------------------
# Download key extraction
# ---------------------------------------------------------------------------

# Patterns to find the download key on a preset page (tried in order)
_KEY_PATTERNS = [
    # data-* attributes on any element
    re.compile(r'data-(?:download-id|key|id)=["\']([A-Za-z0-9_\-]{20,})["\']'),
    # href="/download/index?id=KEY"
    re.compile(r'/download/index\?id=([A-Za-z0-9_\-]{20,})'),
    # JavaScript variable: var downloadId = "KEY" or downloadId: "KEY"
    re.compile(r'(?:downloadId|download_id|downloadKey|fileKey)\s*[=:]\s*["\']([A-Za-z0-9_\-]{20,})["\']'),
    # Generic: id=KEY in any URL-like context that looks like a hash
    re.compile(r'[?&]id=([A-Za-z0-9_\-]{30,})'),
]


def extract_download_keys(html: str) -> list[str]:
    """Return download keys for all free (non-subscriber-only) download buttons."""
    soup = BeautifulSoup(html, "html.parser")
    keys: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        classes = " ".join(a.get("class") or [])
        if "download-button" not in classes:
            continue
        if "inactive" in classes:
            continue  # subscriber-only
        href: str = a["href"]
        m = re.search(r'[?&]id=([A-Za-z0-9_\-]{20,})', href)
        if m:
            key = m.group(1)
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def extract_download_key(html: str) -> str | None:
    """Return the first free download key (legacy single-key interface)."""
    keys = extract_download_keys(html)
    return keys[0] if keys else None


def get_download_key(session: requests.Session, preset_id: str) -> str | None:
    url = PRESET_URL.format(preset_id=preset_id)
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return extract_download_key(resp.text)


def get_download_keys(session: requests.Session, preset_id: str) -> list[str]:
    """Return all free download keys for a preset page."""
    url = PRESET_URL.format(preset_id=preset_id)
    resp = _request_with_retry(session, "GET", url, timeout=30)
    resp.raise_for_status()
    return extract_download_keys(resp.text)


# ---------------------------------------------------------------------------
# File download
# ---------------------------------------------------------------------------

def filename_from_response(resp: requests.Response, fallback: str) -> str:
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename\*?=["\']?(?:UTF-8\'\')?([^"\';\r\n]+)', cd, re.IGNORECASE)
    if m:
        name = m.group(1).strip().strip('"\'')
        if name:
            return name
    return fallback


def _request_with_retry(session: requests.Session, method: str, url: str, retries: int = 4, **kwargs):
    """Wrapper around session.request with exponential backoff on transient errors and 5xx responses."""
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            resp = session.request(method, url, **kwargs)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = delay * (2 ** attempt)
                print(f"  [retry {attempt+1}/{retries}] HTTP {resp.status_code}, waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == retries:
                raise
            wait = delay * (2 ** attempt)
            print(f"  [retry {attempt+1}/{retries}] {type(e).__name__}, waiting {wait:.0f}s")
            time.sleep(wait)


def download_preset(
    session: requests.Session,
    key: str,
    out_dir: Path,
    preset_id: str,
) -> Path | None:
    """Download one preset by its key. Returns saved path, or None on skip/error."""
    url = DOWNLOAD_URL.format(key=key)
    try:
        resp = _request_with_retry(session, "GET", url, stream=True, timeout=90)
    except Exception as e:
        print(f"  [err]  {preset_id}: {e}")
        return None

    with resp:
        if resp.status_code == 404:
            print(f"  [404] {preset_id} key={key}")
            return None
        resp.raise_for_status()

        fname = filename_from_response(resp, f"{preset_id}.vital")
        fname = re.sub(r'[\\/*?:"<>|]', "_", fname)
        dest = out_dir / fname

        if dest.exists():
            print(f"  [skip] {fname} already exists")
            return dest

        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            tmp.rename(dest)
            print(f"  [ok]   {fname}")
            return dest
        except Exception as e:
            tmp.unlink(missing_ok=True)
            print(f"  [err]  {preset_id}: {e}")
            return None


# ---------------------------------------------------------------------------
# State files — track completed preset IDs and last completed page
# ---------------------------------------------------------------------------

def load_done(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    return set(line for line in state_file.read_text().splitlines() if not line.startswith("#"))


def mark_done(state_file: Path, preset_id: str) -> None:
    with open(state_file, "a") as f:
        f.write(preset_id + "\n")


def load_last_page(state_file: Path) -> int:
    """Return the last fully-completed listing page (0 if none)."""
    page_file = state_file.parent / ".last_page.txt"
    if not page_file.exists():
        return 0
    try:
        return int(page_file.read_text().strip())
    except ValueError:
        return 0


def save_last_page(state_file: Path, page: int) -> None:
    page_file = state_file.parent / ".last_page.txt"
    page_file.write_text(str(page) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk-download Vital presets from presetshare.com")
    p.add_argument("--cookies", default="cookies.txt", help="Path to Netscape cookies.txt")
    p.add_argument("--out-dir", default="~/Downloads/vital_presets", help="Output directory")
    p.add_argument("--start-page", type=int, default=1, help="Listing page to start from")
    p.add_argument("--max-pages", type=int, default=0, help="Max pages to scrape (0 = unlimited)")
    p.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    p.add_argument(
        "--debug-page",
        metavar="PRESET_ID",
        help="Print raw HTML of one preset page (e.g. p19785) then exit — useful for debugging key extraction",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cookies_path = args.cookies
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / ".downloaded.txt"

    if not Path(cookies_path).exists():
        print(
            f"ERROR: cookies file not found at '{cookies_path}'.\n"
            "Export cookies from your browser using 'Get cookies.txt LOCALLY' extension."
        )
        sys.exit(1)

    session = load_session(cookies_path)
    done = load_done(state_file)
    last_page = load_last_page(state_file)
    print(f"Already downloaded: {len(done)} presets")
    print(f"Resuming from page: {last_page + 1}" if last_page else f"Starting from page: {args.start_page}")
    print(f"Output directory:   {out_dir}")

    # ------------------------------------------------------------------
    # Debug mode: inspect one preset page and exit
    # ------------------------------------------------------------------
    if args.debug_page:
        pid = args.debug_page
        url = PRESET_URL.format(preset_id=pid)
        print(f"\nFetching {url} ...")
        resp = session.get(url, timeout=30)
        print(f"Status: {resp.status_code}")
        # Print lines containing 'download' (case-insensitive) for quick inspection
        lines = resp.text.splitlines()
        matches = [l for l in lines if "download" in l.lower()]
        print(f"\nLines containing 'download' ({len(matches)} total):")
        for l in matches[:40]:
            print(l[:200])
        keys = extract_download_keys(resp.text)
        print(f"\nExtracted free keys ({len(keys)}): {keys}")
        return

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    # Resume from after the last fully-completed page, unless overridden
    page = max(args.start_page, last_page + 1)
    pages_scraped = 0
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    while True:
        if args.max_pages and pages_scraped >= args.max_pages:
            print(f"\nReached --max-pages {args.max_pages}, stopping.")
            break

        print(f"\n=== Page {page} ===")
        try:
            preset_ids = get_preset_ids_from_page(session, page)
        except requests.HTTPError as e:
            print(f"HTTP error on listing page {page}: {e}")
            break

        if not preset_ids:
            print("No preset IDs found — reached end of listing.")
            break

        pages_scraped += 1
        already_done = [pid for pid in preset_ids if pid in done]
        new_ids = [pid for pid in preset_ids if pid not in done]
        print(f"Found {len(preset_ids)} presets — {len(already_done)} already done, {len(new_ids)} new")

        for pid in new_ids:
            time.sleep(args.delay)

            # Get download keys (may be >1 if page has multiple free presets)
            try:
                keys = get_download_keys(session, pid)
            except requests.HTTPError as e:
                print(f"  [err]  {pid} preset page: {e}")
                total_failed += 1
                continue

            if not keys:
                print(f"  [warn] {pid}: no free download keys found — run with --debug-page {pid}")
                total_failed += 1
                continue

            any_ok = False
            for key in keys:
                time.sleep(args.delay)
                result = download_preset(session, key, out_dir, pid)
                if result is not None:
                    any_ok = True
                    total_downloaded += 1
                else:
                    total_failed += 1

            if any_ok:
                mark_done(state_file, pid)
                done.add(pid)

        total_skipped += len(already_done)

        # Save page progress so restarts skip fully-processed pages
        save_last_page(state_file, page)

        # Move to next page (rely on empty result to detect end)
        time.sleep(args.delay)
        page += 1

    print(f"\nDone. Downloaded: {total_downloaded}, Skipped: {total_skipped}, Failed: {total_failed}")


if __name__ == "__main__":
    main()

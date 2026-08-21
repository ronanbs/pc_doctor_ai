"""
scrapers.py
Fetches and extracts clean text content from web-based documentation pages
(Microsoft Learn, Arch Wiki, NVIDIA support pages, etc.) for ingestion into
the RAG knowledge base.

Uses `trafilatura` for main-content extraction (strips nav/ads/boilerplate)
with a BeautifulSoup fallback for pages trafilatura can't parse cleanly.

Reddit URLs are handled separately: the modern site is JavaScript-only and
often blocks bots. We try Reddit JSON, then Arctic Shift, then PullPush (with
retries). Successful extractions are cached locally to avoid repeat API calls.
"""

import json
import os
import re
import time
from pathlib import Path

import requests
import trafilatura
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Reddit asks for a descriptive, unique User-Agent for API-style access.
REDDIT_HEADERS = {
    "User-Agent": "PCDoctorAI/1.0 (L2 Python RAG educational project)",
}

REDDIT_POST_RE = re.compile(
    r"(?:https?://)?(?:www\.|old\.|np\.|new\.)?reddit\.com/r/[^/]+/comments/([a-z0-9]+)",
    re.IGNORECASE,
)

PULLPUSH_SUBMISSION_URL = "https://api.pullpush.io/reddit/search/submission/"
PULLPUSH_COMMENT_URL = "https://api.pullpush.io/reddit/search/comment/"
ARCTIC_SHIFT_POSTS_URL = "https://arctic-shift.photon-reddit.com/api/posts/ids"
ARCTIC_SHIFT_COMMENTS_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"
REDDIT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "reddit"

DISCOURSE_TOPIC_RE = re.compile(
    r"(https?://[^/]+)/t/(?:[^/]+/)?(\d+)",
    re.IGNORECASE,
)

PLACEHOLDER_MARKERS = (
    "enable javascript",
    "please stand by",
    "loading nvidia geforce forums",
    "requires javascript",
    "javascript in order to view",
    "javascript in order to access",
)


class ScrapeError(Exception):
    """Raised when a URL cannot be scraped, with a user-facing explanation."""


def fetch_page(url: str, timeout: int = 15) -> str | None:
    """Fetch raw HTML for a URL. Returns None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"[scrapers] Failed to fetch {url}: {e}")
        return None


def extract_with_trafilatura(html: str) -> str | None:
    """Primary extraction method — strips boilerplate, keeps main content."""
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    return extracted


def extract_with_bs4_fallback(html: str) -> str:
    """Fallback extraction if trafilatura returns nothing useful."""
    soup = BeautifulSoup(html, "html.parser")

    # Strip elements that are almost never useful content
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Prefer <main> or <article> if present, else the whole body
    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return soup.get_text(separator="\n", strip=True)

    text = main.get_text(separator="\n", strip=True)
    return text


def is_discourse_url(url: str) -> bool:
    return DISCOURSE_TOPIC_RE.search(url) is not None


def is_nvidia_geforce_forum(url: str) -> bool:
    lowered = url.lower()
    return "nvidia.com" in lowered and "/geforce/forums/" in lowered


def _is_placeholder_content(text: str) -> bool:
    """Detect JavaScript-only shell pages that contain no real article content."""
    cleaned = text.strip()
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if any(marker in lowered for marker in PLACEHOLDER_MARKERS):
        return True
    if len(cleaned) < 300 and "loading" in lowered and "javascript" in lowered:
        return True
    return False


def _format_discourse_topic(payload: dict) -> str:
    """Convert a Discourse topic JSON payload into plain text."""
    parts = [f"Title: {payload.get('title', '').strip()}"]

    posts = payload.get("post_stream", {}).get("posts") or []
    for post in posts:
        body = (post.get("raw") or "").strip()
        if not body and post.get("cooked"):
            body = BeautifulSoup(post["cooked"], "html.parser").get_text("\n", strip=True)
        if not body:
            continue
        parts.append(
            f"\n[Post {post.get('post_number', '?')}] "
            f"@{post.get('username', 'unknown')}\n{body}"
        )

    return "\n".join(parts).strip()


def scrape_discourse_url(url: str) -> str | None:
    """Fetch a Discourse forum topic via its public .json API."""
    match = DISCOURSE_TOPIC_RE.search(url)
    if not match:
        return None

    base_url, topic_id = match.group(1), match.group(2)
    json_candidates = [
        f"{url.split('?')[0].rstrip('/')}.json",
        f"{base_url}/t/{topic_id}.json",
    ]

    payload = None
    for json_url in json_candidates:
        for params in (None, {"print": "true"}):
            payload = _get_json_with_retry(json_url, params=params)
            if payload and payload.get("post_stream", {}).get("posts"):
                break
        if payload and payload.get("post_stream", {}).get("posts"):
            break

    if not payload:
        print(f"[scrapers] Discourse JSON fetch failed for {url}")
        return None

    text = _format_discourse_topic(payload)
    if len(text.strip()) < 50:
        print(f"[scrapers] Discourse topic had too little text: {url}")
        return None

    print(f"[scrapers] Discourse OK for {url} ({len(text)} chars)")
    return text


def is_reddit_url(url: str) -> bool:
    return "reddit.com" in url.lower()


def extract_reddit_post_id(url: str) -> str | None:
    match = REDDIT_POST_RE.search(url)
    return match.group(1) if match else None


def _retry_wait_seconds(response: requests.Response, attempt: int, base_delay: float) -> float:
    for header in ("Retry-After", "X-RateLimit-Reset"):
        value = response.headers.get(header)
        if not value:
            continue
        try:
            return max(float(value), 1.0)
        except ValueError:
            continue
    return base_delay * (2 ** attempt)


def _get_json_with_retry(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 20,
    max_attempts: int = 4,
    base_delay: float = 2.0,
) -> dict | None:
    """GET JSON with exponential backoff when APIs return 429/503."""
    headers = headers or REDDIT_HEADERS
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code in {429, 503}:
                wait = _retry_wait_seconds(resp, attempt, base_delay)
                print(
                    f"[scrapers] {resp.status_code} from {url}, "
                    f"retrying in {wait:.1f}s (attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            last_error = exc
            # Don't retry client errors like 404/422 — try the next URL instead.
            if exc.response is not None and exc.response.status_code in {400, 404, 422}:
                return None
            if attempt < max_attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                time.sleep(base_delay * (2 ** attempt))

    print(f"[scrapers] Request failed for {url}: {last_error}")
    return None


def _reddit_cache_path(post_id: str) -> Path:
    return REDDIT_CACHE_DIR / f"{post_id}.txt"


def _load_reddit_cache(post_id: str) -> str | None:
    path = _reddit_cache_path(post_id)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if len(text) >= 50:
        print(f"[scrapers] Reddit cache hit for t3_{post_id} ({len(text)} chars)")
        return text
    return None


def _save_reddit_cache(post_id: str, text: str) -> None:
    REDDIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _reddit_cache_path(post_id).write_text(text, encoding="utf-8")


def _reddit_json_url(url: str) -> str:
    cleaned = url.split("?")[0].rstrip("/")
    if cleaned.endswith(".json"):
        return cleaned
    return f"{cleaned}.json"


def _format_reddit_post(post: dict, comments: list[dict], max_comments: int = 40) -> str:
    """Turn a Reddit submission + comments into plain text for indexing."""
    parts = [
        f"Title: {post.get('title', '').strip()}",
        f"Subreddit: r/{post.get('subreddit', 'unknown')}",
        f"Author: u/{post.get('author', 'unknown')}",
    ]

    selftext = (post.get("selftext") or "").strip()
    if selftext and selftext not in {"[removed]", "[deleted]"}:
        parts.append(f"\nPost:\n{selftext}")

    usable_comments = [
        c for c in comments
        if (c.get("body") or "").strip()
        and c.get("body") not in {"[removed]", "[deleted]"}
    ]
    usable_comments.sort(key=lambda c: c.get("score", 0), reverse=True)

    if usable_comments:
        parts.append("\n--- Top Comments ---")
        for i, comment in enumerate(usable_comments[:max_comments], start=1):
            parts.append(
                f"\n[Comment {i}] u/{comment.get('author', 'unknown')} "
                f"(score: {comment.get('score', 0)})"
            )
            parts.append(comment.get("body", "").strip())

    return "\n".join(parts).strip()


def _walk_reddit_comments(children: list) -> list[dict]:
    """Recursively collect comment objects from Reddit JSON listings."""
    comments = []
    for child in children:
        if child.get("kind") != "t1":
            continue
        data = child.get("data", {})
        comments.append(data)
        replies = data.get("replies")
        if isinstance(replies, dict):
            reply_children = replies.get("data", {}).get("children", [])
            comments.extend(_walk_reddit_comments(reply_children))
    return comments


def scrape_reddit_via_json(url: str, post_id: str) -> str | None:
    """Try Reddit's public JSON endpoint (works when Reddit allows the request)."""
    json_url = _reddit_json_url(url)
    try:
        resp = requests.get(json_url, headers=REDDIT_HEADERS, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
        print(f"[scrapers] Reddit JSON fetch failed for {url}: {e}")
        return None

    if not isinstance(payload, list) or not payload:
        return None

    try:
        post = payload[0]["data"]["children"][0]["data"]
        comment_children = payload[1]["data"]["children"] if len(payload) > 1 else []
        comments = _walk_reddit_comments(comment_children)
    except (KeyError, IndexError, TypeError) as e:
        print(f"[scrapers] Reddit JSON parse failed for {url}: {e}")
        return None

    text = _format_reddit_post(post, comments)
    return text if len(text.strip()) >= 50 else None


def scrape_reddit_via_arctic_shift(post_id: str) -> str | None:
    """Fetch Reddit post + comments from Arctic Shift archive API."""
    submission_payload = _get_json_with_retry(
        ARCTIC_SHIFT_POSTS_URL,
        params={"ids": post_id},
    )
    if not submission_payload:
        return None

    posts = submission_payload.get("data") or []
    if not posts:
        print(f"[scrapers] Arctic Shift returned no submission for {post_id}")
        return None

    time.sleep(0.5)

    comment_payload = _get_json_with_retry(
        ARCTIC_SHIFT_COMMENTS_URL,
        params={"link_id": post_id, "limit": 100},
    )
    comments = (comment_payload or {}).get("data") or []

    text = _format_reddit_post(posts[0], comments)
    return text if len(text.strip()) >= 50 else None


def scrape_reddit_via_pullpush(post_id: str) -> str | None:
    """
    Fetch Reddit post + comments from PullPush (public archive API).
    Used when Reddit blocks direct scraping.
    """
    link_id = f"t3_{post_id}"

    submission_payload = _get_json_with_retry(
        PULLPUSH_SUBMISSION_URL,
        params={"ids": link_id},
    )
    if not submission_payload:
        return None

    posts = submission_payload.get("data") or []
    if not posts:
        print(f"[scrapers] PullPush returned no submission for {post_id}")
        return None

    time.sleep(1.0)

    comment_payload = _get_json_with_retry(
        PULLPUSH_COMMENT_URL,
        params={"link_id": link_id, "size": 100},
    )
    comments = (comment_payload or {}).get("data") or []

    text = _format_reddit_post(posts[0], comments)
    return text if len(text.strip()) >= 50 else None


def scrape_reddit_url(url: str) -> str | None:
    """Extract Reddit post text and top comments."""
    post_id = extract_reddit_post_id(url)
    if not post_id:
        print(f"[scrapers] Could not parse Reddit post id from {url}")
        return None

    cached = _load_reddit_cache(post_id)
    if cached:
        return cached

    fetchers = (
        ("Reddit JSON", lambda: scrape_reddit_via_json(url, post_id)),
        ("Arctic Shift", lambda: scrape_reddit_via_arctic_shift(post_id)),
        ("PullPush", lambda: scrape_reddit_via_pullpush(post_id)),
    )

    for source_name, fetch in fetchers:
        text = fetch()
        if text:
            print(f"[scrapers] {source_name} OK for {url} ({len(text)} chars)")
            _save_reddit_cache(post_id, text)
            return text

    print(f"[scrapers] All Reddit extraction methods failed for {url}")
    return None


def scrape_url_text(url: str) -> str | None:
    """
    Fetch a URL and return extracted text without saving to disk.
    Returns None if fetching or extraction fails.
    Raises ScrapeError for known unsupported pages with a helpful message.
    """
    if is_reddit_url(url):
        text = scrape_reddit_url(url)
        if not text:
            raise ScrapeError(
                "Could not extract this Reddit thread. Wait a minute and try again, "
                "or copy the thread into a .txt file and upload it."
            )
        return text

    if is_discourse_url(url):
        text = scrape_discourse_url(url)
        if not text:
            raise ScrapeError(f"Could not load Discourse topic: {url}")
        return text

    if is_nvidia_geforce_forum(url):
        raise ScrapeError(
            "NVIDIA GeForce forums load content via JavaScript and cannot be scraped "
            "automatically. Copy the thread into a .txt file and upload it, or use a "
            "forums.developer.nvidia.com link if the same topic exists there."
        )

    html = fetch_page(url)
    if html is None:
        return None

    text = extract_with_trafilatura(html)
    if not text or len(text.strip()) < 200:
        text = extract_with_bs4_fallback(html)

    if not text or len(text.strip()) < 50 or _is_placeholder_content(text):
        if _is_placeholder_content(text or ""):
            raise ScrapeError(
                "This page requires JavaScript to display its content, so automatic "
                "extraction failed. Upload a .txt or PDF copy of the page instead."
            )
        print(f"[scrapers] Extraction produced little/no content for {url}")
        return None

    return text


def scrape_url(url: str, save_dir: str = "data/raw") -> str | None:
    """
    Fetch a URL, extract clean text, and save it to disk as a .txt file.
    Returns the path to the saved file, or None if scraping failed.
    """
    text = scrape_url_text(url)
    if text is None:
        return None

    os.makedirs(save_dir, exist_ok=True)
    filename = _url_to_filename(url)
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[scrapers] Saved {url} -> {filepath} ({len(text)} chars)")
    return filepath


def _url_to_filename(url: str) -> str:
    """Turn a URL into a safe filename."""
    cleaned = url.replace("https://", "").replace("http://", "")
    cleaned = cleaned.rstrip("/")
    cleaned = cleaned.replace("/", "_").replace(":", "_").replace("?", "_")
    if not cleaned.endswith(".txt"):
        cleaned += ".txt"
    return cleaned


def scrape_all(urls: list[str], save_dir: str = "/home/ronan/Insync/rbacinskiskippen18@gmail.com/OneDrive/L2 python/pc_doctor_ai/data/raw/", delay: float = 1.0) -> list[str]:
    """
    Scrape a list of URLs with a polite delay between requests.
    Returns list of saved file paths (skips failures).
    """
    saved_paths = []
    for url in urls:
        path = scrape_url(url, save_dir=save_dir)
        if path:
            saved_paths.append(path)
        time.sleep(delay)
    return saved_paths


# Example source list per category — edit/expand as needed.
SOURCE_URLS = {
    "windows_bsod": [
        "https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-code-reference2",
        "https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/advanced-troubleshooting-boot-problems",
    ],
    "gpu_drivers": [
        # Add NVIDIA support / DDU guide URLs here
    ],
    "linux_kernel": [
        "https://wiki.archlinux.org/title/NVIDIA",
        "https://wiki.archlinux.org/title/Kernel_panic",
    ],
}


if __name__ == "__main__":
    for category, urls in SOURCE_URLS.items():
        if not urls:
            continue
        print(f"\n=== Scraping category: {category} ===")
        scrape_all(urls, save_dir=f"data/raw/{category}")

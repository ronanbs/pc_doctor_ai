"""
scrapers.py
Fetches and extracts clean text content from web-based documentation pages
(Microsoft Learn, Arch Wiki, NVIDIA support pages, etc.) for ingestion into
the RAG knowledge base.

Uses `trafilatura` for main-content extraction (strips nav/ads/boilerplate)
with a BeautifulSoup fallback for pages trafilatura can't parse cleanly.
"""

import os
import time
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


def scrape_url(url: str, save_dir: str = "data/raw") -> str | None:
    """
    Fetch a URL, extract clean text, and save it to disk as a .txt file.
    Returns the path to the saved file, or None if scraping failed.
    """
    html = fetch_page(url)
    if html is None:
        return None

    text = extract_with_trafilatura(html)
    if not text or len(text.strip()) < 200:
        # trafilatura came back empty/too short — try the fallback
        text = extract_with_bs4_fallback(html)

    if not text or len(text.strip()) < 50:
        print(f"[scrapers] Extraction produced little/no content for {url}")
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


def scrape_all(urls: list[str], save_dir: str = "data/raw", delay: float = 1.0) -> list[str]:
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

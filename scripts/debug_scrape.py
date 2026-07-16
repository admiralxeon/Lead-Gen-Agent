"""One-off smoke test: does Website() actually load lingscars.com?
Run: python debug_scrape.py
"""

from scraper import Website
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

site = Website("https://lingscars.com")
print("ok:", site.ok)
print("error:", site.error)
print("title:", site.title)
print("text length:", len(site.text))
print("first 300 chars:", site.text[:300])

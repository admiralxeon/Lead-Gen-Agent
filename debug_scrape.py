"""One-off smoke test: does Website() actually load lingscars.com?
Run: python debug_scrape.py
"""

from scraper import Website

site = Website("https://lingscars.com")
print("ok:", site.ok)
print("error:", site.error)
print("title:", site.title)
print("text length:", len(site.text))
print("first 300 chars:", site.text[:300])

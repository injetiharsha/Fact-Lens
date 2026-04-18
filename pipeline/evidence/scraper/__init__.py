"""Tiered scraping adapters."""

from pipeline.evidence.scraper.trafilatura_scraper import TrafilaturaScraper
from pipeline.evidence.scraper.playwright_scraper import PlaywrightScraper
from pipeline.evidence.scraper.beautifulsoup_scraper import BeautifulSoupScraper

__all__ = ["TrafilaturaScraper", "PlaywrightScraper", "BeautifulSoupScraper"]


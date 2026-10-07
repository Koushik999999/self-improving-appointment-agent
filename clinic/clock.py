"""Fixed clock so relative dates ("next Tuesday") resolve the same way on every run."""
from datetime import date, datetime

TODAY = date(2026, 10, 7)  # a Wednesday
NOW = datetime(2026, 10, 7, 8, 0)

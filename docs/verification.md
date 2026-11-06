# Verification — 6 November 2026

Executed locally on Windows with Python 3.12.14.

- Full pipeline: 659318 source rows across nine tables.
- All 19 SQL quality checks passed; typed ingestion and primary-key checks also passed.
- Output row counts: {"fact_orders": 100000, "mart_daily_sales": 1096, "mart_customer_value": 25000, "mart_product_sales": 550}.
- `python -m unittest discover -s tests -v`: 3 tests passed, including four failure-recovery scenarios.
- Full run elapsed approximately 41 seconds on this machine.
- Source hashes and counts: [full-run-report.json](full-run-report.json).

GitHub Actions configuration is included; remote CI execution and full generator regeneration were not validated during local development. Full-dataset replay was not timed; replay and failure preservation were tested on the 200-order fixture.

# Data Boundary

The formal study uses licensed local market data that is not redistributed in
this repository. This directory contains only acquisition-interface
documentation and clearly labeled deterministic synthetic fixtures.

The frozen OOS figures in `results/` are derived research aggregates and must not
be interpreted as live performance.

Run `python scripts/generate_demo_data.py` to rebuild the public factor panel.
The generated dates, securities, factors, returns, industries, and market values
are fictitious. See `SCHEMA.md` and `DATA_MANIFEST.md` for the public contract and
rights classification.

After installing `requirements.txt`, run `python scripts/run_public_demo.py` for
a read-only Rank IC and quintile-return contract check. This command never runs
the protected formal OOS workflow.

`pytest -q` runs the public synthetic contract tests by default. The historical
`test_p*.py` modules validate licensed inputs and frozen private research
artifacts, so they are skipped in a clean public clone. Maintainers with the
complete private workspace may opt in explicitly with
`RUN_PRIVATE_RESEARCH_TESTS=1`; that flag does not make the private data public
or rerun the protected OOS workflow by itself.

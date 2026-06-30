# Package 3 — QA Plan

## Automated controls

The final guarded closure validates:

- branch `feat/streamlit-visual-intelligence-enhancement`;
- baseline commit `a60dcbd`;
- the exact restored 16-file Package 3 working state;
- three intentionally Git-ignored evidence CSV files by SHA-256;
- payload hashes and Unix line endings;
- Python compilation;
- Package 1–3 focused tests;
- 204-row matrix totals and status counts;
- Git whitespace;
- the exact final staged file list;
- no unexpected staged, unstaged, or untracked files.

## Browser-capture exception

Automated browser capture is not part of the final commit because all three
controlled local approaches failed for environment-specific reasons. No
screenshots are claimed.

The complete decision and failure evidence is documented in:

`docs/streamlit/PACKAGE_03_CAPTURE_EXCEPTION.md`

## Final acceptance rule

Package 3 is accepted as **VERIFIED WITH A DOCUMENTED QA EXCEPTION** when the
final tests pass, the exact file guard passes, the closure commit succeeds, and
the branch push succeeds.

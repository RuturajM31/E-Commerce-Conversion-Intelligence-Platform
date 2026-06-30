# Package 3 — Browser Capture QA Exception

## Decision

Package 3 is closed as **VERIFIED WITH A DOCUMENTED QA EXCEPTION**.

The exception applies only to automated browser screenshots. It does not waive
the source inventory, quarantine classification, matrix reconciliation,
compilation, focused tests, Git whitespace checks, or exact-file commit guard.

## Why the screenshot gate was waived

Three controlled browser-capture approaches failed on the local Mac:

1. **Playwright driver incompatibility**
   Playwright's bundled Node driver required a C++ runtime symbol unavailable
   in the installed macOS environment. Chromium installation and Playwright
   startup therefore could not complete.

2. **Chrome DOM-export mode timeout**
   Local Google Chrome opened against the temporary Streamlit server, but the
   DOM-export command did not exit within 120 seconds.

3. **Chrome screenshot mode timeout**
   Both the modern and fallback headless screenshot commands timed out before
   producing a valid page image.

Every failed attempt reported that repository files were not modified. The
one-go closure attempt also restored the exact Package 3 pre-commit state.

## Evidence retained

- Current visual components inventoried: **175**
- Quarantine items classified: **33**
- Package 3 AUDIT rows reconciled: **13**
- Streamlit application files changed by Package 3: **0**
- Final focused tests required before commit: **45**
- Browser screenshots produced: **0**
- Broken browser-capture scripts committed: **0**

## Acceptance authority

On 30 June 2026, after the repeated environment-specific capture failures, the
project owner instructed that Package 3 be finished in one controlled run.
Running the final closure installer accepts this narrow screenshot exception.

## Boundary

This exception does not declare the future feature packages visually accepted.
Each feature package must still receive its own functional and visual QA using
a method that works in the target environment.

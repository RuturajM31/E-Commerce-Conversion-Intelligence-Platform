# Streamlit Closure Sprint — Consolidated QA Plan

## One-go automated checks

1. Verify branch `feat/streamlit-visual-intelligence-enhancement` and commit
   `3f63f8b`.
2. Verify the exact clean Package 3 tree and preserve the existing stash.
3. Verify every payload file and SHA-256 hash.
4. Compile application, source, scripts, and tests.
5. Run the focused Package 1–3 and closure-intelligence tests.
6. Run the consolidated closure regression suite.
7. Validate the 204-row matrix and manifest terminal states.
8. Validate Streamlit Community Cloud coordinates, app-local dependencies, and
   configuration.
9. Run `git diff --check` and exact staged-file review.
10. Commit and push the enhancement branch.
11. Merge into `main` because the project owner explicitly requested complete
    one-go implementation closure.
12. Run post-merge regression tests and push `main`.
13. Confirm the existing stash is unchanged.

## Browser review exception

The Package 3 browser-capture exception remains in force on this Mac. The final
closure uses function tests, data-contract tests, source review, and the
readable review pack. It does not invent screenshots.

## Streamlit Community Cloud boundary

The automated closure prepares and merges the repository. GitHub authorization
and the final Deploy click are completed by the project owner in the Streamlit
Community Cloud interface. A missing public URL does not roll back validated
code.

## Completion gate

The automated result succeeds when the code and matrix are committed and
pushed, `main` contains the merge, the exact test and Git controls pass, and
the repository is ready for Streamlit Community Cloud deployment.
